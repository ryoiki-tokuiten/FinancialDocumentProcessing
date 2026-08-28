"""
Agent node implementations for the financial document processing graph.

Contains:
- supervisor_node: The Chief Financial Data Supervisor
- extractor_node: The document extraction worker
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage, 
    SystemMessage, 
    AIMessage,
    ToolMessage,
    BaseMessage
)

from .state import AgentState, broadcast_live_event, serialize_message
from .prompts import SUPERVISOR_PROMPT, get_extractor_prompt
from .tools import get_all_tools, set_state_context
from .utils import load_document

logger = logging.getLogger(__name__)


# ============================================================================
# SUPERVISOR NODE
# ============================================================================

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Runs the Supervisor LLM with tools bound.
    """
    from .parallel_ui import end_parallel_session, supervisor_thought, supervisor_action
    
    # Close any ongoing parallel extractor session before supervisor reasons
    end_parallel_session()
    
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.1,
        max_retries=3,
    )
    
    tools = get_all_tools()
    model_with_tools = model.bind_tools(tools)
    
    # Build the message history - ALWAYS start with System + Document
    messages: List[BaseMessage] = [SystemMessage(content=SUPERVISOR_PROMPT)]
    
    # ALWAYS add the document as the initial human message
    document_path = state.get("document_path", "")
    if document_path:
        try:
            from .tampering import inspect_document_tampering
            tampering_info = inspect_document_tampering(document_path)
            tampering_note = f"\n\nDOCUMENT METADATA & TAMPERING ANALYSIS:\n{json.dumps(tampering_info, indent=2)}" if tampering_info else ""
            
            document_content = load_document(document_path)
            messages.append(HumanMessage(content=[
                {"type": "text", "text": f"Please analyze this financial document and process it according to your protocol.{tampering_note}"},
                *document_content
            ]))
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
            messages.append(HumanMessage(content=f"Error loading document at {document_path}: {str(e)}"))
    
    # Get existing supervisor messages and add them
    existing_messages = state.get("supervisor_messages", [])
    new_messages_to_return = []
    
    if existing_messages:
        messages.extend(existing_messages)
        
        # Check if we need to insert the extraction result
        last_msg = existing_messages[-1]
        
        if (isinstance(last_msg, AIMessage) and 
            last_msg.tool_calls and 
            last_msg.tool_calls[0]['name'] == 'delegate_extraction'):
            
            gemini_data = state.get("gemini_extraction")
            sarvam_data = state.get("sarvam_extraction")
            
            content_parts = []
            if gemini_data:
                content_parts.append(f"Gemini Extractor returned:\n{json.dumps(gemini_data, indent=2)}")
            if sarvam_data:
                content_parts.append(f"Sarvam Extractor returned:\n{json.dumps(sarvam_data, indent=2)}")
            if not content_parts and state.get("extracted_data"):
                content_parts.append(f"Extractor returned:\n{json.dumps(state['extracted_data'], indent=2)}")
                
            if content_parts:
                logger.info("Injecting extraction data as ToolMessage for delegate_extraction")
                tool_call_id = last_msg.tool_calls[0]['id']
                tool_msg = ToolMessage(
                    content="Delegation complete. Extractor Agent(s) returned:\n\n" + "\n\n".join(content_parts),
                    tool_call_id=tool_call_id,
                    name="delegate_extraction"
                )
                messages.append(tool_msg)
                new_messages_to_return.append(tool_msg)
                     
    set_state_context(
        document_path=state.get("document_path", ""),
        fraud_risk_score=state.get("fraud_risk_score", 0.0)
    )
    
    sanitized_messages = _sanitize_messages_for_gemini(messages)
    
    logger.info("Invoking Supervisor model...")
    
    # Tool call enforcement: retry up to MAX_TOOL_CALL_RETRIES times if model doesn't make a tool call
    MAX_TOOL_CALL_RETRIES = 3
    retry_count = 0
    
    while retry_count < MAX_TOOL_CALL_RETRIES:
        try:
            response = model_with_tools.invoke(sanitized_messages)
        except Exception as e:
            logger.error(f"Model invocation failed: {e}")
            if new_messages_to_return:
                return {"supervisor_messages": new_messages_to_return}
            raise e
        
        # Check if a valid tool call was made
        if response.tool_calls and len(response.tool_calls) > 0:
            if response.content:
                supervisor_thought(str(response.content))
            for tc in response.tool_calls:
                supervisor_action(tc["name"], tc.get("args", {}))
            
            new_messages_to_return.append(response)
            broadcast_live_event({
                "type": "agent_message",
                "node": "supervisor",
                "message": serialize_message(response, "supervisor")
            })
            return {"supervisor_messages": new_messages_to_return}
        
        # No tool call made - silently inject error and retry
        retry_count += 1
        logger.debug(f"No tool call in response (attempt {retry_count}/{MAX_TOOL_CALL_RETRIES})")
        
        if retry_count < MAX_TOOL_CALL_RETRIES:
            # Add the failed response to history
            sanitized_messages.append(response)
            
            # Add feedback message asking for tool call
            feedback_msg = HumanMessage(content="""ERROR: Your response did not include a valid tool call.

As the Chief Financial Data Supervisor, you MUST call one of your available tools in EVERY response:
- delegate_extraction: For extracting data from documents
- verify_extraction: For validating mathematical consistency
- check_fraud_vectors: For checking duplicate/fraud patterns
- inspect_past_records: For looking up historical records
- update_records: For saving validated records
- ignore_request: For rejecting non-financial documents

Please analyze the document again and make an appropriate tool call. Do NOT respond with just text - you must call a tool.""")
            
            sanitized_messages.append(feedback_msg)
            logger.info("Injected tool call enforcement message, retrying...")
        else:
            # Max retries reached - return what we have with error
            logger.error("Max retries reached without valid tool call")
            new_messages_to_return.append(response)
            new_messages_to_return.append(AIMessage(content="[SYSTEM ERROR: Failed to get valid tool response after multiple attempts. Please try again.]"))
            return {"supervisor_messages": new_messages_to_return}
    
    # Fallback (shouldn't reach here)
    return {"supervisor_messages": new_messages_to_return}


def _sanitize_messages_for_gemini(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Sanitize messages for Gemini's strict requirements while preserving thought text."""
    sanitized = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else ""
            sanitized.append(AIMessage(content=content, tool_calls=msg.tool_calls, id=msg.id))
        else:
            sanitized.append(msg)
    return sanitized


# ============================================================================
# EXTRACTOR NODE - MULTI-TURN WITH PYTHON CODE EXECUTION
# ============================================================================

# Maximum iterations for the extractor tool loop
MAX_EXTRACTOR_ITERATIONS = 20


def extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    Runs the Extractor LLM for document parsing with Python code execution.
    
    NEW ARCHITECTURE:
    - Extractor has access to execute_python and Final_Extraction tools
    - Runs in a multi-turn loop until Final_Extraction returns PASS
    - The agent can analyze images using OpenCV, PIL, etc.
    - Libraries are auto-installed on-the-fly
    - History persists across supervisor re-invocations
    """
    import os
    from .extractor_tools import execute_python, Final_Extraction, run_verification
    
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.1,
        max_retries=3,
    )
    
    # Bind tools to the model
    tools = [execute_python, Final_Extraction]
    model_with_tools = model.bind_tools(tools)
    
    doc_type = state.get("doc_type", "invoice")
    try:
        extraction_prompt = get_extractor_prompt(doc_type)
    except ValueError:
        logger.warning(f"Unknown doc_type '{doc_type}', defaulting to invoice")
        extraction_prompt = get_extractor_prompt("invoice")
    
    # ========== BUILD MESSAGES ==========
    messages: List[BaseMessage] = [SystemMessage(content=extraction_prompt)]
    
    # Load the document
    document_path = state.get("document_path", "")
    if not document_path:
        logger.error("No document path provided")
        return {
            "extracted_data": {"error": "No document path provided"},
            "extractor_messages": [AIMessage(content="ERROR: No document path provided")],
            "extraction_verified": False
        }
    
    # Set document path in environment for execute_python tool
    os.environ["CURRENT_DOCUMENT_PATH"] = document_path
    
    try:
        document_content = load_document(document_path)
    except Exception as e:
        logger.error(f"Failed to load document: {e}")
        return {
            "extracted_data": {"error": str(e)},
            "extractor_messages": [AIMessage(content=f"ERROR: {e}")],
            "extraction_verified": False
        }
    
    # Build initial message with document PATH included
    instructions = _extract_latest_instructions(state.get("supervisor_messages", []))

    initial_text = f"""Document path: {document_path}

Please analyze this document and extract the required data.
{f'Supervisor instructions: {instructions}' if instructions else ''}

MANDATORY PROTOCOL:
In EVERY turn, you MUST first write your observation and reasoning text before invoking any tool. Explain what you are inspecting and what Python code you are executing.

Remember:
1. The Python environment is STATEFUL across turns (variables, imports, models persist).
2. Pass view_images=['crop.png'] in execute_python to inspect up to 5 visual crops as multi-modal vision input.
3. Pre-warmed forensic toolkit: forensics.deskew, forensics.ela, forensics.extract_table_grid, forensics.normalize_currency.
4. Pass reset_state=True if you need to clear all variables and reset environment.
5. FINALLY call Final_Extraction with your structured JSON.
"""
    
    messages.append(HumanMessage(content=[
        {"type": "text", "text": initial_text},
        *document_content
    ]))
    
    # Add existing extractor conversation history (for re-invocations)
    existing_messages = state.get("extractor_messages", [])
    new_messages_to_return = []
    
    if existing_messages:
        messages.extend(existing_messages)
        
        # If this is a continuation from supervisor feedback
        verification_feedback = state.get("verification_feedback")
        if verification_feedback:
            logger.info("Including supervisor feedback for re-extraction")
            feedback_msg = HumanMessage(content=f"""
The supervisor has requested corrections. Previous issues:

{verification_feedback}

Please re-analyze the document and fix these specific errors.
""")
            messages.append(feedback_msg)
            new_messages_to_return.append(feedback_msg)
    
    # ========== MULTI-TURN TOOL EXECUTION LOOP ==========
    iterations = state.get("extractor_iterations", 0)
    
    # Import parallel UI for Gemini logging
    from .parallel_ui import get_parallel_ui
    ui = get_parallel_ui()
    
    for iteration in range(MAX_EXTRACTOR_ITERATIONS):
        iterations += 1
        
        # Sanitize messages before invocation
        sanitized_messages = _sanitize_messages_for_gemini(messages)
        
        try:
            response = model_with_tools.invoke(sanitized_messages)
        except Exception as e:
            logger.error(f"Extractor model failed: {e}")
            ui.add_gemini_harness("gemini_api", success=False, error=str(e))
            ui.add_gemini_verification(False, [f"Model API error: {str(e)}"])
            return {
                "gemini_extraction": {"error": f"Extraction failed: {str(e)}"},
                "extractor_messages": new_messages_to_return + [AIMessage(content=f"Error: {e}")],
                "extractor_iterations": iterations,
                "extraction_verified": False
            }
        
        if response is None:
            response = AIMessage(content="")
        
        # Extract agent thoughts from content or metadata
        thought_content = ""
        if isinstance(response.content, str) and response.content.strip():
            thought_content = response.content.strip()
        elif isinstance(response.content, list):
            thought_parts = []
            for p in response.content:
                if isinstance(p, dict):
                    if p.get("type") in ["text", "thought", "reasoning"] and "text" in p:
                        thought_parts.append(p["text"])
                    elif "thought" in p:
                        thought_parts.append(str(p["thought"]))
                elif isinstance(p, str):
                    thought_parts.append(p)
            thought_content = "\n".join(thought_parts).strip()
            
        if not thought_content and hasattr(response, "additional_kwargs"):
            if "thought" in response.additional_kwargs:
                thought_content = str(response.additional_kwargs["thought"]).strip()
            elif "reasoning_content" in response.additional_kwargs:
                thought_content = str(response.additional_kwargs["reasoning_content"]).strip()
        
        # Extract reasoning thought from response or comments
        action_thought = thought_content
        if not action_thought and response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] == "execute_python":
                    code = tc.get("args", {}).get("code", "")
                    comments = [line.lstrip("# ").strip() for line in code.split("\n") if line.strip().startswith("#") and len(line.strip()) > 3]
                    if comments:
                        action_thought = "\n".join(comments[:4])
                    else:
                        action_thought = "Executing forensic Python script to crop regions, enhance image features, and analyze text."
                    break
                elif tc["name"] == "Final_Extraction":
                    action_thought = "Synthesized visual forensic inspection into structured financial data. Submitting for mathematical verification."
                    break

        if action_thought:
            if not response.additional_kwargs:
                response.additional_kwargs = {}
            response.additional_kwargs["thought"] = action_thought

        messages.append(response)
        new_messages_to_return.append(response)
        broadcast_live_event({
            "type": "agent_message",
            "node": "extractor",
            "message": serialize_message(response, "extractor")
        })
        
        # Check for tool calls
        if not response.tool_calls:
            if action_thought:
                ui.add_gemini_thought(action_thought)
            nudge_msg = HumanMessage(content="You MUST use a tool. Plain text responses are not allowed. Either use execute_python to analyze the document with code, OR call Final_Extraction to submit your extracted JSON. Do not respond with just text.")
            messages.append(nudge_msg)
            new_messages_to_return.append(nudge_msg)
            continue
        
        # Process each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call.get("id", f"call_{iteration}")
            
            # Ensure thoughts are displayed for this action
            if tool_name == "execute_python":
                code = tool_args.get("code", "")
                view_images = tool_args.get("view_images")
                reset_state = tool_args.get("reset_state", False)
                ui.add_gemini_thought(action_thought or "Executing forensic Python script.")
                ui.add_gemini_action("execute_python", code)
                
                try:
                    result = execute_python.invoke({
                        "code": code,
                        "view_images": view_images,
                        "reset_state": reset_state
                    })
                except Exception as e:
                    result = {"success": False, "error": str(e), "stdout": "", "stderr": "", "images": []}
                
                images = [img["filepath"] for img in result.get("images", []) if img.get("filepath")]
                ui.add_gemini_harness(
                    "execute_python",
                    stdout=result.get("stdout", ""),
                    images=images,
                    success=result.get("success", True),
                    error=result.get("error", "")
                )
                
                output_parts = []
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                error = result.get("error", "")
                
                if stdout:
                    output_parts.append(f"STDOUT:\n{stdout}")
                if stderr:
                    output_parts.append(f"STDERR:\n{stderr}")
                if error:
                    output_parts.append(f"ERROR:\n{error}")
                if result.get("installed_packages"):
                    output_parts.append(f"Installed packages: {', '.join(result['installed_packages'])}")
                if images:
                    output_parts.append("Generated/View Images:\n" + "\n".join([f"- file://{img_path}" for img_path in images]))
                
                if not output_parts:
                    output_parts.append("Code executed successfully (no output produced).")
                
                from .token_limiter import truncate_tool_output
                full_terminal_output = truncate_tool_output("\n\n".join(output_parts))
                
                # Include generated/view images (capped at max 5) in the tool result
                tool_content = []
                tool_content.append({"type": "text", "text": full_terminal_output})
                
                for img in result.get("images", [])[:5]:
                    tool_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{img['mime_type']};base64,{img['base64']}"}
                    })
                
                tool_msg = ToolMessage(
                    content=tool_content if len(tool_content) > 1 else tool_content[0]["text"],
                    tool_call_id=tool_id,
                    name=tool_name
                )
                messages.append(tool_msg)
                new_messages_to_return.append(tool_msg)
                broadcast_live_event({
                    "type": "agent_message",
                    "node": "extractor",
                    "message": serialize_message(tool_msg, "extractor")
                })
                
            elif tool_name == "Final_Extraction":
                # Run verification on the extracted data
                data = tool_args.get("data", {})
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        data = {}
                
                action_thought = thought_content
                if not action_thought:
                    action_thought = "Synthesized visual forensic inspection into structured financial data. Submitting for mathematical verification."
                ui.add_gemini_thought(action_thought)
                thought_content = ""
                
                ui.add_gemini_action("Final_Extraction", data)
                verification_result = run_verification(data)
                
                if verification_result["status"] == "PASS":
                    ui.add_gemini_verification(True)
                    
                    tool_msg = ToolMessage(
                        content=json.dumps({"status": "PASS", "message": "Extraction verified successfully!"}),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    messages.append(tool_msg)
                    new_messages_to_return.append(tool_msg)
                    broadcast_live_event({
                        "type": "agent_message",
                        "node": "extractor",
                        "message": serialize_message(tool_msg, "extractor")
                    })
                    broadcast_live_event({
                        "type": "data_update",
                        "data": data,
                        "node": "extractor"
                    })
                    
                    # Return successful extraction
                    return {
                        "gemini_extraction": data,
                        "extractor_messages": new_messages_to_return,
                        "extractor_iterations": iterations,
                        "extraction_verified": True,
                        "verification_feedback": None
                    }
                else:
                    ui.add_gemini_verification(False, verification_result.get("errors", []))
                    error_details = "\n".join([f"- {e}" for e in verification_result["errors"]])
                    
                    tool_msg = ToolMessage(
                        content=json.dumps({
                            "status": "FAIL",
                            "errors": verification_result["errors"],
                            "message": f"Verification failed. Please fix these errors and try again:\n{error_details}"
                        }),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    messages.append(tool_msg)
                    new_messages_to_return.append(tool_msg)
                    broadcast_live_event({
                        "type": "agent_message",
                        "node": "extractor",
                        "message": serialize_message(tool_msg, "extractor")
                    })
            else:
                tool_msg = ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    tool_call_id=tool_id,
                    name=tool_name
                )
                messages.append(tool_msg)
                new_messages_to_return.append(tool_msg)
    
    # Max iterations reached without successful extraction
    logger.warning(f"Max iterations ({MAX_EXTRACTOR_ITERATIONS}) reached without verified extraction")
    
    # Try to parse any JSON from the conversation
    parsed_data = None
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            parsed_data = _parse_json_response(msg.content if isinstance(msg.content, str) else str(msg.content))
            if parsed_data:
                break
    
    return {
        "gemini_extraction": parsed_data or {"error": "Max iterations reached without verified extraction"},
        "extractor_messages": new_messages_to_return,
        "extractor_iterations": iterations,
        "extraction_verified": False,
        "verification_feedback": "Max iterations reached - extraction may be incomplete"
    }



def _extract_latest_instructions(messages: List[BaseMessage]) -> Optional[str]:
    """Extract the latest instructions from supervisor tool calls."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get("name") == "delegate_extraction":
                    args = tool_call.get("args", {})
                    return args.get("instructions", "")
    return None


def _parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from response text."""
    if not text:
        return None
    
    # Try markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try finding raw JSON object
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    
    return None



# ============================================================================
# SUPERVISOR TOOL EXECUTION NODE
# ============================================================================

def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes tool calls requested by the Supervisor.
    """
    supervisor_messages = state.get("supervisor_messages", [])
    if not supervisor_messages:
        return {"supervisor_messages": []}
    
    last_message = supervisor_messages[-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"supervisor_messages": []}
    
    from .extractor_tools import execute_python
    from .authority import verify_authority
    from .token_limiter import truncate_tool_output
    from .parallel_ui import supervisor_harness
    from .tools import (
        delegate_extraction,
        verify_extraction,
        check_fraud_vectors,
        inspect_past_records,
        update_records,
        ignore_request
    )
    
    tool_map = {
        "delegate_extraction": delegate_extraction,
        "execute_python": execute_python,
        "verify_authority": verify_authority,
        "verify_extraction": verify_extraction,
        "check_fraud_vectors": check_fraud_vectors,
        "inspect_past_records": inspect_past_records,
        "update_records": update_records,
        "ignore_request": ignore_request
    }
    
    tool_messages = []
    reconciled_extracted_data = None
    fraud_risk_score = None
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call.get("id", "")
        
        logger.info(f"Executing tool: {tool_name}")
        
        if tool_name == "execute_python":
            code = tool_args.get("code", "")
            view_images = tool_args.get("view_images")
            reset_state = tool_args.get("reset_state", False)
            try:
                result = execute_python.invoke({
                    "code": code,
                    "view_images": view_images,
                    "reset_state": reset_state
                })
            except Exception as e:
                result = {"success": False, "error": str(e), "stdout": "", "stderr": "", "images": []}
            
            images = [img["filepath"] for img in result.get("images", []) if img.get("filepath")]
            supervisor_harness("execute_python", result.get("stdout", "") or "Code executed successfully", status="success" if result.get("success", True) else "error")

            output_parts = []
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            error = result.get("error", "")
            
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            if error:
                output_parts.append(f"ERROR:\n{error}")
            if result.get("installed_packages"):
                output_parts.append(f"Installed packages: {', '.join(result['installed_packages'])}")
            if images:
                output_parts.append("Generated/View Images:\n" + "\n".join([f"- file://{img_path}" for img_path in images]))
            
            if not output_parts:
                output_parts.append("Code executed successfully (no output produced).")
            
            full_terminal_output = truncate_tool_output("\n\n".join(output_parts))
            
            tool_content = []
            tool_content.append({"type": "text", "text": full_terminal_output})
            for img in result.get("images", [])[:5]:
                tool_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime_type']};base64,{img['base64']}"}
                })

            t_msg = ToolMessage(
                content=tool_content if len(tool_content) > 1 else tool_content[0]["text"],
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(t_msg)
            broadcast_live_event({
                "type": "agent_message",
                "node": "supervisor",
                "message": serialize_message(t_msg, "supervisor")
            })
            continue

        if tool_name in tool_map:
            tool_func = tool_map[tool_name]
            
            # Enforce check_fraud_vectors prerequisite before update_records
            if tool_name == "update_records":
                has_checked_fraud = False
                for m in state.get("supervisor_messages", []):
                    if isinstance(m, ToolMessage) and m.name == "check_fraud_vectors":
                        has_checked_fraud = True
                        break
                if not has_checked_fraud:
                    err_msg = "TOOL REJECTED: Security violation. You MUST call check_fraud_vectors before calling update_records. Run check_fraud_vectors first to inspect for duplicates and template fraud."
                    supervisor_harness(tool_name, err_msg, status="error")
                    tool_messages.append(ToolMessage(
                        content=err_msg,
                        tool_call_id=tool_id,
                        name=tool_name
                    ))
                    continue
            
            try:
                result = tool_func.invoke(tool_args)
                
                # Check fraud risk score
                if tool_name == "check_fraud_vectors":
                    try:
                        data_arg = tool_args.get("data")
                        if data_arg and isinstance(data_arg, dict):
                            reconciled_extracted_data = data_arg
                        res_obj = json.loads(result) if isinstance(result, str) else result
                        if isinstance(res_obj, dict):
                            fraud_risk_score = res_obj.get("risk_score", 0.0)
                    except Exception:
                        pass
                elif tool_name == "update_records":
                    try:
                        data_arg = tool_args.get("data")
                        if data_arg and isinstance(data_arg, dict):
                            reconciled_extracted_data = data_arg
                    except Exception:
                        pass
                
                is_err = str(result).startswith("FAIL") or str(result).startswith("Tool execution error")
                supervisor_harness(tool_name, result, status="error" if is_err else "success")
                
            except Exception as e:
                result = f"Tool execution error: {str(e)}"
                logger.error(f"Tool {tool_name} failed: {e}")
            t_msg = ToolMessage(
                content=result if isinstance(result, str) else json.dumps(result),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(t_msg)
            broadcast_live_event({
                "type": "agent_message",
                "node": "supervisor",
                "message": serialize_message(t_msg, "supervisor")
            })
        else:
            t_msg = ToolMessage(
                content=f"Unknown tool: {tool_name}",
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(t_msg)
            broadcast_live_event({
                "type": "agent_message",
                "node": "supervisor",
                "message": serialize_message(t_msg, "supervisor")
            })
    
    result = {
        "supervisor_messages": tool_messages,
        "fraud_risk_score": fraud_risk_score
    }
    
    if reconciled_extracted_data:
        result["extracted_data"] = reconciled_extracted_data
    
    return result


# ============================================================================
# DELEGATE EXTRACTION HANDLER
# ============================================================================

def handle_delegate_extraction(state: AgentState) -> Dict[str, Any]:
    """Handle the delegate_extraction tool call by updating state."""
    from .parallel_ui import start_parallel_session
    
    messages = state.get("supervisor_messages", [])
    if not messages:
        return {}
    
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "delegate_extraction":
                doc_type = tool_call["args"].get("doc_type", "invoice")
                target_agents = tool_call["args"].get("target_agents", ["gemini", "sarvam"])
                
                # Sanitize
                if not isinstance(target_agents, list):
                    target_agents = ["gemini", "sarvam"]
                
                # Start parallel extraction session
                start_parallel_session(state.get("document_path", ""))
                
                tool_msg = ToolMessage(
                    content=f"DELEGATION_SIGNAL: Extraction delegated to {target_agents} for {doc_type}.",
                    tool_call_id=tool_call.get("id", "call_del"),
                    name="delegate_extraction"
                )
                broadcast_live_event({
                    "type": "agent_message",
                    "node": "supervisor",
                    "message": serialize_message(tool_msg, "supervisor")
                })
                
                logger.info(f"Setting doc_type to: {doc_type}, active_agents={target_agents}")
                return {
                    "doc_type": doc_type,
                    "active_agents": target_agents
                }
    
    return {}


# ============================================================================
# SARVAM NODE (Doc AI Extraction)
# ============================================================================

SARVAM_DOC_AI_SCHEMA = {
    "type": "object",
    "description": "Extracted financial document data",
    "properties": {
        "vendor_name": {
            "type": "string",
            "description": "Name of the merchant, vendor, or store"
        },
        "vendor_address": {
            "type": "string",
            "description": "Address of the merchant or store"
        },
        "tax_id": {
            "type": "string",
            "description": "Tax identification number, GSTIN, PAN, VAT, or EIN"
        },
        "invoice_number": {
            "type": "string",
            "description": "Invoice number, receipt number, or bill number"
        },
        "invoice_date": {
            "type": "string",
            "description": "Date of the invoice or receipt in YYYY-MM-DD format"
        },
        "due_date": {
            "type": "string",
            "description": "Payment due date in YYYY-MM-DD format"
        },
        "currency_code": {
            "type": "string",
            "description": "Currency code such as INR, USD, EUR"
        },
        "subtotal": {
            "type": "number",
            "description": "Subtotal amount before tax or discounts"
        },
        "tax_amount": {
            "type": "number",
            "description": "Total tax amount"
        },
        "total_amount": {
            "type": "number",
            "description": "Total final amount due or paid"
        },
        "line_items": {
            "type": "array",
            "description": "List of individual items or products purchased",
            "items": {
                "type": "object",
                "description": "An individual purchased item",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Name or description of the product or item"
                    },
                    "quantity": {
                        "type": "number",
                        "description": "Quantity purchased"
                    },
                    "unit_price": {
                        "type": "number",
                        "description": "Price per single unit"
                    },
                    "row_total": {
                        "type": "number",
                        "description": "Total price for this line item"
                    },
                    "product_code": {
                        "type": "string",
                        "description": "Product code, SKU, or HSN code if present"
                    }
                }
            }
        }
    }
}


def sarvam_node(state: AgentState) -> Dict[str, Any]:
    """
    Runs the Sarvam Extractor using the official Sarvam Doc AI v1 API.
    """
    import os
    import time
    import requests
    from .extractor_tools import run_verification
    from .parallel_ui import get_parallel_ui
    
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        logger.error("SARVAM_API_KEY not found in environment")
        return {"sarvam_extraction": {"error": "SARVAM_API_KEY missing"}}

    document_path = state.get("document_path", "")
    if not document_path or not os.path.exists(document_path):
        return {"sarvam_extraction": {"error": f"Document not found at path: {document_path}"}}

    sarvam_ui = get_parallel_ui()
    sarvam_ui.add_sarvam_event("Submitting document to Sarvam Doc AI v1 Extract API...", title="[REQUEST: doc_ai]")
    broadcast_live_event({
        "type": "agent_message",
        "node": "sarvam",
        "message": serialize_message(AIMessage(content="Submitting document to Sarvam Doc AI v1 Extract API..."), "sarvam")
    })

    BASE_URL = "https://api.sarvam.ai/doc-ai/v1"
    headers = {"api-subscription-key": api_key}
    TERMINAL = {"completed", "partially_completed", "failed", "rejected"}
    file_name = os.path.basename(document_path)
    
    try:
        start_time = time.time()
        
        # 1. Submit extraction job with automatic retry on network glitches
        submit = None
        for attempt in range(1, 4):
            try:
                with open(document_path, "rb") as doc_file:
                    submit = requests.post(
                        f"{BASE_URL}/job/extract",
                        headers=headers,
                        files={"file": (file_name, doc_file)},
                        data={
                            "schema": json.dumps(SARVAM_DOC_AI_SCHEMA),
                            "language": "en-IN",
                            "output_format": "json",
                            "classification": "true",
                            "auto_orient": "true",
                        },
                        timeout=(15.0, 60.0),
                    )
                if submit.status_code in (200, 201):
                    break
                elif submit.status_code >= 500:
                    time.sleep(2 * attempt)
                    continue
                else:
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
                logger.warning(f"Sarvam connection retry {attempt}/3: {conn_err}")
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)
        
        if not submit or submit.status_code not in (200, 201):
            err_text = submit.text if submit else "No response from Sarvam"
            sarvam_ui.add_sarvam_event(f"Submit failed ({submit.status_code if submit else 'ERR'}):\n{err_text[:150]}", title="[SARVAM ERROR]", style="bold red")
            sarvam_ui.add_sarvam_verification(False, [f"Sarvam API error: HTTP {submit.status_code if submit else 'ERR'}"])
            return {"sarvam_extraction": {"error": f"Submit failed: {err_text}"}}
            
        job_data = submit.json()
        job_id = job_data.get("job_id")
        
        # Poll until terminal status
        max_wait = 300
        elapsed = 0
        poll_interval = 3
        status_data = {}
        
        while elapsed < max_wait:
            try:
                status_resp = requests.get(f"{BASE_URL}/job/{job_id}/status", headers=headers, timeout=15.0)
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    current_status = status_data.get("status", "").lower()
                    if current_status in TERMINAL:
                        break
            except requests.exceptions.RequestException:
                pass
            time.sleep(poll_interval)
            elapsed += poll_interval
            
        final_status = status_data.get("status", "").lower()
        if final_status not in {"completed", "partially_completed"}:
            sarvam_ui.add_sarvam_event(f"Job ended with status: {final_status or 'TIMEOUT'}", title="[SARVAM ERROR]", style="bold red")
            sarvam_ui.add_sarvam_verification(False, [f"Sarvam job ended with status: {final_status or 'TIMEOUT'}"])
            return {"sarvam_extraction": {"error": f"Sarvam Job {final_status or 'TIMEOUT'}"}}
            
        # Get results
        results_resp = requests.get(
            f"{BASE_URL}/job/{job_id}/results",
            headers=headers,
            params={"format": "json"},
            timeout=15.0,
        )
        
        if results_resp.status_code != 200:
            sarvam_ui.add_sarvam_event(f"Failed to fetch results:\n{results_resp.text[:150]}", title="[SARVAM ERROR]", style="bold red")
            sarvam_ui.add_sarvam_verification(False, [f"Failed to fetch results: HTTP {results_resp.status_code}"])
            return {"sarvam_extraction": {"error": f"Fetch results failed: {results_resp.text}"}}
            
        raw_result = results_resp.json().get("result", {})
        
        # Format to internal ExtractorPayload schema
        vendor_name = raw_result.get("vendor_name") or "Unknown Vendor"
        vendor_address = raw_result.get("vendor_address")
        tax_id = raw_result.get("tax_id")
        
        invoice_number = raw_result.get("invoice_number")
        invoice_date = raw_result.get("invoice_date")
        due_date = raw_result.get("due_date")
        currency_code = raw_result.get("currency_code") or "INR"
        
        total_amount = 0.0
        try:
            total_amount = float(raw_result.get("total_amount", 0.0) or 0.0)
        except (ValueError, TypeError):
            total_amount = 0.0
            
        subtotal = None
        if raw_result.get("subtotal") is not None:
            try:
                subtotal = float(raw_result.get("subtotal"))
            except (ValueError, TypeError):
                subtotal = None
                
        tax_amount = None
        if raw_result.get("tax_amount") is not None:
            try:
                tax_amount = float(raw_result.get("tax_amount"))
            except (ValueError, TypeError):
                tax_amount = None
                
        raw_line_items = raw_result.get("line_items", [])
        line_items = []
        if isinstance(raw_line_items, list):
            for item in raw_line_items:
                if isinstance(item, dict):
                    try:
                        row_tot = float(item.get("row_total", 0.0) or 0.0)
                    except (ValueError, TypeError):
                        row_tot = 0.0
                    try:
                        qty = float(item.get("quantity")) if item.get("quantity") is not None else None
                    except (ValueError, TypeError):
                        qty = None
                    try:
                        unit_p = float(item.get("unit_price")) if item.get("unit_price") is not None else None
                    except (ValueError, TypeError):
                        unit_p = None
                    line_items.append({
                        "description": item.get("description") or "Item",
                        "quantity": qty,
                        "unit_price": unit_p,
                        "row_total": row_tot,
                        "product_code": item.get("product_code")
                    })
                    
        line_items_sum = sum(item["row_total"] for item in line_items) if line_items else None
        
        extracted_data = {
            "vendor": {
                "raw_name": vendor_name,
                "address": vendor_address,
                "tax_id": tax_id
            },
            "invoice_details": {
                "invoice_number": invoice_number,
                "invoice_date": invoice_date,
                "due_date": due_date,
                "currency_code": currency_code
            },
            "financials": {
                "subtotal": subtotal,
                "tax_amount": tax_amount,
                "total_amount": total_amount,
                "line_items_sum": line_items_sum
            },
            "line_items": line_items
        }
        
        # Display extracted result and run verification
        sarvam_ui.add_sarvam_result(extracted_data, duration=time.time() - start_time)
        verification = run_verification(extracted_data)
        sarvam_ui.add_sarvam_verification(verification["status"] == "PASS", verification.get("errors", []))
        
        sarvam_msg = AIMessage(content=f"Sarvam Doc AI extraction completed.\n```json\n{json.dumps(extracted_data, indent=2)}\n```")
        broadcast_live_event({
            "type": "agent_message",
            "node": "sarvam",
            "message": serialize_message(sarvam_msg, "sarvam")
        })
        
        tool_v_msg = ToolMessage(
            content=json.dumps(verification),
            tool_call_id="sarvam_verify",
            name="run_verification"
        )
        broadcast_live_event({
            "type": "agent_message",
            "node": "sarvam",
            "message": serialize_message(tool_v_msg, "sarvam")
        })

        messages = [sarvam_msg, tool_v_msg]
        
        output_state = {
            "sarvam_extraction": extracted_data,
            "sarvam_messages": messages
        }
        
        return output_state
        
    except Exception as e:
        logger.error(f"Sarvam Doc AI error: {e}", exc_info=True)
        sarvam_ui.add_sarvam_event(f"Sarvam Exception:\n{str(e)[:200]}", title="[SARVAM ERROR]", style="bold red")
        sarvam_ui.add_sarvam_verification(False, [f"Sarvam error: {str(e)[:100]}"])
        return {"sarvam_extraction": {"error": f"Sarvam Doc AI error: {e}"}}
