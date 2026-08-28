"""
AgentState TypedDict for LangGraph workflow.

Defines the shared state that persists across the graph execution,
containing all necessary information for the Supervisor-Worker pattern.
"""

from typing import TypedDict, Annotated, List, Optional
import operator
import time

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state for the financial document processing graph.
    
    This state is passed between nodes and accumulates information
    throughout the document processing workflow.
    """
    
    # The Supervisor's conversation history (decisions, tool calls, tool results)
    # Uses operator.add to accumulate messages across invocations
    supervisor_messages: Annotated[List[BaseMessage], operator.add]
    
    # The Gemini Worker's conversation history
    extractor_messages: Annotated[List[BaseMessage], operator.add]

    # The Sarvam Worker's conversation history
    sarvam_messages: Annotated[List[BaseMessage], operator.add]
    
    # The raw document path (image, PDF, video, or audio file)
    # Set at graph invocation time
    document_path: str
    
    # Extracted data from Gemini agent
    gemini_extraction: Optional[dict]
    
    # Extracted data from Sarvam agent
    sarvam_extraction: Optional[dict]
    
    # Combined/Final extracted data (for downstream tools)
    extracted_data: Optional[dict]
    
    # Classification (set by Supervisor via delegate_extraction tool)
    # Valid values: 'invoice', 'receipt', 'statement'
    doc_type: Optional[str]
    
    # List of agents currently active/delegated to
    # e.g., ["gemini", "sarvam"]
    active_agents: List[str]
    
    # Fraud score (written by check_fraud_vectors tool)
    # Range: 0.0 (no risk) to 1.0 (high risk)
    fraud_risk_score: float
    
    # Verification feedback from the last verify_extraction call
    # Passed to the Extractor on re-extraction attempts
    verification_feedback: Optional[str]

    # Counter for extractor tool iterations within a single extraction session
    # Resets when extraction is delegated again
    extractor_iterations: int
    
    # Flag indicating if the current extraction passed verification
    extraction_verified: bool


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# Active WebSocket queues for real-time live event streaming
_active_ws_queues = set()

def register_ws_queue(queue_pair):
    """Register (asyncio.Queue, asyncio.AbstractEventLoop) pair."""
    _active_ws_queues.add(queue_pair)

def unregister_ws_queue(queue_pair):
    """Unregister queue pair."""
    _active_ws_queues.discard(queue_pair)

def broadcast_live_event(event_data: dict):
    """Broadcast an event to all connected active WebSockets immediately."""
    for q, loop in list(_active_ws_queues):
        try:
            loop.call_soon_threadsafe(q.put_nowait, event_data)
        except Exception:
            pass

def serialize_message(msg, node: str) -> dict:
    """Helper to convert LangChain messages to frontend-compatible JSON objects."""
    role = "unknown"
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "agent"
    elif isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, ToolMessage):
        role = "function"
    
    content = msg.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        content = "\n".join(text_parts)
    elif not isinstance(content, str):
        content = str(content) if content is not None else ""
        
    tool_calls = getattr(msg, "tool_calls", []) or []
    additional = getattr(msg, "additional_kwargs", {}) or {}
    thought = additional.get("thought") or additional.get("reasoning_content") or ""
    
    return {
        "role": role,
        "content": content,
        "thought": thought,
        "tool_calls": tool_calls,
        "tool_name": getattr(msg, "name", None),
        "node": node,
        "type": type(msg).__name__,
        "timestamp": int(time.time() * 1000)
    }
