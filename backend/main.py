"""
Main entry point for the financial document processing system.

This module:
- Defines the LangGraph workflow
- Wires the nodes with conditional edges
- Provides CLI interface for document processing
"""

import os
import sys
import warnings
import logging

# Suppress warnings that clutter the terminal UI
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Silence all loggers to prevent debug leakage in CLI UI
logging.basicConfig(level=logging.ERROR)
for logger_name in [
    "google_genai", "google_genai._api_client", "google", "urllib3",
    "httpx", "httpcore", "weaviate", "nebula3", "pydantic", "langchain", "langgraph",
    "backend", "backend.agents", "backend.extractor_tools", "backend.graph_tools", "backend.tools", "backend.main"
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
from typing import Dict, Any
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set API key for Gemini
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from .state import AgentState
from .agents import supervisor_node, extractor_node, sarvam_node, tool_node, handle_delegate_extraction
from .utils import validate_document_path, get_document_info
from .parallel_ui import show_final_summary

# Root logger level
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


# ============================================================================
# GRAPH ROUTING LOGIC
# ============================================================================

def supervisor_router(state: AgentState) -> str:
    """
    Routes from Supervisor node based on the last tool call.
    
    Routing logic:
    - If delegate_extraction was called -> go to 'delegate_handler' then 'extractor'
    - If update_records or ignore_request was called -> END
    - If any other tool was called -> go to 'tools'
    - If no tool call (should rarely happen) -> END
    
    Args:
        state: The current graph state
        
    Returns:
        The name of the next node to route to
    """
    messages = state.get("supervisor_messages", [])
    
    if not messages:
        logger.warning("No messages in state, ending graph")
        return END
    
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if not isinstance(last_message, AIMessage):
        logger.info("Last message is not from AI, ending graph")
        return END
    
    if not last_message.tool_calls:
        logger.info("No tool calls in last message, ending graph")
        return END
    
    # Get the tool name from the first tool call
    tool_name = last_message.tool_calls[0]["name"]
    logger.info(f"Routing based on tool call: {tool_name}")
    
    # Route based on tool name
    if tool_name == "delegate_extraction":
        return "delegate_handler"
    elif tool_name in ["update_records", "ignore_request"]:
        # These are terminal actions - execute them first, then end
        return "tools"
    else:
        # verify_extraction, check_fraud_vectors, inspect_past_records
        return "tools"


def tools_router(state: AgentState) -> str:
    """
    Routes from Tools node back to Supervisor or END.
    
    After tool execution, check if it was a terminal action.
    
    Args:
        state: The current graph state
        
    Returns:
        The name of the next node
    """
    messages = state.get("supervisor_messages", [])
    
    # Find the last AI message to check what tool was called
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_name = msg.tool_calls[0]["name"]
            
            if tool_name in ["update_records", "ignore_request"]:
                logger.info(f"Terminal action {tool_name} completed, ending graph")
                return END
            break
    
    return "supervisor"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_graph() -> StateGraph:
    """
    Create and compile the document processing graph.
    
    Graph structure:
    - Entry: supervisor
    - Supervisor can route to: delegate_handler, tools, or END
    - delegate_handler routes to: extractor
    - Extractor always routes back to: supervisor
    - Tools route to: supervisor or END (for terminal actions)
    
    Returns:
        Compiled LangGraph
    """
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("sarvam", sarvam_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("delegate_handler", handle_delegate_extraction)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "delegate_handler": "delegate_handler",
            "tools": "tools",
            END: END
        }
    )
    
    # Delegate handler routes based on active_agents
    def delegate_router(state: AgentState) -> list[str]:
        agents = state.get("active_agents", ["gemini", "sarvam"])
        routes = []
        if "gemini" in agents:
            routes.append("extractor")
        if "sarvam" in agents:
            routes.append("sarvam")
        return routes

    workflow.add_conditional_edges(
        "delegate_handler",
        delegate_router,
        ["extractor", "sarvam"]
    )
    
    # Extractor always routes back to supervisor
    workflow.add_edge("extractor", "supervisor")
    workflow.add_edge("sarvam", "supervisor")
    
    # Tools route conditionally (back to supervisor or END)
    workflow.add_conditional_edges(
        "tools",
        tools_router,
        {
            "supervisor": "supervisor",
            END: END
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    logger.info("Graph compiled successfully")
    return app


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_document(document_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Process a financial document through the graph.
    
    Args:
        document_path: Path to the document file
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with processing results
    """
    # Validate the document path
    is_valid, error = validate_document_path(document_path)
    if not is_valid:
        return {
            "status": "ERROR",
            "message": error,
            "document_path": document_path
        }
    
    # Get document info
    doc_info = get_document_info(document_path)
    logger.info(f"Processing document: {doc_info}")
    
    # Pre-flight Database Verification: BOTH databases must be active
    from .db.weaviate_client import get_weaviate_client
    from .db.nebula_client import get_nebula_client

    weaviate_ok = False
    nebula_ok = False
    try:
        w_client = get_weaviate_client()
        if not w_client.is_ready():
            w_client.connect()
        weaviate_ok = w_client.is_ready()
    except Exception:
        weaviate_ok = False

    try:
        n_client = get_nebula_client()
        if not n_client.is_connected():
            n_client.connect()
        nebula_ok = n_client.is_connected()
    except Exception:
        nebula_ok = False

    if not weaviate_ok or not nebula_ok:
        err_msg = (
            f"Pre-flight Database Check Failed: Cannot start document processing run because databases are offline!\n"
            f"• Weaviate (Vector DB): {'ONLINE' if weaviate_ok else 'OFFLINE'}\n"
            f"• NebulaGraph (Graph DB): {'ONLINE' if nebula_ok else 'OFFLINE'}\n"
            f"Both databases are strictly required before starting a run."
        )
        logger.error(err_msg)
        return {
            "status": "ERROR",
            "message": err_msg,
            "document_path": document_path
        }

    # Create the graph
    app = create_graph()
    
    # Initialize the state
    initial_state: AgentState = {
        "supervisor_messages": [],
        "extractor_messages": [],
        "document_path": str(Path(document_path).resolve()),
        "extracted_data": None,
        "doc_type": None,
        "fraud_risk_score": 0.0,
        "verification_feedback": None,
        "extractor_iterations": 0,
        "extraction_verified": False
    }
    
    # Run the graph
    try:
        if verbose:
            # Stream events for verbose output
            final_state = None
            for event in app.stream(initial_state, config={"recursion_limit": 150}):
                for node_name, node_state in event.items():
                    logger.info(f"Node '{node_name}' completed")
                    print(f"\n{'='*60}")
                    print(f"NODE: {node_name}")
                    print(f"{'='*60}")
                    if node_state.get("supervisor_messages"):
                        for msg in node_state["supervisor_messages"]:
                            print(f"  Message type: {type(msg).__name__}")
                            if hasattr(msg, 'content') and msg.content:
                                print(f"  Content: {str(msg.content)[:200]}...")
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                print(f"  Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
                    final_state = node_state
        else:
            # Run directly
            final_state = app.invoke(initial_state, config={"recursion_limit": 150})
        
        # Extract results
        extracted = None
        if final_state:
            extracted = final_state.get("extracted_data") or final_state.get("gemini_extraction") or final_state.get("sarvam_extraction")
            
        return {
            "status": "COMPLETED",
            "document_path": document_path,
            "doc_type": final_state.get("doc_type") if final_state else None,
            "extracted_data": extracted,
            "fraud_risk_score": final_state.get("fraud_risk_score", 0.0) if final_state else 0.0,
            "message_count": len(final_state.get("supervisor_messages", [])) if final_state else 0
        }
        
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        return {
            "status": "ERROR",
            "message": str(e),
            "document_path": document_path
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Financial Document Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backend.main --input invoice.pdf
  python -m backend.main --input receipt.jpg --verbose
  python -m backend.main --input statement.png --init-db

Supported file types:
  Images: PNG, JPEG, WEBP, HEIC, HEIF
  Documents: PDF, TXT
  Video: MP4, WEBM, MOV
  Audio: MP3, WAV, FLAC
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the financial document to process"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output showing graph execution"
    )
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database schemas before processing"
    )
    
    parser.add_argument(
        "--weaviate-host",
        type=str,
        default="localhost",
        help="Weaviate host (default: localhost)"
    )
    
    parser.add_argument(
        "--weaviate-port",
        type=int,
        default=8080,
        help="Weaviate HTTP port (default: 8080)"
    )
    
    parser.add_argument(
        "--nebula-host",
        type=str,
        default="localhost",
        help="NebulaGraph host (default: localhost)"
    )
    
    parser.add_argument(
        "--nebula-port",
        type=int,
        default=9669,
        help="NebulaGraph port (default: 9669)"
    )
    
    args = parser.parse_args()
    
    # Initialize databases if requested
    if args.init_db:
        print("Initializing databases...")
        try:
            from .db.weaviate_client import init_weaviate
            from .db.nebula_client import init_nebula
            
            print(f"  Connecting to Weaviate at {args.weaviate_host}:{args.weaviate_port}...")
            init_weaviate(
                host=args.weaviate_host,
                port=args.weaviate_port
            )
            print("  Weaviate schema created.")
            
            print(f"  Connecting to NebulaGraph at {args.nebula_host}:{args.nebula_port}...")
            init_nebula(
                host=args.nebula_host,
                port=args.nebula_port
            )
            print("  NebulaGraph schema created.")
            print("Database initialization complete.\n")
        except Exception as e:
            print(f"Database initialization failed: {e}")
            print("Continuing with document processing...\n")
    
    # Process the document
    result = process_document(args.input, verbose=args.verbose)
    
    # Render modern summary card
    show_final_summary(result)
    
    return 0 if result.get('status') == 'COMPLETED' else 1


if __name__ == "__main__":
    sys.exit(main())
