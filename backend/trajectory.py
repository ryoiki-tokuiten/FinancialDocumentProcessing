"""
Trajectory recorder and persistence module.
Stores complete token-by-token agent trajectories in JSONL format inside distillation/{trace_id}/
"""

import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

DISTILLATION_DIR = Path(__file__).parent.parent / "distillation"

def _clean_content(content: Any) -> Any:
    """Removes huge base64 image strings and keeps clean text and image paths."""
    if isinstance(content, str):
        if "data:image/" in content and ";base64," in content:
            return re.sub(r'data:image\/[a-zA-Z0-9+]+;base64,[A-Za-z0-9+/=]+', '[IMAGE_DATA_OMITTED]', content)
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    pass  # Omit base64 image_url
            elif isinstance(part, str):
                if not part.startswith("data:image"):
                    text_parts.append(part)
        if text_parts:
            return "\n\n".join(text_parts)
        return ""
    elif isinstance(content, dict):
        cleaned = {}
        for k, v in content.items():
            if k == "image_url" and isinstance(v, dict):
                url = v.get("url", "")
                if url.startswith("data:image"):
                    continue
            cleaned[k] = _clean_content(v)
        return cleaned
    return str(content) if content is not None else ""

def message_to_dict(msg: Any) -> dict:
    """Serialize any LangChain message to a clean dictionary without base64 or thought signatures."""
    if isinstance(msg, dict):
        role = msg.get("role", "unknown")
        msg_type = msg.get("type", "Message")
        thought = msg.get("thought", "")
        raw_kwargs = msg.get("additional_kwargs", {})
        cleaned_kwargs = {
            k: v for k, v in raw_kwargs.items()
            if k not in ("__gemini_function_call_thought_signatures__", "thought_signature", "thought_signatures", "thought_signatures_map", "signatures")
        } if isinstance(raw_kwargs, dict) else {}
        
        raw_meta = msg.get("response_metadata", {})
        cleaned_meta = {
            k: v for k, v in raw_meta.items()
            if "signature" not in k.lower()
        } if isinstance(raw_meta, dict) else {}
        
        return {
            "role": role,
            "type": msg_type,
            "content": _clean_content(msg.get("content", "")),
            "thought": thought,
            "name": msg.get("name"),
            "tool_call_id": msg.get("tool_call_id"),
            "tool_calls": msg.get("tool_calls"),
            "additional_kwargs": cleaned_kwargs,
            "response_metadata": cleaned_meta,
            "timestamp": msg.get("timestamp", int(time.time() * 1000))
        }
        
    role = "unknown"
    if isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, ToolMessage):
        role = "tool"
        
    thought = ""
    cleaned_kwargs = {}
    if hasattr(msg, "additional_kwargs") and isinstance(msg.additional_kwargs, dict):
        thought = msg.additional_kwargs.get("thought") or msg.additional_kwargs.get("reasoning_content") or ""
        for k, v in msg.additional_kwargs.items():
            if k in ("__gemini_function_call_thought_signatures__", "thought_signature", "thought_signatures", "thought_signatures_map", "signatures"):
                continue
            cleaned_kwargs[k] = v
            
    cleaned_meta = {}
    if hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
        for k, v in msg.response_metadata.items():
            if "signature" in k.lower():
                continue
            cleaned_meta[k] = v

    raw_content = getattr(msg, "content", "")
    content = _clean_content(raw_content)

    return {
        "role": role,
        "type": type(msg).__name__,
        "content": content,
        "thought": thought,
        "name": getattr(msg, "name", None),
        "tool_call_id": getattr(msg, "tool_call_id", None),
        "tool_calls": getattr(msg, "tool_calls", None),
        "additional_kwargs": cleaned_kwargs,
        "response_metadata": cleaned_meta,
        "timestamp": int(time.time() * 1000)
    }

def _dedupe_message_list(msgs: List[Any]) -> List[dict]:
    """Serializes and removes any identical duplicate consecutive messages."""
    deduped = []
    for msg in msgs:
        d = message_to_dict(msg)
        if not deduped:
            deduped.append(d)
        else:
            prev = deduped[-1]
            if (prev.get("type") == d.get("type") and
                prev.get("role") == d.get("role") and
                prev.get("content") == d.get("content") and
                prev.get("thought") == d.get("thought") and
                prev.get("tool_calls") == d.get("tool_calls") and
                prev.get("name") == d.get("name")):
                continue
            deduped.append(d)
    return deduped

def save_run_trajectory(
    trace_id: str,
    final_state: Dict[str, Any],
    document_path: str,
    status: str = "COMPLETED",
    weaviate_uuid: Optional[str] = None
) -> str:
    """
    Saves the full token-by-token trajectory of a successful run in distillation/{trace_id}/
    Containing:
    - SupervisorAgent.jsonl
    - ExtractorAgent.jsonl
    - SarvamDocAI.jsonl
    - metadata.json
    """
    DISTILLATION_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = DISTILLATION_DIR / trace_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. SupervisorAgent.jsonl
    supervisor_msgs = _dedupe_message_list(final_state.get("supervisor_messages", []))
    sup_file = run_dir / "SupervisorAgent.jsonl"
    with open(sup_file, "w", encoding="utf-8") as f:
        for d in supervisor_msgs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 2. ExtractorAgent.jsonl
    extractor_msgs = _dedupe_message_list(final_state.get("extractor_messages", []))
    ext_file = run_dir / "ExtractorAgent.jsonl"
    with open(ext_file, "w", encoding="utf-8") as f:
        for d in extractor_msgs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 3. SarvamDocAI.jsonl
    sarvam_msgs = _dedupe_message_list(final_state.get("sarvam_messages", []))
    sarvam_file = run_dir / "SarvamDocAI.jsonl"
    with open(sarvam_file, "w", encoding="utf-8") as f:
        for d in sarvam_msgs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 4. metadata.json
    filename = Path(document_path).name if document_path else "unknown"
    metadata = {
        "trace_id": trace_id,
        "weaviate_uuid": weaviate_uuid,
        "document_name": filename,
        "document_path": str(document_path),
        "document_url": f"/uploads/{filename}",
        "status": status,
        "doc_type": final_state.get("doc_type"),
        "fraud_risk_score": final_state.get("fraud_risk_score", 0.0),
        "extraction_verified": final_state.get("extraction_verified", False),
        "total_supervisor_steps": len(supervisor_msgs),
        "total_extractor_steps": len(extractor_msgs),
        "total_sarvam_steps": len(sarvam_msgs),
        "created_at": int(time.time() * 1000)
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return str(run_dir)

def load_run_trajectory(run_id_or_trace_id: str) -> Optional[dict]:
    """Load the full trajectory for a given run from distillation directory."""
    if not DISTILLATION_DIR.exists():
        return None

    # Check direct directory match first
    direct_path = DISTILLATION_DIR / run_id_or_trace_id
    if direct_path.is_dir():
        return _read_trajectory_dir(direct_path)

    # Search metadata files
    for run_dir in DISTILLATION_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        meta_file = run_dir / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("trace_id") == run_id_or_trace_id or meta.get("weaviate_uuid") == run_id_or_trace_id:
                    return _read_trajectory_dir(run_dir)
            except Exception:
                pass
    return None

def _read_trajectory_dir(run_dir: Path) -> dict:
    meta = {}
    if (run_dir / "metadata.json").exists():
        with open(run_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

    def read_jsonl(filename):
        msgs = []
        path = run_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            msgs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return msgs

    return {
        "metadata": meta,
        "supervisor_messages": _dedupe_message_list(read_jsonl("SupervisorAgent.jsonl")),
        "extractor_messages": _dedupe_message_list(read_jsonl("ExtractorAgent.jsonl")),
        "sarvam_messages": _dedupe_message_list(read_jsonl("SarvamDocAI.jsonl"))
    }

def delete_run_trajectory(trace_id_or_uuid: str) -> bool:
    """Deletes trajectory directory for a deleted run."""
    if not DISTILLATION_DIR.exists():
        return False

    direct_path = DISTILLATION_DIR / trace_id_or_uuid
    if direct_path.is_dir():
        shutil.rmtree(direct_path, ignore_errors=True)
        return True

    for run_dir in DISTILLATION_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        meta_file = run_dir / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("trace_id") == trace_id_or_uuid or meta.get("weaviate_uuid") == trace_id_or_uuid:
                    shutil.rmtree(run_dir, ignore_errors=True)
                    return True
            except Exception:
                pass
    return False
