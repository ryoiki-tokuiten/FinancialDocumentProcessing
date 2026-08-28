"""
FastAPI Server for Financial Document Processing & Knowledge Explorer.

Provides REST & WebSocket APIs for:
1. /api/runs - All processed financial document runs with metadata & thumbnails
2. /api/runs/{id} - Full parsed JSON payload, line items, audit trail & document image
3. DELETE /api/runs/{id} - Full deletion from Weaviate & NebulaGraph
4. /api/graph - Live NebulaGraph knowledge graph (nodes, edges, cross-vendor links)
5. /api/vectors - Weaviate vector collections, stats, and embeddings
6. /api/vectors/search - Semantic vector similarity search
7. /upload - Handle file uploads
8. /ws/process/{filename} - Live streaming WebSocket for Supervisor & Extractor agent activity
9. /uploads & /outputs - Static document image & crop serving
"""

import os
import shutil
import json
import logging
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db.weaviate_client import get_weaviate_client
from .db.nebula_client import get_nebula_client
from .main import create_graph
from .state import AgentState, register_ws_queue, unregister_ws_queue
from langchain_core.messages import ToolMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# Silence noisy uvicorn access logs (e.g. GET /outputs/... 200 OK)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

app = FastAPI(title="Financial Document Processing & Knowledge Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


def _get_image_url(source_file: str) -> Optional[str]:
    """Helper to generate web accessible image URL for a document file."""
    if not source_file:
        return None
    fname = os.path.basename(source_file)
    if (UPLOAD_DIR / fname).exists():
        return f"/uploads/{fname}"
    if (OUTPUT_DIR / fname).exists():
        return f"/outputs/{fname}"
    return f"/uploads/{fname}"


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle document file uploads."""
    try:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File uploaded: {file_path}")
        return {
            "filename": file.filename,
            "path": str(file_path.absolute()),
            "url": f"/uploads/{file.filename}"
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/process/{filename}")
async def websocket_endpoint(websocket: WebSocket, filename: str):
    """
    WebSocket endpoint for real-time agent processing.
    Streams events sequentially and instantly for both Supervisor and Extractor agents.
    """
    await websocket.accept()
    
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        await websocket.send_json({"type": "error", "message": f"File not found: {filename}"})
        await websocket.close()
        return

    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    pair = (queue, loop)
    register_ws_queue(pair)

    # Background task to send messages from queue to WebSocket
    async def ws_writer():
        try:
            while True:
                msg = await queue.get()
                await websocket.send_json(msg)
                queue.task_done()
        except Exception:
            pass

    writer_task = asyncio.create_task(ws_writer())

    try:
        # Pre-flight Database Verification: BOTH databases must be active before starting a run
        weaviate_ok = False
        nebula_ok = False
        try:
            w_client = get_weaviate_client()
            if not w_client.is_ready():
                w_client.connect()
            weaviate_ok = w_client.is_ready()
        except Exception as e:
            logger.error(f"Weaviate pre-flight check failed: {e}")

        try:
            n_client = get_nebula_client()
            if not n_client.is_connected():
                n_client.connect()
            nebula_ok = n_client.is_connected()
        except Exception as e:
            logger.error(f"NebulaGraph pre-flight check failed: {e}")

        if not weaviate_ok or not nebula_ok:
            status_text = (
                f"Cannot start document processing run because databases are offline!\n"
                f"• Weaviate (Vector DB): {'ONLINE' if weaviate_ok else 'OFFLINE'}\n"
                f"• NebulaGraph (Graph DB): {'ONLINE' if nebula_ok else 'OFFLINE'}\n\n"
                f"Both databases are strictly required for fraud detection, vector verification, and ledger ledger consistency. "
                f"Please ensure both services are running and retry."
            )
            logger.error(status_text)
            await websocket.send_json({
                "type": "error",
                "message": status_text,
                "database_status": {
                    "weaviate": weaviate_ok,
                    "nebulagraph": nebula_ok
                }
            })
            await websocket.close()
            return

        logger.info(f"Starting processing for {filename} (Databases: Weaviate=OK, NebulaGraph=OK)")
        graph_app = create_graph()
        
        initial_state: AgentState = {
            "supervisor_messages": [],
            "extractor_messages": [],
            "sarvam_messages": [],
            "document_path": str(file_path.absolute()),
            "gemini_extraction": None,
            "sarvam_extraction": None,
            "extracted_data": None,
            "doc_type": None,
            "active_agents": ["gemini", "sarvam"],
            "fraud_risk_score": 0.0,
            "verification_feedback": None,
            "extractor_iterations": 0,
            "extraction_verified": False
        }
        
        # Execute workflow in worker thread to prevent blocking event loop
        def run_sync_workflow():
            return graph_app.invoke(initial_state, config={"recursion_limit": 150})

        final_state = await asyncio.to_thread(run_sync_workflow)
        
        extracted = None
        if final_state:
            extracted = final_state.get("extracted_data") or final_state.get("gemini_extraction") or final_state.get("sarvam_extraction")
            
            # Save full token-by-token trajectory in distillation/
            try:
                trace_id = None
                weaviate_uuid = None
                status = "COMPLETED"
                
                for msg in final_state.get("supervisor_messages", []):
                    if isinstance(msg, ToolMessage):
                        if msg.name == "update_records":
                            try:
                                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                                if isinstance(data, dict):
                                    trace_id = data.get("trace_id")
                                    weaviate_uuid = data.get("weaviate_uuid")
                            except Exception:
                                pass
                        elif msg.name == "ignore_request":
                            status = "REJECTED"
                
                if not trace_id:
                    import uuid
                    trace_id = str(uuid.uuid4())
                    
                from .trajectory import save_run_trajectory
                save_run_trajectory(
                    trace_id=trace_id,
                    final_state=final_state,
                    document_path=str(file_path.absolute()),
                    status=status,
                    weaviate_uuid=weaviate_uuid
                )
            except Exception as e:
                logger.error(f"Error saving trajectory: {e}")
            
        await websocket.send_json({
            "type": "completed",
            "data": extracted,
            "fraud_risk_score": final_state.get("fraud_risk_score", 0.0) if final_state else 0.0,
            "status": "COMPLETED"
        })
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Processing error in websocket: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        unregister_ws_queue(pair)
        writer_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/api/runs")
async def get_all_runs():
    """Fetch all processed runs (both committed in Weaviate and rejected in distillation)."""
    runs = []
    seen_ids = set()
    
    # 1. Fetch committed records from Weaviate
    try:
        w = get_weaviate_client()
        if not w.is_ready():
            w.connect()
            
        col = w.client.collections.get(w.COLLECTION_NAME)
        res = col.query.fetch_objects(limit=100)
        
        for obj in res.objects:
            props = obj.properties or {}
            full_blob_str = props.get("full_json_blob")
            full_blob = json.loads(full_blob_str) if full_blob_str else {}
            
            content = full_blob.get("content", {})
            metadata = full_blob.get("metadata", {})
            audit = full_blob.get("audit_trail", {})
            
            source_file = metadata.get("source_file") or ""
            image_url = _get_image_url(source_file)
            
            inv_details = content.get("invoice_details", {})
            financials = content.get("financials", {})
            vendor = content.get("vendor", {})
            
            math_pass = audit.get("math_verified", True)
            fraud_score = audit.get("fraud_risk_score", 0.0)
            status = "VERIFIED" if math_pass and fraud_score < 0.4 else ("FLAGGED" if fraud_score >= 0.4 else "UNVERIFIED")
            
            trace_id = full_blob.get("trace_id", str(obj.uuid))
            seen_ids.add(str(obj.uuid))
            seen_ids.add(trace_id)
            
            runs.append({
                "id": str(obj.uuid),
                "trace_id": trace_id,
                "vendor_name": props.get("vendor_name") or vendor.get("raw_name") or "Unknown Vendor",
                "invoice_number": inv_details.get("invoice_number") or "-",
                "invoice_date": inv_details.get("invoice_date") or str(props.get("invoice_date") or "")[:10] or "-",
                "total_amount": props.get("total_amount") or financials.get("total_amount") or 0.0,
                "currency": inv_details.get("currency_code") or "INR",
                "source_file": source_file,
                "image_url": image_url,
                "received_at": metadata.get("received_at", ""),
                "math_verified": math_pass,
                "fraud_risk_score": fraud_score,
                "status": status,
                "line_items_count": len(content.get("line_items", []))
            })
    except Exception as e:
        logger.error(f"Error fetching runs from Weaviate: {e}")
        
    # 2. Fetch all runs from distillation/ (including REJECTED or uncommitted runs)
    try:
        from .trajectory import DISTILLATION_DIR
        if DISTILLATION_DIR.exists():
            for run_dir in DISTILLATION_DIR.iterdir():
                if not run_dir.is_dir():
                    continue
                meta_file = run_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        trace_id = meta.get("trace_id", run_dir.name)
                        weaviate_uuid = meta.get("weaviate_uuid")
                        
                        if trace_id not in seen_ids and (not weaviate_uuid or weaviate_uuid not in seen_ids):
                            seen_ids.add(trace_id)
                            source_file = meta.get("document_path", "")
                            status = meta.get("status", "REJECTED")
                            created_ts = meta.get("created_at", 0) / 1000.0 if meta.get("created_at") else time.time()
                            formatted_date = time.strftime("%Y-%m-%d", time.localtime(created_ts))
                            formatted_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_ts))
                            
                            runs.append({
                                "id": trace_id,
                                "trace_id": trace_id,
                                "vendor_name": f"[{status}] {meta.get('document_name', 'Document')}",
                                "invoice_number": "-",
                                "invoice_date": formatted_date,
                                "total_amount": 0.0,
                                "currency": "INR",
                                "source_file": source_file,
                                "image_url": _get_image_url(source_file),
                                "received_at": formatted_iso,
                                "math_verified": meta.get("extraction_verified", False),
                                "fraud_risk_score": meta.get("fraud_risk_score", 0.0),
                                "status": status,
                                "line_items_count": 0
                            })
                    except Exception as ex:
                        logger.debug(f"Error reading trajectory meta: {ex}")
    except Exception as e:
        logger.error(f"Error scanning distillation runs: {e}")
        
    return {"runs": runs, "total": len(runs)}


@app.get("/api/runs/{run_id}")
async def get_run_detail(run_id: str):
    """Fetch complete structured payload, line items, and audit info for a specific run."""
    try:
        # 1. Try Weaviate
        w = get_weaviate_client()
        if not w.is_ready():
            w.connect()
            
        col = w.client.collections.get(w.COLLECTION_NAME)
        
        obj = None
        try:
            obj = col.query.fetch_object_by_id(run_id)
        except Exception:
            pass
            
        if not obj:
            res = col.query.fetch_objects(limit=100)
            for item in res.objects:
                blob = json.loads(item.properties.get("full_json_blob", "{}"))
                if blob.get("trace_id") == run_id:
                    obj = item
                    break
                    
        if obj:
            props = obj.properties or {}
            full_blob = json.loads(props.get("full_json_blob", "{}"))
            source_file = full_blob.get("metadata", {}).get("source_file", "")
            
            return {
                "id": str(obj.uuid),
                "trace_id": full_blob.get("trace_id", str(obj.uuid)),
                "source_file": source_file,
                "image_url": _get_image_url(source_file),
                "metadata": full_blob.get("metadata", {}),
                "audit_trail": full_blob.get("audit_trail", {}),
                "content": full_blob.get("content", {}),
                "raw_payload": full_blob
            }
            
        # 2. Fallback to distillation directory (for REJECTED / uncommitted runs)
        from .trajectory import load_run_trajectory
        traj = load_run_trajectory(run_id)
        if traj:
            meta = traj.get("metadata", {})
            source_file = meta.get("document_path", "")
            created_ts = meta.get("created_at", 0) / 1000.0 if meta.get("created_at") else time.time()
            formatted_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_ts))
            
            return {
                "id": meta.get("trace_id", run_id),
                "trace_id": meta.get("trace_id", run_id),
                "source_file": source_file,
                "image_url": _get_image_url(source_file),
                "metadata": {
                    "source_file": source_file,
                    "received_at": formatted_iso,
                    "document_type": meta.get("doc_type", "document"),
                    "status": meta.get("status", "REJECTED")
                },
                "audit_trail": {
                    "math_verified": meta.get("extraction_verified", False),
                    "fraud_risk_score": meta.get("fraud_risk_score", 0.0),
                    "status": meta.get("status", "REJECTED")
                },
                "content": {},
                "raw_payload": meta
            }
            
        raise HTTPException(status_code=404, detail="Run not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching run detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a run completely from Weaviate and NebulaGraph."""
    try:
        w = get_weaviate_client()
        if not w.is_ready():
            w.connect()
        weaviate_deleted = w.delete_record(run_id)

        neb = get_nebula_client()
        if not neb._pool:
            neb.connect()
        nebula_deleted = neb.delete_invoice_and_edges(run_id)

        from .trajectory import delete_run_trajectory
        delete_run_trajectory(run_id)

        logger.info(f"Deleted run {run_id}: weaviate={weaviate_deleted}, nebula={nebula_deleted}")
        return {
            "status": "SUCCESS",
            "message": f"Run {run_id} deleted successfully",
            "weaviate_deleted": weaviate_deleted,
            "nebula_deleted": nebula_deleted
        }
    except Exception as e:
        logger.error(f"Error deleting run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete run: {str(e)}")


@app.get("/api/runs/{run_id}/trajectory")
async def get_run_trajectory_endpoint(run_id: str):
    """Fetch full token-by-token trajectory across all agents."""
    from .trajectory import load_run_trajectory
    traj = load_run_trajectory(run_id)
    if not traj:
        raise HTTPException(status_code=404, detail=f"Trajectory not found for run: {run_id}")
    return traj


@app.get("/api/graph")
async def get_knowledge_graph():
    """Query NebulaGraph knowledge graph to return nodes and edges for visualization."""
    try:
        client = get_nebula_client()
        if not client._pool:
            client.connect()
            
        client._use_space()
        
        nodes_dict: Dict[str, Dict[str, Any]] = {}
        edges_list: List[Dict[str, Any]] = []
        
        # 1. Fetch Vendors
        try:
            v_res = client._execute("MATCH (v:Vendor) RETURN id(v), v.name, v.tax_id, v.address LIMIT 100;")
            if v_res.is_succeeded():
                for i in range(v_res.row_size()):
                    r = v_res.row_values(i)
                    vid = str(r[0].as_string()).strip('"')
                    name = str(r[1].as_string()).strip('"') if not r[1].is_null() else vid
                    tax = str(r[2].as_string()).strip('"') if not r[2].is_null() else ""
                    addr = str(r[3].as_string()).strip('"') if not r[3].is_null() else ""
                    nodes_dict[vid] = {
                        "id": vid,
                        "label": name or vid,
                        "type": "Vendor",
                        "color": "#0169CC",
                        "properties": {"name": name, "tax_id": tax, "address": addr}
                    }
        except Exception as e:
            logger.debug(f"Vendor query failed: {e}")
            
        # 2. Fetch Invoices
        try:
            i_res = client._execute("MATCH (i:Invoice) RETURN id(i), i.invoice_number, i.total_amount, i.invoice_date LIMIT 100;")
            if i_res.is_succeeded():
                for i in range(i_res.row_size()):
                    r = i_res.row_values(i)
                    iid = str(r[0].as_string()).strip('"')
                    num = str(r[1].as_string()).strip('"') if not r[1].is_null() else "-"
                    tot = float(r[2].as_double()) if not r[2].is_null() else 0.0
                    dt = str(r[3].as_string()).strip('"') if not r[3].is_null() else "-"
                    nodes_dict[iid] = {
                        "id": iid,
                        "label": f"Inv #{num}" if num != "-" else iid[:14],
                        "type": "Invoice",
                        "color": "#7c3aed",
                        "properties": {"invoice_number": num, "total_amount": tot, "date": dt}
                    }
        except Exception as e:
            logger.debug(f"Invoice query failed: {e}")
            
        # 3. Fetch References / Projects / Persons
        for tag, color in [("Reference", "#d97706"), ("Project", "#059669"), ("Person", "#db2777"), ("Department", "#4f46e5")]:
            try:
                e_res = client._execute(f"MATCH (e:{tag}) RETURN id(e), e.name LIMIT 50;")
                if e_res.is_succeeded():
                    for i in range(e_res.row_size()):
                        r = e_res.row_values(i)
                        eid = str(r[0].as_string()).strip('"')
                        name = str(r[1].as_string()).strip('"') if not r[1].is_null() else eid
                        nodes_dict[eid] = {
                            "id": eid,
                            "label": name or eid,
                            "type": tag,
                            "color": color,
                            "properties": {"name": name}
                        }
            except Exception:
                pass
                
        # 4. Fetch All Edges
        try:
            r_res = client._execute("MATCH (src)-[e]->(dst) RETURN id(src), type(e), id(dst) LIMIT 200;")
            if r_res.is_succeeded():
                for i in range(r_res.row_size()):
                    r = r_res.row_values(i)
                    src = str(r[0].as_string()).strip('"')
                    etype = str(r[1].as_string()).strip('"')
                    dst = str(r[2].as_string()).strip('"')
                    
                    if src in nodes_dict and dst in nodes_dict:
                        edges_list.append({
                            "id": f"{src}_{etype}_{dst}_{i}",
                            "source": src,
                            "target": dst,
                            "label": etype
                        })
        except Exception as e:
            logger.debug(f"Edge query failed: {e}")
            
        return {
            "nodes": list(nodes_dict.values()),
            "edges": edges_list,
            "total_nodes": len(nodes_dict),
            "total_edges": len(edges_list)
        }
    except Exception as e:
        logger.error(f"Error fetching knowledge graph: {e}")
        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0, "error": str(e)}


@app.get("/api/vectors")
async def get_vectors_overview():
    """Fetch Weaviate vector collection stats, schema, and sample vectorized records."""
    try:
        w = get_weaviate_client()
        if not w.is_ready():
            w.connect()
            
        col = w.client.collections.get(w.COLLECTION_NAME)
        count = col.aggregate.over_all(total_count=True).total_count
        res = col.query.fetch_objects(limit=15, include_vector=True)
        
        records = []
        for obj in res.objects:
            props = obj.properties or {}
            vector_preview = []
            if hasattr(obj, "vector") and obj.vector:
                vec = obj.vector if isinstance(obj.vector, list) else obj.vector.get("default", [])
                vector_preview = [round(float(x), 4) for x in vec[:8]]
                
            records.append({
                "id": str(obj.uuid),
                "summary_text": props.get("summary_text", ""),
                "vendor_name": props.get("vendor_name", ""),
                "total_amount": props.get("total_amount", 0.0),
                "line_item_fingerprint": str(props.get("line_item_fingerprint") or ""),
                "vector_dimensions": 384,
                "vector_preview": vector_preview
            })
            
        return {
            "collection": w.COLLECTION_NAME,
            "total_objects": count,
            "vector_dimensions": 384,
            "model": "sentence-transformers-multi-qa-MiniLM-L6-cos-v1",
            "distance_metric": "cosine",
            "records": records
        }
    except Exception as e:
        logger.error(f"Error fetching vector overview: {e}")
        return {"collection": "FinancialRecord", "total_objects": 0, "records": [], "error": str(e)}


class SearchQuery(BaseModel):
    query: str
    limit: int = 5


@app.post("/api/vectors/search")
async def search_vectors(query_req: SearchQuery):
    """Execute semantic vector similarity search directly against Weaviate embeddings."""
    try:
        w = get_weaviate_client()
        if not w.is_ready():
            w.connect()
            
        col = w.client.collections.get(w.COLLECTION_NAME)
        search_res = col.query.near_text(
            query=query_req.query,
            limit=query_req.limit,
            return_metadata=["certainty", "distance"]
        )
        
        results = []
        for obj in search_res.objects:
            props = obj.properties or {}
            score = obj.metadata.certainty if hasattr(obj.metadata, "certainty") and obj.metadata.certainty else 0.0
            dist = obj.metadata.distance if hasattr(obj.metadata, "distance") and obj.metadata.distance else 0.0
            
            results.append({
                "id": str(obj.uuid),
                "vendor_name": props.get("vendor_name", ""),
                "total_amount": props.get("total_amount", 0.0),
                "summary_text": props.get("summary_text", ""),
                "invoice_date": str(props.get("invoice_date") or ""),
                "similarity_score": round(score, 4),
                "distance": round(dist, 4)
            })
            
        return {
            "query": query_req.query,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return {"query": query_req.query, "total_results": 0, "results": [], "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, access_log=False)
