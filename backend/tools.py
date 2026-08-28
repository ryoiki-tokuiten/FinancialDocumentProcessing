"""
Tool definitions for the financial document processing system.

Contains all 6 tools bound to the Supervisor agent:
1. delegate_extraction - Routes to the Extractor agent
2. verify_extraction - Math validation
3. check_fraud_vectors - Weaviate similarity search
4. inspect_past_records - Fetch full records by ID
5. update_records - Persist final record
6. ignore_request - Reject document
"""

import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool

from .schemas import TransientSearchRecord, FinalEnrichedRecord
from .db.weaviate_client import get_weaviate_client
from .db.nebula_client import get_nebula_client

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL 1: DELEGATE_EXTRACTION
# Routes to the Extractor agent for document parsing
# ============================================================================

@tool
def delegate_extraction(doc_type: str, instructions: str, target_agents: List[str] = ["gemini", "sarvam"]) -> str:
    """
    Delegates the task of text extraction to specialist worker agents.
    
    Can route to:
    - "gemini": Visual understanding + Python code execution (Best for layout analysis)
    - "sarvam": Sarvam AI Document Intelligence (Best for OCR and table structure)
    
    Args:
        doc_type: The document type - 'invoice', 'receipt', or 'statement'
        instructions: Feedback or initial instructions for the extractors.
        target_agents: List of agents to spawn. Defaults to ["gemini", "sarvam"].
                       Pass ["gemini"] or ["sarvam"] to target specific ones.
    
    Returns:
        A signal string indicating successful delegation.
    """
    valid_types = {"invoice", "receipt", "statement", "financial_statement"}
    
    if doc_type.lower() not in valid_types:
        return f"ERROR: Invalid doc_type '{doc_type}'. Must be one of: {valid_types}"
    
    # Validate target agents
    valid_agents = {"gemini", "sarvam"}
    sanitized_agents = [a.lower() for a in target_agents if a.lower() in valid_agents]
    if not sanitized_agents:
        sanitized_agents = ["gemini", "sarvam"]
    
    logger.info(f"Delegating extraction: type={doc_type}, agents={sanitized_agents}, instructions={instructions[:50]}...")
    
    # This return value is what the Supervisor sees in its tool message
    return f"DELEGATION_SIGNAL: Extraction delegated to {sanitized_agents} for {doc_type}. Instructions: {instructions}"


# ============================================================================
# TOOL 2: VERIFY_EXTRACTION
# Mathematical validation of extracted data
# ============================================================================

@tool
def verify_extraction(data: Dict[str, Any]) -> str:
    """
    Validates mathematical consistency of the extracted data.
    
    Performs the following checks:
    1. Sum of line item row_totals == line_items_sum (if provided)
    2. subtotal + tax_amount == total_amount (with tolerance)
    3. All required fields are present
    
    Args:
        data: The extracted JSON data from the Extractor agent.
              Should match the ExtractorPayload schema.
    
    Returns:
        "PASS" if all validations succeed, or 
        "FAIL: <detailed error message>" if any check fails.
    """
    result = run_verification(data)
    if result["status"] == "PASS":
        return "PASS"
    return "FAIL: " + "; ".join(result["errors"])


# ============================================================================
# TOOL 3: CHECK_FRAUD_VECTORS
# Weaviate similarity search for fraud detection
# ============================================================================

@tool
def check_fraud_vectors(data: Dict[str, Any]) -> str:
    """
    Checks Weaviate for duplicates/fraud using vector similarity and composite keys.
    Also queries NebulaGraph for entity-based fraud patterns (Graph RAG).
    
    Implements a strict "Composite Key" check for Template Fraud:
    1. Vendor (Must match)
    2. Total Amount (Must match)
    3. Line Item Fingerprint (Must match)
    
    Additionally checks the knowledge graph for:
    - Cross-vendor collusion (same Person/Project across different vendors)
    - Ghost project detection (projects with suspicious patterns)
    
    If all 3 match:
    - Same Invoice #: DUPLICATE (Reject)
    - Different Invoice #: TEMPLATE FRAUD (Flag)
    
    Args:
        data: The extracted JSON data (ExtractorPayload format)
    
    Returns:
        JSON string containing list of similar records with:
        - id: Record UUID
        - score: Similarity score (0.0 to 1.0)
        - risk_level: LOW, HIGH, or CRITICAL
        - fraud_reason: Specific reason for the flag
        - graph_insights: Entity-based patterns from knowledge graph
    """
    try:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass

        # 0. STRICT DATABASE ENFORCEMENT - ZERO SILENT FALLBACKS
        w_client = get_weaviate_client()
        if not w_client.is_ready():
            try:
                w_client.connect()
            except Exception as e:
                logger.error(f"Weaviate connect error: {e}")
        
        if not w_client.is_ready():
            return json.dumps({
                "status": "DATABASE_ERROR",
                "risk_level": "DATABASE_OFFLINE",
                "error": "Weaviate Vector Database is offline or not configured properly.",
                "message": "CRITICAL: Vector Database is not running. Fraud vector verification cannot proceed safely. Please ignore/reject request.",
                "similar_records": [],
                "graph_insights": []
            })

        n_client = get_nebula_client()
        if not n_client.is_connected():
            try:
                n_client.connect()
            except Exception as e:
                logger.error(f"NebulaGraph connect error: {e}")

        if not n_client.is_connected():
            return json.dumps({
                "status": "DATABASE_ERROR",
                "risk_level": "DATABASE_OFFLINE",
                "error": "NebulaGraph Database is offline or not configured properly.",
                "message": "CRITICAL: Knowledge Graph Database is not running. Graph RAG fraud inspection cannot proceed safely. Please ignore/reject request.",
                "similar_records": [],
                "graph_insights": []
            })

        # Build transient search record from payload
        transient = TransientSearchRecord.from_payload(data)
        
        # 1. Broad Vector Search (Recall phase)
        similar_records = w_client.query_similar(transient, limit=6)
        
        current_fingerprint = str(transient.line_item_fingerprint) if transient.line_item_fingerprint else ""
        current_amount = transient.total_amount
        current_inv_num = str(data.get("invoice_details", {}).get("invoice_number", "")).strip()
        current_vendor = str(transient.vendor_name or "").strip()
        
        if not similar_records:
            return json.dumps({
                "status": "CLEAR",
                "risk_level": "LOW",
                "message": "No similar records found in vector database.",
                "similar_records": [],
                "graph_insights": []
            })
            
        # 2. Comprehensive Precision & Duplicate Matching
        flagged_records = []
        highest_risk = "LOW"
        risk_message = "Found similar records in ledger"
        
        for record in similar_records:
            rec_fingerprint = str(record.get("line_item_fingerprint", "")).strip()
            rec_amount = record.get("total_amount")
            rec_inv_num = str(record.get("invoice_number", "")).strip()
            rec_vendor = str(record.get("vendor_name", "")).strip()
            rec_uuid = record.get("id")
            rec_score = record.get("score", 0.0)
            
            amount_match = False
            if rec_amount is not None and current_amount is not None:
                try:
                    amount_match = abs(float(rec_amount) - float(current_amount)) < 0.05
                except (ValueError, TypeError):
                    amount_match = False
            
            inv_num_match = bool(current_inv_num and rec_inv_num and current_inv_num.lower() == rec_inv_num.lower())
            fingerprint_match = bool(rec_fingerprint and current_fingerprint and rec_fingerprint == current_fingerprint)
            vendor_match = bool(
                current_vendor and rec_vendor and (
                    current_vendor.lower() in rec_vendor.lower() or rec_vendor.lower() in current_vendor.lower()
                )
            )

            # RULE 1: EXACT DUPLICATE INVOICE NUMBER (CRITICAL RISK)
            # If the invoice number matches an existing record in the database:
            if inv_num_match:
                record["fraud_type"] = "DUPLICATE"
                record["reason"] = (
                    f"CRITICAL DUPLICATE: Invoice #{rec_inv_num} is already registered in the ledger! "
                    f"(Record UUID: {rec_uuid}, Existing Vendor: '{rec_vendor}', Amount: ₹{rec_amount})"
                )
                highest_risk = "CRITICAL"
                risk_message = f"CONFIRMED DUPLICATE DETECTED: Invoice #{rec_inv_num} is already in database"
                flagged_records.append(record)
                continue

            # RULE 2: TEMPLATE FRAUD / AMOUNT + FINGERPRINT / EXACT LINE ITEMS (HIGH RISK)
            if amount_match and (fingerprint_match or vendor_match or rec_score >= 0.75):
                record["fraud_type"] = "TEMPLATE_FRAUD"
                record["reason"] = (
                    f"SUSPECTED TEMPLATE FRAUD: Identical financial amount (₹{current_amount}) and items "
                    f"matching past record {rec_uuid} under different invoice #{rec_inv_num or 'Unknown'}"
                )
                if highest_risk != "CRITICAL":
                    highest_risk = "HIGH"
                    risk_message = "SUSPECTED TEMPLATE FRAUD DETECTED"
                flagged_records.append(record)
                continue

            # RULE 3: HIGH VECTOR SIMILARITY (Score >= 0.85)
            if rec_score >= 0.85:
                record["fraud_type"] = "HIGH_SEMANTIC_SIMILARITY"
                record["reason"] = f"High vector semantic similarity ({rec_score:.2f}) with record {rec_uuid}"
                if highest_risk == "LOW":
                    highest_risk = "MEDIUM"
                flagged_records.append(record)
        
        # 3. Graph RAG: Query knowledge graph for entity-based fraud patterns
        graph_insights = []
        try:
            from .graph_tools import query_graph_insights
            
            for record in similar_records[:3]:
                rec_invoice_id = record.get("invoice_id") or f"invoice_{str(record.get('id', ''))[:16]}"
                insights = query_graph_insights(rec_invoice_id)
                
                if insights.get("cross_vendor_links"):
                    for link in insights["cross_vendor_links"]:
                        graph_insights.append({
                            "type": "CROSS_VENDOR_ENTITY",
                            "description": f"Shared entity '{link['shared_entity']}' found across {link.get('total_related_invoices', 0)} other invoices",
                            "entity_type": link.get("entity_type", "unknown"),
                            "severity": "MEDIUM"
                        })
                
                if insights.get("suspicious_patterns"):
                    for pattern in insights["suspicious_patterns"]:
                        graph_insights.append(pattern)
                        if pattern.get("severity") in ["HIGH", "CRITICAL"] and highest_risk != "CRITICAL":
                            highest_risk = "HIGH"
                            risk_message = f"Graph pattern detected: {pattern.get('type')}"
                            
        except Exception as graph_error:
            logger.error(f"Graph insights query failed: {graph_error}")
            return json.dumps({
                "status": "DATABASE_ERROR",
                "risk_level": "DATABASE_OFFLINE",
                "error": f"NebulaGraph query error: {graph_error}",
                "message": "CRITICAL: Graph Database error during Graph RAG check. Cannot proceed safely.",
                "similar_records": [],
                "graph_insights": []
            })
        
        # If we found strict matches, return those. Otherwise return general similar ones.
        return_records = flagged_records if flagged_records else similar_records
        
        logger.info(f"Fraud check: {risk_message} (Risk: {highest_risk})")
        
        return json.dumps({
            "status": "REVIEW_REQUIRED" if highest_risk != "LOW" else "CLEAR",
            "risk_level": highest_risk,
            "message": risk_message,
            "similar_records": return_records,
            "graph_insights": graph_insights
        }, default=str)
        
    except Exception as e:
        logger.error(f"Fraud check error: {e}")
        return json.dumps({
            "status": "DATABASE_ERROR",
            "error": f"Fraud check exception: {str(e)}",
            "message": f"CRITICAL: Fraud check failed due to database or system error: {str(e)}",
            "similar_records": [],
            "risk_level": "DATABASE_OFFLINE",
            "graph_insights": []
        })


# ============================================================================
# TOOL 4: INSPECT_PAST_RECORDS
# Fetch full records from database by ID
# ============================================================================

@tool
def inspect_past_records(record_ids: List[str]) -> str:
    """
    Retrieves full metadata of specific past records for comparison.
    Each record includes its knowledge graph connections for rich context.
    
    Used after check_fraud_vectors returns high similarity scores.
    Fetches complete record data to allow the Supervisor to distinguish
    between duplicates and legitimate recurring expenses.
    
    Args:
        record_ids: List of record UUIDs (from check_fraud_vectors results)
    
    Returns:
        JSON with records, each enriched with:
        - payload: The original invoice data
        - graph_connections: Entities this record is connected to
        - shared_with: Other invoices sharing the same entities
        - network_summary: Quick textual summary of connections
    """
    try:
        if not record_ids:
            return json.dumps({
                "status": "ERROR",
                "message": "No record IDs provided",
                "records": []
            })
        
        # Try Weaviate first (primary storage)
        client = get_weaviate_client()
        
        if not client.is_ready():
            client.connect()
        
        records = client.get_records_by_ids(record_ids)
        
        # Enrich each record with graph context
        enriched_records = []
        try:
            from .graph_tools import query_graph_insights
            
            for i, record in enumerate(records):
                record_id = record_ids[i] if i < len(record_ids) else None
                enriched = {
                    "record_id": record_id,
                    "payload": record,
                    "graph_connections": [],
                    "shared_with": [],
                    "network_summary": "No graph data available"
                }
                
                if record_id:
                    invoice_id = f"invoice_{record_id[:16]}"
                    insights = query_graph_insights(invoice_id)
                    
                    # Direct connections (entities linked to this invoice)
                    entities = insights.get("connected_entities", [])
                    enriched["graph_connections"] = [
                        {
                            "type": e.get("edge_type", "RELATED_TO"),
                            "entity": e.get("name", e.get("entity_id", "unknown")),
                            "entity_type": e.get("entity_id", "").split("_")[0] if "_" in e.get("entity_id", "") else "unknown"
                        }
                        for e in entities
                    ]
                    
                    # Other invoices sharing same entities
                    cross_links = insights.get("cross_vendor_links", [])
                    enriched["shared_with"] = [
                        {
                            "entity": link.get("shared_entity"),
                            "total_invoices": link.get("total_related_invoices", len(link.get("sample_invoices", []))),
                            "samples": link.get("sample_invoices", [])[:3]
                        }
                        for link in cross_links
                    ]
                    
                    # Generate network summary for quick comprehension
                    connections = len(entities)
                    shared_entities = len(cross_links)
                    total_shared_invoices = sum(link.get("total_related_invoices", 0) for link in cross_links)
                    
                    if connections == 0:
                        enriched["network_summary"] = "Isolated invoice - no entity connections"
                    elif total_shared_invoices == 0:
                        enriched["network_summary"] = f"Connected to {connections} entities, no shared invoices"
                    else:
                        entity_names = [e.get("name", "?") for e in entities[:3]]
                        enriched["network_summary"] = (
                            f"Connected to {connections} entities ({', '.join(entity_names)}). "
                            f"Shares {shared_entities} entities with {total_shared_invoices} other invoices."
                        )
                        
                        # Add warning if high sharing
                        if total_shared_invoices >= 10:
                            enriched["network_summary"] += " ⚠️ HIGH OVERLAP"
                
                enriched_records.append(enriched)
                
        except ImportError:
            logger.debug("graph_tools not available, returning raw records")
            enriched_records = [{"record_id": rid, "payload": rec, "graph_connections": [], "shared_with": [], "network_summary": "Graph unavailable"} 
                               for rid, rec in zip(record_ids, records)]
        except Exception as graph_error:
            logger.warning(f"Entity context query failed: {graph_error}")
            enriched_records = [{"record_id": rid, "payload": rec, "graph_connections": [], "shared_with": [], "network_summary": f"Graph error: {graph_error}"} 
                               for rid, rec in zip(record_ids, records)]
        
        if enriched_records:
            return json.dumps({
                "status": "SUCCESS",
                "message": f"Retrieved {len(enriched_records)} records with graph context",
                "records": enriched_records
            }, default=str)
        
        # Fallback to NebulaGraph for trace_id based lookups
        try:
            nebula_client = get_nebula_client()
            nebula_records = nebula_client.fetch_records(record_ids)
            
            if nebula_records:
                return json.dumps({
                    "status": "SUCCESS",
                    "message": f"Retrieved {len(nebula_records)} records from graph",
                    "records": [{"record_id": rid, "payload": rec, "graph_connections": [], "shared_with": [], "network_summary": "From NebulaGraph"} 
                               for rid, rec in zip(record_ids, nebula_records)]
                }, default=str)
        except Exception as nebula_error:
            logger.warning(f"NebulaGraph lookup failed: {nebula_error}")
        
        return json.dumps({
            "status": "NOT_FOUND",
            "message": f"No records found for IDs: {record_ids}",
            "records": []
        }, default=str)
        
    except Exception as e:
        logger.error(f"Record inspection error: {e}")
        return json.dumps({
            "status": "ERROR",
            "message": f"Failed to fetch records: {str(e)}",
            "records": []
        })


# ============================================================================
# TOOL 5: UPDATE_RECORDS
# Finalize and persist the record to databases
# ============================================================================

# Global state storage for passing context from graph
_current_state_context: Dict[str, Any] = {}


def set_state_context(document_path: str, fraud_risk_score: float):
    """Set the current state context for update_records tool."""
    global _current_state_context
    _current_state_context = {
        "document_path": document_path,
        "fraud_risk_score": fraud_risk_score
    }


def get_state_context() -> Dict[str, Any]:
    """Get the current state context."""
    return _current_state_context


@tool
def update_records(
    data: Dict[str, Any],
    entities: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Commits the validated data to the final ledger and enriches the knowledge graph.
    
    This tool:
    1. Builds the FinalEnrichedRecord (merging Payload + Audit Stats)
    2. Creates a TransientSearchRecord for Weaviate
    3. Persists to Weaviate (Vector DB)
    4. Persists to NebulaGraph (Knowledge Graph - Invoice + Vendor)
    5. Creates entity connections in NebulaGraph (Projects, People, etc.)
    
    Only call this when you are 100% confident in the data.
    This ends the processing session.
    
    Args:
        data: The validated extracted JSON data
        entities: REQUIRED - List of entities you identified in the document.
                  Each entity must have:
                  - entity_type: "project", "person", "department", or "reference"
                  - entity_name: The entity name (e.g., "project apollo")
                  - edge_type: "FOR_PROJECT", "ATTENTION_OF", "FROM_DEPARTMENT", or "REFERENCES"
                  - context: Brief context (optional)
                  
                  Example:
                  [
                      {"entity_type": "project", "entity_name": "project apollo", "edge_type": "FOR_PROJECT"},
                      {"entity_type": "person", "entity_name": "john doe", "edge_type": "ATTENTION_OF"},
                      {"entity_type": "reference", "entity_name": "po-12345", "edge_type": "REFERENCES"}
                  ]
                  
                  If no entities found, pass an empty list [].
    
    Returns:
        Success message with trace_id and relationships created.
    """
    try:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
                
        # Get state context for document path and fraud score
        context = get_state_context()
        document_path = context.get("document_path", "unknown")
        fraud_risk_score = context.get("fraud_risk_score", 0.0)
        
        # Build the final enriched record
        final_record = FinalEnrichedRecord.from_state(
            document_path=document_path,
            payload=data,
            fraud_risk_score=fraud_risk_score,
            math_verified=True
        )
        
        # Build transient record for Weaviate indexing
        transient = TransientSearchRecord.from_payload(data)
        
        # Persist to Weaviate
        weaviate_uuid = None
        weaviate_error = None
        
        try:
            weaviate_client = get_weaviate_client()
            if not weaviate_client.is_ready():
                weaviate_client.connect()
                weaviate_client.create_schema()
            weaviate_uuid = weaviate_client.insert_record(final_record, transient)
        except Exception as w_err:
            weaviate_error = str(w_err)
            logger.warning(f"Weaviate persistence failed: {w_err}")
        
        # Persist to NebulaGraph
        nebula_ids = {}
        nebula_error = None
        relationships_created = 0
        entity_errors = []
        
        try:
            nebula_client = get_nebula_client()
            nebula_client.connect()
            nebula_client.create_schema()
            nebula_ids = nebula_client.insert_entities(final_record)
            
            # Enrich the knowledge graph with entity connections
            if entities:
                from .graph_tools import add_to_graph
                invoice_id = nebula_ids.get("invoice_id", final_record.trace_id)
                
                try:
                    result_json = add_to_graph.invoke({
                        "invoice_id": invoice_id,
                        "entities": entities
                    })
                    result = json.loads(result_json)
                    relationships_created = result.get("relationships_created", 0)
                    if result.get("status") == "ERROR":
                        entity_errors.append(result.get("message", "Unknown error"))
                except Exception as entity_error:
                    entity_errors.append(str(entity_error))
                    logger.warning(f"Entity enrichment failed: {entity_error}")
                    
        except Exception as n_err:
            nebula_error = str(n_err)
            logger.warning(f"NebulaGraph persistence failed: {n_err}")
            
        logger.info(
            f"Record process result: trace_id={final_record.trace_id}, "
            f"weaviate_uuid={weaviate_uuid}, nebula_ids={nebula_ids}, "
            f"relationships={relationships_created}"
        )
        
        # Format comprehensive response
        if weaviate_uuid and not nebula_error:
            response = {
                "status": "COMMITTED",
                "trace_id": final_record.trace_id,
                "weaviate_uuid": weaviate_uuid,
                "nebula_vertex_id": nebula_ids.get("invoice_id"),
                "vendor_vertex_id": nebula_ids.get("vendor_id"),
                "relationships_created": relationships_created,
                "message": "Record successfully committed to both databases"
            }
            if entity_errors:
                response["entity_warnings"] = entity_errors
            return json.dumps(response, default=str)
        elif weaviate_uuid:
            return json.dumps({
                "status": "PARTIAL_COMMIT",
                "trace_id": final_record.trace_id,
                "weaviate_uuid": weaviate_uuid,
                "message": f"Committed to Weaviate. NebulaGraph offline/failed: {nebula_error}"
            }, default=str)
        elif not nebula_error and nebula_ids:
            return json.dumps({
                "status": "PARTIAL_COMMIT",
                "trace_id": final_record.trace_id,
                "nebula_vertex_id": nebula_ids.get("invoice_id"),
                "message": f"Committed to NebulaGraph. Weaviate offline/failed: {weaviate_error}"
            }, default=str)
        else:
            return json.dumps({
                "status": "LOCAL_AUDIT_PASS",
                "trace_id": final_record.trace_id,
                "message": f"Audit completed and validated. Databases offline (Weaviate: {weaviate_error}, Nebula: {nebula_error})"
            }, default=str)
        
    except Exception as e:
        logger.error(f"Update records error: {e}")
        return json.dumps({
            "status": "FAILED",
            "message": f"Failed to commit record: {str(e)}"
        })


# ============================================================================
# TOOL 6: IGNORE_REQUEST
# Reject the document with a reason
# ============================================================================

@tool
def ignore_request(reason: str) -> str:
    """
    Rejects the document with a specified reason.
    
    Use this tool for:
    - Spam or non-financial documents
    - Unreadable or corrupt files
    - Confirmed duplicates
    - Suspected fraud
    
    This ends the processing session.
    
    Args:
        reason: The reason for rejecting the document.
                Be specific (e.g., "Duplicate submission - same invoice 
                as record ID xxx from 2024-01-15")
    
    Returns:
        Confirmation message with the rejection reason.
    """
    logger.info(f"Document rejected: {reason}")
    
    return json.dumps({
        "status": "REJECTED",
        "reason": reason,
        "message": f"Document rejected: {reason}"
    })


# ============================================================================
# TOOL COLLECTION
# All tools bound to the Supervisor agent
# ============================================================================

# Import execute_python for supervisor's forensic analysis
from .extractor_tools import execute_python, run_verification
from .authority import verify_authority

ALL_TOOLS = [
    delegate_extraction,
    execute_python,  # For forensic analysis when agents have conflicting outputs
    verify_authority, # Validates tax/registration IDs (GSTIN, VAT, ABN, EIN) & bank accounts (IBAN)
    check_fraud_vectors,
    inspect_past_records,
    update_records,  # Includes entity enrichment via add_to_graph internally
    ignore_request,
]


def get_all_tools():
    """Get all tools for binding to the Supervisor agent."""
    return ALL_TOOLS
