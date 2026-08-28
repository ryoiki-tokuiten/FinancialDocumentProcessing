"""
Graph RAG tools for knowledge graph operations.

Provides:
- add_to_graph: Called internally by update_records() to store entity connections
- query_graph_insights: Called internally by check_fraud_vectors() to retrieve patterns

The Supervisor doesn't call these directly. They're integrated into the main tools.
"""

import json
import logging
from typing import List, Dict, Any

from langchain_core.tools import tool

from .db.nebula_client import get_nebula_client

logger = logging.getLogger(__name__)


def _sanitize_id(text: str) -> str:
    """Sanitize a string for use as a vertex ID."""
    sanitized = "".join(c if c.isalnum() else "_" for c in text.lower())
    return sanitized[:32]


def _escape(text: str) -> str:
    """Escape special characters for nGQL strings."""
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")


@tool
def add_to_graph(
    invoice_id: str,
    entities: List[Dict[str, str]]
) -> str:
    """
    Stores entity-relationship triplets in NebulaGraph for knowledge graph enrichment.
    
    The Supervisor uses its document context to identify and provide entities.
    This enables advanced fraud detection by finding cross-vendor patterns.
    
    Args:
        invoice_id: The invoice vertex ID (e.g., "invoice_abc123")
        entities: List of entities to link. Each entity must have:
                  - entity_type: "project", "person", "department", or "reference"
                  - entity_name: The entity name (e.g., "project apollo")
                  - edge_type: "FOR_PROJECT", "ATTENTION_OF", "FROM_DEPARTMENT", or "REFERENCES"
                  - context: Brief context (optional)
    
    Returns:
        JSON string with status and list of created relationships.
        
    Example:
        add_to_graph(
            invoice_id="invoice_abc123",
            entities=[
                {"entity_type": "project", "entity_name": "project apollo", "edge_type": "FOR_PROJECT", "context": "Development costs"},
                {"entity_type": "person", "entity_name": "john doe", "edge_type": "ATTENTION_OF", "context": "Finance contact"}
            ]
        )
    """
    try:
        if not entities:
            return json.dumps({
                "status": "NO_ENTITIES",
                "message": "No entities provided",
                "relationships_created": 0
            })
        
        # Get NebulaGraph client
        client = get_nebula_client()
        if not client._pool:
            client.connect()
        
        client._use_space()
        
        created_relationships = []
        
        valid_entity_types = {"project", "person", "department", "reference"}
        
        # Map edge types
        edge_map = {
            "FOR_PROJECT": "FOR_PROJECT",
            "ATTENTION_OF": "ATTENTION_OF",
            "FROM_DEPARTMENT": "FROM_DEPARTMENT",
            "BELONGS_TO": "BELONGS_TO",
            "REFERENCES": "REFERENCES"
        }
        
        for entity in entities:
            entity_type = entity.get("entity_type", "").lower()
            entity_name = entity.get("entity_name", "")
            edge_type = entity.get("edge_type", "RELATED_TO")
            context = entity.get("context", "")
            
            if not entity_name or entity_type not in valid_entity_types:
                continue
            
            edge_name = edge_map.get(edge_type, "RELATED_TO")
            
            # Generate vertex ID for the entity
            entity_id = f"{entity_type}_{_sanitize_id(entity_name)}"
            
            # Insert entity node (upsert)
            if entity_type == "project":
                insert_query = f'''
                    INSERT VERTEX IF NOT EXISTS Project (name)
                    VALUES "{entity_id}": ("{_escape(entity_name)}");
                '''
            elif entity_type == "person":
                insert_query = f'''
                    INSERT VERTEX IF NOT EXISTS Person (name, role)
                    VALUES "{entity_id}": ("{_escape(entity_name)}", "");
                '''
            elif entity_type == "department":
                insert_query = f'''
                    INSERT VERTEX IF NOT EXISTS Department (name)
                    VALUES "{entity_id}": ("{_escape(entity_name)}");
                '''
            elif entity_type == "reference":
                insert_query = f'''
                    INSERT VERTEX IF NOT EXISTS Reference (ref_type, ref_value)
                    VALUES "{entity_id}": ("", "{_escape(entity_name)}");
                '''
            else:
                continue
            
            try:
                client._execute(insert_query)
            except Exception as e:
                logger.warning(f"Failed to insert entity {entity_id}: {e}")
                continue
            
            # Create edge from invoice to entity
            edge_query = f'''
                INSERT EDGE IF NOT EXISTS {edge_name} (context)
                VALUES "{invoice_id}" -> "{entity_id}": ("{_escape(context[:200])}");
            '''
            
            try:
                client._execute(edge_query)
                created_relationships.append({
                    "from": invoice_id,
                    "to": entity_id,
                    "edge": edge_name,
                    "entity_type": entity_type
                })
            except Exception as e:
                logger.warning(f"Failed to create edge {invoice_id} -> {entity_id}: {e}")
        
        logger.info(f"Created {len(created_relationships)} graph relationships for {invoice_id}")
        
        return json.dumps({
            "status": "SUCCESS",
            "message": f"Created {len(created_relationships)} relationships",
            "relationships_created": len(created_relationships),
            "relationships": created_relationships
        })
        
    except Exception as e:
        logger.error(f"add_to_graph failed: {e}")
        return json.dumps({
            "status": "ERROR",
            "message": f"Failed to add entities to graph: {str(e)}",
            "relationships_created": 0
        })


def query_graph_insights(invoice_id: str) -> Dict[str, Any]:
    """
    Query the knowledge graph for fraud-relevant insights about an invoice.
    
    This function is called internally by check_fraud_vectors to enrich
    fraud detection with graph-based patterns.
    
    Args:
        invoice_id: The invoice vertex ID
        
    Returns:
        Dictionary with graph insights including:
        - connected_entities: All entities linked to this invoice
        - cross_vendor_links: Other invoices sharing the same entities
        - suspicious_patterns: Detected anomalies
    """
    try:
        client = get_nebula_client()
        if not client._pool:
            client.connect()
        
        client._use_space()
        
        insights = {
            "connected_entities": [],
            "cross_vendor_links": [],
            "suspicious_patterns": []
        }
        
        # Find all entities connected to this invoice
        clean_inv_id = invoice_id.strip('"')
        entity_query = f'GO FROM "{clean_inv_id}" OVER FOR_PROJECT, ATTENTION_OF, FROM_DEPARTMENT, REFERENCES, ISSUED_BY YIELD dst(edge) as entity_id, type(edge) as edge_type, properties($$) as props;'
        
        try:
            result = client._execute(entity_query)
            for row in result:
                raw_entity_id = str(row.values()[0]).strip('"')
                raw_edge_type = str(row.values()[1]).strip('"')
                props = row.values()[2].as_map() if row.values()[2] else {}
                
                insights["connected_entities"].append({
                    "entity_id": raw_entity_id,
                    "edge_type": raw_edge_type,
                    "name": str(props.get("name", raw_entity_id)).strip('"')
                })
        except Exception as e:
            logger.debug(f"Entity query failed (may not exist): {e}")
        
        # For each connected entity, find other invoices linked to it (with limit)
        for entity in insights["connected_entities"]:
            entity_id = entity["entity_id"].strip('"')
            
            # First get count of related invoices (WHERE before YIELD in nGQL)
            count_query = f'GO FROM "{entity_id}" OVER FOR_PROJECT, ATTENTION_OF, FROM_DEPARTMENT, REFERENCES, ISSUED_BY REVERSELY WHERE src(edge) != "{clean_inv_id}" YIELD src(edge) as other_invoice | YIELD COUNT(*) as total;'
            
            total_related = 0
            try:
                count_result = client._execute(count_query)
                if count_result and count_result.row_size() > 0:
                    total_related = int(count_result.row_values(0)[0].as_int())
            except Exception as e:
                logger.debug(f"Count query failed for {entity_id}: {e}")
            
            # Then get sample invoices (WHERE before YIELD in nGQL)
            cross_query = f'GO FROM "{entity_id}" OVER FOR_PROJECT, ATTENTION_OF, FROM_DEPARTMENT, REFERENCES, ISSUED_BY REVERSELY WHERE src(edge) != "{clean_inv_id}" YIELD src(edge) as other_invoice, type(edge) as edge_type | LIMIT 10;'
            
            try:
                result = client._execute(cross_query)
                sample_invoices = []
                for i in range(result.row_size()):
                    other_invoice = str(result.row_values(i)[0].as_string()).strip('"')
                    if other_invoice not in sample_invoices:
                        sample_invoices.append(other_invoice)
                
                if sample_invoices or total_related > 0:
                    insights["cross_vendor_links"].append({
                        "shared_entity": entity["name"],
                        "entity_type": entity_id.split("_")[0] if "_" in entity_id else "unknown",
                        "total_related_invoices": total_related,
                        "sample_invoices": sample_invoices[:5]  # Show max 5 samples
                    })
            except Exception as e:
                logger.debug(f"Cross-link query failed: {e}")
        
        # Detect suspicious patterns with summarization
        total_cross_links = sum(link.get("total_related_invoices", 0) for link in insights["cross_vendor_links"])
        
        if total_cross_links >= 3:
            severity = "LOW"
            if total_cross_links >= 10:
                severity = "MEDIUM"
            if total_cross_links >= 25:
                severity = "HIGH"
            if total_cross_links >= 50:
                severity = "CRITICAL"
            
            insights["suspicious_patterns"].append({
                "type": "MULTI_INVOICE_ENTITY",
                "description": f"Entities shared across {total_cross_links} other invoices total",
                "severity": severity,
                "total_count": total_cross_links,
                "breakdown": [
                    {"entity": link["shared_entity"], "count": link["total_related_invoices"]}
                    for link in insights["cross_vendor_links"]
                ]
            })
        
        return insights
        
    except Exception as e:
        logger.error(f"Graph insights query failed: {e}")
        return {
            "connected_entities": [],
            "cross_vendor_links": [],
            "suspicious_patterns": [],
            "error": str(e)
        }
