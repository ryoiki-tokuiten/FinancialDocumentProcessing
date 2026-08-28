"""
Pydantic models for structured data validation.

Defines the schemas for:
- Extractor Payload (output from the Extractor agent)
- Transient Search Record (for fraud vector searches, not persisted)
- Final Enriched Record (complete record for storage)
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import hashlib
import json


# ============================================================================
# EXTRACTOR PAYLOAD MODELS
# These represent the structured output from the Extractor agent
# ============================================================================

class VendorInfo(BaseModel):
    """Vendor/Merchant information extracted from the document."""
    raw_name: str = Field(..., description="The vendor/merchant name as it appears on the document")
    address: Optional[str] = Field(None, description="The vendor's address if present")
    tax_id: Optional[str] = Field(None, description="Tax identification number (EIN, VAT, etc.)")


class InvoiceDetails(BaseModel):
    """Invoice-specific metadata extracted from the document."""
    invoice_number: Optional[str] = Field(None, description="The unique invoice identifier")
    invoice_date: Optional[str] = Field(None, description="Date of the invoice (YYYY-MM-DD format)")
    due_date: Optional[str] = Field(None, description="Payment due date (YYYY-MM-DD format)")
    currency_code: str = Field("USD", description="ISO 4217 currency code")


class Financials(BaseModel):
    """Financial summary extracted from the document."""
    subtotal: Optional[float] = Field(None, description="Sum of line items before tax")
    tax_amount: Optional[float] = Field(None, description="Total tax amount")
    total_amount: float = Field(..., description="Final total amount due")
    line_items_sum: Optional[float] = Field(None, description="Calculated sum of line item row totals")


class LineItem(BaseModel):
    """Individual line item from an invoice or statement."""
    description: str = Field(..., description="Description of the product/service")
    quantity: Optional[float] = Field(None, description="Quantity purchased")
    unit_price: Optional[float] = Field(None, description="Price per unit")
    row_total: float = Field(..., description="Total for this line item")
    product_code: Optional[str] = Field(None, description="SKU or product code if present")


class ExtractorPayload(BaseModel):
    """
    Complete payload output from the Extractor Agent.
    
    This is the structured data extracted from a financial document
    before enrichment and storage.
    """
    vendor: VendorInfo
    invoice_details: InvoiceDetails
    financials: Financials
    line_items: List[LineItem] = Field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()


# ============================================================================
# TRANSIENT SEARCH RECORD
# Used for fraud vector searches - NOT persisted to database
# ============================================================================

class TransientSearchRecord(BaseModel):
    """
    Transient record for fraud vector searches.
    
    This is built programmatically from the ExtractorPayload and used
    for similarity searches in Weaviate. It is NOT stored.
    """
    summary_text: str = Field(..., description="Concatenated summary for vector embedding")
    vendor_name: str = Field(..., description="Vendor name for exact match filtering")
    total_amount: float = Field(..., description="Total amount for range filtering")
    invoice_date: Optional[str] = Field(None, description="Invoice date for time filtering")
    line_item_fingerprint: str = Field(..., description="MD5 hash of sorted line item descriptions")
    
    @classmethod
    def from_payload(cls, payload: dict) -> "TransientSearchRecord":
        """
        Build a transient search record from an Extractor Payload.
        
        This is the programmatic construction as defined in Plan.md Section 3.B
        """
        vendor_data = payload.get("vendor", {})
        if isinstance(vendor_data, str):
            vendor = vendor_data
        elif isinstance(vendor_data, dict):
            vendor = vendor_data.get("raw_name") or vendor_data.get("name") or "Unknown"
        else:
            vendor = "Unknown"
            
        financials = payload.get("financials", {})
        if isinstance(financials, dict):
            total_val = financials.get("total_amount", 0.0)
        elif isinstance(financials, (int, float)):
            total_val = financials
        else:
            total_val = 0.0
        try:
            total = float(total_val)
        except (ValueError, TypeError):
            total = 0.0
            
        invoice_details = payload.get("invoice_details", {})
        if isinstance(invoice_details, dict):
            date = str(invoice_details.get("invoice_date") or invoice_details.get("date") or "")
        elif isinstance(invoice_details, str):
            date = invoice_details
        else:
            date = ""
        
        # Build summary text for vector similarity search
        summary_text = f"Invoice from {vendor} for {total} on {date}"
        
        # Build line item fingerprint for template fraud detection
        line_items = payload.get("line_items", [])
        descriptions = []
        if isinstance(line_items, list):
            for item in line_items:
                if isinstance(item, dict):
                    descriptions.append(str(item.get("description", "")))
                elif isinstance(item, str):
                    descriptions.append(item)
            descriptions.sort()
            
        fingerprint = hashlib.md5("".join(descriptions).encode()).hexdigest()
        
        return cls(
            summary_text=summary_text,
            vendor_name=vendor,
            total_amount=total,
            invoice_date=date if date else None,
            line_item_fingerprint=fingerprint
        )


# ============================================================================
# FINAL ENRICHED RECORD
# The complete record stored in Weaviate and NebulaGraph
# ============================================================================

class RecordMetadata(BaseModel):
    """Metadata about the document processing."""
    source_file: str = Field(..., description="Path to the original document")
    received_at: str = Field(..., description="ISO timestamp when document was received")
    schema_version: str = Field("1.0", description="Schema version for forward compatibility")


class AuditTrail(BaseModel):
    """Audit information about the validation process."""
    math_verified: bool = Field(..., description="Whether math validation passed")
    fraud_risk_score: float = Field(..., description="Fraud risk score from vector search")
    validation_errors: List[str] = Field(default_factory=list, description="List of validation errors if any")


class FinalEnrichedRecord(BaseModel):
    """
    The complete enriched record for storage.
    
    This is the final artifact that gets persisted to both
    Weaviate (for vector search) and NebulaGraph (for entity relationships).
    """
    trace_id: str = Field(..., description="Unique identifier for this record")
    metadata: RecordMetadata
    content: dict = Field(..., description="The Extractor's output payload")
    audit_trail: AuditTrail
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string for storage."""
        return json.dumps(self.model_dump())
    
    @classmethod
    def from_state(cls, document_path: str, payload: dict, fraud_risk_score: float, 
                   math_verified: bool = True, validation_errors: List[str] = None) -> "FinalEnrichedRecord":
        """
        Build a final enriched record from processing state.
        
        This is the programmatic construction as defined in Plan.md Section 3.B
        """
        import uuid
        
        return cls(
            trace_id=str(uuid.uuid4()),
            metadata=RecordMetadata(
                source_file=document_path,
                received_at=datetime.now().isoformat(),
                schema_version="1.0"
            ),
            content=payload,
            audit_trail=AuditTrail(
                math_verified=math_verified,
                fraud_risk_score=fraud_risk_score,
                validation_errors=validation_errors or []
            )
        )
