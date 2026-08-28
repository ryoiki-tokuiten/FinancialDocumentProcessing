#!/usr/bin/env python3
"""
Insert a realistic Indian grocery store invoice into Weaviate.
This is for demonstration purposes.
"""

from datetime import datetime
from .db.weaviate_client import init_weaviate
from .schemas import (
    FinalEnrichedRecord, TransientSearchRecord, 
    LineItem, ExtractorPayload, VendorInfo, InvoiceDetails, Financials,
    RecordMetadata, AuditTrail
)
import hashlib
import json
import uuid

def main():
    # Initialize Weaviate and create schema
    print("Connecting to Weaviate...")
    client = init_weaviate(host='localhost', port=8080, grpc_port=50051, create_schema=True)
    
    # Create realistic Indian grocery store data
    line_items = [
        LineItem(
            description="Amul Taaza Milk 1L",
            quantity=2,
            unit_price=56.0,
            row_total=112.0
        ),
        LineItem(
            description="Britannia Good Day Biscuits 200g",
            quantity=3,
            unit_price=35.0,
            row_total=105.0
        ),
        LineItem(
            description="Tata Tea Gold 500g",
            quantity=1,
            unit_price=245.0,
            row_total=245.0
        ),
        LineItem(
            description="Fortune Sunlite Oil 1L",
            quantity=1,
            unit_price=185.0,
            row_total=185.0
        ),
        LineItem(
            description="Everest Haldi Powder 100g",
            quantity=2,
            unit_price=42.0,
            row_total=84.0
        )
    ]
    
    subtotal = sum(item.row_total for item in line_items)
    tax = round(subtotal * 0.05, 2)  # 5% GST
    total = subtotal + tax
    
    # Create ExtractorPayload
    extractor_payload = ExtractorPayload(
        vendor=VendorInfo(
            raw_name="Reliance Fresh",
            address="Koramangala, Bangalore - 560034"
        ),
        invoice_details=InvoiceDetails(
            invoice_number="RF2024-00782",
            invoice_date="2024-01-15",
            currency_code="INR"
        ),
        financials=Financials(
            subtotal=subtotal,
            tax_amount=tax,
            total_amount=total,
            line_items_sum=subtotal
        ),
        line_items=line_items
    )
    
    # Create the final enriched record
    final_record = FinalEnrichedRecord(
        trace_id=str(uuid.uuid4()),
        metadata=RecordMetadata(
            source_file="uploads/reliance_fresh_invoice_20240115.jpg",
            received_at=datetime.now().isoformat(),
            schema_version="1.0"
        ),
        content=extractor_payload.to_dict(),
        audit_trail=AuditTrail(
            math_verified=True,
            fraud_risk_score=0.0,
            validation_errors=[]
        )
    )
    
    # Create transient search record
    line_item_fingerprint = hashlib.md5(
        json.dumps([item.description for item in line_items], sort_keys=True).encode()
    ).hexdigest()
    
    summary_text = f"Invoice from Reliance Fresh for ₹{total:.2f}. Items: Amul milk, Britannia biscuits, Tata tea, Fortune oil, Everest spices. Date: 15-Jan-2024. Location: Koramangala, Bangalore."
    
    transient = TransientSearchRecord(
        summary_text=summary_text,
        vendor_name="Reliance Fresh",
        total_amount=total,
        invoice_date="2024-01-15T00:00:00Z",
        line_item_fingerprint=line_item_fingerprint
    )
    
    # Insert into Weaviate
    print("Inserting record into Weaviate...")
    record_uuid = client.insert_record(final_record, transient)
    print(f"✓ Successfully inserted record with UUID: {record_uuid}")
    print(f"  Vendor: Reliance Fresh")
    print(f"  Total: ₹{total:.2f}")
    print(f"  Date: 2024-01-15")
    print(f"  Items: {len(line_items)}")
    
    client.close()
    print("\n✓ Done! Refresh the Database Records tab in the UI to see the data.")

if __name__ == "__main__":
    main()
