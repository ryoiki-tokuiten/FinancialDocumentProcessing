export interface RunSummary {
    id: string;
    trace_id: string;
    vendor_name: string;
    invoice_number: string;
    invoice_date: string;
    total_amount: number;
    currency: string;
    source_file: string;
    image_url: string;
    received_at: string;
    math_verified: boolean;
    fraud_risk_score: number;
    status: string;
    line_items_count: number;
}

export interface LineItem {
    description?: string;
    quantity?: number;
    unit_price?: number;
    row_total?: number;
    product_code?: string;
}

export interface RunDetail {
    id: string;
    trace_id: string;
    source_file: string;
    image_url: string;
    metadata: {
        source_file?: string;
        received_at?: string;
        schema_version?: string;
    };
    audit_trail: {
        math_verified?: boolean;
        fraud_risk_score?: number;
        validation_errors?: string[];
    };
    content: {
        vendor?: {
            raw_name?: string;
            address?: string;
            tax_id?: string;
        };
        invoice_details?: {
            invoice_number?: string;
            invoice_date?: string;
            currency_code?: string;
        };
        financials?: {
            subtotal?: number;
            tax_amount?: number;
            total_amount?: number;
        };
        line_items?: LineItem[];
    };
    raw_payload: any;
}

export interface Message {
    role: string;
    content?: string;
    thought?: string;
    tool_calls?: any[];
    tool_name?: string;
    node?: string;
    type?: string;
    timestamp?: number;
}

export interface GraphNode {
    id: string;
    label: string;
    type: string;
    color?: string;
    properties?: Record<string, any>;
    x?: number;
    y?: number;
}

export interface GraphEdge {
    id: string;
    source: string;
    target: string;
    label: string;
}

export interface KnowledgeGraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
    total_nodes: number;
    total_edges: number;
}

export interface VectorRecord {
    id: string;
    summary_text: string;
    vendor_name: string;
    total_amount: number;
    line_item_fingerprint: string;
    vector_dimensions: number;
    vector_preview: number[];
}

export interface VectorOverview {
    collection: string;
    total_objects: number;
    vector_dimensions: number;
    model: string;
    distance_metric: string;
    records: VectorRecord[];
}

export interface VectorSearchResult {
    id: string;
    vendor_name: string;
    total_amount: number;
    summary_text: string;
    invoice_date: string;
    similarity_score: number;
    distance: number;
}
