import React, { useState, useEffect, useRef } from 'react';
import Prism from 'prismjs';
import 'prismjs/components/prism-json';
import { RunSummary, RunDetail, KnowledgeGraphData, VectorOverview, VectorSearchResult, Message } from './types';
import { GraphViewer } from './components/GraphViewer';
import { AgentActivityPanel } from './components/AgentActivityPanel';
import { FaTimes } from 'react-icons/fa';

export default function App() {
    const [activeTab, setActiveTab] = useState<'runs' | 'live_run' | 'graph' | 'vectors'>('runs');
    
    // Runs state
    const [runs, setRuns] = useState<RunSummary[]>([]);
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
    const [selectedRunDetail, setSelectedRunDetail] = useState<RunDetail | null>(null);
    const [runsLoading, setRunsLoading] = useState<boolean>(true);
    const [runSearch, setRunSearch] = useState<string>('');
    const [detailViewMode, setDetailViewMode] = useState<'structured' | 'json'>('structured');
    const [deletingId, setDeletingId] = useState<string | null>(null);
    const [deleteModalOpen, setDeleteModalOpen] = useState<boolean>(false);

    // New Run & Live WebSocket State
    const [newRunModalOpen, setNewRunModalOpen] = useState<boolean>(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [liveFilename, setLiveFilename] = useState<string>('');
    const [isLiveProcessing, setIsLiveProcessing] = useState<boolean>(false);
    const [supervisorMessages, setSupervisorMessages] = useState<Message[]>([]);
    const [extractorMessages, setExtractorMessages] = useState<Message[]>([]);
    const [sarvamMessages, setSarvamMessages] = useState<Message[]>([]);
    const [showExtractors, setShowExtractors] = useState<boolean>(false);
    const [previewImage, setPreviewImage] = useState<{ url: string; title: string } | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    // Trajectory Viewer State
    const [trajectoryModalOpen, setTrajectoryModalOpen] = useState<boolean>(false);
    const [trajectoryData, setTrajectoryData] = useState<any>(null);
    const [trajectoryLoading, setTrajectoryLoading] = useState<boolean>(false);

    const handlePreviewImage = (url: string, title?: string) => {
        setPreviewImage({ url, title: title || 'Forensic Image' });
    };

    const handleOpenTrajectory = async (runId: string) => {
        if (!runId) return;
        setTrajectoryLoading(true);
        setTrajectoryModalOpen(true);
        setTrajectoryData(null);
        try {
            const res = await fetch(`/api/runs/${runId}/trajectory`);
            if (res.ok) {
                const data = await res.json();
                setTrajectoryData(data);
            } else {
                alert(`No trajectory recorded for run: ${runId}`);
                setTrajectoryModalOpen(false);
            }
        } catch (e) {
            console.error('Error fetching trajectory:', e);
            setTrajectoryModalOpen(false);
        } finally {
            setTrajectoryLoading(false);
        }
    };

    // Auto-open both extractors when extraction begins
    useEffect(() => {
        if ((extractorMessages.length > 0 || sarvamMessages.length > 0) && isLiveProcessing) {
            setShowExtractors(true);
        }
    }, [extractorMessages.length, sarvamMessages.length, isLiveProcessing]);

    // Graph state
    const [graphData, setGraphData] = useState<KnowledgeGraphData>({ nodes: [], edges: [], total_nodes: 0, total_edges: 0 });
    const [graphLoading, setGraphLoading] = useState<boolean>(false);

    // Vector state
    const [vectorOverview, setVectorOverview] = useState<VectorOverview | null>(null);
    const [vectorQuery, setVectorQuery] = useState<string>('');
    const [searchResults, setSearchResults] = useState<VectorSearchResult[]>([]);
    const [searchingVectors, setSearchingVectors] = useState<boolean>(false);

    // Fetch Runs on load
    useEffect(() => {
        fetchRuns();
    }, []);

    const fetchRuns = async () => {
        setRunsLoading(true);
        try {
            const res = await fetch('/api/runs');
            if (res.ok) {
                const data = await res.json();
                const runList: RunSummary[] = data.runs || [];
                setRuns(runList);
                if (runList.length > 0) {
                    if (!selectedRunId || !runList.some(r => r.id === selectedRunId || r.trace_id === selectedRunId)) {
                        setSelectedRunId(runList[0].id);
                    }
                } else {
                    setSelectedRunId(null);
                    setSelectedRunDetail(null);
                }
            }
        } catch (e) {
            console.error('Error fetching runs:', e);
        } finally {
            setRunsLoading(false);
        }
    };

    // Fetch Run Detail when selected
    useEffect(() => {
        if (!selectedRunId) {
            setSelectedRunDetail(null);
            return;
        }
        const fetchDetail = async () => {
            try {
                const res = await fetch(`/api/runs/${selectedRunId}`);
                if (res.ok) {
                    const data = await res.json();
                    setSelectedRunDetail(data);
                }
            } catch (e) {
                console.error('Error fetching run detail:', e);
            }
        };
        fetchDetail();
    }, [selectedRunId]);

    // Fetch Graph Data
    const fetchGraph = async () => {
        setGraphLoading(true);
        try {
            const res = await fetch('/api/graph');
            if (res.ok) {
                const data = await res.json();
                setGraphData(data);
            }
        } catch (e) {
            console.error('Error fetching graph:', e);
        } finally {
            setGraphLoading(false);
        }
    };

    // Fetch Vectors Data
    const fetchVectors = async () => {
        try {
            const res = await fetch('/api/vectors');
            if (res.ok) {
                const data = await res.json();
                setVectorOverview(data);
            }
        } catch (e) {
            console.error('Error fetching vectors:', e);
        }
    };

    useEffect(() => {
        if (activeTab === 'graph') {
            fetchGraph();
        } else if (activeTab === 'vectors') {
            fetchVectors();
        }
    }, [activeTab]);

    // Start a New Run
    const handleStartNewRun = async () => {
        const file = selectedFile;
        if (!file) {
            alert('Please select a document file to process.');
            return;
        }

        setNewRunModalOpen(false);
        setActiveTab('live_run');
        setIsLiveProcessing(true);
        setSupervisorMessages([]);
        setExtractorMessages([]);
        setSarvamMessages([]);
        setLiveFilename(file.name);
        setShowExtractors(false);

        // 1. Upload File
        const formData = new FormData();
        formData.append('file', file);

        try {
            const uploadRes = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadRes.ok) throw new Error('Upload failed');
            const uploadData = await uploadRes.json();
            const filename = uploadData.filename;

            // 2. Connect WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname === 'localhost' ? 'localhost:8000' : window.location.host;
            const ws = new WebSocket(`${protocol}//${host}/ws/process/${filename}`);
            wsRef.current = ws;

            ws.onmessage = (event) => {
                const payload = JSON.parse(event.data);

                if (payload.type === 'agent_message') {
                    const msg: Message = payload.message;
                    if (payload.node === 'supervisor' || msg.node === 'supervisor') {
                        setSupervisorMessages(prev => [...prev, msg]);
                    } else if (payload.node === 'sarvam' || msg.node === 'sarvam') {
                        setSarvamMessages(prev => [...prev, msg]);
                        setShowExtractors(true);
                    } else {
                        setExtractorMessages(prev => [...prev, msg]);
                        setShowExtractors(true);
                    }
                } else if (payload.type === 'completed' || payload.type === 'complete') {
                    setIsLiveProcessing(false);
                    fetchRuns();
                    fetchGraph();
                } else if (payload.type === 'error') {
                    setIsLiveProcessing(false);
                    console.error('Processing error:', payload.message);
                }
            };

            ws.onerror = () => {
                setIsLiveProcessing(false);
                fetchRuns();
            };

            ws.onclose = () => {
                setIsLiveProcessing(false);
                fetchRuns();
                fetchGraph();
            };

        } catch (e) {
            console.error('Run execution error:', e);
            setIsLiveProcessing(false);
            alert(`Execution failed: ${String(e)}`);
        }
    };

    // Handle Delete Run Completely
    const handleDeleteRun = async () => {
        const idToDelete = deletingId || selectedRunId;
        if (!idToDelete) return;

        try {
            const res = await fetch(`/api/runs/${idToDelete}`, {
                method: 'DELETE'
            });

            if (res.ok) {
                const updated = runs.filter(r => r.id !== idToDelete && r.trace_id !== idToDelete);
                setRuns(updated);
                setDeleteModalOpen(false);
                setDeletingId(null);

                if (updated.length > 0) {
                    setSelectedRunId(updated[0].id);
                } else {
                    setSelectedRunId(null);
                    setSelectedRunDetail(null);
                }

                if (activeTab === 'graph') {
                    fetchGraph();
                }
            } else {
                alert('Failed to delete run from database.');
            }
        } catch (e) {
            console.error('Error deleting run:', e);
            alert('Error deleting run: ' + String(e));
        }
    };

    // Handle Vector Search
    // Filter state for Runs tab
    const [runFilter, setRunFilter] = useState<'committed' | 'rejected' | 'all'>('committed');

    const handleNavigateToRun = (runId: string) => {
        if (!runId) return;
        const cleanId = runId.startsWith('invoice_') ? runId.replace(/^invoice_/, '') : runId;
        const matchingRun = runs.find(r => r.id === cleanId || r.trace_id === cleanId);
        if (matchingRun && matchingRun.status === 'REJECTED') {
            setRunFilter('rejected');
        } else if (matchingRun) {
            setRunFilter('committed');
        }
        setSelectedRunId(cleanId);
        setActiveTab('runs');
    };

    const handleVectorSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!vectorQuery.trim()) return;
        setSearchingVectors(true);
        try {
            const res = await fetch('/api/vectors/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: vectorQuery, limit: 6 })
            });
            if (res.ok) {
                const data = await res.json();
                setSearchResults(data.results || []);
            }
        } catch (e) {
            console.error('Vector search failed:', e);
        } finally {
            setSearchingVectors(false);
        }
    };

    const committedCount = runs.filter(r => r.status !== 'REJECTED').length;
    const rejectedCount = runs.filter(r => r.status === 'REJECTED').length;

    const filteredRuns = runs.filter(r => {
        if (runFilter === 'committed' && r.status === 'REJECTED') return false;
        if (runFilter === 'rejected' && r.status !== 'REJECTED') return false;

        if (!runSearch.trim()) return true;
        const q = runSearch.toLowerCase();
        return (
            (r.vendor_name || '').toLowerCase().includes(q) ||
            (r.invoice_number || '').toLowerCase().includes(q) ||
            (r.trace_id || '').toLowerCase().includes(q) ||
            (r.id || '').toLowerCase().includes(q) ||
            (r.source_file || '').toLowerCase().includes(q)
        );
    });

    return (
        <div className="app-shell">
            {/* Top Navigation */}
            <header className="top-header">
                <div className="brand-section">
                    <div className="brand-title">FINANCIAL DOCUMENT INTELLIGENCE</div>
                </div>

                <div className="header-right-actions">
                    <button className="new-run-header-btn" onClick={() => setNewRunModalOpen(true)}>
                        + New Run
                    </button>

                    <nav className="nav-tabs">
                        {(isLiveProcessing || supervisorMessages.length > 0) && (
                            <button
                                className={`nav-tab ${activeTab === 'live_run' ? 'active' : ''}`}
                                onClick={() => setActiveTab('live_run')}
                            >
                                Live Execution
                                {isLiveProcessing && <span className="tab-count">RUNNING</span>}
                            </button>
                        )}
                                                
                        <button
                            className={`nav-tab ${activeTab === 'runs' ? 'active' : ''}`}
                            onClick={() => setActiveTab('runs')}
                        >
                            Runs & Documents
                            <span className="tab-count">{runs.length}</span>
                        </button>
                        


                        <button
                            className={`nav-tab ${activeTab === 'graph' ? 'active' : ''}`}
                            onClick={() => setActiveTab('graph')}
                        >
                            Knowledge Graph
                            <span className="tab-badge">NebulaGraph</span>
                        </button>
                        <button
                            className={`nav-tab ${activeTab === 'vectors' ? 'active' : ''}`}
                            onClick={() => setActiveTab('vectors')}
                        >
                            Vector Store
                            <span className="tab-badge">Weaviate</span>
                        </button>
                    </nav>
                </div>
            </header>

            {/* Main Content Area */}
            <main className="main-content">
                {/* TAB 1: RUNS & DOCUMENTS */}
                {activeTab === 'runs' && (
                    <div className="runs-layout">
                        {/* Runs Sidebar */}
                        <aside className="runs-sidebar">
                            <div className="sidebar-header">
                                <span className="sidebar-title">HISTORICAL RUNS</span>
                                <button className="refresh-btn" onClick={fetchRuns} disabled={runsLoading}>
                                    {runsLoading ? '...' : 'Refresh'}
                                </button>
                            </div>

                            <div className="sidebar-search">
                                <input
                                    type="text"
                                    placeholder="Filter by vendor, invoice#..."
                                    value={runSearch}
                                    onChange={e => setRunSearch(e.target.value)}
                                />
                            </div>

                            <div className="sidebar-filter-tabs">
                                <button
                                    className={`filter-pill ${runFilter === 'committed' ? 'active' : ''}`}
                                    onClick={() => setRunFilter('committed')}
                                >
                                    Committed ({committedCount})
                                </button>
                                <button
                                    className={`filter-pill ${runFilter === 'rejected' ? 'active' : ''}`}
                                    onClick={() => setRunFilter('rejected')}
                                >
                                    Rejected ({rejectedCount})
                                </button>
                                <button
                                    className={`filter-pill ${runFilter === 'all' ? 'active' : ''}`}
                                    onClick={() => setRunFilter('all')}
                                >
                                    All ({runs.length})
                                </button>
                            </div>

                            <div className="runs-list">
                                {runsLoading && <div className="loading-state">Loading runs from database...</div>}
                                {!runsLoading && filteredRuns.length === 0 && (
                                    <div className="empty-state">No matching runs found. Click "+ New Run" to start one.</div>
                                )}
                                {filteredRuns.map(run => {
                                    const isSelected = selectedRunId === run.id || selectedRunId === run.trace_id;
                                    return (
                                        <div
                                            key={run.id}
                                            className={`run-card ${isSelected ? 'selected' : ''}`}
                                            onClick={() => setSelectedRunId(run.id)}
                                        >
                                            <div className="run-card-header">
                                                <span className="run-vendor">{run.vendor_name}</span>
                                                <span className="run-amount">₹{run.total_amount.toFixed(2)}</span>
                                            </div>
                                            <div className="run-card-meta">
                                                <span>Inv: {run.invoice_number}</span>
                                                <span>Date: {run.invoice_date}</span>
                                            </div>
                                            <div className="run-card-footer">
                                                <span className={`status-badge ${run.status.toLowerCase()}`}>
                                                    {run.status}
                                                </span>
                                                <span className="items-count">{run.line_items_count} items</span>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </aside>

                        {/* Document & Payload Detail */}
                        <section className="run-detail-view">
                            {selectedRunDetail ? (
                                <div className="detail-container">
                                    {/* Top Metadata Header */}
                                    <div className="detail-header-bar">
                                        <div>
                                            <h3 className="detail-title">
                                                {selectedRunDetail.content?.vendor?.raw_name || 'Document Record'}
                                            </h3>
                                            <span className="trace-tag font-mono">
                                                Trace: {selectedRunDetail.trace_id}
                                            </span>
                                        </div>

                                        <div className="detail-header-right">
                                            <button
                                                className="view-trajectory-btn"
                                                onClick={() => handleOpenTrajectory(selectedRunDetail.trace_id || selectedRunDetail.id)}
                                            >
                                                View Trajectory
                                            </button>

                                            <div className="view-toggle">
                                                <button
                                                    className={`toggle-btn ${detailViewMode === 'structured' ? 'active' : ''}`}
                                                    onClick={() => setDetailViewMode('structured')}
                                                >
                                                    Structured Data
                                                </button>
                                                <button
                                                    className={`toggle-btn ${detailViewMode === 'json' ? 'active' : ''}`}
                                                    onClick={() => setDetailViewMode('json')}
                                                >
                                                    Raw JSON
                                                </button>
                                            </div>

                                            <button
                                                className="delete-run-btn"
                                                onClick={() => {
                                                    setDeletingId(selectedRunDetail.id);
                                                    setDeleteModalOpen(true);
                                                }}
                                            >
                                                Delete Run
                                            </button>
                                        </div>
                                    </div>

                                    {/* Split Screen: Document Image on Left, Parsed Data on Right */}
                                    <div className="split-workspace">
                                        {/* Left Pane: Document Image */}
                                        <div className="document-preview-pane">
                                            <div className="pane-header">
                                                <span>ORIGINAL DOCUMENT</span>
                                                <span className="doc-path font-mono">
                                                    {selectedRunDetail.source_file ? selectedRunDetail.source_file.split('/').pop() : 'Document Preview'}
                                                </span>
                                            </div>

                                            <div className="image-viewport">
                                                {selectedRunDetail.image_url ? (
                                                    (selectedRunDetail.image_url.toLowerCase().endsWith('.pdf') || (selectedRunDetail.source_file && selectedRunDetail.source_file.toLowerCase().endsWith('.pdf'))) ? (
                                                        <iframe
                                                            src={selectedRunDetail.image_url}
                                                            title="Original document PDF"
                                                            className="document-pdf-frame"
                                                        />
                                                    ) : (
                                                        <img
                                                            src={selectedRunDetail.image_url}
                                                            alt="Original document"
                                                            className="document-image"
                                                            onError={(e) => {
                                                                (e.target as HTMLElement).style.display = 'none';
                                                            }}
                                                        />
                                                    )
                                                ) : (
                                                    <div className="image-placeholder">No document image attached</div>
                                                )}
                                            </div>
                                        </div>

                                        {/* Right Pane: Structured Financial Payload */}
                                        <div className="payload-pane">
                                            {detailViewMode === 'structured' ? (
                                                <div className="structured-content">
                                                    {/* Audit & Verification Card */}
                                                    <div className="info-card">
                                                        <div className="card-title">VERIFICATION & AUDIT TRAIL</div>
                                                        <div className="grid-2col">
                                                            <div className="field-group">
                                                                <span className="field-label">Mathematical Verification:</span>
                                                                <span className="field-value success">
                                                                    {selectedRunDetail.audit_trail?.math_verified ? 'PASSED (Subtotal + Tax = Total)' : 'FAILED'}
                                                                </span>
                                                            </div>
                                                            <div className="field-group">
                                                                <span className="field-label">Fraud Risk Score:</span>
                                                                <span className="field-value font-mono">
                                                                    {selectedRunDetail.audit_trail?.fraud_risk_score?.toFixed(2) || '0.00'} (LOW RISK)
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    {/* Vendor & Invoice Metadata */}
                                                    <div className="grid-2col">
                                                        <div className="info-card">
                                                            <div className="card-title">VENDOR DETAILS</div>
                                                            <div className="field-group">
                                                                <span className="field-label">Vendor Name:</span>
                                                                <span className="field-value highlight">{selectedRunDetail.content?.vendor?.raw_name || '-'}</span>
                                                            </div>
                                                            <div className="field-group">
                                                                <span className="field-label">Address:</span>
                                                                <span className="field-value">{selectedRunDetail.content?.vendor?.address || '-'}</span>
                                                            </div>
                                                            <div className="field-group">
                                                                <span className="field-label">Tax ID / GSTIN:</span>
                                                                <span className="field-value font-mono">{selectedRunDetail.content?.vendor?.tax_id || '-'}</span>
                                                            </div>
                                                        </div>

                                                        <div className="info-card">
                                                            <div className="card-title">INVOICE METADATA</div>
                                                            <div className="field-group">
                                                                <span className="field-label">Invoice Number:</span>
                                                                <span className="field-value font-mono highlight">{selectedRunDetail.content?.invoice_details?.invoice_number || '-'}</span>
                                                            </div>
                                                            <div className="field-group">
                                                                <span className="field-label">Invoice Date:</span>
                                                                <span className="field-value">{selectedRunDetail.content?.invoice_details?.invoice_date || '-'}</span>
                                                            </div>
                                                            <div className="field-group">
                                                                <span className="field-label">Currency:</span>
                                                                <span className="field-value font-mono">{selectedRunDetail.content?.invoice_details?.currency_code || 'INR'}</span>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    {/* Financial Breakdown Card */}
                                                    <div className="info-card">
                                                        <div className="card-title">FINANCIAL TOTALS</div>
                                                        <div className="totals-grid">
                                                            <div className="total-box">
                                                                <span className="total-label">Subtotal</span>
                                                                <span className="total-val">₹{selectedRunDetail.content?.financials?.subtotal?.toFixed(2) || '0.00'}</span>
                                                            </div>
                                                            <div className="total-box">
                                                                <span className="total-label">Tax Amount</span>
                                                                <span className="total-val">₹{selectedRunDetail.content?.financials?.tax_amount?.toFixed(2) || '0.00'}</span>
                                                            </div>
                                                            <div className="total-box grand-total">
                                                                <span className="total-label">Total Amount</span>
                                                                <span className="total-val highlight">₹{selectedRunDetail.content?.financials?.total_amount?.toFixed(2) || '0.00'}</span>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    {/* Line Items Table */}
                                                    <div className="info-card">
                                                        <div className="card-title">
                                                            LINE ITEMS ({selectedRunDetail.content?.line_items?.length || 0})
                                                        </div>
                                                        <div className="table-wrapper">
                                                            <table className="items-table">
                                                                <thead>
                                                                    <tr>
                                                                        <th>#</th>
                                                                        <th>Description</th>
                                                                        <th>Qty</th>
                                                                        <th>Unit Price</th>
                                                                        <th>Row Total</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody>
                                                                    {selectedRunDetail.content?.line_items?.map((item, idx) => (
                                                                        <tr key={idx}>
                                                                            <td className="font-mono">{idx + 1}</td>
                                                                            <td>{item.description || '-'}</td>
                                                                            <td className="font-mono">{item.quantity ?? '-'}</td>
                                                                            <td className="font-mono">₹{item.unit_price?.toFixed(2) ?? '-'}</td>
                                                                            <td className="font-mono highlight">₹{item.row_total?.toFixed(2) ?? '-'}</td>
                                                                        </tr>
                                                                    ))}
                                                                </tbody>
                                                            </table>
                                                        </div>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="json-container">
                                                    <pre className="json-code font-mono">
                                                        <code dangerouslySetInnerHTML={{
                                                            __html: Prism.highlight(
                                                                JSON.stringify(selectedRunDetail.raw_payload || selectedRunDetail.content || selectedRunDetail, null, 2),
                                                                Prism.languages.json || Prism.languages.javascript,
                                                                'json'
                                                            )
                                                        }} />
                                                    </pre>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="empty-selection">Select a run from the sidebar to inspect its document and extraction data.</div>
                            )}
                        </section>
                    </div>
                )}

                {/* TAB: LIVE RUN WORKSPACE */}
                {activeTab === 'live_run' && (
                    <div className="live-run-workspace">
                        <div className="live-panels-scroll-container custom-scrollbar">
                            <div className="live-panels-row">
                                <AgentActivityPanel
                                    title={`Supervisor Agent${liveFilename ? ` — ${liveFilename}` : ''}`}
                                    messages={supervisorMessages}
                                    isProcessing={isLiveProcessing}
                                    isWaiting={isLiveProcessing && (showExtractors || supervisorMessages.some(m => m.tool_calls?.some(tc => tc.name === 'delegate_extraction')))}
                                    onStop={() => {
                                        wsRef.current?.close();
                                        setIsLiveProcessing(false);
                                    }}
                                    showExtractor={showExtractors}
                                    onToggleExtractor={() => setShowExtractors(prev => !prev)}
                                    onPreviewImage={handlePreviewImage}
                                />

                                {showExtractors && (
                                    <>
                                        <AgentActivityPanel
                                            title="Gemini Extractor Agent"
                                            messages={extractorMessages}
                                            isProcessing={isLiveProcessing && extractorMessages.length === 0}
                                            onClose={() => setShowExtractors(false)}
                                            onPreviewImage={handlePreviewImage}
                                        />

                                        <AgentActivityPanel
                                            title="Sarvam AI Agent (Doc AI)"
                                            messages={sarvamMessages}
                                            isProcessing={isLiveProcessing && sarvamMessages.length === 0}
                                            onClose={() => setShowExtractors(false)}
                                            onPreviewImage={handlePreviewImage}
                                        />
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* TAB 2: KNOWLEDGE GRAPH */}
                {activeTab === 'graph' && (
                    <div className="graph-tab-wrapper">
                        <GraphViewer
                            data={graphData}
                            onRefresh={fetchGraph}
                            loading={graphLoading}
                            onSelectRun={handleNavigateToRun}
                        />
                    </div>
                )}

                {/* TAB 3: VECTOR STORE */}
                {activeTab === 'vectors' && (
                    <div className="vectors-layout">
                        {/* Vector Search Header */}
                        <div className="vectors-header-card">
                            <div className="vector-meta-strip">
                                <div>Collection: <strong>{vectorOverview?.collection || 'FinancialRecord'}</strong></div>
                                <div>Total Objects: <strong>{vectorOverview?.total_objects || 0}</strong></div>
                                <div>Dimensions: <strong>384 Dense Float Vector</strong></div>
                                <div>Model: <strong>sentence-transformers</strong></div>
                                <div>Metric: <strong>Cosine Distance</strong></div>
                            </div>

                            <form onSubmit={handleVectorSearch} className="vector-search-bar">
                                <input
                                    type="text"
                                    placeholder="Enter natural language query for semantic vector search (e.g. 'Dmart bill for Syska bulb')..."
                                    value={vectorQuery}
                                    onChange={e => setVectorQuery(e.target.value)}
                                />
                                <button type="submit" className="search-btn" disabled={searchingVectors}>
                                    {searchingVectors ? 'Searching...' : 'Search'}
                                </button>
                            </form>
                        </div>

                        {/* Search Results */}
                        {searchResults.length > 0 && (
                            <div className="search-results-section">
                                <h4>VECTOR SEARCH RESULTS ({searchResults.length})</h4>
                                <div className="results-grid">
                                    {searchResults.map((res, idx) => (
                                        <div key={idx} className="result-card">
                                            <div className="result-header">
                                                <span className="result-vendor">{res.vendor_name}</span>
                                                <span className="score-badge">Similarity: {res.similarity_score.toFixed(4)}</span>
                                            </div>
                                            <div className="result-summary">{res.summary_text}</div>
                                            <div className="result-meta">
                                                <span>Amount: ₹{res.total_amount.toFixed(2)}</span>
                                                <span>Distance: {res.distance.toFixed(4)}</span>
                                                <button
                                                    className="link-to-run-btn"
                                                    onClick={() => handleNavigateToRun(res.id)}
                                                >
                                                    View Run Document →
                                                </button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Stored Vectors Table */}
                        <div className="stored-vectors-section">
                            <h4>INGESTED VECTOR OBJECTS</h4>
                            <div className="table-wrapper">
                                <table className="items-table">
                                    <thead>
                                        <tr>
                                            <th>UUID</th>
                                            <th>Vendor</th>
                                            <th>Amount</th>
                                            <th>Summary Text</th>
                                            <th>Vector Preview (8 / 384 dims)</th>
                                            <th>Action</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {vectorOverview?.records?.map(rec => (
                                            <tr key={rec.id}>
                                                <td className="font-mono">{rec.id.substring(0, 12)}…</td>
                                                <td>{rec.vendor_name}</td>
                                                <td className="font-mono">₹{rec.total_amount.toFixed(2)}</td>
                                                <td>{rec.summary_text}</td>
                                                <td className="font-mono text-muted">
                                                    [{rec.vector_preview?.join(', ')}…]
                                                </td>
                                                <td>
                                                    <button
                                                        className="link-to-run-btn"
                                                        onClick={() => handleNavigateToRun(rec.id)}
                                                    >
                                                        View Run →
                                                    </button>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}
            </main>

            {/* New Run Modal */}
            {newRunModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-card">
                        <h4 className="modal-title">New Document Processing Run</h4>
                        <p className="modal-text">
                            Upload an invoice, receipt, or financial statement (JPG, PNG, PDF) to trigger autonomous dual-agent extraction and fraud auditing.
                        </p>

                        <div
                            className="file-dropzone"
                            onClick={() => document.getElementById('file-upload-input')?.click()}
                        >
                            <span className="dropzone-label">
                                {selectedFile ? selectedFile.name : 'Click to select or drop document file'}
                            </span>
                            <span className="dropzone-sub">Supports receipts, invoices, bills (JPEG, PNG, PDF)</span>
                            <input
                                id="file-upload-input"
                                type="file"
                                accept="image/*,application/pdf"
                                className="file-input-hidden"
                                onChange={(e) => {
                                    if (e.target.files && e.target.files.length > 0) {
                                        setSelectedFile(e.target.files[0]);
                                    }
                                }}
                            />
                        </div>

                        <div className="modal-actions">
                            <button
                                className="modal-cancel-btn"
                                onClick={() => {
                                    setNewRunModalOpen(false);
                                    setSelectedFile(null);
                                }}
                            >
                                Cancel
                            </button>
                            <button
                                className="new-run-header-btn"
                                disabled={!selectedFile}
                                onClick={handleStartNewRun}
                            >
                                Start Processing Run
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {deleteModalOpen && (
                <div className="modal-overlay">
                    <div className="modal-card">
                        <h4 className="modal-title">Delete Financial Run</h4>
                        <p className="modal-text">
                            Are you sure you want to permanently delete this run? This will completely remove the document record from <strong>Weaviate</strong>, delete the node and relationships in <strong>NebulaGraph</strong>, and remove the audit trace.
                        </p>
                        <div className="modal-actions">
                            <button
                                className="modal-cancel-btn"
                                onClick={() => {
                                    setDeleteModalOpen(false);
                                    setDeletingId(null);
                                }}
                            >
                                Cancel
                            </button>
                            <button className="modal-confirm-delete-btn" onClick={handleDeleteRun}>
                                Delete Completely
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Forensic Image Lightbox Modal */}
            {previewImage && (
                <div className="image-lightbox-overlay" onClick={() => setPreviewImage(null)}>
                    <div className="lightbox-modal" onClick={e => e.stopPropagation()}>
                        <div className="lightbox-header">
                            <span className="lightbox-title">{previewImage.title}</span>
                            <button className="lightbox-close-btn" onClick={() => setPreviewImage(null)}>
                                <FaTimes />
                            </button>
                        </div>
                        <div className="lightbox-body">
                            <img src={previewImage.url} alt={previewImage.title} className="lightbox-image" />
                        </div>
                    </div>
                </div>
            )}

            {/* Trajectory Viewer Modal */}
            {trajectoryModalOpen && (
                <div className="trajectory-modal-overlay" onClick={() => setTrajectoryModalOpen(false)}>
                    <div className="trajectory-modal-container" onClick={e => e.stopPropagation()}>
                        <div className="trajectory-modal-header">
                            <div>
                                <h3>
                                    Agent Trajectory — {trajectoryData?.metadata?.document_name || selectedRunDetail?.source_file || 'Run Trajectory'}
                                </h3>
                                <span className="trace-tag font-mono" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                                    Trace ID: {trajectoryData?.metadata?.trace_id || selectedRunId} | Supervisor ({trajectoryData?.supervisor_messages?.length || 0} steps) | Gemini ({trajectoryData?.extractor_messages?.length || 0} steps) | Sarvam ({trajectoryData?.sarvam_messages?.length || 0} steps)
                                </span>
                            </div>
                            <button className="close-panel-btn" onClick={() => setTrajectoryModalOpen(false)} title="Close Trajectory">
                                <FaTimes />
                            </button>
                        </div>
                        <div className="trajectory-modal-body custom-scrollbar">
                            {trajectoryLoading ? (
                                <div className="empty-indicator" style={{ marginTop: '60px' }}>Loading agent trajectory traces...</div>
                            ) : trajectoryData ? (
                                <div className="live-panels-row" style={{ height: '100%', minWidth: '100%' }}>
                                    <AgentActivityPanel
                                        title="Supervisor Agent (Historical Trajectory)"
                                        messages={trajectoryData.supervisor_messages || []}
                                        isProcessing={false}
                                        onPreviewImage={handlePreviewImage}
                                    />
                                    <AgentActivityPanel
                                        title="Gemini Extractor Agent (Historical Trajectory)"
                                        messages={trajectoryData.extractor_messages || []}
                                        isProcessing={false}
                                        onPreviewImage={handlePreviewImage}
                                    />
                                    <AgentActivityPanel
                                        title="Sarvam AI Agent (Historical Trajectory)"
                                        messages={trajectoryData.sarvam_messages || []}
                                        isProcessing={false}
                                        onPreviewImage={handlePreviewImage}
                                    />
                                </div>
                            ) : (
                                <div className="empty-indicator" style={{ marginTop: '60px' }}>No trajectory data available for this run.</div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
