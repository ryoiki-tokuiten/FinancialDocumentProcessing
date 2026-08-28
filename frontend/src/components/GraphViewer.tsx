import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Network } from 'vis-network/standalone';
import { DataSet } from 'vis-data/standalone';
import { KnowledgeGraphData, GraphNode } from '../types';

interface GraphViewerProps {
    data: KnowledgeGraphData;
    onRefresh?: () => void;
    loading?: boolean;
    onSelectRun?: (runId: string) => void;
}

export const GraphViewer: React.FC<GraphViewerProps> = ({ data, onRefresh, loading, onSelectRun }) => {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const networkRef = useRef<Network | null>(null);
    const [selectedType, setSelectedType] = useState<string>('ALL');
    const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
    const [searchTerm, setSearchTerm] = useState<string>('');

    // Color mapping based on node type (Codex / professional palette)
    const typeColorMap: Record<string, { background: string; border: string; highlight: { background: string; border: string } }> = {
        Vendor: {
            background: '#0169CC',
            border: '#38bdf8',
            highlight: { background: '#0284c7', border: '#ffffff' }
        },
        Invoice: {
            background: '#7c3aed',
            border: '#a855f7',
            highlight: { background: '#9333ea', border: '#ffffff' }
        },
        Reference: {
            background: '#d97706',
            border: '#f59e0b',
            highlight: { background: '#b45309', border: '#ffffff' }
        },
        Project: {
            background: '#059669',
            border: '#10b981',
            highlight: { background: '#047857', border: '#ffffff' }
        },
        Person: {
            background: '#db2777',
            border: '#ec4899',
            highlight: { background: '#be185d', border: '#ffffff' }
        },
        Department: {
            background: '#4f46e5',
            border: '#6366f1',
            highlight: { background: '#4338ca', border: '#ffffff' }
        }
    };

    // Filter nodes and edges based on type and search
    const filteredGraph = useMemo(() => {
        let activeNodes = data.nodes;

        if (selectedType !== 'ALL') {
            // Find nodes of selected type
            const directNodes = data.nodes.filter(n => n.type.toLowerCase() === selectedType.toLowerCase());
            const directIds = new Set(directNodes.map(n => n.id));
            
            // Also find directly connected neighbor nodes
            const connectedIds = new Set<string>();
            data.edges.forEach(e => {
                if (directIds.has(e.source)) connectedIds.add(e.target);
                if (directIds.has(e.target)) connectedIds.add(e.source);
            });

            activeNodes = data.nodes.filter(n => directIds.has(n.id) || connectedIds.has(n.id));
        }

        if (searchTerm.trim()) {
            const term = searchTerm.toLowerCase();
            activeNodes = activeNodes.filter(n =>
                n.label.toLowerCase().includes(term) ||
                n.id.toLowerCase().includes(term) ||
                (n.properties && Object.values(n.properties).some(v => String(v).toLowerCase().includes(term)))
            );
        }

        const activeNodeIds = new Set(activeNodes.map(n => n.id));
        const activeEdges = data.edges.filter(e => activeNodeIds.has(e.source) && activeNodeIds.has(e.target));

        return { nodes: activeNodes, edges: activeEdges };
    }, [data, selectedType, searchTerm]);

    // Initialize & update Vis Network
    useEffect(() => {
        if (!containerRef.current) return;

        const visNodes = new DataSet(
            filteredGraph.nodes.map(n => {
                const colors = typeColorMap[n.type] || {
                    background: '#475569',
                    border: '#94a3b8',
                    highlight: { background: '#64748b', border: '#ffffff' }
                };
                const radius = n.type === 'Vendor' ? 24 : n.type === 'Invoice' ? 18 : 14;

                return {
                    id: n.id,
                    label: n.label,
                    title: `${n.type}: ${n.label}\nID: ${n.id}`,
                    size: radius,
                    color: colors,
                    font: {
                        color: '#FCFCFC',
                        size: 11,
                        face: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                    },
                    borderWidth: 1.5,
                    borderWidthSelected: 2.5
                };
            })
        );

        const visEdges = new DataSet(
            filteredGraph.edges.map(e => ({
                id: e.id,
                from: e.source,
                to: e.target,
                label: e.label,
                color: {
                    color: '#333333',
                    highlight: '#0169CC',
                    hover: '#555555'
                },
                font: {
                    color: '#888888',
                    size: 9,
                    align: 'middle',
                    face: 'monospace',
                    background: '#090909',
                    strokeWidth: 0
                },
                arrows: {
                    to: { enabled: true, scaleFactor: 0.5 }
                },
                smooth: {
                    enabled: true,
                    type: 'continuous',
                    roundness: 0.5
                }
            }))
        );

        const options = {
            autoResize: true,
            height: '100%',
            width: '100%',
            nodes: {
                shape: 'dot'
            },
            physics: {
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -60,
                    centralGravity: 0.015,
                    springLength: 110,
                    springConstant: 0.08,
                    damping: 0.4
                },
                stabilization: {
                    iterations: 150
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true
            }
        };

        const network = new Network(containerRef.current, { nodes: visNodes as any, edges: visEdges as any }, options as any);
        networkRef.current = network;

        network.on('click', (params: any) => {
            if (params.nodes && params.nodes.length > 0) {
                const clickedId = params.nodes[0];
                const nodeObj = data.nodes.find(n => n.id === clickedId) || null;
                setSelectedNode(nodeObj);
            } else {
                setSelectedNode(null);
            }
        });

        return () => {
            network.destroy();
        };
    }, [filteredGraph]);

    const handleZoomIn = () => {
        if (!networkRef.current) return;
        const currentScale = networkRef.current.getScale();
        networkRef.current.moveTo({ scale: currentScale * 1.25 });
    };

    const handleZoomOut = () => {
        if (!networkRef.current) return;
        const currentScale = networkRef.current.getScale();
        networkRef.current.moveTo({ scale: currentScale * 0.8 });
    };

    const handleFit = () => {
        if (!networkRef.current) return;
        networkRef.current.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
    };

    const types = ['ALL', 'Vendor', 'Invoice', 'Reference', 'Project', 'Person', 'Department'];

    return (
        <div className="graph-container">
            {/* Toolbar */}
            <div className="graph-toolbar">
                <div className="toolbar-group">
                    <span className="toolbar-label">Filter:</span>
                    {types.map(t => (
                        <button
                            key={t}
                            className={`filter-btn ${selectedType === t ? 'active' : ''}`}
                            onClick={() => setSelectedType(t)}
                        >
                            {t}
                        </button>
                    ))}
                </div>

                <div className="toolbar-group">
                    <input
                        type="text"
                        className="search-input"
                        placeholder="Search entities or properties..."
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                    />
                    <button className="tool-btn" onClick={handleZoomIn}>+</button>
                    <button className="tool-btn" onClick={handleZoomOut}>-</button>
                    <button className="tool-btn" onClick={handleFit}>Fit</button>
                    {onRefresh && (
                        <button className="tool-btn primary" onClick={onRefresh} disabled={loading}>
                            {loading ? 'Refreshing...' : 'Refresh Graph'}
                        </button>
                    )}
                </div>
            </div>

            {/* Canvas Workspace */}
            <div className="graph-workspace">
                <div ref={containerRef} className="vis-network-container" />

                {/* Node Detail Inspector */}
                {selectedNode && (
                    <div className="node-detail-panel">
                        <div className="panel-header">
                            <span
                                className="node-type-badge"
                                style={{ backgroundColor: typeColorMap[selectedNode.type]?.background || '#0169CC' }}
                            >
                                {selectedNode.type}
                            </span>
                            <button className="close-btn" onClick={() => setSelectedNode(null)}>×</button>
                        </div>
                        <h4 className="node-title">{selectedNode.label}</h4>
                        <div className="node-props">
                            <div className="prop-row">
                                <span className="prop-key">Vertex ID:</span>
                                <span className="prop-val font-mono">{selectedNode.id}</span>
                            </div>
                            {selectedNode.properties && Object.entries(selectedNode.properties).map(([k, v]) => v ? (
                                <div className="prop-row" key={k}>
                                    <span className="prop-key">{k}:</span>
                                    <span className="prop-val">{String(v)}</span>
                                </div>
                            ) : null)}
                        </div>

                        {onSelectRun && (selectedNode.type === 'Invoice' || selectedNode.id.startsWith('invoice_')) && (
                            <button
                                className="node-action-run-btn"
                                onClick={() => onSelectRun(selectedNode.id.replace(/^invoice_/, ''))}
                            >
                                📄 View Invoice Document & Run →
                            </button>
                        )}

                        <div className="connected-edges-summary">
                            <h5>Connections ({data.edges.filter(e => e.source === selectedNode.id || e.target === selectedNode.id).length}):</h5>
                            {data.edges.filter(e => e.source === selectedNode.id || e.target === selectedNode.id).map(e => {
                                const otherId = e.source === selectedNode.id ? e.target : e.source;
                                const isInvoice = otherId.startsWith('invoice_');
                                const cleanInvId = otherId.replace(/^invoice_/, '');

                                return (
                                    <div key={e.id} className="edge-item">
                                        <span className="edge-label">{e.label}</span>
                                        {isInvoice && onSelectRun ? (
                                            <button
                                                className="edge-link-btn font-mono"
                                                onClick={() => onSelectRun(cleanInvId)}
                                                title={`Click to view run: ${cleanInvId}`}
                                            >
                                                {e.source === selectedNode.id ? `→ ${e.target}` : `← ${e.source}`}
                                            </button>
                                        ) : (
                                            <span className="edge-target font-mono">{e.source === selectedNode.id ? `→ ${e.target}` : `← ${e.source}`}</span>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>

            {/* Footer Stats */}
            <div className="graph-stats-footer">
                <span>Entities Showing: <strong>{filteredGraph.nodes.length}</strong> / {data.total_nodes}</span>
                <span>Relationships: <strong>{filteredGraph.edges.length}</strong> / {data.total_edges}</span>
                <span>Database: <strong>NebulaGraph Space: financial_records</strong></span>
            </div>
        </div>
    );
};
