import React, { useEffect, useRef, useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-json';
import { Message } from '../types';
import { FaRobot, FaStop, FaTimes, FaCode, FaCheck, FaTimesCircle, FaImage } from 'react-icons/fa';
import '../styles/AgentActivityPanel.css';

interface AgentActivityPanelProps {
    messages: Message[];
    isProcessing: boolean;
    isWaiting?: boolean;
    onStop?: () => void;
    title?: string;
    onToggleExtractor?: () => void;
    showExtractor?: boolean;
    onClose?: () => void;
    onPreviewImage?: (imageUrl: string, title?: string) => void;
}

// Prism.js Python Code Highlighter Component
const HighlightedCode: React.FC<{ code: string; language?: string }> = ({ code, language = 'python' }) => {
    const grammar = Prism.languages[language] || Prism.languages.python || Prism.languages.javascript;
    const highlighted = Prism.highlight(code, grammar, language);

    return (
        <pre className="tool-args-code font-mono">
            <code dangerouslySetInnerHTML={{ __html: highlighted }} />
        </pre>
    );
};

// Visual tool argument card matching Deepthink architecture
const ToolArgumentsCard: React.FC<{
    toolName: string;
    args?: any;
    showExtractor?: boolean;
    onToggleExtractor?: () => void;
}> = ({ toolName, args, showExtractor, onToggleExtractor }) => {
    if (!args || typeof args !== 'object') return null;

    switch (toolName) {
        case 'delegate_extraction': {
            const docType = args.doc_type || 'document';
            const instructions = args.instructions || '';
            const targetAgents = args.target_agents || ['gemini', 'sarvam'];

            return (
                <div className="tool-args-card">
                    <div className="tool-args-header">
                        <div style={{ display: 'flex', gap: '6px', alignItems: 'center', minWidth: 0, overflow: 'hidden' }}>
                            <span className="tool-badge-pill">doc_type: {docType}</span>
                            <span className="tool-badge-pill">targets: {targetAgents.join(', ')}</span>
                        </div>
                        {onToggleExtractor && (
                            <button
                                className="toggle-extractor-btn"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onToggleExtractor();
                                }}
                            >
                                {showExtractor ? 'Collapse Extractors' : 'View Extractors →'}
                            </button>
                        )}
                    </div>
                    {instructions && (
                        <div className="tool-args-body">
                            <span className="tool-args-label">Instructions:</span>
                            <div className="tool-args-text">{instructions}</div>
                        </div>
                    )}
                </div>
            );
        }

        case 'execute_python': {
            const code = args.code || '';
            return (
                <div className="tool-args-card">
                    <div className="tool-args-header">
                        <span className="tool-args-label"><FaCode style={{ marginRight: '4px' }} /> Python Forensic Script</span>
                    </div>
                    <HighlightedCode code={code} language="python" />
                </div>
            );
        }

        case 'Final_Extraction': {
            const data = args.data || args;
            return (
                <div className="tool-args-card">
                    <div className="tool-args-header">
                        <span className="tool-args-label font-mono"><FaCode style={{ marginRight: '4px' }} /> Final_Extraction Payload</span>
                    </div>
                    <HighlightedCode code={JSON.stringify(data, null, 2)} language="json" />
                </div>
            );
        }

        case 'check_fraud_vectors': {
            const docPath = args.document_path || '';
            const data = args.data || {};
            const vendor = data.vendor?.raw_name || data.vendor || '';

            return (
                <div className="tool-args-card">
                    <div className="tool-args-header">
                        <span className="tool-badge-pill">Check Vector DB</span>
                        {vendor && <span className="tool-badge-pill">Vendor: {vendor}</span>}
                    </div>
                    {docPath && (
                        <div className="tool-args-body">
                            <span className="tool-args-label font-mono" style={{ fontSize: '11px' }}>{docPath}</span>
                        </div>
                    )}
                </div>
            );
        }

        case 'verify_authority': {
            const identifier = args.identifier || '';
            const docTypeOrCountry = args.doc_type_or_country || 'IN';
            return (
                <div className="tool-args-card">
                    <div className="tool-args-header">
                        <span className="tool-args-label font-mono"><FaCode style={{ marginRight: '4px' }} /> Verify Authority / Tax ID</span>
                        <span className="tool-badge-pill">{identifier}</span>
                        <span className="tool-badge-pill">{docTypeOrCountry}</span>
                    </div>
                </div>
            );
        }

        case 'update_records': {
            const entities = Array.isArray(args.entities) ? args.entities : [];
            const data = args.data || {};
            const vendor = data.vendor?.raw_name || data.vendor || 'Unknown';

            return (
                <div className="tool-args-card">
                    <div className="tool-args-header">
                        <span className="tool-badge-pill success">Commit to Ledger & Graph</span>
                        <span className="tool-badge-pill">Vendor: {vendor}</span>
                    </div>
                    {entities.length > 0 && (
                        <div className="tool-args-body" style={{ marginTop: '6px' }}>
                            <span className="tool-args-label">Enriched Graph Entities ({entities.length}):</span>
                            <div className="entities-chip-row">
                                {entities.map((e: any, idx: number) => (
                                    <span key={idx} className="entity-chip font-mono">
                                        {e.edge_type || 'REL'} → {e.entity_name}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        default: {
            return (
                <div className="tool-args-card">
                    <HighlightedCode code={JSON.stringify(args, null, 2)} language="json" />
                </div>
            );
        }
    }
};

// Custom Markdown Renderer that syntax highlights code blocks (including JSON from Sarvam)
const MarkdownRenderer: React.FC<{ content: string }> = ({ content }) => {
    return (
        <ReactMarkdown
            components={{
                code({ inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    const lang = match ? match[1] : '';
                    const codeStr = String(children).replace(/\n$/, '');
                    if (!inline && (lang || codeStr.includes('\n') || codeStr.trim().startsWith('{') || codeStr.trim().startsWith('['))) {
                        const effectiveLang = lang || (codeStr.trim().startsWith('{') || codeStr.trim().startsWith('[') ? 'json' : 'python');
                        return <HighlightedCode code={codeStr} language={effectiveLang} />;
                    }
                    return <code className={className} {...props}>{children}</code>;
                }
            }}
        >
            {content}
        </ReactMarkdown>
    );
};

// Component for rendering Environment/Tool Output with Image Links & Lightbox
const ToolResultDisplay: React.FC<{
    content: any;
    toolName?: string;
    onPreviewImage?: (imageUrl: string, title?: string) => void;
}> = ({ content, toolName, onPreviewImage }) => {
    let parsed: any = null;
    let rawStr = typeof content === 'string' ? content : JSON.stringify(content, null, 2);

    if (typeof content === 'string') {
        try {
            parsed = JSON.parse(content);
        } catch {
            parsed = null;
        }
    } else if (typeof content === 'object') {
        parsed = content;
    }

    const isPass = parsed?.status === 'PASS' || parsed?.status === 'COMMITTED' || parsed?.checksum_passed === true || parsed?.is_valid === true;
    const isFail = parsed?.status === 'FAIL' || parsed?.status === 'FAILED' || parsed?.status === 'ERROR' || parsed?.checksum_passed === false || parsed?.is_valid === false;

    // Extract any image paths referenced in content and deduplicate by filename
    const rawMatches = rawStr.match(/(\/[^\s"']+\.(?:png|jpg|jpeg|webp)|file:\/\/[^\s"']+\.(?:png|jpg|jpeg|webp))/gi) || [];
    const seenFilenames = new Set<string>();
    const imageMatches = rawMatches.filter(imgPath => {
        const fname = imgPath.replace(/^file:\/\//, '').split('/').pop() || '';
        if (!fname || seenFilenames.has(fname)) return false;
        seenFilenames.add(fname);
        return true;
    });

    return (
        <div className="tool-result-wrapper">
            <div className="tool-result-header">
                <span className="tool-result-title">
                    ENVIRONMENT OUTPUT {toolName ? `— [${toolName}]` : ''}
                </span>
                {isPass && <span className="status-pill pass"><FaCheck /> PASSED</span>}
                {isFail && <span className="status-pill fail"><FaTimesCircle /> FAILED</span>}
            </div>

            {/* Render Image Links as Interactive Forensic Preview Chips */}
            {imageMatches.length > 0 && onPreviewImage && (
                <div className="image-links-container">
                    {imageMatches.map((imgPath, idx) => {
                        const cleanPath = imgPath.replace(/^file:\/\//, '');
                        const filename = cleanPath.split('/').pop() || 'image.png';
                        // Map absolute paths to /outputs or /uploads
                        const serveUrl = (cleanPath.includes('/outputs/') || filename.startsWith('extracted_') || filename.startsWith('crop_') || cleanPath.includes('/scratch/'))
                            ? `/outputs/${filename}`
                            : `/uploads/${filename}`;

                        return (
                            <button
                                key={idx}
                                className="image-preview-chip"
                                onClick={() => onPreviewImage(serveUrl, filename)}
                            >
                                <FaImage style={{ marginRight: '5px' }} />
                                <span>Forensic Crop: {filename}</span>
                            </button>
                        );
                    })}
                </div>
            )}

            <div className="tool-result-content">
                {parsed ? (
                    <HighlightedCode code={JSON.stringify(parsed, null, 2)} language="json" />
                ) : (
                    <pre className="stdout-result font-mono">{rawStr}</pre>
                )}
            </div>
        </div>
    );
};

export const AgentActivityPanel: React.FC<AgentActivityPanelProps> = ({
    messages,
    isProcessing,
    isWaiting = false,
    onStop,
    title = 'Agent Activity',
    onToggleExtractor,
    showExtractor,
    onClose,
    onPreviewImage
}) => {
    const scrollRef = useRef<HTMLDivElement>(null);
    const [isUserScrolledUp, setIsUserScrolledUp] = useState(false);

    const handleScroll = useCallback(() => {
        const el = scrollRef.current;
        if (el) {
            const { scrollTop, scrollHeight, clientHeight } = el;
            const isAtBottom = Math.abs(scrollHeight - clientHeight - scrollTop) <= 4;
            setIsUserScrolledUp(!isAtBottom);
        }
    }, []);

    useEffect(() => {
        if (!isUserScrolledUp && scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages.length, isUserScrolledUp]);

    return (
        <div className="agent-activity-panel-container">
            <div className="agent-panel-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FaRobot style={{ color: 'var(--accent-blue)', fontSize: '15px' }} />
                    <h3>{title}</h3>
                </div>

                <div className="header-info">
                    {isProcessing && onStop && (
                        <button className="stop-button" onClick={onStop}>
                            <FaStop /> Stop
                        </button>
                    )}
                    {onClose && (
                        <button className="close-panel-btn" onClick={onClose} title="Collapse Panel">
                            <FaTimes />
                        </button>
                    )}
                </div>
            </div>

            <div
                className="agent-messages-container custom-scrollbar"
                ref={scrollRef}
                onScroll={handleScroll}
            >
                {messages.length === 0 && !isProcessing && (
                    <div className="empty-indicator">Waiting for execution events...</div>
                )}

                {messages.map((msg, idx) => {
                    // Agent Messages (Thoughts, Code generation, Tool calls)
                    if (msg.role === 'agent' || msg.type === 'AIMessage') {
                        const hasThought = Boolean(msg.thought && msg.thought.trim());
                        const hasContent = Boolean(msg.content && typeof msg.content === 'string' && msg.content.trim());
                        const toolCalls = msg.tool_calls || [];

                        return (
                            <div key={idx} className="message-card agent-message-card">
                                {/* Reasoning / Thought Narrative */}
                                {hasThought && (
                                    <div className="thought-block">
                                        <MarkdownRenderer content={String(msg.thought)} />
                                    </div>
                                )}

                                {hasContent && !hasThought && (
                                    <div className="narrative-block">
                                        <MarkdownRenderer content={String(msg.content)} />
                                    </div>
                                )}

                                {/* Specific Visual Tool Argument Cards */}
                                {toolCalls.map((tc, tcIdx) => {
                                    const toolName = tc.name || tc.function?.name || 'tool';
                                    const toolArgs = tc.args || (typeof tc.function?.arguments === 'string' ? JSON.parse(tc.function.arguments) : tc.function?.arguments) || {};
                                    return (
                                        <ToolArgumentsCard
                                            key={tcIdx}
                                            toolName={toolName}
                                            args={toolArgs}
                                            showExtractor={showExtractor}
                                            onToggleExtractor={onToggleExtractor}
                                        />
                                    );
                                })}
                            </div>
                        );
                    }

                    // Environment Output / Tool Result
                    if (msg.role === 'function' || msg.type === 'ToolMessage' || msg.role === 'system') {
                        return (
                            <div key={idx} className="message-card tool-message-card">
                                <ToolResultDisplay
                                    content={msg.content}
                                    toolName={msg.tool_name}
                                    onPreviewImage={onPreviewImage}
                                />
                            </div>
                        );
                    }

                    return null;
                })}

                {/* Processing Indicator */}
                {isProcessing && (
                    <div className="processing-indicator">
                        <div className="spinner"></div>
                        <span>{isWaiting ? 'Agent is waiting...' : 'Agent is processing...'}</span>
                    </div>
                )}
            </div>
        </div>
    );
};
