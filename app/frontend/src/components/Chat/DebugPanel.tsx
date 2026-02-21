import { QueryResponse, SourceResponse } from '../../api/models';

interface DebugPanelProps {
    lastResponse: QueryResponse | null;
}

export const DebugPanel: React.FC<DebugPanelProps> = ({ lastResponse }) => {

    if (!lastResponse) {
        return (
            <div className="debug-panel">
                <h2>Observability</h2>
                <div style={{ color: '#64748b', fontSize: '14px', marginTop: '20px' }}>
                    System idle. Awaiting query...
                </div>
            </div>
        );
    }

    const {
        metadata
    } = lastResponse;

    const {
        model_used,
        classification,
        tokens,
        latency_ms,
        retrieval_latency_ms,
        num_chunks_retrieved,
        avg_similarity_score,
        evaluator_flags,
    } = metadata;

    const sources = lastResponse.sources;
    const hasFlags = evaluator_flags.length > 0;

    return (
        <div className="debug-panel">
            <h2>Observability</h2>

            <div className="debug-section">
                <div className="debug-section-title">System Status</div>
                <div className="metric-row">
                    <span className="metric-label">Integrity</span>
                    {hasFlags ? (
                        <span className="status-badge low">Low Confidence</span>
                    ) : (
                        <span className="status-badge high">Grounded</span>
                    )}
                </div>
                {hasFlags && (
                    <div style={{ fontSize: '11px', color: '#dc2626', marginTop: '4px' }}>
                        Flags: {evaluator_flags.join(", ")}
                    </div>
                )}
            </div>

            <div className="debug-section">
                <div className="debug-section-title">Intelligence Layer</div>
                <div className="metric-row">
                    <span className="metric-label">Core LLM</span>
                    <span className="metric-value">{model_used}</span>
                </div>
                <div className="metric-row">
                    <span className="metric-label">Classifier</span>
                    <span className="metric-value">{classification}</span>
                </div>
                <div className="metric-row">
                    <span className="metric-label">Token Load</span>
                    <span className="metric-value">{tokens.input} In / {tokens.output} Out</span>
                </div>
            </div>

            <div className="debug-section">
                <div className="debug-section-title">Retrieval & RAG</div>
                <div className="metric-row">
                    <span className="metric-label">FAISS Chunks</span>
                    <span className="metric-value">{num_chunks_retrieved}</span>
                </div>
                <div className="metric-row">
                    <span className="metric-label">Avg Similarity</span>
                    <span className="metric-value">{avg_similarity_score.toFixed(3)}</span>
                </div>
                <div className="metric-row">
                    <span className="metric-label">RAG Latency</span>
                    <span className="metric-value">{retrieval_latency_ms} ms</span>
                </div>
            </div>

            <div className="debug-section">
                <div className="debug-section-title">Performance</div>
                <div className="metric-row">
                    <span className="metric-label">Total E2E</span>
                    <span className="metric-value" style={{ color: latency_ms < 500 ? '#16a34a' : 'inherit' }}>
                        {latency_ms} ms
                    </span>
                </div>
            </div>

            {sources && sources.length > 0 && (
                <div className="debug-section">
                    <div className="debug-section-title">Context Sources</div>
                    {sources.slice(0, 3).map((s: SourceResponse, i: number) => (
                        <div key={i} className="source-item">
                            <span className="name">{s.document}</span>
                            <div style={{ display: 'flex', justifyContent: 'space-between', opacity: 0.7 }}>
                                <span>{s.section || 'General'}</span>
                                {s.page && <span>Page {s.page}</span>}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
