import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { askApi } from '../../api/api';
import { QueryResponse, SourceResponse } from '../../api/models';

interface ChatProps {
    conversationId: string;
    onMetadataUpdate: (response: QueryResponse) => void;
    onFirstMessage: (text: string) => void;
}

interface Message {
    id: string;
    role: 'user' | 'bot';
    text: string;
    sources?: SourceResponse[];
}

const Citation: React.FC<{ index: string; sources?: SourceResponse[] }> = ({ index, sources }) => {
    const [isHovered, setIsHovered] = useState(false);
    const [isClicked, setIsClicked] = useState(false);

    const idx = parseInt(index);
    const source = sources ? sources[idx - 1] : undefined;

    if (!source) return <span className="citation">[{index}]</span>;
    return (
        <span
            className={`citation clickable ${(isHovered || isClicked) ? 'active' : ''}`}
            title={`${source.document}${source.page ? ` (Page ${source.page})` : ''} - Relevance: ${source.relevance_score}`}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            onClick={() => setIsClicked(!isClicked)}
        >
            {(isHovered || isClicked) ? '?' : `[${index}]`}
        </span>
    );
};

export const Chat: React.FC<ChatProps> = ({ conversationId, onMetadataUpdate, onFirstMessage }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const formatCitations = (text: string) => {
        // Replace [1], [2] with <cite>1</cite>, <cite>2</cite>
        return text.replace(/\[(\d+)\]/g, '<cite>$1</cite>');
    };

    // Load messages from localStorage on mount/conversation change
    useEffect(() => {
        if (!conversationId) return;
        const saved = localStorage.getItem(`nexus_msgs_${conversationId}`);
        if (saved) {
            setMessages(JSON.parse(saved));
        } else {
            setMessages([]);
        }
    }, [conversationId]);

    // Persist messages to localStorage
    useEffect(() => {
        if (!conversationId || messages.length === 0) return;
        localStorage.setItem(`nexus_msgs_${conversationId}`, JSON.stringify(messages));
    }, [messages, conversationId]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!inputValue.trim() || isLoading) return;

        const userText = inputValue;
        setInputValue('');

        if (messages.length === 0) {
            onFirstMessage(userText);
        }

        const newUserMessage: Message = { id: Date.now().toString(), role: 'user', text: userText };
        setMessages(prev => [...prev, newUserMessage]);
        setIsLoading(true);

        try {
            // Map our internal Message format to API ChatMessage format
            const chatHistory = messages.slice(-6).map(m => ({
                role: m.role,
                text: m.text
            }));

            const response = await askApi({
                question: userText,
                history: chatHistory
            });

            // Pass full response to parent (Debug Panel)
            onMetadataUpdate(response);

            setMessages(prev => [...prev, {
                id: (Date.now() + 1).toString(),
                role: 'bot',
                text: response.answer,
                sources: response.sources // Now storing sources per message
            }]);
        } catch (error) {
            setMessages(prev => [...prev, {
                id: (Date.now() + 1).toString(),
                role: 'bot',
                text: "An error occurred while fetching the response."
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <>
            <div className="chat-history">
                {messages.length === 0 && (
                    <div className="empty-state">
                        <h3>Clearpath Nexus</h3>
                        <p>
                            Grounded Technical Support Assistant. Powered by hybrid retrieval
                            and cross-encoder reranking over internal documentation.
                        </p>
                        <p style={{ marginTop: '12px', opacity: 0.7, fontSize: '12px' }}>
                            Start by asking about clearing paths, feature comparison, or troubleshooting.
                        </p>
                    </div>
                )}
                {messages.map(msg => (
                    <div key={msg.id} className={`message ${msg.role}`}>
                        {msg.role === 'bot' && (
                            <span className="model-badge">Assistant • Llama 3.3 70B</span>
                        )}
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            rehypePlugins={[rehypeRaw]}
                            components={{
                                // @ts-ignore
                                cite: ({ children }) => (
                                    <Citation index={String(children)} sources={msg.sources} />
                                )
                            }}
                        >
                            {msg.role === 'bot' ? formatCitations(msg.text) : msg.text}
                        </ReactMarkdown>
                    </div>
                ))}
                {isLoading && (
                    <div className="message bot" style={{ opacity: 0.7 }}>
                        Thinking...
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="chat-input">
                <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="Type your question..."
                    disabled={isLoading}
                />
                <button onClick={handleSend} disabled={isLoading || !inputValue.trim()}>
                    Send
                </button>
            </div>
        </>
    );
};
