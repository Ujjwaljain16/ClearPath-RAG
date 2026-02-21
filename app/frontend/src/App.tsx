import React, { useState, useEffect } from 'react';
import { Chat } from './components/Chat/Chat';
import { DebugPanel } from './components/Chat/DebugPanel';
import { QueryResponse } from './api/models';

interface ConversationMetadata {
    id: string;
    title: string;
    lastUpdated: number;
}

export const App = () => {
    const [lastResponse, setLastResponse] = useState<QueryResponse | null>(null);
    const [conversations, setConversations] = useState<ConversationMetadata[]>([]);
    const [currentConvId, setCurrentConvId] = useState<string | null>(null);

    // Initialize/Load conversation list from localStorage
    useEffect(() => {
        const saved = localStorage.getItem('nexus_conversations');
        if (saved) {
            const list = JSON.parse(saved);
            setConversations(list);
            if (list.length > 0) setCurrentConvId(list[0].id);
        } else {
            createNewChat();
        }
    }, []);

    const createNewChat = () => {
        const id = `chat_${Date.now()}`;
        const newConv = { id, title: 'New Chat', lastUpdated: Date.now() };
        setConversations(prev => [newConv, ...prev]);
        setCurrentConvId(id);
        localStorage.setItem('nexus_conversations', JSON.stringify([newConv, ...conversations]));
    };

    const updateTitle = (id: string, text: string) => {
        setConversations(prev => prev.map(c =>
            c.id === id ? { ...c, title: text.slice(0, 30) + '...', lastUpdated: Date.now() } : c
        ));
    };

    const deleteChat = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        const updated = conversations.filter(c => c.id !== id);
        setConversations(updated);
        localStorage.removeItem(`nexus_msgs_${id}`);
        if (currentConvId === id) {
            setCurrentConvId(updated.length > 0 ? updated[0].id : null);
        }
    };

    useEffect(() => {
        if (conversations.length > 0) {
            localStorage.setItem('nexus_conversations', JSON.stringify(conversations));
        }
    }, [conversations]);

    return (
        <div className="chat-container">
            <div className="chat-sidebar">
                <button className="new-chat-btn" onClick={createNewChat}>
                    <span>+</span> New Chat
                </button>
                <div className="chat-list">
                    {conversations.map(conv => (
                        <div
                            key={conv.id}
                            className={`history-item ${currentConvId === conv.id ? 'active' : ''}`}
                            onClick={() => setCurrentConvId(conv.id)}
                        >
                            <span className="title">{conv.title}</span>
                            <button className="delete-btn" onClick={(e) => deleteChat(e, conv.id)} title="Delete Chat">
                                ✕
                            </button>
                        </div>
                    ))}
                </div>
            </div>

            <div className="chat-main">
                <header className="header-area">
                    <h1 className="header-title">Clearpath Nexus</h1>
                    <p className="header-subtitle">Grounded Hybrid RAG System • FAISS + BM25 + Cross-Encoder</p>
                </header>

                <Chat
                    key={currentConvId}
                    conversationId={currentConvId || ''}
                    onMetadataUpdate={(res) => setLastResponse(res as unknown as QueryResponse)} // Hack for quick fix if Chat.tsx still passes just metadata
                    onFirstMessage={(text) => currentConvId && updateTitle(currentConvId, text)}
                />
            </div>

            <DebugPanel lastResponse={lastResponse} />
        </div>
    );
};

export default App;
