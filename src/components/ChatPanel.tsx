import React, { useState, useRef, useEffect, useCallback } from "react";
import { streamChat, getChatHistory, startNewSession, clearChatHistory } from "../lib/sidecar";
import MarkdownPreview from "./MarkdownPreview";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  sessionId?: number;
}

const IconTrash = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/>
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
    <path d="M10 11v6"/><path d="M14 11v6"/>
    <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
  </svg>
);

const IconX = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);

const IconSend = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);

const IconPlus = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
);

const IconBot = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 8V4H8" />
    <rect width="16" height="12" x="4" y="8" rx="2" />
    <path d="M2 14h2" />
    <path d="M20 14h2" />
    <path d="M15 13v2" />
    <path d="M9 13v2" />
  </svg>
);

const IconUser = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
);

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex gap-3 fade-in-up w-full ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {/* Avatar */}
      <div className={`shrink-0 w-7 h-7 rounded-lg flex items-center justify-center mt-1 border ${
        isUser 
          ? "bg-[var(--surface-alt)] text-[var(--text-secondary)] border-[var(--border)]" 
          : "bg-[var(--accent-subtle)] text-[var(--accent-text)] border-[var(--border-glow)]"
      }`}>
        {isUser ? <IconUser /> : <IconBot />}
      </div>

      {/* Bubble */}
      <div className={`max-w-[85%] text-[13px] leading-relaxed ${
        isUser 
          ? "bg-[var(--btn-bg)] text-[var(--text-primary)] border border-[var(--border-hover)] rounded-2xl rounded-tr-sm px-4 py-2.5 shadow-sm" 
          : "text-[var(--text-secondary)] py-1.5"
      }`}>
        {isUser ? (
          <p className="m-0 whitespace-pre-wrap">{message.content}</p>
        ) : message.streaming ? (
          <div>
            <p className="m-0 whitespace-pre-wrap inline">{message.content}</p>
            <span className="chat-cursor" />
          </div>
        ) : (
          <div className="text-[13px]">
            <MarkdownPreview content={message.content} compact />
          </div>
        )}
      </div>
    </div>
  );
}

interface ChatPanelProps {
  docId: string;
  onClose: () => void;
  onDocumentEdited?: () => void;
}

export default function ChatPanel({ docId, onClose, onDocumentEdited }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<number>(1);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, input]); // also scroll when input changes if needed

  useEffect(() => { inputRef.current?.focus(); }, []);

  // Auto resize input
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
  };

  // Load chat history on mount or when docId changes
  useEffect(() => {
    // Clear stale state immediately so previous doc's messages don't linger
    setMessages([]);
    setSessionId(1);
    setError(null);

    getChatHistory(docId).then((history) => {
      if (history.length > 0) {
        const msgs: Message[] = history.map((h) => ({
          id: String(h.id),
          role: h.role,
          content: h.content,
          sessionId: h.session_id,
        }));
        setMessages(msgs);
        setSessionId(Math.max(...history.map(h => h.session_id)));
      }
    }).catch(() => { /* ignore load errors */ });
  }, [docId]);

  // Close drawer on Escape
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  const sendMessage = useCallback(async () => {
    const question = input.trim();
    if (!question || streaming) return;
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    setError(null);

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: question, sessionId };
    const assistantMsg: Message = { id: crypto.randomUUID(), role: "assistant", content: "", streaming: true, sessionId };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    try {
      await streamChat(docId, question, (token) => {
        setMessages((prev) =>
          prev.map((m) => m.id === assistantMsg.id ? { ...m, content: m.content + token } : m)
        );
      }, sessionId, onDocumentEdited);
    } catch (err) {
      setError(String(err));
      setMessages((prev) => prev.filter((m) => m.id !== assistantMsg.id));
    } finally {
      setMessages((prev) =>
        prev.map((m) => m.id === assistantMsg.id ? { ...m, streaming: false } : m)
      );
      setStreaming(false);
    }
  }, [docId, input, streaming, sessionId, onDocumentEdited]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  return (
    <div className="flex flex-col h-full bg-[var(--surface)] relative overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)] shrink-0 bg-[var(--surface-glass)] backdrop-blur-md z-10">
        <div className="flex items-center gap-2.5">
          <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent)] shadow-[0_0_8px_var(--accent-glow)] pulse" />
          <span className="text-[11px] font-medium text-[var(--text-secondary)] tracking-wide uppercase">
            Chat with Document
          </span>
        </div>
        <div className="flex gap-1">
          <button
            onClick={async () => {
              try {
                await clearChatHistory(docId);
                setMessages([]);
                setError(null);
                setSessionId(1);
              } catch (e) {
                setError(String(e));
              }
            }}
            title="Clear conversation"
            className="btn-icon"
          >
            <IconTrash />
          </button>
          <button onClick={onClose} title="Close chat" className="btn-icon">
            <IconX />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 flex flex-col gap-5 pb-40">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center px-4 py-12 fade-in">
            <div className="w-12 h-12 rounded-2xl bg-[var(--accent-subtle)] border border-[var(--border-glow)] flex items-center justify-center mb-5 glow-pulse text-[var(--accent-text)]">
              <IconBot />
            </div>
            <p className="text-[13px] font-medium text-[var(--text-secondary)] mb-1.5">
              Ask anything about this document
            </p>
            <p className="text-[11px] text-[var(--text-faint)] m-0 leading-relaxed max-w-[240px]">
              Answers are grounded in this document only.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => {
          const prevSession = idx > 0 ? messages[idx - 1].sessionId : msg.sessionId;
          const showDivider = msg.sessionId !== prevSession;
          return (
            <React.Fragment key={msg.id}>
              {showDivider && (
                <div className="flex items-center gap-2 py-2 text-[var(--text-faint)] text-[10px] uppercase tracking-wide">
                  <div className="flex-1 h-px bg-[var(--border)]" />
                  <span>New Session</span>
                  <div className="flex-1 h-px bg-[var(--border)]" />
                </div>
              )}
              <MessageBubble message={msg} />
            </React.Fragment>
          );
        })}

        {error && (
          <div className="p-3 rounded-lg border border-[rgba(248,113,113,0.2)] bg-[var(--error-bg)] text-xs text-[var(--error)] fade-in">
            {error}
          </div>
        )}
      </div>

      {/* Input Area (Floating) */}
      <div className="absolute bottom-0 left-0 right-0 px-4 pb-4 pt-12 bg-gradient-to-t from-[var(--surface)] via-[var(--surface)] to-transparent pointer-events-none">
        <div className="flex flex-col gap-2 max-w-2xl mx-auto pointer-events-auto">
          {/* New Session Button */}
          {messages.length > 0 && (
            <div className="flex justify-center mb-1">
              <button
                onClick={async () => {
                  try {
                    const newId = await startNewSession(docId);
                    setSessionId(newId);
                  } catch (e) {
                    setError(String(e));
                  }
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[var(--surface-alt)] border border-[var(--border-strong)] text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--border-hover)] transition-all scale-in shadow-sm hover:shadow-md"
                title="Start a new session with fresh context"
              >
                <IconPlus />
                <span>New Topic</span>
              </button>
            </div>
          )}

          {/* Input Box */}
          <div className="relative flex items-end gap-2 bg-[var(--surface-alt)] border border-[var(--border-strong)] rounded-xl p-1.5 shadow-lg focus-within:border-[var(--accent)] focus-within:ring-2 focus-within:ring-[var(--accent-subtle)] transition-all">
            <textarea
              ref={inputRef}
              id="chat-input"
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question…"
              rows={1}
              disabled={streaming}
              className="flex-1 resize-none bg-transparent border-none text-[13px] text-[var(--text-primary)] placeholder-[var(--text-faint)] py-2 px-3 outline-none focus:outline-none focus-visible:outline-none focus:ring-0 min-h-[40px] max-h-[120px]"
              style={{ overflowY: input.length > 100 ? "auto" : "hidden", outline: "none" }}
            />
            <button
              id="chat-send-btn"
              onClick={sendMessage}
              disabled={!input.trim() || streaming}
              className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all ${
                input.trim() && !streaming 
                  ? "bg-gradient-to-br from-[var(--accent)] to-[#818cf8] text-white shadow-[0_2px_10px_var(--accent-glow)] hover:scale-105" 
                  : "bg-[var(--surface-hover)] text-[var(--text-faint)]"
              }`}
            >
              {streaming ? (
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full spin" />
              ) : (
                <IconSend />
              )}
            </button>
          </div>
          
          <p className="text-[10px] text-center text-[var(--text-faint)] m-0 mt-1 tracking-wide uppercase">
            Grounded in this document only · Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}

