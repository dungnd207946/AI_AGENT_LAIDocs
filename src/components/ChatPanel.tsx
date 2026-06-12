import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { streamChat, getChatHistory, startNewSession, clearChatHistory, deleteSession, listDocuments, compareRetrieval } from "../lib/sidecar";
import type { Evidence, CompareResult, DocSummary } from "../lib/sidecar";
import MarkdownPreview from "./MarkdownPreview";
import CitationChips from "./CitationChips";
import ReasoningChain from "./ReasoningChain";
import CompareDrawer from "./CompareDrawer";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  sessionId?: number;
  evidence?: Evidence[];
  chain?: string;
}

const DEMO_MODE_KEY = "laidocs-demo-mode";

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

const IconChevronDown = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9"/>
  </svg>
);

const IconTrashSmall = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/>
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
    <path d="M10 11v6"/><path d="M14 11v6"/>
    <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
  </svg>
);

const IconChat = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
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

const IconScale = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v18" /><path d="M5 7h14" /><path d="M5 7l-3 6a4 4 0 0 0 6 0z" /><path d="M19 7l3 6a4 4 0 0 1-6 0z" />
    <path d="M8 21h8" />
  </svg>
);

function MessageBubble({ message, onJumpToSource }: { message: Message; onJumpToSource?: (ev: Evidence) => void }) {
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
      <div className={`max-w-[85%] min-w-0 text-[13px] leading-relaxed ${
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
            {message.chain && <ReasoningChain chain={message.chain} />}
            {message.evidence && message.evidence.length > 0 && (
              <CitationChips evidence={message.evidence} onJump={onJumpToSource} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface ChatPanelProps {
  initialDocId: string;            // file the chat was opened from (seeds scope)
  onClose: () => void;
  onDocumentEdited?: () => void;
  onJumpToSource?: (ev: Evidence) => void;
}

export default function ChatPanel({ initialDocId, onClose, onDocumentEdited, onJumpToSource }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<number>(1);
  const [sessionMenuOpen, setSessionMenuOpen] = useState(false);

  // Scope: which documents this turn may read/edit. Transient (not persisted).
  const [scopeDocIds, setScopeDocIds] = useState<string[]>([initialDocId]);
  const [allDocs, setAllDocs] = useState<DocSummary[]>([]);
  const [scopeMenuOpen, setScopeMenuOpen] = useState(false);

  // Demo mode — surfaces the RAG-vs-GraphRAG compare tool. Persisted locally.
  const [demoMode, setDemoMode] = useState<boolean>(() => {
    try { return localStorage.getItem(DEMO_MODE_KEY) === "1"; } catch { return false; }
  });
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [compareQuestion, setCompareQuestion] = useState("");

  useEffect(() => {
    try { localStorage.setItem(DEMO_MODE_KEY, demoMode ? "1" : "0"); } catch { /* ignore */ }
  }, [demoMode]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const sessionMenuRef = useRef<HTMLDivElement>(null);
  const scopeMenuRef = useRef<HTMLDivElement>(null);

  const docTitle = useCallback(
    (id: string) => {
      const d = allDocs.find((x) => x.id === id);
      return d ? (d.title || d.filename) : id;
    },
    [allDocs],
  );

  // Show one conversation at a time: only the active session's messages.
  const visibleMessages = useMemo(
    () => messages.filter((m) => (m.sessionId ?? 1) === sessionId),
    [messages, sessionId],
  );

  // Every session that exists for this doc, plus the active one (a freshly
  // created session has no saved messages yet but must still be selectable).
  const sessionList = useMemo(() => {
    const ids = new Set<number>(messages.map((m) => m.sessionId ?? 1));
    ids.add(sessionId);
    return Array.from(ids).sort((a, b) => a - b);
  }, [messages, sessionId]);

  // Label a session by its first user message so switching is meaningful.
  const sessionLabel = useCallback(
    (sid: number): string => {
      const firstUser = messages.find((m) => (m.sessionId ?? 1) === sid && m.role === "user");
      if (!firstUser) return `Session ${sid} · empty`;
      const snippet = firstUser.content.trim().replace(/\s+/g, " ").slice(0, 32);
      return `Session ${sid} · ${snippet}${firstUser.content.length > 32 ? "…" : ""}`;
    },
    [messages],
  );

  // Close the session menu on outside click
  useEffect(() => {
    if (!sessionMenuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (sessionMenuRef.current && !sessionMenuRef.current.contains(e.target as Node)) {
        setSessionMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [sessionMenuOpen]);

  // Close the scope menu on outside click
  useEffect(() => {
    if (!scopeMenuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (scopeMenuRef.current && !scopeMenuRef.current.contains(e.target as Node)) {
        setScopeMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [scopeMenuOpen]);

  const toggleScopeDoc = useCallback((id: string) => {
    setScopeDocIds((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id],
    );
  }, []);

  // Delete a single session; if it was active, fall back to another one.
  const handleDeleteSession = useCallback(async (sid: number) => {
    try {
      await deleteSession(sid);
      const remaining = messages
        .map((m) => m.sessionId ?? 1)
        .filter((id) => id !== sid);
      setMessages((prev) => prev.filter((m) => (m.sessionId ?? 1) !== sid));
      if (sid === sessionId) {
        const next = remaining.length ? Math.max(...remaining) : 1;
        setSessionId(next);
      }
    } catch (e) {
      setError(String(e));
    }
  }, [sessionId, messages]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [visibleMessages, input]); // also scroll when input changes if needed

  useEffect(() => { inputRef.current?.focus(); }, []);

  // Auto resize input
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
  };

  // Load the document list once (for the scope picker).
  useEffect(() => {
    listDocuments().then(setAllDocs).catch(() => { /* ignore */ });
  }, []);

  // Seed scope with the file the panel was opened from.
  useEffect(() => {
    setScopeDocIds([initialDocId]);
  }, [initialDocId]);

  // Load GLOBAL chat history on mount (sessions are not tied to a doc).
  useEffect(() => {
    setMessages([]);
    setSessionId(1);
    setError(null);

    getChatHistory().then((history) => {
      if (history.length > 0) {
        const msgs: Message[] = history.map((h) => ({
          id: String(h.id),
          role: h.role,
          content: h.content,
          sessionId: h.session_id,
          evidence: h.evidence,
          chain: h.chain,
        }));
        setMessages(msgs);
        setSessionId(Math.max(...history.map(h => h.session_id)));
      }
    }).catch(() => { /* ignore load errors */ });
  }, []);

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
    if (scopeDocIds.length === 0) { setError("Hãy chọn ít nhất một tài liệu."); return; }
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    setError(null);

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: question, sessionId };
    const assistantMsg: Message = { id: crypto.randomUUID(), role: "assistant", content: "", streaming: true, sessionId };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    try {
      await streamChat(scopeDocIds, question, {
        onChunk: (token) => {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantMsg.id ? { ...m, content: m.content + token } : m)
          );
        },
        onEvidence: (evidence) => {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantMsg.id ? { ...m, evidence } : m)
          );
        },
        onChain: (chain) => {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantMsg.id ? { ...m, chain } : m)
          );
        },
        onEdited: onDocumentEdited,
      }, sessionId);
    } catch (err) {
      setError(String(err));
      setMessages((prev) => prev.filter((m) => m.id !== assistantMsg.id));
    } finally {
      setMessages((prev) =>
        prev.map((m) => m.id === assistantMsg.id ? { ...m, streaming: false } : m)
      );
      setStreaming(false);
    }
  }, [scopeDocIds, input, streaming, sessionId, onDocumentEdited]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  // Run the RAG-vs-GraphRAG compare on the current input, or the last question asked.
  const runComparison = useCallback(async () => {
    const lastUser = [...visibleMessages].reverse().find((m) => m.role === "user");
    const question = input.trim() || lastUser?.content?.trim() || "";
    if (!question) {
      setCompareError("Type a question (or ask one first) to compare.");
      setCompareQuestion("");
      setCompareResult(null);
      setCompareOpen(true);
      return;
    }
    setCompareQuestion(question);
    setCompareResult(null);
    setCompareError(null);
    setCompareLoading(true);
    setCompareOpen(true);
    try {
      const result = await compareRetrieval(scopeDocIds[0], question);
      setCompareResult(result);
    } catch (e) {
      setCompareError(String(e));
    } finally {
      setCompareLoading(false);
    }
  }, [scopeDocIds, input, visibleMessages]);

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
        <div className="flex items-center gap-1">
          {/* Demo mode toggle — reveals the RAG-vs-GraphRAG compare tool */}
          <button
            onClick={() => setDemoMode((v) => !v)}
            title={demoMode ? "Demo mode on — hides the compare tool when off" : "Demo mode off — turn on to compare RAG vs GraphRAG"}
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium uppercase tracking-wide transition-all border ${
              demoMode
                ? "text-[var(--accent-text)] bg-[var(--accent-subtle)] border-[var(--border-glow)]"
                : "text-[var(--text-faint)] bg-transparent border-[var(--border)] hover:text-[var(--text-muted)]"
            }`}
          >
            <IconScale />
            Demo
          </button>
          <button onClick={onClose} title="Close chat" className="btn-icon">
            <IconX />
          </button>
        </div>
      </div>

      {/* Session switcher — always visible; resume or delete any conversation */}
      <div className="flex items-center gap-2.5 px-4 py-2.5 border-b border-[var(--border)] shrink-0 bg-[var(--surface-glass)] backdrop-blur-md z-20">
        <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-faint)] shrink-0">
          Conversation
        </span>

        {/* Custom dropdown */}
        <div ref={sessionMenuRef} className="relative flex-1 min-w-0">
          <button
            type="button"
            onClick={() => setSessionMenuOpen((v) => !v)}
            disabled={streaming}
            aria-haspopup="listbox"
            aria-expanded={sessionMenuOpen}
            title="Switch or delete a conversation"
            className="w-full flex items-center gap-2 bg-[var(--surface-alt)] border border-[var(--border-strong)] rounded-lg text-[12px] text-[var(--text-primary)] pl-2.5 pr-2 py-1.5 outline-none hover:border-[var(--border-hover)] focus-visible:border-[var(--accent)] focus-visible:ring-2 focus-visible:ring-[var(--accent-subtle)] transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="text-[var(--accent-text)] shrink-0 flex"><IconChat /></span>
            <span className="flex-1 min-w-0 truncate text-left">{sessionLabel(sessionId)}</span>
            <span
              className="text-[var(--text-faint)] shrink-0 flex transition-transform duration-200"
              style={{ transform: sessionMenuOpen ? "rotate(180deg)" : "rotate(0deg)" }}
            >
              <IconChevronDown />
            </span>
          </button>

          {sessionMenuOpen && (
            <div
              role="listbox"
              className="absolute left-0 right-0 top-[calc(100%+6px)] z-30 max-h-72 overflow-y-auto rounded-xl border border-[var(--border-strong)] bg-[var(--surface)] shadow-2xl shadow-black/40 p-1.5 scale-in origin-top"
            >
              {[...sessionList].reverse().map((sid) => {
                const isActive = sid === sessionId;
                return (
                  <div
                    key={sid}
                    role="option"
                    aria-selected={isActive}
                    onClick={() => { setSessionId(sid); setSessionMenuOpen(false); }}
                    className={`group flex items-center gap-2 rounded-lg pl-2.5 pr-1.5 py-2 cursor-pointer transition-colors ${
                      isActive
                        ? "bg-[var(--accent-subtle)] text-[var(--text-primary)]"
                        : "text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]"
                    }`}
                  >
                    <span
                      className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                        isActive ? "bg-[var(--accent)] shadow-[0_0_6px_var(--accent-glow)]" : "bg-transparent"
                      }`}
                    />
                    <span className="flex-1 min-w-0 truncate text-[12px] leading-snug">
                      {sessionLabel(sid)}
                    </span>
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); handleDeleteSession(sid); }}
                      disabled={streaming}
                      title="Delete this conversation"
                      aria-label="Delete this conversation"
                      className="shrink-0 w-7 h-7 -mr-0.5 rounded-md flex items-center justify-center text-[var(--text-faint)] opacity-0 group-hover:opacity-100 focus:opacity-100 hover:bg-[var(--error-bg)] hover:text-[var(--error)] transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      <IconTrashSmall />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Scope picker — which documents are in chat scope */}
      <div ref={scopeMenuRef} className="relative flex items-center gap-2 px-4 py-2 border-b border-[var(--border)] shrink-0 bg-[var(--surface-glass)] z-10">
        <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-faint)] shrink-0">
          Scope
        </span>
        <div className="flex flex-wrap items-center gap-1.5 flex-1 min-w-0">
          {scopeDocIds.map((id) => (
            <span key={id} className="inline-flex items-center gap-1 max-w-[160px] px-2 py-0.5 rounded-full bg-[var(--accent-subtle)] border border-[var(--border-glow)] text-[11px] text-[var(--accent-text)]">
              <span className="truncate">{docTitle(id)}</span>
              {scopeDocIds.length > 1 && (
                <button
                  type="button"
                  onClick={() => toggleScopeDoc(id)}
                  disabled={streaming}
                  className="shrink-0 hover:text-[var(--error)] disabled:opacity-40"
                  title="Remove from scope"
                >
                  <IconX />
                </button>
              )}
            </span>
          ))}
          <button
            type="button"
            onClick={() => setScopeMenuOpen((v) => !v)}
            disabled={streaming}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border border-[var(--border-strong)] text-[11px] text-[var(--text-muted)] hover:border-[var(--border-hover)] hover:text-[var(--text-primary)] disabled:opacity-40"
            title="Add files to scope"
          >
            <IconPlus /> File
          </button>
        </div>

        {scopeMenuOpen && (
          <div role="listbox" className="absolute left-4 right-4 top-[calc(100%+4px)] z-30 max-h-72 overflow-y-auto rounded-xl border border-[var(--border-strong)] bg-[var(--surface)] shadow-2xl shadow-black/40 p-1.5 scale-in origin-top">
            {allDocs.length === 0 && (
              <div className="px-2.5 py-2 text-[12px] text-[var(--text-faint)]">No documents</div>
            )}
            {allDocs.map((d) => {
              const checked = scopeDocIds.includes(d.id);
              return (
                <div
                  key={d.id}
                  role="option"
                  aria-selected={checked}
                  onClick={() => toggleScopeDoc(d.id)}
                  className={`flex items-center gap-2 rounded-lg px-2.5 py-2 cursor-pointer transition-colors ${
                    checked ? "bg-[var(--accent-subtle)] text-[var(--text-primary)]" : "text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]"
                  }`}
                >
                  <input type="checkbox" readOnly checked={checked} className="accent-[var(--accent)]" />
                  <span className="flex-1 min-w-0 truncate text-[12px]">{d.title || d.filename}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 flex flex-col gap-5 pb-40">
        {visibleMessages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center px-4 py-12 fade-in">
            <div className="w-12 h-12 rounded-2xl bg-[var(--accent-subtle)] border border-[var(--border-glow)] flex items-center justify-center mb-5 glow-pulse text-[var(--accent-text)]">
              <IconBot />
            </div>
            <p className="text-[13px] font-medium text-[var(--text-secondary)] mb-1.5">
              Ask anything about the selected documents
            </p>
            <p className="text-[11px] text-[var(--text-faint)] m-0 leading-relaxed max-w-[240px]">
              Answers are grounded in the selected documents.
            </p>
          </div>
        )}

        {visibleMessages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onJumpToSource={onJumpToSource} />
        ))}

        {error && (
          <div className="p-3 rounded-lg border border-[rgba(248,113,113,0.2)] bg-[var(--error-bg)] text-xs text-[var(--error)] fade-in">
            {error}
          </div>
        )}
      </div>

      {/* Input Area (Floating) */}
      <div className="absolute bottom-0 left-0 right-0 px-4 pb-4 pt-12 bg-gradient-to-t from-[var(--surface)] via-[var(--surface)] to-transparent pointer-events-none">
        <div className="flex flex-col gap-2 max-w-2xl mx-auto pointer-events-auto">
          {/* Action row — New Topic + (demo) Compare RAG vs GraphRAG */}
          {(messages.length > 0 || demoMode) && (
            <div className="flex justify-center gap-2 mb-1">
              {messages.length > 0 && (
                <button
                  onClick={async () => {
                    try {
                      const newId = await startNewSession();
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
              )}
              {demoMode && (
                <button
                  onClick={runComparison}
                  disabled={streaming || compareLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[var(--accent-subtle)] border border-[var(--border-glow)] text-[11px] text-[var(--accent-text)] hover:brightness-110 transition-all scale-in shadow-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Answer the current question with plain RAG and with GraphRAG, side by side"
                >
                  <IconScale />
                  <span>Compare RAG vs GraphRAG</span>
                </button>
              )}
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
            Grounded in selected documents · Shift+Enter for new line
          </p>
        </div>
      </div>

      {/* RAG vs GraphRAG compare overlay (demo mode) */}
      {compareOpen && (
        <CompareDrawer
          question={compareQuestion}
          loading={compareLoading}
          error={compareError}
          result={compareResult}
          onClose={() => setCompareOpen(false)}
        />
      )}
    </div>
  );
}

