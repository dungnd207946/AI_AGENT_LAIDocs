export const API_BASE = "http://localhost:8008";

// ── Generic HTTP helpers ───────────────────────────────────────────

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function apiPost<T, B = unknown>(path: string, body: B): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function apiPut<T, B = unknown>(path: string, body: B): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`PUT ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`DELETE ${path} failed: ${res.status} ${res.statusText}`);
  }
  if (res.status === 204) {
    return {} as T;
  }
  return res.json() as Promise<T>;
}

// ── Streaming chat (SSE) ──────────────────────────────────────────

/** A document unit cited as the source for an assistant answer. */
export interface Evidence {
  unit_id: string;
  title: string;
  kind: "text" | "image" | "table" | string;
  heading_path: string[];
  preview: string;
}

/** Payload of an edit-confirmation gate: the agent paused before writing. */
export interface EditConfirmation {
  type: "edit_confirmation";
  file: string;
  action: "REPLACE" | "DELETE" | string;
  old_string: string;
  new_string: string;
}

export interface StreamHandlers {
  onChunk: (text: string) => void;
  onEdited?: () => void;
  /** Citation evidence emitted once the answer is grounded. */
  onEvidence?: (evidence: Evidence[]) => void;
  /** Graph-of-thought reasoning chain (multi-hop questions). */
  onChain?: (chain: string) => void;
  /** The turn paused at an edit-confirmation gate; resolve via resumeChat. */
  onInterrupt?: (confirmation: EditConfirmation) => void;
}

/** Read an SSE response body, dispatching sentinels + tokens to handlers. */
async function consumeChatStream(res: Response, handlers: StreamHandlers): Promise<void> {
  const { onChunk, onEdited, onEvidence, onChain, onInterrupt } = handlers;
  if (!res.ok || !res.body) {
    throw new Error(`Chat request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  try {
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Keep the last (possibly incomplete) line in the buffer
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;
        const payload = trimmed.slice(6);
        if (payload === "[DONE]") return;
        if (payload === "[EDITED]") {
          // Agent edited the document — let the caller reload it
          onEdited?.();
          continue;
        }
        if (payload.startsWith("[EVIDENCE] ")) {
          try { onEvidence?.(JSON.parse(payload.slice(11)) as Evidence[]); } catch { /* ignore */ }
          continue;
        }
        if (payload.startsWith("[CHAIN] ")) {
          try { onChain?.(JSON.parse(payload.slice(8)) as string); } catch { /* ignore */ }
          continue;
        }
        if (payload.startsWith("[INTERRUPT] ")) {
          try { onInterrupt?.(JSON.parse(payload.slice(12)) as EditConfirmation); } catch { /* ignore */ }
          continue;
        }
        if (payload.startsWith("[ERROR]")) throw new Error(payload.slice(8));
        // Tokens are plain text with escaped newlines
        onChunk(payload.replace(/\\n/g, "\n"));
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function streamChat(
  docIds: string[],
  question: string,
  handlers: StreamHandlers,
  sessionId?: number,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_ids: docIds, question, session_id: sessionId ?? null }),
  });
  await consumeChatStream(res, handlers);
}

/** Resume a turn paused at an edit-confirmation gate (apply_edit interrupt). */
export async function resumeChat(
  docIds: string[],
  decision: "approve" | "reject",
  handlers: StreamHandlers,
  sessionId?: number,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_ids: docIds, decision, session_id: sessionId ?? null }),
  });
  await consumeChatStream(res, handlers);
}

// ── Document list (for the chat scope picker) ─────────────────────

export interface DocSummary {
  id: string;
  title?: string;
  filename: string;
  folder: string;
}

export async function listDocuments(): Promise<DocSummary[]> {
  return apiGet<DocSummary[]>(`/api/documents/`);
}

// ── RAG vs GraphRAG compare (demo) ────────────────────────────────

export interface CompareArm {
  answer: string;
  units: Evidence[];
}

export interface CompareResult {
  doc_id: string;
  question: string;
  rag: CompareArm;
  graph: CompareArm;
  bridge_unit_ids: string[];
}

export async function compareRetrieval(docId: string, question: string): Promise<CompareResult> {
  return apiPost<CompareResult>("/api/chat/compare", { doc_id: docId, question });
}

// ── Chat history & session management (global sessions) ───────────

export interface ChatMessage {
  id: number;
  session_id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  evidence?: Evidence[];
  chain?: string;
}

export async function getChatHistory(): Promise<ChatMessage[]> {
  const res = await apiGet<{ messages: ChatMessage[] }>(`/api/chat/history`);
  return res.messages;
}

export async function startNewSession(): Promise<number> {
  const res = await apiPost<{ session_id: number }>(`/api/chat/new-session`, {});
  return res.session_id;
}

export async function clearChatHistory(): Promise<void> {
  await apiDelete(`/api/chat/history`);
}

export async function deleteSession(sessionId: number): Promise<void> {
  await apiDelete(`/api/chat/session/${sessionId}`);
}

// ── Sidecar health check ──────────────────────────────────────────

const HEALTH_PATH = `${API_BASE}/api/health`;
const POLL_INTERVAL_MS = 500;
const MAX_WAIT_MS = 30_000;

/**
 * Wait until the sidecar backend is healthy.
 *
 * - Dev mode (no Tauri): polls /api/health until 200.
 * - Tauri mode: listens for the `sidecar-stdout` event from Tauri's shell plugin.
 *
 * Returns a promise that resolves on success or rejects on timeout / error.
 */
export function waitForSidecar(): Promise<void> {
  // Detect whether we're running inside Tauri
  const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

  if (isTauri) {
    return waitForSidecarTauri();
  }
  return waitForSidecarDev();
}

// ── Dev mode: poll health endpoint ────────────────────────────────

function waitForSidecarDev(): Promise<void> {
  return new Promise((resolve, reject) => {
    const start = Date.now();

    const poll = async () => {
      try {
        const res = await fetch(HEALTH_PATH);
        if (res.ok) {
          resolve();
          return;
        }
      } catch {
        // backend not up yet
      }

      if (Date.now() - start > MAX_WAIT_MS) {
        reject(new Error("Sidecar did not become ready in time"));
        return;
      }

      setTimeout(poll, POLL_INTERVAL_MS);
    };

    poll();
  });
}

// ── Tauri mode: listen for sidecar-stdout event ───────────────────

async function waitForSidecarTauri(): Promise<void> {
  const { listen } = await import("@tauri-apps/api/event");

  return new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error("Sidecar did not become ready in time"));
    }, MAX_WAIT_MS);

    listen<string>("sidecar-stdout", (event) => {
      if (typeof event.payload === "string" && event.payload.includes("ready")) {
        clearTimeout(timeout);
        resolve();
      }
    });

    // Also poll as a fallback
    const start = Date.now();
    const poll = async () => {
      try {
        const res = await fetch(HEALTH_PATH);
        if (res.ok) {
          clearTimeout(timeout);
          resolve();
          return;
        }
      } catch {
        // not up yet
      }
      if (Date.now() - start > MAX_WAIT_MS) {
        reject(new Error("Sidecar did not become ready in time"));
      }
      setTimeout(poll, POLL_INTERVAL_MS);
    };
    poll();
  });
}
