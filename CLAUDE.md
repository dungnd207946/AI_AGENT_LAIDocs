# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# LAIDocs

Local AI-powered document manager: convert files/URLs to Markdown, organize in folders, and chat with documents using a LangGraph ReAct agent with SOUL (document-grounded only), durable conversation memory, session management, and optional vision + document-editing tools. Fully local — only connects to your configured LLM API (Gemini by default, or any OpenAI-compatible / Anthropic endpoint).

## Commands

```bash
# Frontend (React + Vite)
pnpm install
pnpm dev          # Vite dev server on :5173

# Backend (Python FastAPI sidecar)
python3 backend/main.py --dev   # starts on localhost:8008

# Full Tauri app (frontend + sidecar)
pnpm tauri dev

# Build
pnpm build                       # TypeScript + Vite build
pnpm tauri build                 # full production build
python3 build_sidecar.py         # PyInstaller sidecar binary

# Backend venv (if needed)
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt

# Tests
pytest tests/                    # all backend tests
pytest tests/test_converter_fallback.py  # single file
```

## Architecture

```
Tauri v2 (Rust shell)
├── React 19 + TypeScript + Tailwind (WebView, port 5173)
├── Python FastAPI sidecar (localhost:8008)
│   ├── Docling — document → Markdown conversion (PDF/DOCX/PPTX/HTML)
│   ├── MarkItDown — XLSX conversion (merged-cell-safe, replaces Docling for Excel)
│   ├── Crawl4AI — web crawling
│   ├── PageIndex — hierarchical tree index (reasoning-based RAG)
│   ├── LangGraph create_react_agent — chat agent: SOUL, durable memory, vision + edit tools
│   ├── LangGraph + LangChain — agent framework + checkpointer
│   └── SQLite — metadata, tree index, chat history
└── Vault — filesystem storage at ~/.laidocs/vault/
```

Frontend communicates with the sidecar via HTTP REST + SSE on `localhost:8008`. The Tauri shell plugin spawns the sidecar process.

## Chat System

The chat is powered by a **LangGraph `create_react_agent`** (`backend/services/agent.py`) — provider-agnostic across Gemini / OpenAI-compatible / Anthropic. (It replaced DeepAgents, whose Anthropic-only middleware returned empty responses on Gemini.) It has:
- **SOUL prompt** — document-grounded only, no fabrication, cite sections/figures/tables
- **Tools** — `retrieve_context` (hybrid retrieval via `retrieval.agentic_retrieve_context`), `reason_over_graph` (GraphRAG: explicit relation chains for multi-hop "how is X connected to Y" questions), `read_image` (VLM analysis of figures/charts embedded in the doc), `preview_edit` + `apply_edit` (edit the document, gated on user confirmation), and `create_markdown_file` (export/save document-grounded content as a downloadable `.md`)
- **Conversation memory** — durable `AsyncSqliteSaver` checkpointer at `~/.laidocs/data/checkpoints.db`, keyed per `thread_id = "doc-{doc_id}-s{session_id}"`; survives backend restarts
- **Conversation compaction** — `backend/services/compactor.py` keeps token usage bounded: `compact_if_needed` (called before each stream in `chat.py`) rolls older display history into an LLM summary once it exceeds a token threshold, keeping the last few Q&A pairs verbatim
- **Retrieved-evidence tracking** — each turn's retrieved units are saved per message (`save_message_evidence`); the agent reads them back via `get_retrieved_evidence`, and stale/unverified prior evidence is flagged so it is never reused as a document fact. `evidence_from_units` also carries `heading_path` + a one-line `preview` so the UI can render citations and jump-to-source.
- **Demo UI** — the chat panel surfaces the retrieval/graph work for demos (see `src/components/CitationChips.tsx`, `ReasoningChain.tsx`, `CompareDrawer.tsx`): (1) **citations + grounding badge** under each answer, click a chip to scroll the document preview to that section (`DocumentEditor.handleJumpToSource`); (2) **reasoning-path** chips when `reason_over_graph` ran; (3) a **Demo-mode toggle** (chat header, `localStorage` key `laidocs-demo-mode`) that reveals a **Compare RAG vs GraphRAG** button. The stream emits `[EVIDENCE]` and `[CHAIN]` SSE events; chains persist in the `chat_message_chains` table and are replayed on history reload via `chat_history.get_display_messages`.
- **User preferences** — `~/.laidocs/memories/preferences.md` is read at agent build time and injected into the system prompt (read-only seeding; there is no live write-back store)
- **Display history** — separate `chat_messages` SQLite table (survives session reset)
- **Session management** — per-(doc, session) threads; the ChatPanel UI shows one session at a time with a switcher dropdown to resume any past session
- **Agent singleton** — created lazily on first request; call `reset_agent()` after settings change so the next request rebuilds with new LLM config. The checkpointer persists across resets and is closed on app shutdown via `close_checkpointer()`.

API endpoints in `backend/api/chat.py`:
- `POST /api/chat/stream` — SSE stream with `session_id` support (also emits `[EVIDENCE]` / `[CHAIN]` / `[EDITED]` sentinels before `[DONE]`)
- `POST /api/chat/compare` — stateless RAG-vs-GraphRAG A/B: deep-copies settings, toggles `graph_rag.enabled` off then on (same model + grounded prompt), returns both answers + ranked units + `bridge_unit_ids` (units only the graph walk recovered). Powers the Demo-mode compare drawer.
- `GET /api/chat/history/{doc_id}` — load all display messages (assistant rows carry their `evidence` + `chain`)
- `POST /api/chat/new-session/{doc_id}` — start fresh session
- `DELETE /api/chat/history/{doc_id}` — clear all history
- `DELETE /api/chat/session/{doc_id}/{session_id}` — delete a single session
- `GET /api/download/{filename}` — download a `create_markdown_file` export (`backend/api/downloads.py`)

## LLM Configuration

Settings are persisted to `~/.laidocs/config.json` and read via `backend/core/config.py`. LLM can also be seeded from a `.env` file (at project root) using:

```
DEFAULT_LLM_PROVIDER=gemini            # or "openai" (incl. local Ollama/LM Studio) / "anthropic"
DEFAULT_LLM_API_KEY=...
DEFAULT_LLM_MODEL=gemini-2.5-flash
# OpenAI-compatible / local example instead:
#   DEFAULT_LLM_PROVIDER=openai
#   DEFAULT_LLM_BASE_URL=http://localhost:11434/v1
#   DEFAULT_LLM_MODEL=llama3
# Optional vision model for the read_image tool:
#   DEFAULT_VLM_BASE_URL=...  DEFAULT_VLM_API_KEY=...  DEFAULT_VLM_MODEL=...
```

The `active_llm` property on `Settings` merges the persisted `llm` config with these env defaults (`llm.*` takes precedence); `active_vlm` does the same for the optional vision model, `active_reranker` for the optional reranker (`RerankerConfig`, disabled by default; Jina-compatible `/v1/rerank` endpoint), and `active_graph_rag` for GraphRAG (`GraphRagConfig`, enabled by default). Every LLM call — the agent, node selection in `retrieval.select_node_ids`, and embeddings — goes through the provider-agnostic factory in `backend/services/llm.py` (`create_chat_model` / `create_embeddings`). Default embedding model for Gemini is `gemini-embedding-001`.

## Retrieval

`backend/services/retrieval.py` builds a shared retrieval corpus of "units" from the current document state and runs hybrid (dense + lexical + tree + graph) ranking, with an optional cross-encoder rerank pass when `active_reranker.enabled`. Dense embeddings are cached in the `document_embeddings` SQLite table keyed by **per-unit content hash** (`unit_hash`) — only units whose hash changed are re-embedded on the next query, so edits and re-indexing don't trigger a full re-embed. `agentic_retrieve_context` wraps this with a multi-hop + self-critique loop and returns the units as citable evidence.

### GraphRAG

`hybrid_rank` fuses a fourth signal — `graph_search` — alongside tree/BM25/dense. It walks an **entity-relation graph** built from the document (`backend/services/knowledge_graph.py`) starting from the question's entities, surfacing passages connected through the graph even when no single passage matches lexically (the multi-hop "X founded by Y who created Z" case). Triples are **extracted once per retrieval unit and cached** in the `document_graph_units` SQLite table, keyed by `unit_hash` + extractor `model` (same incremental pattern as the embedding cache) — so query time costs ~one LLM call (query-entity extraction) plus a deterministic walk, not a full re-extraction. The cache is built proactively in the ingest background task (after the tree index) via `ensure_graph_index_async`, and lazily on first query otherwise. The agent also exposes `reason_over_graph` (graph-of-thought) to render explicit relation chains. Gated by `active_graph_rag.enabled`; degrades to a no-op when no LLM is configured. Measure the lift with `scripts/ai_eval/run_retrieval_benchmark.py` on `datasets/multihop_graph.json`.

## Document Conversion Pipeline

`backend/services/converter.py` uses a hybrid strategy:
- **XLSX** → MarkItDown (avoids Docling's merged-cell duplication)
- **PDF** → Docling with full layout pipeline + optional VLM picture description (requires LLM configured)
- **DOCX / PPTX / HTML** → Docling (image extraction, no VLM)

If no LLM is configured, VLM description and post-conversion LLM refinement are both skipped (graceful degradation). The `_refine()` method on `DoclingConverter` passes raw markdown through an LLM cleanup pass — it never raises, always falls back to the raw output.

## Backup / Export-Import

`backend/api/backup.py` and `backend/services/backup.py` handle `.laidocs-backup` archive files:
- `GET /api/backup/stats` — current vault statistics
- `POST /api/backup/export` — create archive at absolute `target_path`
- `POST /api/backup/preview` — read manifest without modifying data
- `POST /api/backup/import` — import with mode `"replace"` or `"merge"`

The UI surface is `src/components/DataTab.tsx`. Export uses the Tauri save-file dialog and writes via the Tauri backend to support native file paths on all platforms.

## Key Paths

| Path | Purpose |
|------|---------|
| `~/.laidocs/config.json` | Persisted settings (LLM + VLM config) |
| `~/.laidocs/vault/<folder>/<doc>.md` | Converted Markdown documents |
| `~/.laidocs/vault/<folder>/<doc>.md.meta.json` | Document metadata sidecar |
| `~/.laidocs/vault/assets/<doc_id>_N.png` | Extracted images |
| `~/.laidocs/data/laidocs.db` | SQLite database (metadata, tree index, chat history, per-message citation evidence + reasoning chains, embedding cache, knowledge-graph triple cache) |
| `~/.laidocs/data/checkpoints.db` | Durable conversation memory — `AsyncSqliteSaver` checkpointer, one thread per `doc+session`; survives restarts |
| `~/.laidocs/memories/preferences.md` | Initial agent-learned user preferences |

## Project Structure

```
src/                    # React frontend
├── pages/              # Documents, DocumentEditor, Settings, WelcomePanel
├── components/         # Sidebar, ChatPanel, UploadDialog, CrawlDialog,
│                       #   MarkdownPreview, DataTab (export/import), TopBar, FileTree,
│                       #   CitationChips, ReasoningChain, CompareDrawer (demo UI)
├── context/            # FolderContext, UploadContext (React state)
├── hooks/              # useSidecar (Tauri invoke wrappers)
└── lib/                # sidecar.ts (HTTP helpers, SSE, health polling, chat history API)
                        # api-upload.ts (multipart upload helpers)
backend/                # Python FastAPI sidecar
├── api/                # REST routers: documents, folders, chat, settings,
│                       #   backup, downloads (serve exported .md files)
├── core/               # config, database (SQLite), exceptions, vault, telemetry
├── models/             # Pydantic document model
└── services/           # agent, llm, retrieval, compactor, tree_index,
                        #   knowledge_graph, evaluation, document_store,
                        #   chat_history, converter, crawler, backup,
                        #   picture_serializer
src-tauri/              # Tauri v2 (Rust)
└── src/main.rs         # Sidecar spawn/shutdown, Tauri commands
tests/                  # pytest backend unit tests
telemetry_server/       # Standalone telemetry collection server (optional)
```

## Sidecar Lifecycle

- **Spawn**: Tauri auto-starts the sidecar on app launch via `spawn_sidecar()`. Dev mode runs `python3 backend/main.py --dev`; release mode uses the bundled PyInstaller binary from `src-tauri/bin/api/main`.
- **Shutdown**: ALWAYS via stdin — sends `"sidecar shutdown\n"`. Never call `process.kill()`. The Python backend's stdin listener handles graceful exit.
- **Health check**: Frontend polls `GET /api/health` until 200. In Tauri mode, also listens for `sidecar-stdout` events containing `"ready"`.

## Frontend ↔ Backend Protocol

- REST API at `http://localhost:8008` — see `src/lib/sidecar.ts` for `apiGet`/`apiPost`/`apiPut`/`apiDelete` helpers.
- SSE streaming for chat (`POST /api/chat/stream`) and upload progress stages. `streamChat` takes a handlers object `{ onChunk, onEdited, onEvidence, onChain }`; the stream's `[EVIDENCE] {json}` / `[CHAIN] {json}` sentinels carry citation + reasoning-chain payloads.
- Chat history API: `getChatHistory`, `startNewSession`, `clearChatHistory`, `compareRetrieval` in `sidecar.ts`.
- Assets served at `/assets/<filename>` via FastAPI `StaticFiles` mount.

## Gotchas

- **UTF-8 on Windows**: `main.py` forces `PYTHONUTF8=1` and reconfigures stdout/stderr to UTF-8 — needed for CJK content.
- **Dev mode Python**: If `backend/.venv/bin/python3` exists, Tauri uses it; otherwise falls back to system `python3`.
- **Design system**: Warp-inspired warm dark theme — see `DESIGN.md` for colors, typography, and component patterns.
- **Tauri dev CWD**: During `tauri dev`, Tauri sets CWD to the project root (where `package.json` lives). The Rust code resolves paths relative to this.
- **Tree index build**: On document upload/crawl, the tree index is built asynchronously in a background task. The agent falls back to raw document content if no tree index exists (e.g., document has no headings).
- **Agent concurrency**: `agent.py` uses `contextvars.ContextVar` (not module-level dict) for per-request tool context isolation — safe for concurrent requests.
- **Agent streaming**: `chat.py` uses `astream_events(version="v2")` (dict chunks). The tools node and tool-call chunks are filtered out to emit only AI content tokens; Gemini's `google-genai` SDK may return `content` as a list of content blocks, which the stream flattens to text.
- **Document editing via chat**: `apply_edit` writes through `document_store.persist_document_content` (keeps the `.md` file, SQLite `content`, and tree index in sync) and sets a per-request `edited` flag; the chat stream then emits a `data: [EDITED]` SSE event so the frontend reloads the document.
- **PageIndex**: The tree index implementation is adapted from [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — a vectorless, reasoning-based RAG system that builds a hierarchical tree from markdown headings with LLM-generated summaries per node.
- **Settings change → agent reset**: After saving new LLM settings via `POST /api/settings`, the API must call `reset_agent()` so the singleton is rebuilt with the updated model on the next chat request.
