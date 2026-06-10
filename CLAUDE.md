# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# LAIDocs

Local AI-powered document manager: convert files/URLs to Markdown, organize in folders, and chat with documents using a DeepAgents-powered assistant with SOUL (document-grounded only), conversation memory, and session management. Fully local — only connects to your configured LLM API.

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
│   ├── DeepAgents — chat agent with SOUL, memory, sessions
│   ├── LangGraph + LangChain — agent framework + checkpointer
│   └── SQLite — metadata, tree index, chat history
└── Vault — filesystem storage at ~/.laidocs/vault/
```

Frontend communicates with the sidecar via HTTP REST + SSE on `localhost:8008`. The Tauri shell plugin spawns the sidecar process.

## Chat System

The chat is powered by a **DeepAgents** agent (`backend/services/agent.py`) with:
- **SOUL prompt** — document-grounded only, no fabrication, cite sections
- **`retrieve_context` tool** — wraps PageIndex tree reasoning as a LangChain `@tool`
- **Conversation memory** — LangGraph `MemorySaver` checkpointer (per-doc sessions)
- **User preference learning** — `StoreBackend` routing `/memories/` writes to `InMemoryStore`; seeded from `~/.laidocs/memories/preferences.md`
- **Display history** — separate `chat_messages` SQLite table (survives session reset)
- **Session management** — new-session button resets agent context, all messages remain visible
- **Agent singleton** — created lazily on first request; call `reset_agent()` after settings change so the next request rebuilds with new LLM config

API endpoints in `backend/api/chat.py`:
- `POST /api/chat/stream` — SSE stream with `session_id` support
- `GET /api/chat/history/{doc_id}` — load all display messages
- `POST /api/chat/new-session/{doc_id}` — start fresh session
- `DELETE /api/chat/history/{doc_id}` — clear all history

## LLM Configuration

Settings are persisted to `~/.laidocs/config.json` and read via `backend/core/config.py`. LLM can also be seeded from a `.env` file (at project root) using:

```
DEFAULT_LLM_BASE_URL=http://localhost:11434/v1
DEFAULT_LLM_API_KEY=sk-...
DEFAULT_LLM_MODEL=llama3
```

The `active_llm` property on `Settings` merges the persisted `llm` config with these env defaults (`llm.*` takes precedence). Node selection inside `retrieve_context` uses a direct `openai.OpenAI` call with `active_llm` settings, separate from the LangChain agent model.

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
| `~/.laidocs/config.json` | Persisted settings (LLM config) |
| `~/.laidocs/vault/<folder>/<doc>.md` | Converted Markdown documents |
| `~/.laidocs/vault/<folder>/<doc>.md.meta.json` | Document metadata sidecar |
| `~/.laidocs/vault/assets/<doc_id>_N.png` | Extracted images |
| `~/.laidocs/data/laidocs.db` | SQLite database (metadata, tree index, chat history) |
| `~/.laidocs/data/checkpoints.db` | LangGraph MemorySaver (path defined in agent.py but currently uses InMemory) |
| `~/.laidocs/memories/preferences.md` | Initial agent-learned user preferences |

## Project Structure

```
src/                    # React frontend
├── pages/              # Documents, DocumentEditor, Settings, WelcomePanel
├── components/         # Sidebar, ChatPanel, UploadDialog, CrawlDialog,
│                       #   MarkdownPreview, DataTab (export/import), TopBar, FileTree
├── context/            # FolderContext, UploadContext (React state)
├── hooks/              # useSidecar (Tauri invoke wrappers)
└── lib/                # sidecar.ts (HTTP helpers, SSE, health polling, chat history API)
                        # api-upload.ts (multipart upload helpers)
backend/                # Python FastAPI sidecar
├── api/                # REST routers: documents, folders, chat, settings, backup
├── core/               # config, database (SQLite), exceptions, vault, telemetry
├── models/             # Pydantic document model
└── services/           # agent, chat_history, converter, crawler, tree_index, rag,
                        #   backup, picture_serializer
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
- SSE streaming for chat (`POST /api/chat/stream`) and upload progress stages.
- Chat history API: `getChatHistory`, `startNewSession`, `clearChatHistory` in `sidecar.ts`.
- Assets served at `/assets/<filename>` via FastAPI `StaticFiles` mount.

## Gotchas

- **UTF-8 on Windows**: `main.py` forces `PYTHONUTF8=1` and reconfigures stdout/stderr to UTF-8 — needed for CJK content.
- **Dev mode Python**: If `backend/.venv/bin/python3` exists, Tauri uses it; otherwise falls back to system `python3`.
- **Design system**: Warp-inspired warm dark theme — see `DESIGN.md` for colors, typography, and component patterns.
- **Tauri dev CWD**: During `tauri dev`, Tauri sets CWD to the project root (where `package.json` lives). The Rust code resolves paths relative to this.
- **Tree index build**: On document upload/crawl, the tree index is built asynchronously in a background task. The agent falls back to raw document content if no tree index exists (e.g., document has no headings).
- **Agent concurrency**: `agent.py` uses `contextvars.ContextVar` (not module-level dict) for per-request tool context isolation — safe for concurrent requests.
- **Agent streaming**: Uses LangGraph v2 streaming format (`version="v2"`) with dict-based chunks. Subagent/tool-call chunks are filtered out to emit only AI content tokens.
- **PageIndex**: The tree index implementation is adapted from [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — a vectorless, reasoning-based RAG system that builds a hierarchical tree from markdown headings with LLM-generated summaries per node.
- **Settings change → agent reset**: After saving new LLM settings via `POST /api/settings`, the API must call `reset_agent()` so the singleton is rebuilt with the updated model on the next chat request.
