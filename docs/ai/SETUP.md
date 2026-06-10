# AI Layer — Installation & Environment Setup

How to get the AI subsystem running and testable, on Windows, Linux, and macOS.
This covers the two environments you actually need: the **full sidecar** (to run
the app) and the **isolated `.venv-ai`** (to run AI tests and benchmarks without
the heavy ingest dependencies).

> For frontend/Tauri build instructions see the root [README.md](../../README.md).
> This doc is the AI layer only.

---

## 1. Prerequisites

| Tool | Version | Why |
|------|---------|-----|
| Python | 3.11+ | backend sidecar + AI services |
| [uv](https://docs.astral.sh/uv/) | latest | fast, isolated env for AI tests (`.venv-ai`) |
| An LLM provider | — | Gemini API key (default), OR any OpenAI-compatible endpoint, OR a local Ollama/LM Studio server |

**GPU is not required.** The AI layer gets embeddings and completions from the
provider API; there is no local GPU inference path in this codebase. CPU-only is
the normal case. A local Ollama server can run on CPU or GPU independently of
LAIDocs.

**Storage.** Plan for: converted Markdown + assets under `~/.laidocs/vault/`
(scales with your corpus), the SQLite DBs under `~/.laidocs/data/` (metadata,
tree index, chat history, embeddings, memory store — typically tens of MB), and
the cached dense vectors in `document_embeddings` (a few KB per unit).

---

## 2. The isolated AI test/benchmark env (`.venv-ai`)

This is the recommended way to run AI tests and the evaluation harness. It is
**gitignored** and intentionally **excludes** the heavy ingest deps
(`docling`, `torch`) so it installs fast and stays lean.

### Create it (once)

```bash
# Windows (PowerShell) / Linux / macOS — uv is cross-platform
uv venv .venv-ai --python 3.11

# install only the AI-layer runtime deps
uv pip install --python .venv-ai \
    langchain langchain-core "langchain[google-genai]" langchain-google-genai \
    langchain-openai langgraph langgraph-checkpoint-sqlite \
    deepagents rank_bm25 numpy pydantic pydantic-settings pytest
```

### Use it

```bash
# Windows
.venv-ai/Scripts/python.exe -m pytest tests/ -v

# Linux / macOS
.venv-ai/bin/python -m pytest tests/ -v
```

> Throughout this handbook commands use the Windows path
> `.venv-ai/Scripts/python.exe`. On Linux/macOS substitute `.venv-ai/bin/python`.

### What the AI test suite covers

```bash
.venv-ai/Scripts/python.exe -m pytest \
    tests/test_phase4_multimodal.py \
    tests/test_phase5_evaluation.py \
    tests/test_phase6_knowledge_graph.py -v
# → 41 tests, fully offline (no network, no model calls)
```

These set `HOME`/`USERPROFILE` to a temp dir before importing backend modules,
so they never touch your real `~/.laidocs`.

---

## 3. Full sidecar env (to run the app/live mode)

For live retrieval against real ingested documents you need the full backend
(which includes the ingest stack):

```bash
python3 -m venv backend/.venv
# Windows: backend\.venv\Scripts\activate    Linux/macOS: source backend/.venv/bin/activate
pip install -r backend/requirements.txt
python3 backend/main.py --dev      # sidecar on http://localhost:8008
```

---

## 4. Provider configuration

Settings resolve from **`~/.laidocs/config.json`** first, then fall back to
**`.env`** defaults (`backend/core/config.py` → `Settings.active_llm`). Copy the
template:

```bash
cp .env.example .env
```

### Recommended configurations

| Goal | provider | model | embed_model | Notes |
|------|----------|-------|-------------|-------|
| **Default / best quality-per-cost** | `gemini` | `gemini-2.0-flash` | `models/text-embedding-004` | dense retrieval ✅; cheap multi-round agentic loop |
| Highest reasoning quality | `gemini` | `gemini-1.5-pro` | `models/text-embedding-004` | slower/pricier critique calls |
| Fully local / offline | `openai` | local model (e.g. `llama3.1`) | local embed model | set `DEFAULT_LLM_BASE_URL=http://localhost:11434/v1` |
| OpenAI hosted | `openai` | `gpt-4o-mini` | `text-embedding-3-small` | leave base_url blank |
| Anthropic | `anthropic` | `claude-3-5-sonnet-latest` | — | ⚠️ no embeddings → **dense retrieval auto-disabled**, system falls back to BM25 + tree |

### Performance / quality trade-offs

- **Dense retrieval requires an embedding backend** (Gemini or OpenAI-compatible).
  With Anthropic, `embeddings_supported()` is false and the pipeline silently
  runs BM25 + tree only — expect lower recall on paraphrased questions.
- **Agentic loop cost** scales with rounds × follow-ups (≤ 3 × 2 critique calls
  + retrievals). A Flash-class model keeps this cheap; a Pro/large model makes it
  noticeably slower and costlier. Tune `MAX_RETRIEVAL_ROUNDS` in `retrieval.py`.
- **Embedding index is lazy and cached.** First query on a document pays the
  embedding cost; subsequent queries read cached vectors from SQLite. Switching
  embedding models invalidates and rebuilds the index automatically.

---

## 5. Applying settings changes

After changing the LLM config at runtime, the agent singleton must be rebuilt:

- Via the app: saving settings triggers `reset_agent()` (see
  [`backend/api/settings.py`](../../backend/api/settings.py)).
- In a script: call `backend.services.agent.reset_agent()` or
  `backend.core.config.reload_settings()` before the next request.

---

## 6. Verify the install

```bash
# 1. AI unit tests pass (offline)
.venv-ai/Scripts/python.exe -m pytest tests/ -q

# 2. Evaluation harness runs (offline, no LLM)
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json -k 5

# 3. Provider is reachable (live — needs a key)
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services.llm import is_llm_configured; \
print('LLM configured:', is_llm_configured(get_settings().active_llm))"
```

Expected: tests green; the benchmark prints a variant leaderboard; step 3 prints
`LLM configured: True` once your `.env`/config has a valid provider.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `rank_bm25 not installed; skipping lexical retrieval` (log) | dep missing in env | `uv pip install --python .venv-ai rank_bm25` |
| Dense retrieval returns nothing | provider has no embedding backend (e.g. Anthropic) or no key | switch to Gemini/OpenAI provider, or set an `embed_model`; check `embeddings_supported()` |
| `LLM configured: False` | no model/key resolved | set `DEFAULT_LLM_*` in `.env` or `llm.*` in `~/.laidocs/config.json` |
| Dimension-mismatch units skipped (dense) | embedding model changed | harmless — index auto-rebuilds; or delete rows in `document_embeddings` |
| Agent answers from old model after settings change | singleton not reset | call `reset_agent()` / re-save settings |
| CJK text garbled on Windows | stdout encoding | `main.py` forces `PYTHONUTF8=1`; ensure you launch via it |
| `graph` variant always empty in benchmark | no LLM (extractor) configured | configure a provider; KG extraction needs an LLM |
| Tests touch real `~/.laidocs` | import order | ensure `HOME`/`USERPROFILE` are set before importing backend (the bundled tests already do this) |

Next: [EVALUATION.md](EVALUATION.md) for benchmarking and experiments.
