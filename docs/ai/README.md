# LAIDocs — AI Layer Engineering Handbook

This is the technical documentation for the **AI subsystem** of LAIDocs: the
retrieval pipeline, agent orchestration, multimodal units, long-term memory,
evaluation harness, and knowledge-graph reasoning. It is written for
researchers, contributors, and evaluators who want to **run, benchmark, and
reproduce** the system — not just install it.

> **Scope note.** This handbook documents the AI layer (`backend/services/*`,
> `backend/core/config.py`). The Tauri/React frontend and the FastAPI transport
> plumbing are covered only where they gate AI behaviour; for those see the
> root [`CLAUDE.md`](../../CLAUDE.md) and [`README.md`](../../README.md).

## Documents

| Doc | What's in it |
|-----|--------------|
| [README.md](README.md) (this file) | System overview, architecture, data/flow diagrams, what's local vs. external, what's implemented vs. aspirational |
| [SETUP.md](SETUP.md) | Environment setup (Windows/Linux/macOS), the isolated `.venv-ai`, `.env`, provider config, troubleshooting |
| [EVALUATION.md](EVALUATION.md) | The reproducibility package: retrieval benchmarking, grounding/hallucination eval, agentic & memory experiments, comparison matrices, demo scenarios |
| [`scripts/ai_eval/`](../../scripts/ai_eval/) | Runnable benchmark scripts + sample dataset |

---

## 1. System overview

LAIDocs converts documents to Markdown, indexes them, and answers questions
**grounded only in the document** via a DeepAgents agent. The AI layer is the
part that does retrieval, reasoning, and answer generation.

### 1.1 Component map (AI layer)

```
backend/core/config.py        Settings + LLMConfig + active_llm merge
backend/services/
├── llm.py                    Provider-agnostic factory: chat models + embeddings
│                               (OpenAI-compatible / Gemini / Anthropic)
├── tree_index.py             PageIndex hierarchical tree (headings → tree + summaries)
├── retrieval.py              THE retrieval engine:
│                               • retrieval units (tree nodes / chunks / figures / tables)
│                               • BM25 (lexical)  • dense (embeddings)  • tree reasoning
│                               • RRF fusion → hybrid_rank
│                               • single-shot retrieve_context
│                               • agentic_retrieve_context (multi-hop + self-critique)
├── knowledge_graph.py        Entity-relation graph + multi-hop graph retrieval + graph-of-thought
├── evaluation.py             Retrieval metrics + RAGAS-style grounding/relevance (injectable judge)
└── agent.py                  DeepAgents agent: SOUL prompt, retrieve_context tool,
                                MemorySaver checkpointer, durable SqliteStore memory
```

### 1.2 What runs locally vs. what calls an external API

| Capability | Local (no network) | Needs external API |
|------------|--------------------|--------------------|
| BM25 lexical retrieval | ✅ pure Python (`rank_bm25`) | — |
| Tree structure build | ✅ heading parsing | summaries per node use the LLM |
| Tree node selection | — | ✅ LLM reasons over the tree |
| Dense retrieval | vector math + SQLite cache local | ✅ embeddings come from the provider |
| RRF fusion | ✅ pure Python | — |
| Agentic self-critique loop | — | ✅ LLM critique each round |
| Knowledge-graph traversal | ✅ deterministic core | ✅ triple/entity extraction is LLM |
| Deterministic eval metrics | ✅ fully offline | — |
| RAGAS-style judged metrics | — | ✅ LLM judge (injectable) |
| Conversation + long-term memory | ✅ SQLite on disk | — |

The only external dependency is **the LLM provider you configure** (Gemini by
default; or an OpenAI-compatible endpoint, including a fully local Ollama/LM
Studio server → then *everything* is local). No telemetry is required for the
AI layer to function.

### 1.3 Implemented vs. aspirational

To keep this handbook honest, the following are **implemented and tested**:
hybrid retrieval (BM25 + dense + tree + RRF), agentic multi-hop retrieval,
multimodal figure/table units, the evaluation harness, the knowledge graph and
graph-of-thought.

The following are **referenced in the literature but NOT in this codebase** —
do not expect commands for them: ColBERT / late-interaction retrieval,
cross-encoder reranking, a standalone OCR pipeline (OCR happens inside Docling
at ingest time, which is out of the AI-layer scope), and GPU-accelerated local
embedding/inference (embeddings are obtained via the provider API). Where the
evaluation guide mentions these, it says so explicitly and frames them as
extension points, not features.

> **Live-path wiring.** `evaluation.py` and `knowledge_graph.py` are standalone
> AI libraries with clean entry points. The agent's live `retrieve_context`
> tool currently calls `retrieval.agentic_retrieve_context` only — the eval
> harness and the KG/graph-of-thought are invoked from scripts/tests, not yet
> from the agent answer path. Their hooks (`graph_augmented_units` returns
> RRF-ready ids; `graph_of_thought` returns a scaffold string) are shaped to
> drop in. This is intentional and documented per the project's scope.

---

## 2. Retrieval pipeline

The heart of the system. A single question flows through up to three retrievers
whose rankings are fused with Reciprocal Rank Fusion (RRF), optionally driven by
an agentic multi-hop loop.

### 2.1 The shared unit corpus

Every retriever ranks the **same** list of *retrieval units* so their rankings
fuse cleanly (`get_retrieval_units`). A unit is `{unit_id, title, text, kind}`:

| Source | `unit_id` | `kind` | When |
|--------|-----------|--------|------|
| Tree node (heading section) | `"0007"` | `text` | document has a heading tree |
| Paragraph chunk (~1000 chars) | `"c0001"` | `text` | document has *no* headings |
| Figure (alt + VLM description + caption) | `"img0001"` | `image` | always (parsed lazily) |
| Table (intact markdown cells + caption) | `"tbl0001"` | `table` | always (parsed lazily) |

Multimodal units are parsed **lazily at query time** from the stored Markdown,
so they work for already-converted documents with no re-ingest.

### 2.2 Single-query flow (`hybrid_rank` → `retrieve_context`)

```
                    question
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼                ▼
   Tree reasoning   BM25 lexical    Dense (embeddings)
   (LLM picks      (rank_bm25 over  (cosine over cached
    node_ids)       unit corpus)     vectors in SQLite)
        │              │                │
   ranked ids     ranked ids        ranked ids
        └──────────────┼────────────────┘
                       ▼
            RRF fusion (k=60, top_k=8)        rrf_fuse()
                       ▼
        units → build_context_from_units (≤12k chars)
                       ▼
                  context string
```

Graceful degradation is built in: any retriever that is unavailable (no
embeddings backend, `rank_bm25` missing, tree errors) is simply dropped from the
fusion. If tree reasoning is the *only* signal and it deliberately returns
nothing, the question is treated as out-of-scope and an empty context is
returned — this preserves the anti-hallucination contract.

### 2.3 Agentic multi-hop flow (`agentic_retrieve_context`)

This is what the agent's tool actually calls.

```
round 1: hybrid_rank(question) ──► accumulate evidence (best RRF rank per unit)
   │
   ▼  build context from accumulated units
critique(question, context) ──► {sufficient?, missing, followups[]}
   │
   ├─ sufficient OR no followups OR no new units OR round == max ─► STOP
   └─ else: queries := followups ─► next round (≤ 3 rounds total)
   ▼
final context = top accumulated units (≤ 12)
```

Bounds: `MAX_RETRIEVAL_ROUNDS=3`, `MAX_FOLLOWUPS_PER_ROUND=2`,
`MAX_ACCUMULATED_UNITS=12`. Evidence accumulates across rounds and sub-queries,
scored by the best RRF rank seen — so a fact split across sections is gathered
hop by hop.

### 2.4 Knowledge-graph retrieval (parallel track)

`knowledge_graph.graph_augmented_units(doc_id, question)` builds an
entity-relation graph from the same units, finds the question's entities, walks
≤2 hops, and returns the source `unit_id`s of every reached entity — **shaped to
fuse as one more ranked list into `rrf_fuse`**. `graph_of_thought(...)` instead
renders the relation paths between question entities as explicit
`A --[founded by]--> B --[located in]--> C` reasoning chains.

---

## 3. Memory pipeline

Two independent memory systems, by design:

```
┌─ Conversation memory (working) ──────────────────────────────┐
│ LangGraph MemorySaver checkpointer (in-process, per session) │
│ Holds the active turn-by-turn state for one chat session.    │
└──────────────────────────────────────────────────────────────┘
┌─ Long-term memory (durable) ─────────────────────────────────┐
│ SqliteStore at ~/.laidocs/data/memory_store.db               │
│ CompositeBackend routes /memories/ writes here; survives      │
│ restarts. Seeded from ~/.laidocs/memories/preferences.md.     │
│ The agent learns repeated user preferences (language, detail).│
└──────────────────────────────────────────────────────────────┘
```

Display history (what the UI shows) is a separate `chat_messages` SQLite table
that survives a session reset — it is **not** the checkpointer. See
[EVALUATION.md §With/Without Memory](EVALUATION.md#7-long-term-memory-evaluation)
for how to measure memory's effect.

---

## 4. Agent orchestration

`agent.py` builds a single DeepAgents agent (lazy singleton):

- **SOUL system prompt** — document-grounded only, no fabrication, cite
  sections/figures/tables, retrieve before answering.
- **One tool** — `retrieve_context(question)` → `agentic_retrieve_context`.
- **Checkpointer** — `MemorySaver` for working memory.
- **Store** — `SqliteStore` for durable `/memories/`.
- **Concurrency** — per-request `doc_id`/`settings` live in a `ContextVar`
  (`set_tool_context`), so concurrent requests never collide.
- **Reset** — `reset_agent()` after a settings change rebuilds the singleton
  with the new model on the next request.

Request flow:

```
chat request ─► set_tool_context(doc_id, settings) ─► agent.astream(...)
   agent reads /memories/preferences.md
   agent calls retrieve_context(question)
        └─► agentic_retrieve_context → hybrid_rank loop → context
   agent composes a grounded, cited answer ─► SSE tokens to the client
   agent may persist a learned preference to /memories/
```

---

## 5. Where to go next

- **Set it up:** [SETUP.md](SETUP.md)
- **Benchmark & reproduce:** [EVALUATION.md](EVALUATION.md)
- **Run the harness:** [`scripts/ai_eval/`](../../scripts/ai_eval/)
