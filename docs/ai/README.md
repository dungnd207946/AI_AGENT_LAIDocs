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
| [README.md](README.md) (this file) | System overview, architecture, data/flow diagrams, what's local vs. external, what's implemented vs. aspirational, **how to run the system**, and **runnable demo scenarios** (§5–§6) |
| [SETUP.md](SETUP.md) | Environment setup (Windows/Linux/macOS), the isolated `.venv-ai`, `.env`, provider config, troubleshooting |
| [EVALUATION.md](EVALUATION.md) | The reproducibility package: retrieval benchmarking, grounding/hallucination eval, agentic & memory experiments, comparison matrices, demo scenarios |
| [`scripts/ai_eval/`](../../scripts/ai_eval/) | Runnable benchmark scripts + sample dataset |

---

## 1. System overview

LAIDocs converts documents to Markdown, indexes them, and answers questions
**grounded only in the document** via a LangGraph ReAct agent. The AI layer is the
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
│                               • graph_search (GraphRAG) — fused as a 4th signal
│                               • RRF fusion → hybrid_rank  • optional cross-encoder rerank
│                               • per-unit-hash embedding cache (only re-embed changed units)
│                               • single-shot retrieve_context
│                               • agentic_retrieve_context (multi-hop + self-critique)
├── compactor.py              Rolling-summary conversation compactor (token budget control)
├── knowledge_graph.py        Entity-relation graph + multi-hop graph retrieval + graph-of-thought.
│                               LIVE: fused into hybrid_rank + exposed as the reason_over_graph
│                               tool. Triples cached in document_graph_units (per unit_hash),
│                               built at ingest / lazily; ~1 LLM call per query at run time.
├── evaluation.py             Retrieval metrics + RAGAS-style grounding/relevance (injectable judge)
└── agent.py                  LangGraph create_react_agent: SOUL prompt; tools =
                                retrieve_context / reason_over_graph / read_image / preview_edit /
                                apply_edit / create_markdown_file; durable AsyncSqliteSaver
                                checkpointer (per session); per-message retrieved-evidence tracking
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
graph-of-thought, and the **demo UI** that surfaces all of the above in the
chat panel (see below).

The following are **referenced in the literature but NOT in this codebase** —
do not expect commands for them: ColBERT / late-interaction retrieval,
cross-encoder reranking, a standalone OCR pipeline (OCR happens inside Docling
at ingest time, which is out of the AI-layer scope), and GPU-accelerated local
embedding/inference (embeddings are obtained via the provider API). Where the
evaluation guide mentions these, it says so explicitly and frames them as
extension points, not features.

> **Live-path wiring.** `knowledge_graph.py` is now **wired into the live agent
> path**: `retrieval.hybrid_rank` fuses `graph_search` (cache-backed
> `graph_augmented_units_cached`) as a fourth signal, and the agent exposes
> `reason_over_graph` (graph-of-thought) as a tool. Triples are persisted in the
> `document_graph_units` cache (per `unit_hash`), built at ingest / lazily, so
> the query-time cost is ~one LLM call. `evaluation.py` remains a standalone
> harness invoked from scripts/tests (not the answer path) — an intentional,
> documented boundary.

> **Demo UI (in the chat panel).** The retrieval/graph work is now visible in
> the app, not just headless scripts:
> - **Citations + grounding badge** — every answer shows a "Grounded · N sources"
>   badge and clickable source chips (hover for a snippet); clicking a chip
>   scrolls the document preview to that section (jump-to-source). Backed by the
>   SSE `[EVIDENCE]` event + `evidence_from_units` (now carries `heading_path` +
>   `preview`); persisted/rebuilt for history via `chat_history.get_display_messages`.
> - **Reasoning-path view** — when the agent calls `reason_over_graph`, the
>   relation chain (`A --[rel]--> B --[rel]--> C`) renders as node/edge chips.
>   Backed by the SSE `[CHAIN]` event + the `chat_message_chains` table.
> - **RAG-vs-GraphRAG compare** — a **Demo-mode** toggle (chat header) reveals a
>   "Compare RAG vs GraphRAG" button that calls the stateless `POST
>   /api/chat/compare` endpoint and shows both answers side by side, with the
>   **bridge passage** GraphRAG recovered highlighted. This is Scenario 2's
>   headless A/B, made clickable.
>
> Frontend: `CitationChips.tsx`, `ReasoningChain.tsx`, `CompareDrawer.tsx`,
> wired through `ChatPanel.tsx` + `DocumentEditor.tsx`.

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

### 2.4 Knowledge-graph retrieval (GraphRAG — fused into the live path)

`knowledge_graph` builds an entity-relation graph from the same units, finds the
question's entities, walks ≤`hops` edges, and returns the source `unit_id`s of
every reached entity — fused as a fourth ranked list into `rrf_fuse` via
`retrieval.graph_search`. `graph_of_thought(...)` instead renders the relation
paths between question entities as explicit
`A --[founded by]--> B --[located in]--> C` reasoning chains, exposed to the
agent as the `reason_over_graph` tool.

**Caching (what makes it live-viable).** Rebuilding the graph per query means one
LLM triple-extraction call *per unit, per question* — too slow for the answer
path. So triples are extracted **once per retrieval unit** and persisted in
`document_graph_units`, keyed by `unit_hash` + extractor `model` (the same
incremental pattern as the embedding cache). `ensure_graph_index` re-extracts
only changed units; `graph_augmented_units_cached` / `graph_of_thought_cached`
then load the graph deterministically and spend just ~one LLM call (query-entity
extraction) at run time. The cache is built proactively in the ingest background
task (after the tree index) and lazily on first query otherwise. Gated by
`Settings.active_graph_rag.enabled` (default on); a no-op without an LLM.

---

## 3. Memory pipeline

Two memory layers, by design:

```
Conversation memory — DURABLE
  LangGraph AsyncSqliteSaver checkpointer at ~/.laidocs/data/checkpoints.db.
  One thread per "doc-{doc_id}-s{session_id}"; holds turn-by-turn state and
  SURVIVES backend restarts — any past session can be resumed.

User preferences — READ-ONLY SEED
  ~/.laidocs/memories/preferences.md is read at agent build time and injected
  into the SOUL system prompt. There is NO runtime write-back store: the agent
  does not learn or persist new preferences (the former SqliteStore /memories/
  backend was removed in the move off DeepAgents).
```

Display history (what the UI shows) is a separate `chat_messages` SQLite table
that survives a session reset — it is **not** the checkpointer. To keep prompt
size bounded, `compactor.py` rolls older display history into an LLM summary row
once it crosses a token threshold (last few Q&A pairs kept verbatim); the next
request loads `[summary] + [tail]` instead of the full transcript.

---

## 4. Agent orchestration

`agent.py` builds a single LangGraph `create_react_agent` (lazy singleton):

- **SOUL system prompt** — document-grounded only, no fabrication, cite
  sections/figures/tables, retrieve before answering. Preferences from
  `preferences.md` are appended at build time.
- **Tools** — `retrieve_context(question)` → `agentic_retrieve_context`;
  `read_image(image_path, prompt)` (VLM); `preview_edit` / `apply_edit`;
  `create_markdown_file([filename], [content])` (export a `.md` for download).
- **Checkpointer** — durable `AsyncSqliteSaver` (`checkpoints.db`) for
  per-session conversation memory; closed on shutdown via `close_checkpointer()`.
- **Compaction** — `compactor.compact_if_needed` runs before each stream; once
  display history exceeds the token threshold it rolls the older turns into an
  LLM summary, keeping the last few Q&A pairs verbatim, so the prompt stays bounded.
- **Evidence tracking** — the units retrieved on a turn are saved per message
  (`save_message_evidence`) and read back via `get_retrieved_evidence`; stale or
  unverified prior evidence is flagged so it is never reused as a document fact.
- **Concurrency** — per-request `doc_id`/`settings`/`edited` live in a
  `ContextVar` (`set_tool_context`), so concurrent requests never collide.
- **Reset** — `reset_agent()` after a settings change rebuilds the singleton
  with the new model on the next request (the checkpointer is reused).

Request flow:

```
chat request ─► compact_if_needed(doc_id) ─► set_tool_context(doc_id, settings)
   ─► agent.astream_events(question, version="v2", thread_id=doc-…-s…)
   agent calls retrieve_context(question)
        └─► agentic_retrieve_context → hybrid_rank (+ optional rerank) → context
            (cached embeddings reused for unchanged units)
   retrieved units saved as this message's evidence
   agent composes a grounded, cited answer ─► SSE tokens to the client
   (if it called apply_edit, the stream also emits [EDITED])
```

---

## 5. Running the system

There are **two ways to run** the AI layer, and the demos below use both:

| Entry point | What it's for | Setup |
|-------------|---------------|-------|
| **Full app** (Tauri + sidecar) | Interactive chat demos (Scenarios 3 & 7) — upload a doc, chat in the UI | [SETUP.md §3](SETUP.md#3-full-sidecar-env-to-run-the-applive-mode) |
| **AI-only `.venv-ai`** | Every benchmark / eval / headless script demo (Scenarios 1, 2, 4, 5, 6) | [SETUP.md §2](SETUP.md#2-the-isolated-ai-testbenchmark-env-venv-ai) |

> Commands below use the Windows path `.venv-ai/Scripts/python.exe`. On
> Linux/macOS substitute `.venv-ai/bin/python`.

### 5.1 Run the full app (for the interactive demos)

```bash
pip install -r backend/requirements.txt
python backend/main.py --dev        # sidecar only, http://localhost:8008
# — or the whole desktop app (frontend + sidecar):
pnpm install && pnpm tauri dev
```

Then open **Settings**, configure a provider (see [SETUP.md §4](SETUP.md#4-provider-configuration)),
and ingest a document (upload a PDF or crawl a URL). The tree index builds in a
background task after ingest.

### 5.2 Create the AI-only env (for every script demo)

```bash
uv venv .venv-ai --python 3.11
uv pip install --python .venv-ai \
    langchain langchain-core "langchain[google-genai]" langchain-google-genai \
    langchain-openai langgraph langgraph-checkpoint-sqlite \
    deepagents rank_bm25 numpy pydantic pydantic-settings pytest
```

This env is lean (no `docling`/`torch`) and offline-capable. Full detail and
provider config in [SETUP.md §2](SETUP.md#2-the-isolated-ai-testbenchmark-env-venv-ai).

### 5.3 30-second smoke test (offline — no LLM, no documents)

```bash
# the AI suite (the lean .venv-ai excludes the ingest deps, so target the
# three hermetic AI test files rather than `pytest tests/`)
.venv-ai/Scripts/python.exe -m pytest \
    tests/test_phase4_multimodal.py \
    tests/test_phase5_evaluation.py \
    tests/test_phase6_knowledge_graph.py -q
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json -k 5
```

Expect **41 tests green** plus a variant leaderboard. If this works, every
**offline** demo below (Scenario 1, and Scenario 2's dry-run) works from a clean
checkout with no API key.

### 5.4 Find a `doc_id` (needed for the live demos)

Live scenarios run the real retrievers against a document already ingested into
`~/.laidocs`. Get its id either way:

```bash
# A) from the running app's API
curl http://localhost:8008/api/documents/

# B) straight from SQLite (no server needed)
python -c "import sqlite3, os; db=os.path.expanduser('~/.laidocs/data/laidocs.db'); \
print('\n'.join(f'{r[0]}  {r[1] or r[2]}' for r in \
sqlite3.connect(db).execute('SELECT id, title, filename FROM documents')))"
```

Use that id wherever a snippet says `YOUR_DOC_ID`.

---

## 6. Demo scenarios

Repeatable, end-to-end walkthroughs that build **one story**: *this assistant is
grounded, reasons across a document, and you can prove it.* Run them roughly in
order — each scenario adds a claim, and the last two back the claims with numbers.

Every scenario lists **what it proves**, the **subsystem/files**, the **mode**
(offline / live / in-app), **setup**, **exact steps**, the **expected result**
on the shared demo document, **what to point at**, and a one-line **wow**.

### 6.0 The demo document (use this everywhere)

All live scenarios use one small, deterministic file:
[`docs/ai/demo/acme_robotics_handbook.md`](demo/acme_robotics_handbook.md). Its
facts are **chained across sections on purpose** — the founder is *named* in §1
but *described* (PhD city, birthplace) in §3; the product is named in §4 but the
factory city lives in §2. That is exactly the shape where plain RAG fails and
GraphRAG wins, so the contrast is reproducible, not lucky.

```text
§1 Company     Acme Robotics — founded by Dr. Lena Hoffmann (2014)
§2 Headquarters    …headquartered in Berlin; all manufacturing in Berlin
§3 Leadership      Dr. Lena Hoffmann — PhD in Munich, born in Lyon
§4 Products        …manufactures the Atlas-7 warehouse robot
§5 People          Marco Ruiz leads Atlas-7, reports to Dr. Hoffmann
§6 Specifications  table: payload / speed / battery / navigation
§7 Support         warranty handled within 30 days
```

**Ingest it once:** start the app (§5.1), configure a provider, drag the file in
(or convert via the upload dialog). Then grab its id with the §5 helper and use it
wherever a snippet says `YOUR_DOC_ID`. The first chat turn triggers the embedding
and knowledge-graph caches; everything after is fast.

| # | Scenario | Proves | Mode | Needs |
|---|----------|--------|------|-------|
| 1 | Grounded, cited answers + out-of-scope refusal | reliability | **live** | app + doc + LLM |
| 2 | **Plain RAG vs GraphRAG** on a multi-hop question | the differentiator | **live** | doc + LLM |
| 3 | Graph-of-thought reasoning chains | explainability | **live** | doc + LLM |
| 4 | Agentic self-critique retrieval | autonomy | **live** | doc + LLM |
| 5 | Multimodal table / figure QA | breadth | **live** | doc + LLM |
| 6 | Durable memory + preference seeding | stickiness | **live** | app + LLM |
| 7 | Retrieval leaderboard | rigor (reproducible) | **offline** | nothing |
| 8 | Anti-hallucination / faithfulness harness | measurable trust | **offline → live** | LLM for live arm |

> **Tip for a live audience:** run **7 and 8 first on a screen-share to set up
> credibility** (they need no key and produce hard numbers), then switch to the
> app and walk 1 → 6 as the narrative. Scenarios 2 and 3 are the centrepiece.

---

### Scenario 1 — Grounded, cited answers + out-of-scope refusal (the promise)

- **Proves:** every answer is traceable to a section, and the assistant **refuses
  what isn't in the document** instead of inventing it.
- **Subsystem:** [`agent.py`](../../backend/services/agent.py) SOUL prompt + the
  `retrieve_context` tool; the empty-context guard in
  [`retrieval.py`](../../backend/services/retrieval.py).
- **Mode:** live, in the app.

**Steps (app):**
1. Ask **"What is the warranty window?"** → answer cites *§7 Support*: *warranty
   handled within 30 days*.
2. Ask an **out-of-scope** question — **"What is Acme's employee parking policy?"**
   (genuinely absent) → *"I don't see this in the document."* No fabrication.

**Headless variant — show the contract at the retrieval layer, before the model speaks:**

```bash
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services import retrieval as r; s=get_settings(); doc='YOUR_DOC_ID'; \
print('IN-SCOPE :', bool(r.retrieve_context(doc, 'What is the warranty window?', s))); \
print('OUT      :', repr(r.retrieve_context(doc, 'What is the parking policy?', s)[:80]) or '<empty>')"
```

**Expected:** the in-scope question returns labelled context (`[Section: Support …]`);
the out-of-scope question returns **empty context** → the agent refuses.

- **In-app:** the grounded answer shows a **"Grounded · N sources"** badge and
  clickable **citation chips**; click one to jump to that section in the
  document preview. The out-of-scope answer shows **no chips** — nothing to cite.
- **Point at:** the `[Section: …]` citation in the answer; the explicit refusal.
- **Wow:** *"The refusal is not the model being polite — it's enforced at the
  retrieval layer: no evidence, no answer."*

---

### Scenario 2 — Plain RAG vs GraphRAG on a multi-hop question (the centrepiece)

- **Proves:** the headline differentiator. On a question whose answer is split
  across sections, **plain hybrid RAG misses the bridge passage; GraphRAG
  recovers it** by walking the entity-relation graph — so only GraphRAG answers
  correctly.
- **Subsystem:** [`retrieval.hybrid_rank`](../../backend/services/retrieval.py)
  with/without the fused [`graph_search`](../../backend/services/retrieval.py)
  signal, backed by the cached [`knowledge_graph.py`](../../backend/services/knowledge_graph.py).
- **Mode:** live (graph build needs an LLM); plus a fully-offline reproducible arm.

**The question:** **"Where was the founder of Acme Robotics born?"**
The answer needs two hops: §1 says *Acme was founded by Dr. Lena Hoffmann*; §3 says
*Hoffmann was born in Lyon*. The words "founder" and "born" never co-occur in one
section, so BM25/dense rank §1 (or neither) and **drop §3** — the bridge.

**A/B at the retrieval layer (graph OFF vs ON):**

```bash
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services import retrieval as r; s=get_settings(); doc='YOUR_DOC_ID'; \
q='Where was the founder of Acme Robotics born?'; \
s.graph_rag.enabled=False; rag,_=r.hybrid_rank(doc,q,s); print('RAG-only units :', rag); \
s.graph_rag.enabled=True;  g,_  =r.hybrid_rank(doc,q,s); print('GraphRAG units :', g)"
```

**Expected:** the GraphRAG list contains the **Leadership (§3)** unit — the one
carrying *born in Lyon* — that the RAG-only list omits. Feed each context to the
agent and the RAG-only answer is *"the document doesn't say where the founder was
born,"* while GraphRAG answers **"Lyon."**

**Fully reproducible proof (offline, no key):**

```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
  --dataset scripts/ai_eval/datasets/multihop_graph.json -k 2
```

**Expected:** `graph` scores **recall/precision 1.0**; `hybrid`/`dense` **0.5**;
`bm25` lower — the bridge unit only surfaces through the graph walk.

**In-app (clickable A/B):** turn on **Demo mode** (toggle in the chat header),
type the question, and hit **"Compare RAG vs GraphRAG"**. The `CompareDrawer`
shows both answers side by side and highlights the **bridge passage** GraphRAG
recovered — the same A/B as the headless command, no terminal needed.

- **Point at:** the one extra `unit_id` in the GraphRAG list (highlighted green
  in the drawer); the two answers diverging on the *same* question and *same*
  model — only retrieval changed.
- **Wow:** *"Same model, same question — the only difference is whether we walked
  the graph. That single bridge passage is the difference between 'I don't know'
  and the right answer."*

---

### Scenario 3 — Graph-of-thought reasoning chains (explainability)

- **Proves:** the assistant can **show its reasoning path**, not just an answer —
  the explicit relation chain it followed across the document.
- **Subsystem:** the `reason_over_graph` agent tool →
  [`knowledge_graph.graph_of_thought_cached`](../../backend/services/knowledge_graph.py).
- **Mode:** live (LLM for triple/entity extraction; cache built in Scenario 2).

**Steps (app):** ask a relational question — **"How is Marco Ruiz connected to
Lyon?"** The agent calls `reason_over_graph` and grounds the prose answer in the
chain.

**Headless:**

```bash
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services import knowledge_graph as kg; s=get_settings(); doc='YOUR_DOC_ID'; \
print(kg.graph_of_thought_cached(doc, 'How is Marco Ruiz connected to Lyon?', s) \
      or '<no connecting path>')"
```

**Expected:** a rendered chain such as
`Marco Ruiz --[reports to]--> Lena Hoffmann --[born in]--> Lyon` — assembled from
§5 + §3, two sections that share no keywords with each other.

- **In-app:** the answer renders the chain as a **"Reasoning path"** strip of
  node/edge chips (`Marco Ruiz → reports to → Lena Hoffmann → born in → Lyon`),
  collapsible, right above the citation chips.
- **Point at:** the chain hops, each traceable to a source section.
- **Wow:** *"It doesn't just answer — it shows the path it walked, and every hop
  is a sentence in the document."*

---

### Scenario 4 — Agentic self-critique retrieval (autonomy)

- **Proves:** when one retrieval pass is insufficient, the agent **names what's
  missing and fetches it** in a follow-up round, instead of answering half a
  question.
- **Subsystem:** [`retrieval.agentic_retrieve_context`](../../backend/services/retrieval.py)
  (≤3 rounds, ≤2 follow-ups/round, ≤12 accumulated units).
- **Mode:** live (LLM for the critique step).

```bash
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services import retrieval as r; s=get_settings(); doc='YOUR_DOC_ID'; \
q='Who leads the Atlas-7 program, and where did the person they report to study?'; \
print('--- single-shot ---'); print(r.retrieve_context(doc, q, s)[:1200]); \
print('--- agentic ---');     print(r.agentic_retrieve_context(doc, q, s)[:1200])"
```

**Expected:** the question spans §5 (Marco → reports to Hoffmann) and §3 (Hoffmann
→ PhD in Munich). The agentic context accumulates **both**; single-shot may stop
at the first hop. (GraphRAG and the self-critique loop are complementary — the
graph gives recall in one pass, the loop recovers when a pass falls short.)

- **Point at:** the second-round follow-up query the critic generated.
- **Wow:** *"It audits its own evidence and goes back for the missing piece —
  and if the critique ever flakes, it safely defaults to 'sufficient' so it never
  hangs."*

---

### Scenario 5 — Multimodal table / figure QA (breadth)

- **Proves:** tables (`tbl*`) and figures (`img*`) are **first-class retrieval
  units**, parsed lazily at query time — a question about a cell surfaces the
  table, cells intact.
- **Subsystem:** [`retrieval.get_retrieval_units`](../../backend/services/retrieval.py)
  multimodal units + RRF; `read_image` (VLM) for figures.
- **Mode:** live.

```bash
# 1) list units → find the tbl* id (the §6 Specifications table)
.venv-ai/Scripts/python.exe -c "from backend.services import retrieval as r; import json; \
print(json.dumps([{k:u[k] for k in ('unit_id','kind','title')} \
for u in r.get_retrieval_units('YOUR_DOC_ID')], indent=2))"

# 2) ask a question whose answer is a single table cell
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services import retrieval as r; \
print(r.retrieve_context('YOUR_DOC_ID', \
'What is the Atlas-7 battery life?', get_settings())[:1500])"
```

**Expected:** the context includes a `[Table: Specifications …]` block with the
rows intact; the agent answers **"9 hours"** and cites *"Table: Specifications."*
Content-free `Image N` placeholders are skipped (noise control).

- **Point at:** the `[Table: …]` block in the context and the cited cell.
- **Wow:** *"It reads the cell, it doesn't guess — and tables already converted
  work with no re-ingest."*

---

### Scenario 6 — Durable memory + preference seeding (stickiness)

- **Proves:** a session's memory **survives a backend restart**, and a seeded
  preference shapes every new session.
- **Subsystem:** [`agent.py`](../../backend/services/agent.py) `AsyncSqliteSaver`
  checkpointer (`~/.laidocs/data/checkpoints.db`) + preference injection.
- **Mode:** live, in the app.

**Durable memory:**
1. **Turn 1:** "What does Acme manufacture?" → *Atlas-7*. **Turn 2:** "What did I
   just ask?" → it recalls.
2. **Restart the backend.** Reopen the doc, pick the **same session** in the
   switcher, ask "What was my first question?" → still recalls (read back from
   `checkpoints.db`).

**Preference seeding:** add a line to `~/.laidocs/memories/preferences.md` (e.g.
*"Always answer in Vietnamese, max 2 sentences"*), start a fresh session, ask any
question → it complies without restatement.

- **Point at:** the session switcher; the recalled first question after restart.
- **Wow:** *"Close the app, reopen, resume the exact conversation — the memory is
  on disk, not in RAM."*

> **Scope note:** there is **no runtime write-back store** — the agent does not
> auto-learn new preferences across sessions (the old DeepAgents `SqliteStore`
> `/memories/` backend was removed). Preferences are a read-only seed; durability
> comes from the conversation checkpointer.

---

### Scenario 7 — Retrieval leaderboard (offline, fully reproducible)

- **Proves:** retrieval quality is **measured**, not asserted — five retriever
  variants ranked head-to-head on a gold dataset, from a clean checkout, no key.
- **Subsystem:** [`retrieval.py`](../../backend/services/retrieval.py) +
  [`knowledge_graph.py`](../../backend/services/knowledge_graph.py) +
  [`evaluation.py`](../../backend/services/evaluation.py).
- **Mode:** offline.

```bash
# general leaderboard
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json -k 5
# the multi-hop set where graph dominates (pairs with Scenario 2)
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset scripts/ai_eval/datasets/multihop_graph.json -k 2
```

**Expected:** leaderboard sorted by nDCG; on the multi-hop set `graph` tops the
table at recall `1.0`. Recall@k is the headline metric (no evidence retrieved →
no correct answer); hybrid is rarely worse than its best component because RRF
rewards units multiple retrievers agree on.

- **Go live:** put real `doc_id`s on each case, drop `ranked_units`, add
  `--live --variants bm25,dense,tree,hybrid,graph --out runs/retrieval.json` for a
  `latency_s_mean` column too. Full method in [EVALUATION.md §4](EVALUATION.md#4-retrieval-benchmarking).

---

### Scenario 8 — Anti-hallucination / faithfulness harness (measurable trust)

- **Proves:** "grounded" is a **number**. Faithfulness decomposes an answer into
  atomic claims, NLI-checks each against the retrieved context, and **lists the
  unsupported ones** — the concrete hallucinations.
- **Subsystem:** [`evaluation.py`](../../backend/services/evaluation.py) (RAGAS-style judge).
- **Mode:** offline dry-run (wiring) → live (real LLM judge).

```bash
# offline wiring check — neutral judge, zero model calls
.venv-ai/Scripts/python.exe scripts/ai_eval/run_grounding_eval.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json --dry-run
# live — real judge, per-question "Unsupported claims" list
.venv-ai/Scripts/python.exe scripts/ai_eval/run_grounding_eval.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json --out runs/grounding.json
```

**Expected (dry-run):** `faithfulness 1.0000` (neutral), `answer_relevance 0.0000`
— the neutral judge **never reports a false hallucination**, so an unconfigured
environment degrades safely. **(live):** aggregate scores plus per-question
unsupported claims. To make a hallucination visible, edit one case's `answer` to
assert a wrong number and re-run — that claim gets flagged.

- **Point at:** the "Unsupported claims" list; the safe-degradation `1.0`.
- **Wow:** *"This is the harness that turns 'looks fine in chat' into a score we
  can regression-test."* Adversarial tests in [EVALUATION.md §5](EVALUATION.md#5-grounding--hallucination-evaluation).

---

## 7. Where to go next

- **Set it up:** [SETUP.md](SETUP.md)
- **Benchmark & reproduce:** [EVALUATION.md](EVALUATION.md)
- **Run the harness:** [`scripts/ai_eval/`](../../scripts/ai_eval/)
