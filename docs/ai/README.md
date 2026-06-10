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

Repeatable, end-to-end walkthroughs of the AI layer. Each lists **what it
shows**, **which subsystem**, **mode**, the **exact commands**, the **expected
result**, and the **talk track**. (For the one-paragraph "presenter cue"
versions and the deeper measurement protocol behind each, see
[EVALUATION.md §13](EVALUATION.md#13-demo-scenarios).)

| # | Scenario | Subsystem | Mode | Needs |
|---|----------|-----------|------|-------|
| 1 | Retrieval leaderboard | BM25 / dense / tree / hybrid / graph + RRF | **offline** | nothing |
| 2 | Anti-hallucination harness | grounding / faithfulness eval | **offline → live** | LLM for the live arm |
| 3 | Citation-grounded QA & out-of-scope refusal | agent SOUL + `retrieve_context` | **live** | app + doc + LLM |
| 4 | Agentic multi-hop retrieval | self-critique loop | **live** | doc + LLM |
| 5 | Multimodal table / figure QA | `img*` / `tbl*` units | **live** | doc w/ table+figure + LLM |
| 6 | Knowledge graph + graph-of-thought | entity graph reasoning | **live** | doc + LLM |
| 7 | Durable memory / personalization | `SqliteStore` `/memories/` | **live** | app + LLM |

**Start with the offline scenarios** (1, and 2's dry-run) — they reproduce from a
clean checkout with no key. The live scenarios assume §5.1/§5.2 are set up with a
provider configured and a document ingested.

### Scenario 1 — Retrieval leaderboard (offline, fully reproducible)

- **Shows:** the five retriever variants ranked head-to-head on a gold dataset —
  the evidence that hybrid RRF is a robust default and `graph` wins precision on
  the multi-hop entity question.
- **Subsystem:** [`retrieval.py`](../../backend/services/retrieval.py) (BM25 +
  dense + tree + RRF) and [`knowledge_graph.py`](../../backend/services/knowledge_graph.py).
- **Mode:** offline — scores the `ranked_units` already recorded in the dataset;
  no LLM, no DB.

```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json -k 5
```

**Expected** — leaderboard sorted by nDCG (graph precision `0.6667`, the rest
`0.2667`; `tree` lowest recall). Matches [EVALUATION.md §4](EVALUATION.md#4-retrieval-benchmarking).

**Talk track:** recall@k is the headline metric for a grounded assistant (no
evidence retrieved → no correct answer); `graph`'s higher precision on the
"HQ city + founder" question; hybrid is rarely worse than its best component
because RRF rewards units multiple retrievers agree on.

**Go live:** put real `doc_id`s on each case, drop `ranked_units`, and add
`--live --variants bm25,dense,tree,hybrid,graph --out runs/retrieval.json` to
also get a `latency_s_mean` column.

### Scenario 2 — Anti-hallucination / grounding harness (offline → live)

- **Shows:** faithfulness decomposes an answer into atomic claims and NLI-checks
  each against the retrieved context, then **lists the unsupported claims** — the
  concrete hallucinations.
- **Subsystem:** [`evaluation.py`](../../backend/services/evaluation.py) (RAGAS-style judge).
- **Mode:** offline dry-run (wiring check) → live (real LLM judge).

```bash
# offline wiring check — neutral judge, zero model calls
.venv-ai/Scripts/python.exe scripts/ai_eval/run_grounding_eval.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json --dry-run
```

**Expected (dry-run):** `faithfulness 1.0000` (neutral), `answer_relevance 0.0000`,
`context_relevance 0.0000`. The neutral judge **never reports a false
hallucination** — an unconfigured environment degrades safely.

```bash
# real evaluation — needs a configured provider
.venv-ai/Scripts/python.exe scripts/ai_eval/run_grounding_eval.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json --out runs/grounding.json
```

**Expected (live):** aggregate `faithfulness` / `answer_relevance` /
`context_relevance`, plus a per-question **"Unsupported claims"** list. To make a
hallucination visible, edit one case's `answer` to assert a wrong number and
re-run — that claim is flagged.

**Talk track:** the safe-degradation property (no judge → `1.0`, never a false
positive) and that this is the harness which turns "looks fine in chat" into a
number. Full method + adversarial tests in [EVALUATION.md §5](EVALUATION.md#5-grounding--hallucination-evaluation).

### Scenario 3 — Citation-grounded QA & out-of-scope refusal (the headline)

- **Shows:** the core promise — answers cite the section they came from, and the
  assistant refuses what isn't in the document instead of fabricating.
- **Subsystem:** [`agent.py`](../../backend/services/agent.py) SOUL prompt + the
  `retrieve_context` tool.
- **Mode:** live, in the app.

**In the app:**
1. Start the app (§5.1), configure a provider, and ingest a structured PDF
   (a paper, policy, or handbook with clear headings).
2. Ask a factual question answerable from one section → the answer cites it,
   e.g. *"[Section: Refunds] …"*.
3. Ask an **out-of-scope** question (something genuinely absent) → *"I don't see
   this in the document"* — no fabrication.

**Headless variant (no UI), to show the retrieval-layer contract directly:**

```bash
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services import retrieval as r; \
ctx = r.retrieve_context('YOUR_DOC_ID', 'An in-scope question?', get_settings()); \
print(ctx[:1500] or '<empty context -> treated as out-of-scope>')"
```

**Expected:** an in-scope question returns labelled context (`[Section: …]` /
`[Table: …]`); a genuinely out-of-scope question returns **empty context** — the
anti-hallucination contract enforced at the retrieval layer, before the model
ever speaks.

**Talk track:** the empty-context → refusal path *is* the reliability story;
cited sections make every answer auditable.

### Scenario 4 — Agentic multi-hop retrieval (live)

- **Shows:** single-shot vs iterative self-critique. On a question whose answer
  is split across two sections, the agentic loop names the missing piece and
  fetches it in a follow-up round.
- **Subsystem:** [`retrieval.agentic_retrieve_context`](../../backend/services/retrieval.py)
  (≤3 rounds, ≤2 follow-ups/round, ≤12 accumulated units).
- **Mode:** live (needs an LLM for the critique step).

```bash
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services import retrieval as r; s=get_settings(); doc='YOUR_DOC_ID'; \
q='Which city is the company HQ in, and who founded it?'; \
print('--- single-shot ---'); print(r.retrieve_context(doc, q, s)[:1200]); \
print('--- agentic ---');     print(r.agentic_retrieve_context(doc, q, s)[:1200])"
```

**Expected:** the agentic context contains **both** facts (HQ city *and*
founder); single-shot may surface only the first hop. Quantify the recall lift in
[EVALUATION.md §6](EVALUATION.md#6-agentic-retrieval-evaluation).

**Talk track:** worst case degrades to single-shot — a flaky critique defaults to
`sufficient=true`, so the loop never hangs; the cost is up to ~3× the calls,
which the live benchmark's latency column quantifies.

### Scenario 5 — Multimodal table / figure QA (live)

- **Shows:** figures (`img*`) and tables (`tbl*`) are first-class retrieval
  units, parsed **lazily at query time** from the stored Markdown — so a question
  about a table cell or a figure caption surfaces that unit (no re-ingest).
- **Subsystem:** [`retrieval.get_retrieval_units`](../../backend/services/retrieval.py)
  multimodal units + RRF.
- **Mode:** live (a document that actually has a table and/or figure).

```bash
# 1) list the units to find the tbl* / img* ids
.venv-ai/Scripts/python.exe -c "from backend.services import retrieval as r; \
import json; print(json.dumps([{k:u[k] for k in ('unit_id','kind','title')} \
for u in r.get_retrieval_units('YOUR_DOC_ID')], indent=2))"

# 2) ask a question whose answer lives in a cell / caption
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services import retrieval as r; \
print(r.retrieve_context('YOUR_DOC_ID', \
'What were total revenues in the most recent year?', get_settings())[:1500])"
```

**Expected:** the context includes a `[Table: …]` block (cells kept intact) or a
`[Figure: …]` block (caption + VLM description). Content-free `Image N` figures
are skipped (noise control).

**Talk track:** units are parsed at query time so already-converted documents
work with no re-ingest; the model cites *"Table: …"* / *"Figure: …"*. Benchmark
recall on a `tbl*`/`img*` dataset in [EVALUATION.md §8](EVALUATION.md#8-multimodal-evaluation).

### Scenario 6 — Knowledge graph + graph-of-thought (live)

- **Shows:** an entity-relation graph built from the same units. A multi-hop walk
  returns the supporting `unit_id`s (shaped to fuse into RRF), and
  graph-of-thought renders explicit reasoning chains like
  `A --[founded by]--> B --[located in]--> C`.
- **Subsystem:** [`knowledge_graph.py`](../../backend/services/knowledge_graph.py).
- **Mode:** live (LLM needed for triple/entity extraction).

```bash
.venv-ai/Scripts/python.exe -c "from backend.core.config import get_settings; \
from backend.services import knowledge_graph as kg; s=get_settings(); \
doc='YOUR_DOC_ID'; q='Who founded the company and where is it based?'; \
print('augmented units:', kg.graph_augmented_units(doc, q, s)); \
print(kg.graph_of_thought(doc, q, s) or '<no connecting path>')"
```

**Expected:** `graph_augmented_units` returns a ranked list of `unit_id`s;
`graph_of_thought` prints the relation paths between the question's entities
(empty when no LLM/extractor or no connecting path exists).

**Talk track:** this is a **parallel retrieval track** shaped to fuse into
`rrf_fuse`. It currently runs from scripts/tests and is **not yet wired into the
live agent answer path** (see §1.3) — an intentional, documented boundary. Its
precision edge is already visible in Scenario 1's leaderboard.

### Scenario 7 — Durable memory / personalization (live, in the app)

- **Shows:** preferences survive a session reset **and** a restart, via the
  `SqliteStore` at `~/.laidocs/data/memory_store.db` — Turn 3 honours a Turn 1
  preference without restatement.
- **Subsystem:** [`agent.py`](../../backend/services/agent.py) durable store +
  `/memories/`.
- **Mode:** live, in the app.

1. **Turn 1:** "Answer me in Vietnamese and keep answers to 2 sentences."
2. **Turn 2:** a normal document question — observe it complies.
3. **New session** (resets the checkpointer, keeps the store) → **Turn 3:** a
   fresh document question, *without* restating the preference.

**Expected:** Turn 3 is still Vietnamese and ~2 sentences. For the A/B that
*proves* the effect, wipe `memory_store.db` and the seed `preferences.md`, repeat,
and watch Turn 3 revert to the default ([EVALUATION.md §7](EVALUATION.md#7-long-term-memory-evaluation)).

**Talk track:** memory stores **preferences, not document facts** — verify it
never injects an ungrounded claim (faithfulness *with* memory ≈ *without*).

---

## 7. Where to go next

- **Set it up:** [SETUP.md](SETUP.md)
- **Benchmark & reproduce:** [EVALUATION.md](EVALUATION.md)
- **Run the harness:** [`scripts/ai_eval/`](../../scripts/ai_eval/)
