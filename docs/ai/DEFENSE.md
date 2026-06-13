# LAIDocs — Thesis Defense Package

> **Purpose:** everything you need to defend this project orally. Read top-to-bottom
> once, then memorize §A (60-second pitch), §M (critical concepts), and §Q (the
> hardest questions). Every claim here is traceable to real code — file paths are given.

---

## Visual aids

Seven projection-ready diagrams live in [`diagrams/`](diagrams/README.md) — open them
while you study each section:

1. [Architecture](diagrams/01-architecture.svg) (§C) · 2. [Request lifecycle](diagrams/02-request-lifecycle.svg) (§N) · 3. [Hybrid retrieval + RRF](diagrams/03-hybrid-retrieval-rrf.svg) (§E) · 4. [Agentic loop](diagrams/04-agentic-loop.svg) (§E.6) · 5. [Memory layers](diagrams/05-memory-layers.svg) (§G) · 6. [Edit gate](diagrams/06-edit-gate-interrupt.svg) (§H) · 7. [GraphRAG](diagrams/07-graphrag.svg) (§F)

---

## A. The 60-second pitch (memorize this verbatim)

> "LAIDocs is a **local, document-grounded AI assistant**. You convert files or web
> pages into Markdown, organize them, and chat with them. The chat is a **LangGraph
> ReAct agent** with a strict 'SOUL' contract: it answers **only** from content its
> tools retrieve from your selected documents, cites the file and section, and refuses
> to fabricate. Retrieval is **hybrid** — it fuses four signals (a reasoning-based tree
> index, BM25 lexical, dense embeddings, and an entity-relation knowledge graph) with
> **Reciprocal Rank Fusion**. It runs **fully local** except for calls to whatever LLM
> endpoint you configure — Gemini by default, or any OpenAI-compatible/Anthropic
> endpoint, including a fully-local Ollama. It has **durable conversation memory**, can
> **edit your documents through a human-approval gate**, and reads **figures and tables**
> with a vision model."

That paragraph touches every selling point. If you can deliver it confidently, you've
already passed the first impression.

---

## B. What it is, who it's for, what it solves

**What it does.** Convert documents (PDF/DOCX/PPTX/XLSX/HTML) or crawled web pages into
clean Markdown, store them in a local vault, and answer questions about them with
citations. It can also *edit* documents and *export* generated content.

**Target users.** Students, researchers, analysts, lawyers, anyone with a private
document corpus they don't want to upload to a cloud SaaS. The privacy story is central:
the only network egress is to *your* chosen LLM endpoint.

**Problems it solves.**
1. **Hallucination** — a general chatbot invents facts. LAIDocs is *grounded*: every
   factual claim must come from retrieved context, and out-of-scope questions get
   "I don't see this in the document" rather than a confident lie.
2. **Finding the right passage** — keyword search misses paraphrases; pure vector search
   misses exact codes/names. Hybrid retrieval covers both, plus multi-hop questions whose
   answer is split across sections.
3. **Privacy** — no forced cloud upload; can run end-to-end local with Ollama.
4. **Trust/verifiability** — citation chips let the user click back to the exact source
   section, so answers are auditable.

**End-to-end workflow.** Upload → convert to Markdown (Docling/MarkItDown) → build a
hierarchical tree index + (background) knowledge-graph triple cache → user selects
document(s) as scope → asks a question → agent retrieves → answers with citations → user
can approve edits or export.

**Three framings for the committee:**
- *Beginner:* "It's a chatbot that reads your PDFs and answers with sources."
- *Intermediate:* "It's a RAG system with hybrid retrieval and an agentic loop, wrapped
  in a desktop app, with grounding guarantees."
- *Expert/research:* "It's a provider-agnostic, tool-using LangGraph agent over a
  four-signal RRF-fused retriever (reasoning-tree + BM25 + dense + GraphRAG), with
  self-critique multi-hop retrieval, incremental per-unit embedding & triple caches,
  durable checkpointed memory with LLM summarization compaction, and a LangGraph
  `interrupt()`-based human-in-the-loop edit gate."

---

## C. System architecture (the whole stack)

```
Tauri v2 (Rust shell)  — spawns + supervises the sidecar, native file dialogs
├── React 19 + TS + Tailwind  (WebView, Vite dev :5173)
│     └── talks to backend over HTTP REST + SSE on localhost:8008
└── Python FastAPI sidecar  (localhost:8008)
      ├── Conversion:  Docling (PDF/DOCX/PPTX/HTML), MarkItDown (XLSX), Crawl4AI (web)
      ├── Indexing:    PageIndex tree index  +  knowledge-graph triple cache
      ├── Agent:       LangGraph create_react_agent  (SOUL prompt + 6 tools)
      ├── Retrieval:   tree + BM25 + dense + graph  → RRF  → optional rerank
      ├── Memory:      AsyncSqliteSaver checkpointer  + preferences.md
      └── Storage:     SQLite (metadata, tree, embeddings, KG, chat) + filesystem vault
```

**Why a sidecar (two processes, not one)?** The heavy AI/ML Python ecosystem (Docling,
LangChain, numpy) lives in Python; the UI is web tech in a Rust-native shell. They
communicate over localhost HTTP. Tauri spawns the Python process and shuts it down
**gracefully via stdin** (`"sidecar shutdown\n"`), never `kill()`, so the backend can
close its DB/checkpointer cleanly.

**Key paths to know:**
| Path | What |
|------|------|
| `~/.laidocs/vault/` | Markdown documents + extracted image assets |
| `~/.laidocs/data/laidocs.db` | metadata, tree index, chat history, embedding cache, KG cache |
| `~/.laidocs/data/checkpoints.db` | durable conversation memory (LangGraph checkpointer) |
| `~/.laidocs/memories/preferences.md` | seed user preferences injected into the prompt |
| `~/.laidocs/config.json` | persisted LLM/VLM/reranker/graph settings |

---

## D. The agent (`backend/services/agent.py`)

It's a **LangGraph `create_react_agent`** — the prebuilt ReAct (Reason + Act) loop. The
graph is: `LLM node → (tool calls?) → tools node → LLM node → … → final answer`. We chose
the prebuilt agent because it works **identically across Gemini / OpenAI / Anthropic**.
(The CLAUDE.md notes it *replaced* DeepAgents, whose Anthropic-only middleware returned
empty responses on Gemini — a concrete portability lesson.)

**The SOUL system prompt** (the `DOCUMENT_SOUL_PROMPT` constant) is the grounding
contract. Non-negotiable rules: document-grounded only; never fabricate; cite file +
section; **first tool call for any content question must be `retrieve_context`**;
treat prior assistant answers as *intent*, not *evidence* (so stale answers can't
re-enter as facts after an edit).

**The six tools:**
1. `retrieve_context(question)` — the workhorse. Calls
   `retrieval.agentic_retrieve_context_multi_with_evidence`. Returns labelled context
   sections AND records citable evidence units for the UI.
2. `reason_over_graph(question)` — GraphRAG graph-of-thought; renders explicit relation
   chains (`Acme --[founded by]--> Jane --[born in]--> Paris`) for multi-hop questions.
3. `read_image(image_path, prompt)` — sends a base64 image to the **VLM** (vision model)
   for figures/charts.
4. `preview_edit(file, old, new)` — dry-run; locates the unique span, shows the diff.
5. `apply_edit(file, old, new)` — the **human-in-the-loop gate** (see §H).
6. `create_markdown_file(filename?, content?)` — export generated content as a
   downloadable `.md`.

**Per-request isolation.** Tool context (which docs are in scope, settings, titles) is
stored in a `contextvars.ContextVar`, **not** a module-level dict. That's what makes
concurrent requests safe — each async request has its own copy. *Memorize this; it's a
classic "how do you handle concurrency?" question.*

**Singleton + reset.** The agent is built lazily once and cached. After a settings change
(`POST /api/settings`) the API calls `reset_agent()` so the next request rebuilds with the
new model. The checkpointer survives the reset.

---

## E. Retrieval pipeline (`backend/services/retrieval.py`) — the heart

### E.1 Units: the atom of retrieval
Everything ranks **"units."** A unit is either:
- a **tree node** (section, `unit_id` = node id) when the doc has headings, OR
- a **paragraph chunk** (~1000 chars, `unit_id` = `c0001…`) when it has no headings, PLUS
- **multimodal units**: figures (`img0001…`, alt-text + VLM description + caption) and
  tables (`tbl0001…`, the intact markdown table so the model reads cells directly).

This is a deliberate design choice: **the same unit set feeds all four retrievers**, so
their ranked lists fuse cleanly (they're voting on the same candidates).

### E.2 The four signals
| Signal | Function | What it captures | Cost |
|--------|----------|------------------|------|
| **Tree** | `select_node_ids` | LLM reads the heading tree (titles + summaries, *no body text*) and picks relevant node ids — reasoning, not embedding | 1 LLM call |
| **BM25** | `bm25_search` | lexical term overlap (exact names, codes, rare tokens) via `rank_bm25.BM25Okapi` | local, no LLM |
| **Dense** | `dense_search` | embedding cosine similarity (paraphrase, synonyms) | 1 embed call (cached) |
| **Graph** | `graph_search` | entity-relation walk for multi-hop, split-evidence questions | ~1 LLM call (cached triples) |

### E.3 Fusion: Reciprocal Rank Fusion (RRF)
Each retriever returns a *ranked list of unit ids*. RRF fuses them:

> `score(unit) = Σ_retrievers 1 / (k + rank)`, with `k = 60` (`_RRF_K`).

**Why RRF and not weighted score-averaging?** Because the four signals produce
**incomparable scores** — a BM25 score, a cosine in [-1,1], and an LLM's pick are not on
the same scale. RRF only uses **rank position**, so it needs no normalization and no
hand-tuned weights. A unit that several retrievers rank highly wins; the `+k` damps the
influence of any single list's tail. This is the single most defensible design decision
in the retriever — know it cold.

### E.4 Optional cross-encoder rerank
If a reranker is configured (`active_reranker.enabled`, a Jina-compatible `/v1/rerank`
endpoint), the fused top candidates go through a **cross-encoder** that scores
(query, document) *jointly* — more accurate than the bi-encoder cosine, but too expensive
to run on the whole corpus, so it only reranks the already-fused shortlist.

### E.5 Context assembly
Selected units are formatted with `[File: … | Section: …]` headers and concatenated up to
`MAX_CONTEXT_CHARS = 12000`. This bound is why **rank order matters** — low-ranked units
can be truncated away.

### E.6 The agentic loop (multi-hop + self-critique)
`agentic_retrieve_context_multi_with_evidence` wraps the hybrid ranker in a loop
(`MAX_RETRIEVAL_ROUNDS = 3`):
1. Retrieve for the original question.
2. An LLM **critic** judges: *is this context sufficient? If not, what specific
   follow-up sub-queries would fill the gap?* (`_CRITIQUE_PROMPT`).
3. If insufficient, run the follow-ups (up to 2/round) and accumulate evidence, scored by
   best RRF rank seen.
4. Stop when sufficient, no new units appear, no follow-ups, or 3 rounds hit.

**Safety:** a flaky critique call defaults to `sufficient=true` — so the worst case
degrades to single-shot retrieval, never an infinite loop. *This is a great answer to
"what if the LLM misbehaves in your loop?"*

### E.7 Caching (the performance story)
- **Dense:** embeddings are cached in `document_embeddings`, keyed by a **per-unit content
  hash** (`unit_hash`). Edit one paragraph → only that unit is re-embedded, not the whole
  document. Lazy: built on first query, no ingest-time blocking hook.
- **Graph:** triples are extracted **once per unit** and cached in `document_graph_units`,
  keyed by `unit_hash` + extractor model. Query time = one query-entity LLM call + a
  deterministic graph walk. This caching is *the entire reason GraphRAG could be wired
  into the live agent* — without it, every question would re-extract the whole graph.

---

## F. GraphRAG (`backend/services/knowledge_graph.py`) — the research layer

**Motivation.** Lexical and dense retrieval rank *passages*. They fail when the answer is
*split*: "X was founded by Y, who also created Z" — no single passage contains the whole
chain, and "Z" may share no words with the question.

**How it works.**
1. **Extraction:** an LLM turns each unit's text into `(subject, relation, object)`
   triples (`_TRIPLE_PROMPT`). Cached per unit.
2. **Graph build (deterministic, no LLM):** triples become a directed multigraph;
   entities are normalized (lowercased, whitespace-collapsed) for matching; a
   `unit_index` maps each entity back to the source units that mention it — **the bridge
   back to passage retrieval**.
3. **Query:** extract the question's entities (1 LLM call) → match them to graph nodes →
   **BFS up to `MAX_HOPS = 2`** → collect the source units of every reached entity.
4. Those unit ids become **one more ranked list fused into RRF** — graph retrieval doesn't
   replace the others, it augments them.

**Graph-of-thought** (`reason_over_graph` tool / `graph_of_thought`): instead of returning
units, it enumerates the *relation paths* between the question's entities and renders them
as readable chains. It's a **reasoning scaffold** ("a map of where to look"), explicitly
*not* standalone evidence — the agent still grounds claims in retrieved prose.

**Honest limit to state proactively:** the graph is only as good as the LLM's triple
extraction. Bad extraction → missing edges. We mitigate with lenient JSON parsing,
per-unit caching, and by treating graph output as a *complement*, never the sole source.

---

## G. Memory system (`agent.py` + `compactor.py`)

Three distinct layers — **don't conflate them**, examiners love this distinction:

1. **Durable conversation memory** — a LangGraph `AsyncSqliteSaver` **checkpointer** at
   `checkpoints.db`, keyed by `thread_id = "session-{session_id}"`. It persists the full
   message history and **survives backend restarts**. On each turn we send *only the new
   question*; the checkpointer replays the rest.
2. **Working-memory window** — a `pre_model_hook` (`_trim_to_recent_turns`) caps what the
   *model sees* this step to the last `MAX_RECENT_TURNS = 10` user turns. It returns
   `llm_input_messages` (a view), so the **persisted checkpoint is untouched** — the
   window always starts on a `HumanMessage` so a tool-call/tool-result pair is never
   orphaned.
3. **Checkpoint compaction** (`compactor.py`) — the trim above bounds *prompt* tokens but
   not the *stored* checkpoint, which would grow unboundedly (huge `ToolMessage` payloads
   from retrieval). After each turn, if the stored list exceeds
   `CHECKPOINT_COMPACT_THRESHOLD_TOKENS = 4000`, it replaces everything except the last
   `CHECKPOINT_TAIL_PAIRS = 2` exchanges with a single **LLM-generated summary
   `AIMessage`**, then writes the smaller list back.

Plus **preference memory**: `~/.laidocs/memories/preferences.md` is read at agent build
time and injected into the system prompt (read-only seeding — there's no live write-back).

**Two subtle correctness details worth knowing** (they signal depth):
- The compactor generates a **UUID6** checkpoint id, not uuid4, because AsyncSqliteSaver
  picks "latest" by `ORDER BY checkpoint_id DESC` and UUID6 is time-ordered — a uuid4
  could sort *before* the existing id and never be picked up.
- Compaction touches private LangGraph internals (`channel_values`, `channel_versions`),
  so it's best-effort and never raises — an honest, documented fragility.

---

## H. Document editing — human-in-the-loop via `interrupt()`

This is the most sophisticated control-flow piece; expect questions.

`apply_edit` does **not** write the file when called. It calls LangGraph **`interrupt()`**,
which **pauses the graph mid-execution**; the durable checkpointer holds the suspended
state. The diff surfaces to the UI as an `[INTERRUPT]` SSE event. The user approves or
rejects; the app calls `POST /api/chat/resume` with `Command(resume="approve"|"reject")`
on the **same thread_id**. The tool node *re-executes*: its read-only locate logic re-runs
idempotently, and the post-`interrupt()` write runs **exactly once** — on approve it
persists through `document_store.persist_document_content` (keeps the `.md` file, SQLite
`content`, and tree index in sync) and emits `[EDITED]`; on reject the file is untouched.

**Stale-gate guards** (great "edge cases?" material): a pending interrupt *wedges* the
thread — LangGraph keeps it pending even if you feed a new message. So `/stream` calls
`_clear_pending_interrupt` (silently drains a `Command(resume="reject")`, discarding the
abandoned edit) before a new question, and `/resume` no-ops if nothing is suspended
(guards double-clicks). An edit left hanging across an app restart is safely auto-discarded
on the user's next message rather than blocking the session.

---

## I. LLM integration (`backend/services/llm.py`)

**Provider-agnostic factory.** `create_chat_model(cfg)` and `create_embeddings(cfg)` turn
a config into a LangChain model via `init_chat_model`. A one-line `_PROVIDER_MAP` maps
friendly names → LangChain provider ids (`gemini → google_genai`, `openai`, `anthropic`).
**Every** LLM call in the system — agent, tree selection, critique, triple extraction,
summaries, compaction — goes through this one factory. *Adding a provider is a one-line
change.* That's the architectural payoff.

**Default model:** Gemini `gemini-2.5-flash`; default embeddings `gemini-embedding-001`.
**Why Flash?** Cheap + fast + long context — and the architecture makes *many* small LLM
calls (tree select, critique ×N, entity extraction, summaries), so per-call cost dominates.
A fast cheap model is the right default; you can swap to a stronger model in Settings.

**Graceful degradation:** if no LLM is configured, embeddings/critique/tree-summaries are
skipped and the system falls back (truncated raw content, BM25-only). Anthropic has no
embeddings backend here, so dense auto-disables → BM25 + tree only. *Knowing these
fallbacks is how you answer "what happens if X isn't available?"*

**Generation knobs:** `temperature=0` is used for the deterministic sub-tasks (tree
selection, critique, judge, extraction, summaries) to maximize reproducibility; the main
answer uses provider defaults. `max_tokens` is bounded per sub-task (e.g. 200 for node
selection, 300 for critique).

---

## J. Conversion & indexing

**Conversion** (`converter.py`): hybrid strategy — XLSX→MarkItDown (avoids Docling's
merged-cell duplication), PDF→Docling full layout pipeline + optional VLM picture
description, DOCX/PPTX/HTML→Docling. If no LLM: VLM description and an LLM "refine" cleanup
pass are skipped. `_refine()` never raises — always falls back to raw markdown.

**Tree index** (`tree_index.py`, adapted from **PageIndex** — a *vectorless,
reasoning-based* RAG approach): parse markdown headings (respecting code blocks) → build a
nested tree by heading level → generate a 1–2 sentence **LLM summary per node** (concurrent,
semaphore-limited to 5 to avoid 429s; short nodes skip the LLM). Built async in a
background task on upload. **Fallback:** a document with no headings has no tree → retrieval
uses paragraph chunks instead. This is why the tree retriever sends *summaries, not body
text* to the LLM — it's a cheap "table of contents reasoning" step.

---

## K. Streaming & the SSE protocol

Chat streams over **SSE** using `astream_events(version="v2")`. The handler filters to
`on_chat_model_stream` events, drops the `tools` node and tool-call chunks, and emits only
**AI answer tokens**. Gemini's SDK sometimes returns `content` as a list of blocks, which
the stream flattens to text. After the answer, a **trailer of sentinels**:
`[EVIDENCE] {json}` (citation chips), `[CHAIN] {json}` (reasoning path), then **either**
`[INTERRUPT] {json}` (paused at the edit gate) **or** `[EDITED]` (doc changed) + checkpoint
compaction, always closing with `[DONE]`.

---

## L. Honest weaknesses (state these *before* the examiner finds them)

Owning limitations is how you win a defense. These are real, from the code:

1. **No ANN index — dense search is O(N) per query.** `dense_search` loads *all* of a
   doc's embedding rows and computes cosine in a Python/numpy loop. Fine for local,
   single-/few-document scale; **would not scale to millions of vectors** — you'd add
   FAISS/HNSW. Honest and easy to justify: "the workload is one user's personal corpus,
   not a web index."
2. **BM25 rebuilt every query.** `BM25Okapi` is constructed from scratch each call (no
   persisted lexical index). O(corpus) per query — negligible for small docs, wasteful at
   scale.
3. **Latency from many LLM calls.** Tree select + critique×N + entity extraction + the
   answer. The agentic loop can be ~3× a single shot. Mitigated by caching, cheap Flash
   model, and the `sufficient=true` early-exit.
4. **GraphRAG quality ceiling = LLM extraction quality.** Wrong/missing triples → missing
   edges. Mitigated by caching + treating graph as a complement.
5. **Compactor uses private LangGraph internals** — version-fragile by design (best-effort,
   never raises).
6. **No auth / no rate limiting / no multi-tenant isolation.** Correct for a local
   single-user desktop app; a SaaS deployment would need all three.
7. **Tiny-corpus BM25 degeneracy** (negative IDF when a term is in most units) — handled by
   a documented fallback, but worth naming.
8. **Hallucination is reduced, not eliminated.** The SOUL prompt + grounding + out-of-scope
   refusal make it *strongly* grounded, but an LLM can still misread retrieved context. We
   *measure* this (faithfulness/NLI harness in EVALUATION.md) rather than claim it's solved.

---

## M. Critical concepts to memorize (the cram sheet)

- **ReAct agent** = Reason+Act loop: model thinks → calls a tool → reads result → repeats →
  answers. LangGraph `create_react_agent`.
- **RAG** = Retrieval-Augmented Generation: fetch relevant context, put it in the prompt,
  generate grounded on it. Avoids fine-tuning and reduces hallucination.
- **RRF** `1/(k+rank)`, `k=60` — fuses ranked lists without score normalization.
- **Four signals:** tree (LLM reasoning), BM25 (lexical/sparse), dense (embeddings),
  graph (entity walk).
- **Bi-encoder vs cross-encoder:** dense embeddings encode query & doc *separately*
  (fast, ANN-able); the reranker cross-encoder scores them *together* (accurate, expensive,
  shortlist-only).
- **Dense vs sparse:** dense = embeddings (semantics/paraphrase); sparse = BM25 (exact
  terms). Hybrid gets both.
- **Embedding** = a vector capturing meaning; **cosine similarity** = angle between
  vectors; close angle ≈ similar meaning. Default 768-dim-class Gemini embeddings.
- **Agentic retrieval** = self-critique loop that issues follow-up sub-queries for
  multi-hop questions (`MAX_RETRIEVAL_ROUNDS=3`).
- **Checkpointer** = durable per-session memory (survives restart). **pre_model_hook** =
  trims the *view* (10 turns). **Compactor** = summarizes the *stored* history past 4000
  tokens.
- **`interrupt()`** = pauses the graph for human approval of an edit; resume with
  `Command(resume=...)`.
- **ContextVar** = per-async-request tool isolation → concurrency safety.
- **Magic numbers:** `MAX_CONTEXT_CHARS=12000`, `_RRF_K=60`, `_FUSED_TOP_K=8`,
  `_PER_RETRIEVER_TOP_K=10`, chunk≈1000 chars, `MAX_HOPS=2`, `MAX_RECENT_TURNS=10`,
  compact threshold 4000 tokens, tail 2 pairs.

---

## N. The request lifecycle (recite this end-to-end)

1. User selects doc(s) and types a question → `POST /api/chat/stream` (`chat.py`).
2. Guard: LLM configured? docs selected? Resolve `session_id`, doc titles.
3. `set_tool_context(doc_ids, settings, titles)` → ContextVar.
4. Build the singleton agent; build run config with `thread_id = session-{id}`.
5. If a stale edit-interrupt is pending → silently `reject`-drain it.
6. Send **only the new question**; checkpointer supplies prior turns.
7. Agent's LLM node decides to call `retrieve_context` (SOUL forces retrieval-first).
8. `retrieve_context` → `agentic_retrieve_context_multi_with_evidence`: build units →
   per query, `hybrid_rank_multi` (tree + BM25 + dense [+ graph]) → RRF → (rerank) →
   assemble context; critic loop may issue follow-ups (≤3 rounds).
9. Tool returns labelled context + records evidence units.
10. LLM node generates the grounded answer, streamed token-by-token over SSE.
11. (If the user asked to edit → `apply_edit` → `interrupt()` → `[INTERRUPT]` → resume.)
12. Trailer: `[EVIDENCE]`, `[CHAIN]`, then `[EDITED]` or done; **checkpoint compaction**
    runs in `finally`; `[DONE]`.
13. Display history saved to `chat_messages` (separate from the checkpointer); evidence +
    chain saved per message for reload.

---

## O. Comparisons (the "why X not Y?" defense)

| Question | Answer |
|----------|--------|
| **RAG vs fine-tuning?** | RAG injects *current, swappable* knowledge at query time, gives citations, needs no training, and updates instantly when a doc changes. Fine-tuning bakes knowledge into weights — expensive, static, no provenance, prone to confident hallucination. For a personal document corpus that changes, RAG is correct. |
| **RAG vs traditional search?** | Search returns *links/passages*; the user synthesizes. RAG *synthesizes an answer* over retrieved passages and cites them. We give both: an answer **and** clickable citations. |
| **Dense vs sparse (BM25)?** | Dense captures meaning/paraphrase but can miss exact tokens; sparse nails exact names/codes but misses synonyms. We fuse both via RRF — strictly more robust than either alone. |
| **RRF vs weighted scores?** | Scores from BM25, cosine, and an LLM pick are not comparable; weighting needs tuning and normalization. RRF uses only rank → no tuning, robust. |
| **Why a knowledge graph on top of vectors?** | Vectors rank single passages; they fail on multi-hop, split-evidence questions. The graph walks entity relations to pull *connected* passages even with zero lexical overlap. |
| **Why LangGraph not a raw while-loop?** | We need durable, resumable state (memory across restarts) and **`interrupt()`** for human-approval gates — LangGraph gives checkpointed, pausable graph execution for free. |
| **Why provider-agnostic, not just Gemini?** | Privacy (local Ollama path), cost/quality flexibility, vendor independence. One factory, one-line to add a provider. |
| **Why a tree index (PageIndex) at all?** | It's *reasoning-based, vectorless* retrieval: the LLM reasons over a table-of-contents instead of embedding everything. Strong on long, well-structured documents; complements vectors in the fusion. |

---

## P. How we evaluate (so it's "measured, not vibes")

`backend/services/evaluation.py` + `scripts/ai_eval/` (full guide in `EVALUATION.md`):
- **Deterministic, offline:** precision@k, recall@k, hit@k, MRR, nDCG against a gold
  dataset of `(question, gold_unit_ids)`. **recall@k is the headline** — if the evidence
  isn't retrieved, no model can answer correctly.
- **LLM-judged (RAGAS-style):** `faithfulness` decomposes the answer into atomic claims and
  NLI-checks each against the context (`score = supported/total`); plus answer/context
  relevance. Safe default: with no judge configured it returns neutral `1.0` (never
  fabricates false hallucination reports).
- **Adversarial tests:** out-of-scope (must refuse), distractor numbers, contradictions,
  partial support (≈0.5 faithfulness).
- **A/B in-app:** `POST /api/chat/compare` runs the same question with GraphRAG off vs on
  and reports the **bridge units** only the graph walk recovered — live proof the graph
  adds value.

---

## Q. Hardest questions — model answers

Each: a **short** answer (one breath) and an **expert** answer (the depth that scores).

**Q1. How do you prevent hallucination?**
*Short:* Strict grounding — the agent answers only from retrieved context, cites the
source, and says "not in the document" otherwise.
*Expert:* Three layers. (1) The SOUL system prompt makes retrieval-first mandatory and
forbids using prior answers as evidence. (2) Retrieval returns labelled context and the
out-of-scope path returns empty so the agent has nothing to fabricate from. (3) We *measure*
residual hallucination with an NLI faithfulness harness — atomic-claim decomposition,
each claim checked against context. It's reduced and measured, not claimed solved.

**Q2. Walk me through retrieval for one question.**
*Expert:* Build the unit corpus (tree nodes or chunks + figure/table units). Run four
retrievers → four ranked id lists: tree (LLM picks nodes from the TOC), BM25 (lexical),
dense (cached-embedding cosine), graph (entity walk). Fuse with RRF `1/(60+rank)`. If a
reranker is on, cross-encode the shortlist. Assemble context to 12k chars. A critic LLM
checks sufficiency; if not, it emits follow-up sub-queries and we loop up to 3 rounds,
accumulating evidence by best rank.

**Q3. Why RRF specifically?**
*Expert:* The four signals output incomparable scores. RRF depends only on rank, so it
needs no score normalization and no tuned weights; `k=60` damps low-rank noise; agreement
across retrievers is rewarded multiplicatively in the sum. It's the standard, robust
fusion baseline and removes a whole class of tuning bugs.

**Q4. How does GraphRAG actually find an answer pure RAG misses?**
*Expert:* For "X founded by Y who created Z", Z may share no words with the question and
live in a different section. We extract `(subject,relation,object)` triples per unit,
build an entity graph, find the question's entities, BFS 2 hops, and return the *source
units of every reached entity* — including Z's. Those ids join the RRF fusion. The
`/compare` endpoint shows the exact "bridge units" only the graph recovered.

**Q5. Your dense search has no FAISS — isn't that wrong?**
*Expert:* It's a conscious scope decision. The workload is one user's personal corpus —
tens to low-thousands of units — so a brute-force numpy cosine is sub-millisecond and
avoids an index-maintenance dependency. The clean extension point is FAISS/HNSW as a
drop-in for the cosine loop in `dense_search`. I'd add it the moment corpus size crosses
~10⁴–10⁵ vectors. I know exactly where it goes; it's not needed yet.

**Q6. How is conversation memory durable across restarts?**
*Expert:* A LangGraph `AsyncSqliteSaver` checkpointer persists the full message state to
`checkpoints.db`, keyed by `thread_id=session-{id}`. Each turn we send only the new
question; the checkpointer replays history. A `pre_model_hook` trims the *model's view* to
10 turns without touching storage, and a post-turn compactor LLM-summarizes stored history
past 4000 tokens so the checkpoint can't grow unboundedly.

**Q7. Editing through a chatbot is dangerous — how is it safe?**
*Expert:* `apply_edit` never writes on call. It invokes LangGraph `interrupt()`, pausing
the graph with state held in the checkpointer, and surfaces the exact diff to the UI. Only
an explicit `Command(resume="approve")` writes — exactly once, idempotent locate logic
re-runs harmlessly. Reject leaves the file untouched. Stale gates are auto-drained so a
session can't wedge.

**Q8. What happens under concurrent requests?**
*Expert:* Tool context lives in a `contextvars.ContextVar`, so each async request gets an
isolated copy of (doc scope, settings) — no cross-talk. The agent is a stateless singleton;
per-conversation state is isolated by `thread_id` in the checkpointer. So two users/tabs
asking different questions don't collide.

**Q9. Why Gemini Flash and not a bigger model?**
*Expert:* The architecture makes many small LLM calls per question (tree select, critique
×N, entity extraction, node summaries, the answer). Per-call latency/cost dominates, so a
fast cheap long-context model is the right *default*. It's a one-line settings change to a
stronger model, because every call goes through one provider-agnostic factory.

**Q10. What's actually novel vs standard engineering here?**
*Expert:* Standard: FastAPI/React/Tauri plumbing, BM25, embeddings, RRF. Modern-but-known:
the ReAct agent, agentic self-critique retrieval, NLI faithfulness eval. The integrative
contributions: (a) a **four-signal** retriever (reasoning-tree + lexical + dense + graph)
in one RRF fusion over a shared unit corpus; (b) **incremental per-unit caches** for both
embeddings *and* KG triples that make live GraphRAG affordable; (c) a fully
**provider-agnostic** local-first stack; (d) a **`interrupt()`-gated** human-in-the-loop
document editor with stale-gate recovery. The novelty is in the *integration and the
engineering that makes the research techniques cheap enough to run live*, not a new algorithm.

**Q11. How do you chunk, and why does chunk size matter?**
*Expert:* When a doc has headings we don't fixed-size chunk at all — units are tree
*sections*, which preserves semantic boundaries. Only headingless docs get ~1000-char
paragraph chunks (split on blank lines, never mid-paragraph). Chunk size trades recall vs
precision: too big → diluted embeddings and wasted context budget; too small → split
context and broken references. Section-aligned units sidestep most of that.

**Q12. If the tree retriever picks the wrong nodes, are you stuck?**
*Expert:* No — tree is one of four fused signals. If it errs, BM25/dense/graph still vote.
There's a deliberate exception: if the tree is the *only* available signal and it
explicitly returns "nothing relevant," we treat the question as out-of-scope and refuse —
preserving the anti-hallucination contract rather than dumping the raw document.

**Q13. What's your context window strategy / token budget?**
*Expert:* Retrieval context is capped at 12k chars (`MAX_CONTEXT_CHARS`); the model's
conversation view is capped at 10 turns; stored history is summarized past 4k tokens.
Three independent bounds on three different token sources (retrieved context, recent
dialogue, long-term history) keep total prompt size predictable.

**Q14. Multilingual? It's a Vietnamese project.**
*Expert:* The SOUL prompt instructs language-matching (Vietnamese in → Vietnamese out,
including exported files). Retrieval is language-agnostic: BM25 tokenizes Unicode word
chars; embeddings (Gemini) are multilingual; the tree/critique/extraction prompts work in
any language the model supports. Caption regex for figures/tables includes Vietnamese
keywords (`hình`, `bảng`, `biểu đồ`).

**Q15. What's the single biggest risk in production?**
*Expert:* Latency tail from the multi-LLM-call agentic path on a slow/rate-limited
endpoint, and GraphRAG extraction quality on messy documents. Both are bounded (3-round
cap, `sufficient=true` fail-open, cached triples) and measurable via the eval harness; the
fix levers are model choice, rerank, and disabling the agentic loop for simple corpora.

---

## R. Rapid-defense — "If the teacher asks X, say Y"

| If asked… | Say… |
|-----------|------|
| "What is this?" | The 60-second pitch (§A). |
| "Is it just ChatGPT?" | "No — it's grounded RAG over *your* documents with citations and out-of-scope refusal; it can't answer from outside the corpus." |
| "How does it avoid making things up?" | Q1. Lead with "retrieval-first SOUL contract + measured faithfulness." |
| "Explain retrieval." | Q2 + RRF (Q3). Draw the four arrows into one RRF box. |
| "Why four retrievers?" | Each covers a different failure mode: lexical (exact), dense (paraphrase), tree (structure), graph (multi-hop). Fuse, don't pick. |
| "Where's the vector DB / FAISS?" | Q5 — own it: brute-force numpy is correct at this scale; FAISS is a known drop-in. |
| "How's memory durable?" | Q6 — checkpointer + trim hook + compactor, three layers. |
| "Editing via chat is risky." | Q7 — `interrupt()` human-approval gate, writes only on approve. |
| "Concurrency?" | Q8 — ContextVar isolation + per-thread checkpointer. |
| "Why this model?" | Q9 — many small calls → fast cheap default; one-line swap. |
| "What's novel?" | Q10 — integration + caches that make live GraphRAG cheap; honest about no new algorithm. |
| "Biggest weakness?" | §L item 1 (O(N) dense) or item 8 (hallucination reduced not solved) — naming it first earns trust. |
| "Did you evaluate it?" | §P — recall@k + NLI faithfulness + the `/compare` A/B; "measured, not vibes." |
| You don't know an answer | "That's outside what I implemented in the AI layer; the design hook for it is in `<module>`, but I haven't built it." Never bluff — the eval doc's §14 'honest boundaries' models this. |

---

## S. One-line module map (for "where is X?")

| Concern | File |
|---------|------|
| Agent, tools, SOUL prompt, memory hooks | `backend/services/agent.py` |
| Hybrid retrieval, RRF, BM25, dense, agentic loop | `backend/services/retrieval.py` |
| KG triples, graph walk, graph-of-thought | `backend/services/knowledge_graph.py` |
| Checkpoint compaction | `backend/services/compactor.py` |
| Provider-agnostic LLM/embeddings factory | `backend/services/llm.py` |
| PageIndex tree index | `backend/services/tree_index.py` |
| Conversion (Docling/MarkItDown) | `backend/services/converter.py` |
| Chat API, SSE, resume/compare endpoints | `backend/api/chat.py` |
| Metrics + faithfulness harness | `backend/services/evaluation.py` |
| Benchmark runners | `scripts/ai_eval/` |
| Frontend chat + citations + compare UI | `src/components/ChatPanel.tsx`, `CitationChips.tsx`, `ReasoningChain.tsx`, `CompareDrawer.tsx` |

---

*Good luck. Lead with grounding, defend with RRF + the four signals, and win the room by
naming your own weaknesses before they do.*
