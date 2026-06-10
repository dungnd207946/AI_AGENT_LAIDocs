# AI Layer — Experimental Evaluation & Benchmarking Guide

A research-grade reproducibility package for the LAIDocs AI subsystem. It tells
you how to **measure** retrieval quality, answer grounding, agentic behaviour,
multimodal retrieval, and memory — scientifically, with runnable commands and
honest interpretation.

The guiding principle: **deterministic metrics on a fixed dataset before
LLM-judged metrics**. Deterministic metrics (precision/recall/MRR/nDCG) are
cheap, reproducible, and run offline; LLM-judged metrics (faithfulness,
relevance) are slower, non-deterministic, and reserved for the questions
deterministic metrics can't answer.

Everything here is backed by real code:
[`backend/services/evaluation.py`](../../backend/services/evaluation.py),
[`retrieval.py`](../../backend/services/retrieval.py),
[`knowledge_graph.py`](../../backend/services/knowledge_graph.py), driven by
[`scripts/ai_eval/`](../../scripts/ai_eval/).

---

## Table of contents

1. [Why benchmark this system](#1-why-benchmark-this-system)
2. [The evaluation harness](#2-the-evaluation-harness)
3. [Building a gold dataset](#3-building-a-gold-dataset)
4. [Retrieval benchmarking](#4-retrieval-benchmarking)
5. [Grounding & hallucination evaluation](#5-grounding--hallucination-evaluation)
6. [Agentic retrieval evaluation](#6-agentic-retrieval-evaluation)
7. [Long-term memory evaluation](#7-long-term-memory-evaluation)
8. [Multimodal evaluation](#8-multimodal-evaluation)
9. [Performance benchmarking](#9-performance-benchmarking)
10. [Reproducibility protocol](#10-reproducibility-protocol)
11. [Testing infrastructure](#11-testing-infrastructure)
12. [Comparison matrices](#12-comparison-matrices)
13. [Demo scenarios](#13-demo-scenarios)
14. [Not implemented — honest boundaries](#14-not-implemented--honest-boundaries)

---

## 1. Why benchmark this system

A document-grounded assistant has two failure modes that "it looks good in chat"
will never catch:

1. **Retrieval misses** — the right passage is never surfaced, so the answer is
   incomplete regardless of how good the model is.
2. **Hallucination** — the model asserts things the retrieved context does not
   support.

Section 4 measures (1) deterministically; Section 5 measures (2) with an
NLI-style judge. Every architectural change (a new fusion weight, a different
embedding model, an extra agentic round, enabling graph retrieval) should move a
number in one of these, or it isn't worth shipping.

---

## 2. The evaluation harness

`backend/services/evaluation.py` provides two metric families.

### Deterministic retrieval metrics (no LLM, fully offline)

| Function | Meaning | Good for |
|----------|---------|----------|
| `precision_at_k(ranked, gold, k)` | fraction of top-k that are gold | precision-sensitive UIs |
| `recall_at_k(ranked, gold, k)` | fraction of gold found in top-k | "did we surface the evidence?" |
| `hit_at_k(ranked, gold, k)` | 1 if any gold in top-k | coarse coverage |
| `reciprocal_rank(ranked, gold)` | 1/rank of first gold | how high the first hit ranks |
| `ndcg_at_k(ranked, gold, k)` | rank-discounted gain (binary) | overall ranking quality |
| `score_retrieval(ranked, gold, k)` | all of the above as a `RetrievalScore` | one call per query |

### LLM-judged metrics (RAGAS-style, injectable judge)

| Function | Meaning |
|----------|---------|
| `verify_grounding(answer, context)` → `GroundingResult` | decomposes the answer into atomic claims, NLI-checks each against context; `score` = supported fraction, plus the list of unsupported claims |
| `faithfulness(answer, context)` | the grounding score (RAGAS alias) |
| `answer_relevance(question, answer)` | does the answer address the question? (0–1) |
| `context_relevance(question, context)` | did retrieval surface relevant context? (0–1) |

The `judge` is a `Callable[[str], dict]`. In production it's built from your
`Settings` via `make_llm_judge` (provider-agnostic). In tests/CI you inject a
deterministic stub — the whole module is unit-testable offline.

**Critical safety property:** when no judge is configured, `verify_grounding`
returns a neutral `score=1.0` (it never reports false hallucinations), and
relevance returns `0.0`. So an unconfigured environment degrades safely rather
than emitting misleading numbers.

### One-call dataset evaluation

```python
from backend.services.evaluation import EvalCase, evaluate_dataset

cases = [EvalCase(question=..., gold_units=[...], ranked_units=[...],
                  context=..., answer=...)]
report = evaluate_dataset(cases, k=5, judged=True)   # judged=False → offline only
# → {"cases": [...per-case...], "aggregate": {metric: mean}, "k": 5}
```

---

## 3. Building a gold dataset

A dataset is a JSON list; schema is in
[`scripts/ai_eval/README.md`](../../scripts/ai_eval/README.md). The key field is
`gold_units` — the `unit_id`s that *should* be retrieved for each question.

**Step 1 — list a document's units** to know the id space:

```bash
.venv-ai/Scripts/python.exe -c "from backend.services import retrieval as r; \
import json; print(json.dumps([{k:u[k] for k in ('unit_id','kind','title')} \
for u in r.get_retrieval_units('YOUR_DOC_ID')], indent=2))"
```

**Step 2 — write questions** with the gold unit ids (and, for grounding,
`context` + `answer`). Aim for a spread:
- single-hop factual (one gold unit)
- multi-hop (≥2 gold units in different sections) — stresses agentic + graph
- table/figure questions (gold id `tbl*`/`img*`) — stresses multimodal
- **out-of-scope** questions (empty `gold_units`) — must NOT be answered

**Step 3 — version it.** Commit the dataset under
`scripts/ai_eval/datasets/<name>.json`. A dataset is an experiment artifact;
treat it like code.

> **Dataset size guidance.** 20–30 well-chosen questions per document type give
> stable deterministic aggregates. LLM-judged metrics are noisier; average over
> ≥3 runs (see §10) or use a larger set.

---

## 4. Retrieval benchmarking

This is the primary, reproducible experiment: **which retriever variant ranks
the gold evidence best?**

### Variants the harness can compare

| Variant | Code path | Signal |
|---------|-----------|--------|
| `bm25` | `retrieval.bm25_search` | lexical term overlap |
| `dense` | `retrieval.dense_search` | embedding cosine similarity |
| `tree` | `retrieval.select_node_ids` | LLM reasoning over the heading tree |
| `hybrid` | `retrieval.hybrid_rank` | RRF fusion of the above |
| `graph` | `knowledge_graph.graph_augmented_units` | entity multi-hop walk |

### Offline (reproducible, no LLM/DB)

Scores the `ranked_units` already recorded per variant in the dataset:

```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json -k 5
```

Expected output (leaderboard sorted by nDCG):

```
Retrieval benchmark (k=5, n_variants=5)
variant  precision     recall        hit        mrr       ndcg
--------------------------------------------------------------
dense      0.2667     1.0000     1.0000     1.0000     1.0000
hybrid     0.2667     1.0000     1.0000     1.0000     1.0000
graph      0.6667     1.0000     1.0000     1.0000     1.0000
bm25       0.2667     1.0000     1.0000     0.8333     0.8502
tree       0.5000     0.5000     0.6667     0.6667     0.5377
```

### Live (runs the real retrievers against your vault)

```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset my_live_set.json --live \
    --variants bm25,dense,tree,hybrid,graph -k 5 \
    --out runs/2026-06-07_retrieval.json
```

Live mode also reports `latency_s_mean` per variant.

### Interpreting results

- **recall@k is the headline for a grounded assistant** — if the evidence isn't
  retrieved, the answer can't be right. Watch this first.
- **MRR/nDCG** tell you whether the right unit is near the *top* (matters because
  context is truncated at `MAX_CONTEXT_CHARS=12000` — low-ranked units may be cut).
- **precision@k is naturally low** when a question has few gold units and `k=5`;
  don't over-index on it for single-gold questions.
- **Where each method wins** (typical, corpus-dependent):
  - `bm25` — exact terms, codes, names, rare tokens; cheap; degenerate on tiny
    corpora (negative IDF — see the fallback in `bm25_search`).
  - `dense` — paraphrases and synonyms; needs an embedding backend.
  - `tree` — long structured docs where section semantics matter; LLM cost.
  - `hybrid` — robust default; rarely worse than its best component because RRF
    rewards units multiple retrievers agree on.
  - `graph` — multi-hop questions where the answer spans entities/sections.

---

## 5. Grounding & hallucination evaluation

Measures whether answers are **supported by the retrieved context** — the
faithfulness/hallucination axis.

### Wiring check (offline, neutral judge)

```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_grounding_eval.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json --dry-run
# faithfulness=1.0 (neutral), relevance=0.0 — confirms wiring, no model calls
```

### Real evaluation (LLM judge)

```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_grounding_eval.py \
    --dataset my_set.json --out runs/2026-06-07_grounding.json
```

Output reports aggregate `faithfulness`, `answer_relevance`,
`context_relevance`, and **lists the unsupported claims per question** — the
concrete hallucinations to investigate.

### How faithfulness works (and why it's trustworthy)

1. The judge decomposes the answer into atomic, self-contained claims
   (`_CLAIM_PROMPT`).
2. Each claim is NLI-checked against the context with a **strict** prompt —
   "supported" only if verifiable from context alone (`_NLI_PROMPT`).
3. `score = supported / total`. An answer with no checkable claims scores 1.0
   (nothing to contradict).

### Adversarial / stress tests to include

| Test | Construction | Pass condition |
|------|--------------|----------------|
| Out-of-scope | question with `gold_units: []`, context unrelated | answer says "not in the document"; retrieval returns empty (see `retrieve_context` out-of-scope path) |
| Distractor context | context contains a plausible-but-wrong number | faithfulness flags the wrong claim |
| Contradiction | context states X, answer asserts not-X | claim marked unsupported |
| Partial support | answer mixes 1 supported + 1 unsupported claim | faithfulness ≈ 0.5, unsupported claim listed |
| Empty context | context = "" | grounding cannot support claims → low score (judge) |

A regression target: **faithfulness should not drop** when you change retrieval,
and ideally rises as recall rises (better evidence → fewer unsupported claims).

---

## 6. Agentic retrieval evaluation

Compares **single-shot** (`retrieve_context`) vs **iterative multi-hop**
(`agentic_retrieve_context`) — the experiment that justifies the agentic loop.

### Protocol

For each question in a **multi-hop** dataset (gold units in ≥2 sections):

```python
from backend.core.config import get_settings
from backend.services import retrieval as r
from backend.services import evaluation as ev

settings = get_settings()
# Single-shot: derive ranked units from one hybrid_rank
single_ranked, _ = r.hybrid_rank(doc_id, question, settings)
# Agentic: capture the units the loop accumulates (instrument accumulated dict,
# or compare the final CONTEXT each produces for grounding)
single_ctx  = r.retrieve_context(doc_id, question, settings)
agentic_ctx = r.agentic_retrieve_context(doc_id, question, settings)

# Retrieval recall on the ranked ids:
print(ev.recall_at_k(single_ranked, gold, k=8))
# Answer-side: generate an answer from each context, then faithfulness/relevance
```

Measure, single-shot vs agentic:

| Metric | Expectation when agentic helps |
|--------|--------------------------------|
| recall@k (multi-hop questions) | ↑ — follow-ups fetch the second-hop evidence |
| context_relevance | ≈ or ↑ |
| faithfulness | ↑ — more complete evidence → fewer unsupported claims |
| latency / LLM calls | ↑ (cost of the win — quantify it) |
| recall on single-hop questions | ≈ (agentic shouldn't hurt simple cases) |

### When agentic retrieval helps vs. hurts

- **Helps:** "compare A and B", "X did Y, who also did Z", questions whose answer
  is split across sections — the critique loop names the missing piece and
  chases it.
- **Neutral/wasteful:** simple single-fact lookups — the critique returns
  `sufficient=true` after round 1 (good), but you still paid one critique call.
- **Failure mode:** a flaky critique call defaults to `sufficient=true` (the loop
  never hangs) — so the worst case degrades to single-shot, not to an infinite
  loop. Verify this by forcing a judge error and confirming termination.

---

## 7. Long-term memory evaluation

Compares the assistant **with** vs **without** durable preference memory
(`SqliteStore` + `/memories/preferences.md`). This is a conversational
experiment, not a single-query one.

### Setup

- **With memory:** normal config (durable store at
  `~/.laidocs/data/memory_store.db`).
- **Without memory:** start from an empty store / wipe `memory_store.db` and the
  seed `preferences.md` between runs.

### Reproducible test conversation

```
Turn 1 (user): "Answer me in Vietnamese and keep answers to 2 sentences."
Turn 2 (user): <a normal document question>
--- new session (reset checkpointer, keep store) ---
Turn 3 (user): <another document question>      ← no restated preference
```

| Behaviour | With memory | Without memory |
|-----------|-------------|----------------|
| Turn 3 language | Vietnamese (recalled) | default/English |
| Turn 3 length | ~2 sentences (recalled) | unconstrained |

### What to measure

| Dimension | How |
|-----------|-----|
| Personalization | does Turn 3 honour the Turn 1 preference without restatement? (binary per preference) |
| Contextual continuity | manual rubric 1–5 across a session |
| Hallucination | run §5 faithfulness on both arms — memory must **not** raise faithfulness loss (it should not inject unsupported facts) |
| Token cost | memory adds the recalled preferences to the prompt — measure prompt-token delta |
| Latency | store read is local SQLite — should be negligible; confirm |

**Key correctness check:** memory stores *preferences*, not *document facts*.
Verify it never causes a claim ungrounded in the current document — i.e.
faithfulness with memory ≈ faithfulness without.

---

## 8. Multimodal evaluation

Tests retrieval of **figure** (`img*`) and **table** (`tbl*`) units.

### Build a multimodal dataset

Questions whose answers live in a table cell or figure caption/description, with
`gold_units` set to the `tbl*`/`img*` ids (use the unit-listing command in §3).

```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset datasets/multimodal.json --live --variants bm25,dense,hybrid -k 5
```

### What to verify

| Aspect | Check |
|--------|-------|
| Table QA | a question about a cell retrieves the `tbl*` unit (cells kept intact in unit text) |
| Figure retrieval | a question about a figure retrieves the `img*` unit (caption + VLM description) |
| Noise control | content-free `Image N` figures are **skipped** (see `_extract_image_units`) — confirm they don't appear |
| BM25 caveat | on a tiny corpus BM25 has negative-IDF degeneracy; multimodal recall there relies on dense/RRF, not BM25 |

> **Honest boundary:** there is **no separate OCR-grounding or layout-aware
> retrieval** in the AI layer. OCR/figure descriptions are produced by Docling +
> the VLM pass at *ingest* time (out of AI-layer scope). The AI layer consumes
> the resulting Markdown. So "OCR grounding" here means: does retrieval surface
> the figure/table unit whose text came from that pipeline.

---

## 9. Performance benchmarking

What the AI layer can meaningfully measure (and what it can't).

### Retrieval latency (measurable, built in)

Live benchmark reports `latency_s_mean` per variant:

```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset my_live_set.json --live --variants bm25,dense,tree,hybrid -k 5
```

Expected ordering (local cost): `bm25` ≪ `dense` (1 embedding call + cached
cosine) < `tree` (1 LLM call) < `hybrid` (all) ≪ agentic (multiple rounds).

### Indexing latency (embedding build)

The dense index builds lazily on first query. Time it explicitly:

```python
import time
from backend.core.config import get_settings
from backend.services import retrieval as r
t=time.perf_counter(); r.ensure_embedding_index(doc_id, get_settings()); \
print("embed build s:", time.perf_counter()-t)
```

### Token cost

The harness does not bill tokens directly. To track cost, wrap
`llm.create_chat_model` with a LangChain callback/`get_openai_callback`-style
counter, or read your provider dashboard for a benchmark run. Report
tokens-per-question for: tree selection, each critique round, the final answer.

### What is NOT benchmarkable here (and why)

- **GPU utilization** — no local GPU inference path; embeddings/completions are
  remote API calls.
- **Concurrent sessions** — concurrency *correctness* is handled
  (`ContextVar` isolation in `agent.py`), but there is no built-in load
  generator. To stress it, drive `set_tool_context` + agent calls from an
  `asyncio.gather` and measure wall-clock; treat results as indicative only.

---

## 10. Reproducibility protocol

A run is reproducible when someone else gets your numbers from your artifacts.

### Checklist

- [ ] **Dataset** committed under `scripts/ai_eval/datasets/<name>.json`
- [ ] **Config snapshot** — record `provider`, `model`, `embed_model` (copy the
      relevant fields of `~/.laidocs/config.json` / `.env` into the run report)
- [ ] **Code version** — `git rev-parse HEAD`
- [ ] **Env** — `.venv-ai` package versions (`uv pip freeze --python .venv-ai`)
- [ ] **k** and variant list recorded (the harness writes these into `--out`)
- [ ] **≥3 runs** for any LLM-judged metric; report mean ± spread (LLM judges are
      non-deterministic even at temperature 0)
- [ ] Deterministic metrics: 1 run is enough (they're exact)

### Determinism notes

- Deterministic retrieval metrics are **exact** given fixed `ranked_units`.
- `temperature=0` is used for tree selection, critique, judge, and extraction
  (see each module), but hosted models are not bit-reproducible — hence the ≥3-run
  rule for judged metrics.
- The harness's `--out` JSON already records `k`, `n_cases`, `live`, and the full
  per-variant report. Keep these files; they *are* your results.

### Suggested layout

```
runs/
└── 2026-06-07_gemini-flash_hybrid-vs-bm25/
    ├── report.json            # harness --out
    ├── config.json            # provider/model snapshot
    ├── env.txt                # uv pip freeze
    └── commit.txt             # git rev-parse HEAD
```

Naming convention: `YYYY-MM-DD_<model>_<experiment>`.

---

## 11. Testing infrastructure

| Layer | Where | Run |
|-------|-------|-----|
| Deterministic metric correctness | `tests/test_phase5_evaluation.py` (15) | `pytest tests/test_phase5_evaluation.py` |
| KG / graph-of-thought core | `tests/test_phase6_knowledge_graph.py` (18) | `pytest tests/test_phase6_knowledge_graph.py` |
| Multimodal unit parsing | `tests/test_phase4_multimodal.py` (8) | `pytest tests/test_phase4_multimodal.py` |
| Full AI suite (offline) | all of the above | `.venv-ai/Scripts/python.exe -m pytest tests/ -q` → 41 passing |

All AI tests are **hermetic**: they isolate `~/.laidocs` to a temp dir and inject
fake `judge`/`extractor` stubs, so they make **no network calls**. This makes
them suitable for CI gating on every PR.

### Recommended CI gate

```yaml
# pseudo-CI
- uv venv .venv-ai --python 3.11
- uv pip install --python .venv-ai langchain langchain-core langgraph \
    langgraph-checkpoint-sqlite deepagents rank_bm25 numpy pydantic \
    pydantic-settings pytest
- .venv-ai/bin/python -m pytest tests/ -q          # correctness gate
- .venv-ai/bin/python scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json -k 5   # smoke
```

For **regression testing of retrieval quality**, keep a frozen offline dataset
with recorded `ranked_units` and assert aggregate nDCG ≥ a baseline in CI — any
change that lowers it fails the build.

---

## 12. Comparison matrices

Qualitative summary to orient experiments. Validate the quantitative cells with
the harness on *your* corpus — these are directional, corpus-dependent defaults.

### Retrieval methods

| Method | Recall (paraphrase) | Recall (exact term) | Latency | LLM cost | Local-only? | Best for |
|--------|--------------------|--------------------|---------|----------|-------------|----------|
| BM25 | low–med | **high** | **very low** | none | ✅ | codes, names, rare tokens |
| Dense | **high** | med | low | embeddings | needs embed backend | paraphrase, synonyms |
| Tree | med | med | med | 1 chat call | ❌ | long structured docs |
| Hybrid (RRF) | **high** | **high** | med | embeddings + tree | partial | robust default |
| Graph | med | med | high | extraction calls | ❌ | multi-hop / entity questions |

### Retrieval strategy

| Strategy | Recall (multi-hop) | Faithfulness | Latency | Cost | Use when |
|----------|-------------------|--------------|---------|------|----------|
| Single-shot | med | med | low | 1× | simple lookups |
| Agentic multi-hop | **high** | **high** | high | up to ~3× | comparison / split-evidence questions |

### Memory

| Config | Personalization | Continuity | Token overhead | Durability |
|--------|-----------------|------------|----------------|------------|
| No memory | none | within-session only | none | — |
| Checkpointer only | none | within-session | low | lost on restart |
| + SqliteStore (`/memories/`) | **across sessions** | **across sessions** | small (recalled prefs) | survives restart |

### Provider / model

| Provider | Chat | Embeddings (dense) | Local option | Notes |
|----------|------|--------------------|--------------|-------|
| Gemini | ✅ | ✅ `text-embedding-004` | ❌ | recommended default |
| OpenAI-compatible | ✅ | ✅ `text-embedding-3-small` | ✅ (Ollama/LM Studio) | fully local path |
| Anthropic | ✅ | ❌ | ❌ | dense auto-disabled → BM25 + tree only |

---

## 13. Demo scenarios

Polished, repeatable walkthroughs. Each lists the prompt, what to show, and which
subsystem it exercises.

### A. Citation-grounded QA (the core promise)

1. Ingest a structured PDF (e.g. a paper or policy).
2. Ask a factual question answerable from one section.
3. **Show:** the answer cites the section title; run §5 grounding → faithfulness
   high, zero unsupported claims.
4. Ask an out-of-scope question → assistant says "not in the document" (no
   fabrication). **This is the headline reliability demo.**

### B. Multi-document / multi-hop reasoning

1. A document where the answer spans two sections ("founded by X, who also…").
2. Ask the multi-hop question.
3. **Show:** `agentic_retrieve_context` issues a follow-up sub-query (instrument
   the loop or compare recall vs single-shot from §6). Then enable `graph`
   variant in the benchmark and show its precision advantage on this question
   (the sample dataset already demonstrates graph's higher precision).

### C. Multimodal paper QA

1. A PDF with a results table and a figure.
2. Ask "what was the value in row X?" and "what does Figure 2 show?"
3. **Show:** retrieval surfaces the `tbl*`/`img*` unit; the answer reads the cell
   directly and cites "Table: …" / "Figure: …".

### D. Memory-aware conversation

1. State a preference ("answer in Vietnamese, briefly").
2. Start a new session, ask a fresh question.
3. **Show:** the preference is honoured without restatement (§7) — durable
   `SqliteStore` memory.

### E. Research-assistant benchmarking (for technical reviewers)

1. Run the offline retrieval benchmark live in front of the audience:
   ```bash
   .venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
       --dataset scripts/ai_eval/datasets/sample_eval.json -k 5
   ```
2. **Show:** the leaderboard — concrete evidence the system is *measured*, not
   vibes. Then the grounding dry-run to show the hallucination harness.

**Recommended demo datasets:** a research paper (structure + tables + figures),
a policy/handbook (clear sections, good for citation + out-of-scope), and a
financial report (tables → multimodal).

---

## 14. Not implemented — honest boundaries

So reviewers aren't misled, these appear in the RAG literature but are **not in
this codebase**. They are clean extension points, not current features:

| Feature | Status | Where it would slot in |
|---------|--------|------------------------|
| ColBERT / late-interaction retrieval | not implemented | a new ranked-list source feeding `rrf_fuse` |
| Cross-encoder reranking | not implemented | a rerank pass after `rrf_fuse`, before context build |
| Standalone OCR-grounding eval | not implemented | OCR is Docling/VLM at ingest (out of AI scope); AI layer consumes the Markdown |
| GPU-accelerated local inference | not implemented | embeddings/chat are provider API calls |
| Concurrent-session load benchmark | not implemented | correctness is handled (`ContextVar`); no load generator |
| Live wiring of eval + KG into the agent | not wired | `graph_augmented_units` → `rrf_fuse`; `graph_of_thought` → context scaffold (hooks ready) |

Everything *else* in this guide runs against real, tested code.

---

*See also:* [README.md](README.md) (architecture) · [SETUP.md](SETUP.md)
(install) · [`scripts/ai_eval/`](../../scripts/ai_eval/) (harness).
