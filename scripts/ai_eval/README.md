# AI evaluation harness

Reproducible benchmark scripts for the LAIDocs AI layer. They call the real
modules — `backend/services/evaluation.py`, `retrieval.py`,
`knowledge_graph.py` — so the numbers in [`docs/ai/EVALUATION.md`](../../docs/ai/EVALUATION.md)
can be reproduced from a clean checkout.

## Contents

| File | Purpose |
|------|---------|
| `run_retrieval_benchmark.py` | Score retriever variants (bm25 / dense / tree / hybrid / graph) — precision@k, recall@k, hit@k, MRR, nDCG, latency |
| `run_grounding_eval.py` | RAGAS-style answer quality — faithfulness (hallucination), answer/context relevance |
| `datasets/sample_eval.json` | Worked example dataset (replace `doc_id`s for live mode) |

## Quick start (offline, no LLM, no DB)

```bash
# from the repo root, using the AI-only env
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json -k 5

.venv-ai/Scripts/python.exe scripts/ai_eval/run_grounding_eval.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json --dry-run
```

Offline mode scores the `ranked_units` already recorded in the dataset, so it
needs no model and no ingested documents — ideal for CI and regression gating.

## Live mode (runs the real retrievers)

Requires a configured LLM (see [`docs/ai/SETUP.md`](../../docs/ai/SETUP.md)) and
documents already ingested into `~/.laidocs`. Put a real `doc_id` on each case
and drop the `ranked_units` (the script computes them):

```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset my_live_set.json --live --variants bm25,dense,tree,hybrid,graph \
    -k 5 --out runs/2026-06-07.json
```

`graph` requires triple extraction (an LLM); it returns nothing if no LLM is
configured, which the harness scores as 0 rather than crashing.

## Dataset schema

```jsonc
[
  {
    "question": "What is the refund window?",
    "gold_units": ["0007", "tbl0002"],     // ids that SHOULD be retrieved
    "doc_id": "abc123",                      // required for --live
    "ranked_units": {                        // required for OFFLINE retrieval
      "bm25":   ["0007", "0003", ...],
      "hybrid": ["0007", "tbl0002", ...]
    },
    "context": "[Section: Refunds] ...",     // required for grounding metrics
    "answer":  "You can request a refund within 30 days."
  }
]
```

`unit_id`s are the ids `retrieval.get_retrieval_units` assigns: tree node ids
(`"0007"`), chunk ids (`"c0001"`), figure ids (`"img0001"`), table ids
(`"tbl0001"`). Get them for a real document with:

```bash
.venv-ai/Scripts/python.exe -c "from backend.services import retrieval as r; \
import json; print(json.dumps([{k:u[k] for k in ('unit_id','kind','title')} \
for u in r.get_retrieval_units('YOUR_DOC_ID')], indent=2))"
```

See [`docs/ai/EVALUATION.md`](../../docs/ai/EVALUATION.md) for the full
methodology, comparison matrices, and how to interpret each metric.
