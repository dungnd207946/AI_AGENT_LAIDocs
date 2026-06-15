# Defense Diagrams

Projection-friendly SVG diagrams for the thesis defense. Each is self-contained
(no external fonts/assets) — open in a browser, drop into slides, or export to PNG.

| # | File | What it shows | Use it to answer |
|---|------|---------------|------------------|
| 1 | [01-architecture.svg](01-architecture.svg) | Full stack: Tauri → React → FastAPI sidecar → services → storage | "Walk me through the architecture" |
| 2 | [02-request-lifecycle.svg](02-request-lifecycle.svg) | End-to-end flow of one chat message (SSE) | "What happens when a user sends a message?" |
| 3 | [03-hybrid-retrieval-rrf.svg](03-hybrid-retrieval-rrf.svg) | Four retrievers → RRF fusion → context | "Explain retrieval" / "Why RRF?" — **the heart** |
| 4 | [04-agentic-loop.svg](04-agentic-loop.svg) | Self-critique multi-hop retrieval loop | "How do you handle multi-hop questions?" |
| 5 | [05-memory-layers.svg](05-memory-layers.svg) | Checkpointer + trim hook + compactor | "How does memory work / not explode?" |
| 6 | [06-edit-gate-interrupt.svg](06-edit-gate-interrupt.svg) | `interrupt()` human-approval edit gate | "Editing via chat is risky — how is it safe?" |
| 7 | [07-graphrag.svg](07-graphrag.svg) | Entity-graph walk recovering split evidence | "What does GraphRAG add over vectors?" |

## Benchmark tables (illustrative — see caveat)

Rendered as SVG for slides. **These carry expected/illustrative values, not measured
results** — every file says so on its face. Regenerate real numbers with
[`scripts/ai_eval/`](../../../scripts/ai_eval/) before citing them as measured.

| File | What it shows |
|------|---------------|
| [bench-00-gold-set.svg](bench-00-gold-set.svg) | The 30-question gold set composition (read this first — the numbers are meaningless without it) |
| [bench-A-retrieval-variants.svg](bench-A-retrieval-variants.svg) | Per-variant deterministic metrics (P@5, R@5, MRR, nDCG@5, latency) |
| [bench-B-multihop-strategy.svg](bench-B-multihop-strategy.svg) | Single-shot vs agentic vs +graph on the multi-hop subset |
| [bench-C-grounding.svg](bench-C-grounding.svg) | LLM-judged faithfulness / answer-rel / context-rel ablation |
| [bench-D-antihallucination.svg](bench-D-antihallucination.svg) | Out-of-scope refusal — grounded vs ungrounded |
| [bench-E-latency-cost.svg](bench-E-latency-cost.svg) | Per-query wall-clock + LLM/embedding call counts |
| [bench-F-embedding-provider.svg](bench-F-embedding-provider.svg) | Gemini vs OpenAI vs local embeddings |

Reproduce:
```bash
.venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json --live \
    --variants bm25,dense,tree,hybrid,graph -k 5 --out runs/defense_retrieval.json
.venv-ai/Scripts/python.exe scripts/ai_eval/run_grounding_eval.py \
    --dataset scripts/ai_eval/datasets/sample_eval.json --out runs/defense_grounding.json
```

## Suggested slide order

For a **10-minute** talk: 1 → 2 → 3 → 5. For a **20-minute defense**: all seven, in
order. Diagram **3 is the one to spend the most time on** — it's where the technical
substance lives.

## Export to PNG (for slides that don't accept SVG)

```bash
# with rsvg-convert (librsvg)
rsvg-convert -w 2400 docs/ai/diagrams/03-hybrid-retrieval-rrf.svg -o 03.png

# or with Inkscape
inkscape docs/ai/diagrams/03-hybrid-retrieval-rrf.svg --export-type=png -w 2400
```

Or simply open the `.svg` in a browser and screenshot at high zoom.
