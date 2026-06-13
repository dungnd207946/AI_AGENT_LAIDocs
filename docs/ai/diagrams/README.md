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
