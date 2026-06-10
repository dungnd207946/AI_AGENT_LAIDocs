# Phân chia công việc — LAIDocs (4 thành viên)

> Nguyên tắc: chia theo **ranh giới subsystem** để chạy song song, ít xung đột.
> Mỗi người = **(A) Review/củng cố code cũ** + **(B) Phát triển phần mới của một "agent cơ bản nên có"**.

## Bối cảnh hiện trạng
- Retrieval đã mạnh: Phase 1 (multi-provider LLM + durable memory), Phase 2 (hybrid tree+BM25+dense+RRF), Phase 3 (agentic multi-hop), Phase 4 (multimodal figure/table).
- **Lỗ hổng của "agent cơ bản":**
  1. Checkpointer hội thoại đang **in-memory** (`MemorySaver`) → mất khi restart.
  2. **Phase 5 (evaluation) & Phase 6 (knowledge graph) là thư viện rời, CHƯA nối vào request-path.**
  3. Thiếu **guardrails** (chống prompt-injection / lọc output), **observability** nối vào agent, và **test cho Phase 1–3**.

---

## Thành viên 1 — Retrieval & RAG Core
**Sở hữu:** `backend/services/retrieval.py`, `tree_index.py`

**(A) Review code cũ**
- Đọc kỹ Phase 2–4: RRF fusion, `get_retrieval_units`, BM25 fallback (negative-IDF), dense lazy index, parse figure/table.
- Kiểm chứng các knob: `_PER_RETRIEVER_TOP_K`, `_FUSED_TOP_K`, `_RRF_K`, chunk size.
- Bổ sung test cho Phase 2–3 (hiện chưa có file test riêng).

**(B) Phát triển mới**
- **Re-ranking stage** (cross-encoder / LLM rerank) sau RRF.
- **Wire Phase 5 (eval)** thành benchmark chạy được trên dataset cố định → đo regression mỗi lần chỉnh retrieval.
- Cải thiện chunking (semantic chunking), cache embedding index.

---

## Thành viên 2 — Agent, Memory & Orchestration
**Sở hữu:** `backend/services/agent.py`, `api/chat.py`

**(A) Review code cũ**
- 3 lớp memory: `MemorySaver` (working), `chat_messages` (display), `SqliteStore` (preferences).
- SOUL prompt, `contextvars` isolation, singleton + `reset_agent()`.

**(B) Phát triển mới (ưu tiên cao)**
- **Durable checkpointer**: thay `MemorySaver` → `SqliteSaver`/`AsyncSqliteSaver` để hội thoại sống sót restart (xem note trong agent.py về bug context-manager).
- **Guardrails**: chống prompt-injection, lọc output ngoài tài liệu.
- **HITL** (human-in-the-loop) cho hành động nhạy cảm (dùng `langchain-middleware`).
- **Wire Phase 6 (knowledge graph)** vào `retrieve_context` như một tín hiệu fuse thêm.

---

## Thành viên 3 — Ingestion & Data Pipeline
**Sở hữu:** `backend/services/converter.py`, `crawler.py`, `backup.py`, `core/database.py`

**(A) Review code cũ**
- Chiến lược convert hybrid (Docling/MarkItDown/Crawl4AI), VLM picture description, `_refine()`.
- Tree index build async, backup export/import (replace/merge).

**(B) Phát triển mới**
- **Ingest-time index**: build embedding + knowledge-graph lúc upload thay vì lazy mỗi query (hiệu năng).
- Cải thiện trích figure/table (caption đa ngôn ngữ, bảng lồng).
- Hỗ trợ thêm định dạng + xử lý lỗi convert mạnh hơn.

---

## Thành viên 4 — Frontend, UX, Observability & QA
**Sở hữu:** `src/` (React), `telemetry_server/`, `tests/`

**(A) Review code cũ**
- ChatPanel, Settings, DataTab, `sidecar.ts` (SSE, health polling, chat history API).
- Sidecar lifecycle (spawn/shutdown qua stdin).

**(B) Phát triển mới**
- **Citation UI**: hiển thị section/Figure/Table nguồn cho mỗi câu trả lời.
- **Eval dashboard**: trực quan hóa kết quả Phase 5.
- **Nối telemetry** vào agent (latency, token, retrieval hits).
- **E2E tests** + CI; lấp test cho Phase 1–3.

---

## Việc chung (cross-cutting — chia nhỏ, không ai sở hữu riêng)
| Hạng mục | Gợi ý chủ trì |
|---|---|
| Thống nhất schema citation backend↔frontend | TV2 + TV4 |
| Quyết định single-doc vs multi-doc chat | TV1 + TV2 |
| Eval dataset (ground-truth Q&A) | Cả team đóng góp |
| Tài liệu kiến trúc + demo cuối | TV3 + TV4 |

## Mốc đề xuất
1. **Tuần 1:** mỗi người review + viết test bù cho phần mình → báo cáo "code cũ chạy đúng".
2. **Tuần 2–3:** phát triển phần (B); ưu tiên durable checkpointer (TV2) + wire eval (TV1).
3. **Tuần 4:** tích hợp Phase 5/6 vào request-path, citation UI, E2E, demo.
