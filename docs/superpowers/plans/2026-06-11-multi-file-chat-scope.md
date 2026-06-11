# Multi-File Chat Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho phép tích chọn nhiều tài liệu làm *phạm vi* cho chat; agent retrieve/đọc/sửa trong phạm vi đó; session chat tách khỏi doc, chỉ định danh bằng `session_id` toàn cục.

**Architecture:** `doc_ids[]` là tham số **tạm thời theo request** (frontend giữ state, không lưu DB). Retrieval gom units của tất cả doc đã chọn thành **một pool** (unit_id được namespace `doc_id::unit_id`) rồi rank + RRF **một lần** trên toàn pool (cosine dense so sánh được giữa doc; BM25 IDF tính trên pool chung; tree chạy per-doc rồi nối). Edit tools nhận thêm `file` để LLM chỉ định file cần sửa theo intent, tool validate trong phạm vi. Session toàn cục: `chat_messages` key theo `session_id`, `thread_id = session-{id}`. **Không đổi** cấu trúc agent (vẫn `create_react_agent`), không đổi các hàm retrieve single-doc đang có — chỉ **thêm** biến thể `*_multi`.

**Tech Stack:** Python 3 / FastAPI / LangGraph (create_react_agent, AsyncSqliteSaver) / SQLite; React 19 + TypeScript + Vite; pytest.

**Quy ước test:** Mỗi test Python tự cô lập DB bằng `monkeypatch.setattr(database, "DB_PATH", tmp_path/"t.db")` rồi `database.init_db()`. `get_db()` đọc `DB_PATH` ở runtime nên patch có hiệu lực.

---

## File Structure

**Backend (modify):**
- `backend/core/database.py` — thêm guarded migration runner + migration `0003_global_sessions` (recreate `chat_messages`: `doc_id` nullable, `session_id` toàn cục).
- `backend/services/chat_history.py` — đổi mọi hàm từ key `doc_id` sang `session_id` toàn cục.
- `backend/services/compactor.py` — `compact_if_needed(session_id, ...)` thay cho `doc_id`.
- `backend/services/retrieval.py` — **thêm** `get_retrieval_units_multi`, `dense_search_multi`, `hybrid_rank_multi`, `agentic_retrieve_context_multi`, helper `_doc_title`; sửa `build_context_from_units` thêm nhãn `[File: ...]` (backward-compatible).
- `backend/services/agent.py` — `set_tool_context(doc_ids, settings, doc_titles)`; `retrieve_context` dùng `*_multi`; `preview_edit`/`apply_edit` thêm tham số `file` + `_resolve_scope_doc`; cập nhật SOUL prompt.
- `backend/api/chat.py` — `ChatRequest.doc_ids`; session endpoints toàn cục; nạp titles.

**Frontend (modify):**
- `src/lib/sidecar.ts` — `streamChat(docIds[], ...)`, history/session API toàn cục, `listDocuments()`.
- `src/components/ChatPanel.tsx` — scope picker (chips + add-file dropdown), nạp history toàn cục, gửi `docIds`.
- `src/pages/DocumentEditor.tsx` — truyền `initialDocId` thay `docId`.

**Tests (create):**
- `tests/test_chat_history_sessions.py`
- `tests/test_retrieval_multi.py`
- `tests/test_agent_scope.py`

---

## Phase 1 — DB: session toàn cục + migration runner

### Task 1: Guarded migration runner + migration `0003_global_sessions`

**Files:**
- Test: `tests/test_chat_history_sessions.py`
- Modify: `backend/core/database.py` (hàm `init_db`, khoảng dòng 84-96)

- [ ] **Step 1: Viết test thất bại** — `tests/test_chat_history_sessions.py`

```python
import sqlite3
import pytest

from backend.core import database


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = tmp_path / "laidocs.db"
    monkeypatch.setattr(database, "DB_PATH", p)
    database.init_db()
    return p


def _cols(db_path, table):
    with sqlite3.connect(str(db_path)) as c:
        return {r[1]: r for r in c.execute(f"PRAGMA table_info({table})")}


def test_chat_messages_doc_id_is_nullable_after_migration(db):
    info = _cols(db, "chat_messages")
    assert "doc_id" in info
    # PRAGMA table_info column index 3 = notnull flag; 0 means nullable
    assert info["doc_id"][3] == 0


def test_global_session_ids_do_not_collide_across_docs(db):
    # Two docs that historically each had session_id=1 must be re-numbered
    # to distinct global ids by the migration.
    with sqlite3.connect(str(db)) as c:
        c.execute(
            "INSERT INTO chat_messages (doc_id, session_id, role, content) "
            "VALUES ('docA', 1, 'user', 'qA'), ('docB', 1, 'user', 'qB')"
        )
        c.commit()
    # Re-run migrations (idempotent) — must not crash, ids stay valid.
    database.init_db()
    with sqlite3.connect(str(db)) as c:
        rows = c.execute(
            "SELECT doc_id, session_id FROM chat_messages ORDER BY doc_id"
        ).fetchall()
    assert {r[0] for r in rows} == {"docA", "docB"}
    # init_db is idempotent: running twice must not duplicate rows.
    assert len(rows) == 2
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_chat_history_sessions.py -v`
Expected: FAIL — `doc_id` hiện đang `NOT NULL` (notnull flag = 1).

- [ ] **Step 3: Thêm migration runner + migration**

Trong `backend/core/database.py`, **sau** danh sách `_MIGRATIONS` (sau dòng 81) thêm:

```python
# Guarded migrations: run exactly once, tracked in schema_migrations.
# Use this (not the _MIGRATIONS list) for destructive/idempotency-sensitive
# changes such as table rebuilds.
_GUARDED_MIGRATIONS: list[tuple[str, list[str]]] = [
    (
        "0003_global_sessions",
        [
            # Rebuild chat_messages: doc_id becomes nullable (provenance only,
            # no longer a key) and session_id is globalised so ids never
            # collide across documents.
            """CREATE TABLE chat_messages_v3 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                session_id INTEGER NOT NULL DEFAULT 1,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'summary')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            # Globalise: rank each distinct (doc_id, session_id) pair to a new
            # contiguous id, ordered by (doc_id, session_id). Correlated
            # subquery avoids window-function portability concerns.
            """INSERT INTO chat_messages_v3 (id, doc_id, session_id, role, content, created_at)
               SELECT cm.id, cm.doc_id,
                 (SELECT COUNT(*) FROM (SELECT DISTINCT doc_id, session_id FROM chat_messages) d
                  WHERE (d.doc_id < cm.doc_id)
                     OR (d.doc_id = cm.doc_id AND d.session_id <= cm.session_id)) AS new_sid,
                 cm.role, cm.content, cm.created_at
               FROM chat_messages cm""",
            "DROP TABLE chat_messages",
            "ALTER TABLE chat_messages_v3 RENAME TO chat_messages",
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)",
        ],
    ),
]
```

Sau đó sửa thân `init_db` (đoạn dòng 84-96). Thay khối hiện tại:

```python
def init_db() -> None:
    """Create DB file and all tables if they do not exist yet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        # Run migrations (ignore errors for already-applied ones)
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists or table already recreated
```

bằng:

```python
def init_db() -> None:
    """Create DB file and all tables if they do not exist yet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        # Run lightweight migrations (ignore errors for already-applied ones)
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists or table already recreated
        _run_guarded_migrations(conn)
        conn.commit()


def _run_guarded_migrations(conn: sqlite3.Connection) -> None:
    """Run each guarded migration exactly once, tracked in schema_migrations."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "name TEXT PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    applied = {
        row[0]
        for row in conn.execute("SELECT name FROM schema_migrations").fetchall()
    }
    for name, statements in _GUARDED_MIGRATIONS:
        if name in applied:
            continue
        for stmt in statements:
            conn.execute(stmt)
        conn.execute("INSERT INTO schema_migrations (name) VALUES (?)", (name,))
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_chat_history_sessions.py -v`
Expected: 2 passed (cùng các test sẽ thêm ở Phase 2).

- [ ] **Step 5: Commit**

```bash
git add backend/core/database.py tests/test_chat_history_sessions.py
git commit -m "feat(db): globalise chat sessions via guarded migration"
```

---

## Phase 2 — chat_history.py: key theo session_id toàn cục

### Task 2: Viết lại chat_history sang session-based

**Files:**
- Test: `tests/test_chat_history_sessions.py` (bổ sung)
- Modify: `backend/services/chat_history.py` (toàn bộ các hàm)

- [ ] **Step 1: Bổ sung test thất bại** vào `tests/test_chat_history_sessions.py`

```python
from backend.services import chat_history as ch


def test_session_lifecycle_is_global(db):
    assert ch.get_current_session_id() == 1  # empty DB → default 1
    ch.save_message(1, "user", "hello")
    ch.save_message(1, "assistant", "hi")
    assert ch.get_current_session_id() == 1
    new_id = ch.start_new_session()
    assert new_id == 2
    ch.save_message(new_id, "user", "second topic")

    # get_messages() returns ALL rows globally, ordered.
    allm = ch.get_messages()
    assert [m["content"] for m in allm] == ["hello", "hi", "second topic"]
    assert allm[0]["session_id"] == 1 and allm[2]["session_id"] == 2

    # per-session load
    s1 = ch.get_messages_for_session(1)
    assert [m["content"] for m in s1] == ["hello", "hi"]
    s2 = ch.get_messages_for_session(2)
    assert [m["content"] for m in s2] == ["second topic"]


def test_delete_session_global(db):
    ch.save_message(1, "user", "a")
    ch.save_message(2, "user", "b")
    ch.delete_session(2)
    assert [m["content"] for m in ch.get_messages()] == ["a"]


def test_summary_injected_per_session(db):
    ch.save_compact_summary(1, "rolling summary text")
    ch.save_message(1, "user", "follow up")
    msgs = ch.get_messages_for_session(1)
    # summary becomes a synthetic user/assistant pair, then real messages
    assert msgs[0]["role"] == "user"
    assert msgs[1] == {"role": "assistant", "content": "rolling summary text"}
    assert msgs[2] == {"role": "user", "content": "follow up"}
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_chat_history_sessions.py -v`
Expected: FAIL — `get_current_session_id()` hiện yêu cầu tham số `doc_id`.

- [ ] **Step 3: Thay toàn bộ nội dung `backend/services/chat_history.py`**

```python
"""Chat history service for display-layer message persistence.

Sessions are GLOBAL: a session is identified by ``session_id`` alone and is
not tied to any document. The set of documents in scope for a turn is a
transient per-request value (sent by the frontend), never persisted here.

Stores ALL messages across ALL sessions for UI display. Separate from the
agent's LangGraph checkpointer (keyed by thread_id "session-{session_id}").

Compact support:
  - Rows with role='summary' are synthetic summaries from the compactor.
  - get_messages()             — UI display (all rows incl. summaries)
  - get_messages_for_compact() — compactor input (latest summary + rows after)
  - save_compact_summary()     — insert a summary row
  - delete_compacted_messages() — remove rows folded into a summary
"""

from __future__ import annotations

from ..core.database import get_db


def get_current_session_id() -> int:
    """Get the current (latest) global session ID; 1 when there is none."""
    with get_db() as conn:
        row = conn.execute("SELECT MAX(session_id) FROM chat_messages").fetchone()
    return row[0] if row and row[0] else 1


def start_new_session() -> int:
    """Return the next global session ID."""
    return get_current_session_id() + 1


def save_message(session_id: int, role: str, content: str) -> None:
    """Save a single display message (doc_id left NULL — scope is transient)."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def get_messages() -> list[dict]:
    """Load ALL messages globally, ordered by creation time (incl. summaries)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, session_id, role, content, created_at
               FROM chat_messages ORDER BY created_at ASC, id ASC"""
        ).fetchall()
    return [
        {
            "id": r[0],
            "session_id": r[1],
            "role": r[2],
            "content": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


def get_messages_for_session(session_id: int) -> list[dict]:
    """Load messages for agent context injection on session resume.

    Returns 'user'/'assistant' rows for this session. If a compact summary
    exists for this session, prepend it as a synthetic user/assistant exchange
    (LangGraph does not understand role='summary').
    """
    with get_db() as conn:
        summary_row = conn.execute(
            """SELECT content FROM chat_messages
               WHERE session_id = ? AND role = 'summary'
               ORDER BY created_at DESC LIMIT 1""",
            (session_id,),
        ).fetchone()
        rows = conn.execute(
            """SELECT role, content FROM chat_messages
               WHERE session_id = ? AND role IN ('user', 'assistant')
               ORDER BY created_at ASC, id ASC""",
            (session_id,),
        ).fetchall()

    result: list[dict] = []
    if summary_row:
        result.append({"role": "user", "content": "Summarize our conversation so far."})
        result.append({"role": "assistant", "content": summary_row[0]})
    result += [{"role": r[0], "content": r[1]} for r in rows]
    return result


def get_messages_for_compact(session_id: int) -> list[dict]:
    """Load messages for the compactor: latest summary (if any) + rows after it."""
    with get_db() as conn:
        summary_row = conn.execute(
            """SELECT id, session_id, role, content, created_at FROM chat_messages
               WHERE session_id = ? AND role = 'summary'
               ORDER BY created_at DESC LIMIT 1""",
            (session_id,),
        ).fetchone()
        if summary_row:
            rows = conn.execute(
                """SELECT id, session_id, role, content, created_at FROM chat_messages
                   WHERE session_id = ? AND role IN ('user', 'assistant')
                     AND created_at > ?
                   ORDER BY created_at ASC, id ASC""",
                (session_id, summary_row[4]),
            ).fetchall()
            result = [
                {
                    "id": summary_row[0],
                    "session_id": summary_row[1],
                    "role": summary_row[2],
                    "content": summary_row[3],
                    "created_at": summary_row[4],
                }
            ]
        else:
            rows = conn.execute(
                """SELECT id, session_id, role, content, created_at FROM chat_messages
                   WHERE session_id = ? AND role IN ('user', 'assistant')
                   ORDER BY created_at ASC, id ASC""",
                (session_id,),
            ).fetchall()
            result = []

    result += [
        {
            "id": r[0],
            "session_id": r[1],
            "role": r[2],
            "content": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]
    return result


def save_compact_summary(session_id: int, summary: str) -> None:
    """Insert a compacted summary row (role='summary')."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, 'summary', ?)",
            (session_id, summary),
        )


def delete_compacted_messages(message_ids: list[int]) -> None:
    """Delete rows folded into a summary (ids are globally unique)."""
    if not message_ids:
        return
    placeholders = ",".join("?" * len(message_ids))
    with get_db() as conn:
        conn.execute(
            f"DELETE FROM chat_messages WHERE id IN ({placeholders})",
            list(message_ids),
        )


def delete_messages() -> None:
    """Delete ALL messages (every session)."""
    with get_db() as conn:
        conn.execute("DELETE FROM chat_messages")


def delete_session(session_id: int) -> None:
    """Delete all messages belonging to a single session."""
    with get_db() as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_chat_history_sessions.py -v`
Expected: tất cả PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/services/chat_history.py tests/test_chat_history_sessions.py
git commit -m "feat(chat): make chat history global per session_id"
```

---

## Phase 3 — compactor.py: theo session_id

### Task 3: Đổi `compact_if_needed` sang session-based

**Files:**
- Modify: `backend/services/compactor.py` (dòng 90-143)

- [ ] **Step 1: Sửa chữ ký + thân `compact_if_needed`**

Thay đoạn từ `async def compact_if_needed(` (dòng 90) đến hết file bằng:

```python
async def compact_if_needed(
    session_id: int,
    settings: "Settings",
    threshold: int = COMPACT_THRESHOLD_TOKENS,
) -> bool:
    """Check a session's display history; compact if over threshold.

    Returns True if compaction was performed. Reads/writes via chat_history.
    """
    from ..services.chat_history import (
        get_messages_for_compact,
        save_compact_summary,
        delete_compacted_messages,
    )

    messages = get_messages_for_compact(session_id)

    if len(messages) <= TAIL_MESSAGES:
        return False

    total_tokens = estimate_tokens(messages)
    if total_tokens <= threshold:
        return False

    body = messages[:-TAIL_MESSAGES]
    tail = messages[-TAIL_MESSAGES:]

    if estimate_tokens(tail) >= threshold:
        body = messages[:-2]
        tail = messages[-2:]

    logger.info(
        "Compacting %d messages (est. %d tokens) for session %s",
        len(body), total_tokens, session_id,
    )

    history_text = _format_for_compact(body)
    try:
        summary = await _call_llm_compact(history_text, settings)
    except Exception:
        logger.exception("Compaction LLM call failed; skipping compact")
        return False

    body_ids = [m["id"] for m in body]
    save_compact_summary(session_id, summary)
    delete_compacted_messages(body_ids)

    logger.info("Compaction done for session %s; summary saved", session_id)
    return True
```

- [ ] **Step 2: Xác minh import không gãy**

Run: `python -c "import backend.services.compactor"`
Expected: không lỗi.

- [ ] **Step 3: Commit**

```bash
git add backend/services/compactor.py
git commit -m "refactor(compactor): compact per global session_id"
```

---

## Phase 4 — retrieval.py: pool đa-doc, rank một lần

### Task 4: Nhãn nguồn file trong context (backward-compatible)

**Files:**
- Test: `tests/test_retrieval_multi.py`
- Modify: `backend/services/retrieval.py` (hàm `build_context_from_units`, dòng 189-211)

- [ ] **Step 1: Viết test thất bại** — `tests/test_retrieval_multi.py`

```python
from backend.services import retrieval as R


def test_build_context_adds_file_label_when_present():
    units = [
        {"unit_id": "docA::0001", "title": "Intro", "text": "hello",
         "kind": "text", "doc_title": "Report.pdf"},
    ]
    ctx = R.build_context_from_units(units)
    assert "[File: Report.pdf" in ctx
    assert "Intro" in ctx
    # namespaced id is displayed stripped of the doc prefix
    assert "docA::" not in ctx
    assert "0001" in ctx


def test_build_context_unchanged_without_doc_title():
    units = [{"unit_id": "0001", "title": "Intro", "text": "hello", "kind": "text"}]
    ctx = R.build_context_from_units(units)
    assert ctx.startswith("[Section: Intro (node 0001)]")
    assert "File:" not in ctx
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_retrieval_multi.py -v`
Expected: FAIL — nhãn `[File: ...]` chưa tồn tại.

- [ ] **Step 3: Sửa `build_context_from_units`** trong `backend/services/retrieval.py`

Thay thân hàm (dòng 195-211) bằng:

```python
    ctx = ""
    for u in units:
        kind = u.get("kind", "text")
        title = u.get("title") or "Section"
        uid = str(u.get("unit_id", "?")).split("::")[-1]  # strip doc namespace
        text = u.get("text", "")
        src = f"File: {u['doc_title']} | " if u.get("doc_title") else ""
        if kind == "image":
            header = f"[{src}Figure: {title} ({uid})]"
        elif kind == "table":
            header = f"[{src}Table: {title} ({uid})]"
        else:
            header = f"[{src}Section: {title} (node {uid})]"
        section = f"{header}\n{text}\n\n"
        if len(ctx) + len(section) > MAX_CONTEXT_CHARS:
            break
        ctx += section
    return ctx.strip()
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_retrieval_multi.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/services/retrieval.py tests/test_retrieval_multi.py
git commit -m "feat(retrieval): add optional file label to context units"
```

### Task 5: Pool units đa-doc + dense/hybrid đa-doc

**Files:**
- Test: `tests/test_retrieval_multi.py` (bổ sung)
- Modify: `backend/services/retrieval.py` (thêm hàm mới, cuối khu vực hybrid; trước `retrieve_context`)

- [ ] **Step 1: Bổ sung test thất bại** vào `tests/test_retrieval_multi.py`

```python
import sqlite3
import pytest
from backend.core import database


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = tmp_path / "laidocs.db"
    monkeypatch.setattr(database, "DB_PATH", p)
    database.init_db()
    # two docs with headings so tree/units exist
    with sqlite3.connect(str(p)) as c:
        c.execute(
            "INSERT INTO folders (path, name) VALUES ('f', 'f')"
        )
        c.execute(
            "INSERT INTO documents (id, folder, filename, title, source_type, content) "
            "VALUES ('docA','f','a.md','Alpha','file', ?)",
            ("# Alpha\n\nApples are red and sweet.\n",),
        )
        c.execute(
            "INSERT INTO documents (id, folder, filename, title, source_type, content) "
            "VALUES ('docB','f','b.md','Beta','file', ?)",
            ("# Beta\n\nBananas are yellow.\n",),
        )
        c.commit()
    return p


def test_pool_namespaces_unit_ids_and_tags_source(db):
    units = R.get_retrieval_units_multi(["docA", "docB"])
    assert units, "expected pooled units from both docs"
    assert all("::" in u["unit_id"] for u in units)
    assert {u["doc_id"] for u in units} == {"docA", "docB"}
    assert {u["doc_title"] for u in units} == {"Alpha", "Beta"}
    # namespaced id begins with its source doc id
    for u in units:
        assert u["unit_id"].startswith(u["doc_id"] + "::")


def test_bm25_over_pool_ranks_correct_file(db):
    # BM25 needs no LLM/embeddings — pure lexical over the shared corpus.
    units = R.get_retrieval_units_multi(["docA", "docB"])
    ranked = R.bm25_search("docA", "bananas yellow", units=units)
    assert ranked, "expected a lexical hit"
    top = ranked[0]
    # the winning unit must come from docB (where 'bananas' lives)
    assert top.startswith("docB::")
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_retrieval_multi.py -v`
Expected: FAIL — `get_retrieval_units_multi` chưa tồn tại.

- [ ] **Step 3: Thêm các hàm đa-doc** vào `backend/services/retrieval.py`, ngay **trước** `def retrieve_context(` (dòng 561)

```python
# ---------------------------------------------------------------------------
# Multi-document retrieval — scope = several selected docs ranked as one pool
# ---------------------------------------------------------------------------
# Reuses the single-doc primitives above; only ADDS list-aware wrappers so the
# existing single-doc functions and the agent's retrieval contract are intact.

_NS_SEP = "::"  # namespaces unit ids across docs: f"{doc_id}::{unit_id}"


def _doc_title(doc_id: str) -> str:
    """Human-readable title for a doc (title → filename → id)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COALESCE(title, filename, id) FROM documents WHERE id=?",
            (doc_id,),
        ).fetchone()
    return row[0] if row and row[0] else doc_id


def get_retrieval_units_multi(doc_ids: list[str]) -> list[dict]:
    """Pool retrieval units from several docs into one corpus.

    Calls the single-doc get_retrieval_units() per document, namespaces each
    unit_id as f"{doc_id}::{unit_id}" (so ids never collide across docs), and
    tags each unit with its source doc_id + doc_title for citation + edit.
    """
    pooled: list[dict] = []
    for doc_id in doc_ids:
        title = _doc_title(doc_id)
        for u in get_retrieval_units(doc_id):
            pooled.append({
                **u,
                "unit_id": f"{doc_id}{_NS_SEP}{u['unit_id']}",
                "doc_id": doc_id,
                "doc_title": title,
            })
    return pooled


def dense_search_multi(
    doc_ids: list[str],
    question: str,
    settings: Settings,
    units: list[dict] | None = None,
    top_k: int = _PER_RETRIEVER_TOP_K,
) -> list[str]:
    """Dense search across several docs → namespaced unit_ids.

    Cosine scores live in one shared embedding space, so a single global sort
    across all docs is correct. Ensures each doc's index exists (lazily) using
    its slice of the pooled units (de-namespaced for indexing).
    """
    cfg = settings.active_llm
    if not embeddings_supported(cfg):
        return []

    per_doc_units: dict[str, list[dict]] = {}
    for u in (units or []):
        bare = {**u, "unit_id": u["unit_id"].split(_NS_SEP, 1)[-1]}
        per_doc_units.setdefault(u.get("doc_id", ""), []).append(bare)
    for doc_id in doc_ids:
        ensure_embedding_index(doc_id, settings, units=per_doc_units.get(doc_id))

    import numpy as np

    embedder = create_embeddings(cfg)
    q = np.asarray(embedder.embed_query(question), dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-8)

    placeholders = ",".join("?" * len(doc_ids))
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT doc_id, unit_id, vector FROM document_embeddings "
            f"WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        ).fetchall()
    if not rows:
        return []

    scored: list[tuple[str, float]] = []
    for doc_id, unit_id, blob in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.shape != qn.shape:
            continue
        score = float(qn @ (v / (np.linalg.norm(v) + 1e-8)))
        scored.append((f"{doc_id}{_NS_SEP}{unit_id}", score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [uid for uid, _ in scored[:top_k]]


def hybrid_rank_multi(
    doc_ids: list[str],
    question: str,
    settings: Settings,
    units: list[dict] | None = None,
) -> tuple[list[str], list[str] | None]:
    """Rank pooled units across docs by fusing tree + BM25 + dense (one RRF).

    Returns ``(fused_namespaced_unit_ids, tree_selected)`` where tree_selected
    is the concatenated per-doc tree selection (``[]`` = trees existed but
    chose nothing → out-of-scope signal; ``None`` = no doc had a tree).
    """
    units = units if units is not None else get_retrieval_units_multi(doc_ids)
    if not units:
        return [], None

    ranked_lists: list[list[str]] = []

    any_tree = False
    tree_selected: list[str] = []
    for doc_id in doc_ids:
        tree = get_tree_index(doc_id)
        if tree and tree.get("structure"):
            any_tree = True
            try:
                for nid in select_node_ids(tree, question, settings):
                    tree_selected.append(f"{doc_id}{_NS_SEP}{nid}")
            except Exception:
                logger.exception("Tree node selection failed for doc %s", doc_id)
    if tree_selected:
        ranked_lists.append(tree_selected)

    try:
        bm = bm25_search(doc_ids[0], question, units=units)  # doc_id unused (units given)
        if bm:
            ranked_lists.append(bm)
    except Exception:
        logger.exception("BM25 retrieval failed for docs %s", doc_ids)

    try:
        dense = dense_search_multi(doc_ids, question, settings, units=units)
        if dense:
            ranked_lists.append(dense)
    except Exception:
        logger.exception("Dense retrieval failed for docs %s", doc_ids)

    if not ranked_lists:
        return [], (tree_selected if any_tree else None)
    return rrf_fuse(ranked_lists), (tree_selected if any_tree else None)
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_retrieval_multi.py -v`
Expected: tất cả PASS (BM25 không cần LLM).

- [ ] **Step 5: Commit**

```bash
git add backend/services/retrieval.py tests/test_retrieval_multi.py
git commit -m "feat(retrieval): multi-doc pooled units + dense/hybrid ranking"
```

### Task 6: `agentic_retrieve_context_multi`

**Files:**
- Test: `tests/test_retrieval_multi.py` (bổ sung)
- Modify: `backend/services/retrieval.py` (thêm cuối file)

- [ ] **Step 1: Bổ sung test thất bại**

```python
def test_agentic_multi_no_llm_falls_back_to_pooled_singleshot(db, monkeypatch):
    # With no LLM configured, the multi path must still return pooled lexical
    # context grounded in the selected docs (no tree/dense/critique).
    import backend.services.retrieval as RR
    monkeypatch.setattr(RR, "is_llm_configured", lambda cfg: False)
    out = RR.agentic_retrieve_context_multi(["docA", "docB"], "yellow bananas")
    assert "Beta" in out  # file label of docB
    assert "Bananas are yellow" in out


def test_agentic_multi_empty_scope_returns_blank(db):
    assert R.agentic_retrieve_context_multi([], "anything") == ""
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_retrieval_multi.py -v`
Expected: FAIL — `agentic_retrieve_context_multi` chưa tồn tại.

- [ ] **Step 3: Thêm hàm** vào cuối `backend/services/retrieval.py`

```python
def agentic_retrieve_context_multi(
    doc_ids: list[str],
    question: str,
    settings: Settings | None = None,
    max_rounds: int = MAX_RETRIEVAL_ROUNDS,
) -> str:
    """Iterative multi-hop retrieval over a POOL of several documents.

    Mirrors agentic_retrieve_context() but ranks all selected docs as one pool
    via hybrid_rank_multi (units namespaced by doc). Unlike the single-doc
    path, it never falls back to a raw whole-document dump (there is no single
    "the document"): when nothing is retrieved it returns "" — preserving the
    anti-hallucination contract for out-of-scope questions.
    """
    settings = settings or get_settings()
    doc_ids = [d for d in doc_ids if d]
    if not doc_ids:
        return ""

    units = get_retrieval_units_multi(doc_ids)
    if not units:
        return ""
    unit_map = {u["unit_id"]: u for u in units}

    # No LLM: single-shot pooled lexical/dense retrieval, no critique loop.
    if not is_llm_configured(settings.active_llm):
        fused, _ = hybrid_rank_multi(doc_ids, question, settings, units=units)
        if not fused:
            return ""
        return build_context_from_units(
            [unit_map[uid] for uid in fused if uid in unit_map]
        )

    accumulated: dict[str, float] = {}
    seen_queries: set[str] = set()
    first_round_empty_tree_only = False

    queries = [question]
    for round_idx in range(max_rounds):
        new_units_this_round = 0
        for q in queries:
            qkey = q.strip().lower()
            if not qkey or qkey in seen_queries:
                continue
            seen_queries.add(qkey)

            fused, tree_selected = hybrid_rank_multi(doc_ids, q, settings, units=units)
            if round_idx == 0 and q == question and not fused and tree_selected == []:
                first_round_empty_tree_only = True

            for rank, uid in enumerate(fused):
                if uid not in unit_map:
                    continue
                score = 1.0 / (_RRF_K + rank + 1)
                if uid not in accumulated:
                    new_units_this_round += 1
                if score > accumulated.get(uid, 0.0):
                    accumulated[uid] = score

        if not accumulated and first_round_empty_tree_only:
            return ""
        if not accumulated:
            break

        ordered = sorted(accumulated.items(), key=lambda x: x[1], reverse=True)
        selected = [unit_map[uid] for uid, _ in ordered[:MAX_ACCUMULATED_UNITS]]
        context = build_context_from_units(selected)

        if round_idx == max_rounds - 1:
            break

        verdict = _critique(question, context, settings)
        if verdict["sufficient"] or not verdict["followups"]:
            break
        if new_units_this_round == 0:
            break

        queries = verdict["followups"]

    if not accumulated:
        return ""

    ordered = sorted(accumulated.items(), key=lambda x: x[1], reverse=True)
    selected = [unit_map[uid] for uid, _ in ordered[:MAX_ACCUMULATED_UNITS]]
    return build_context_from_units(selected)
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_retrieval_multi.py -v`
Expected: tất cả PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/services/retrieval.py tests/test_retrieval_multi.py
git commit -m "feat(retrieval): agentic multi-doc retrieval over pooled units"
```

---

## Phase 5 — agent.py: scope context, edit theo intent, prompt

### Task 7: Scope context + edit resolver + tool `file`

**Files:**
- Test: `tests/test_agent_scope.py`
- Modify: `backend/services/agent.py`
  - `set_tool_context` (dòng 438-444), `retrieve_context` (203-224),
    `preview_edit` (286-319), `apply_edit` (322-355); thêm `_resolve_scope_doc`;
    cập nhật SOUL prompt (57-100) và đăng ký tool (431).

- [ ] **Step 1: Viết test thất bại** — `tests/test_agent_scope.py`

```python
import pytest
from backend.services import agent as A


def test_resolve_scope_doc_by_title_and_id():
    ctx = {"doc_ids": ["docA", "docB"],
           "doc_titles": {"docA": "Alpha", "docB": "Beta"}}
    assert A._resolve_scope_doc("Beta", ctx) == "docB"     # by title
    assert A._resolve_scope_doc("beta", ctx) == "docB"     # case-insensitive
    assert A._resolve_scope_doc("docA", ctx) == "docA"     # by raw id


def test_resolve_scope_doc_single_scope_defaults():
    ctx = {"doc_ids": ["docA"], "doc_titles": {"docA": "Alpha"}}
    # empty file + single doc in scope → that doc
    assert A._resolve_scope_doc("", ctx) == "docA"


def test_resolve_scope_doc_rejects_out_of_scope():
    ctx = {"doc_ids": ["docA"], "doc_titles": {"docA": "Alpha"}}
    out = A._resolve_scope_doc("Gamma", ctx)
    assert out.startswith("Error")
    assert "Alpha" in out  # lists available files


def test_resolve_scope_doc_ambiguous_empty_multi():
    ctx = {"doc_ids": ["docA", "docB"],
           "doc_titles": {"docA": "Alpha", "docB": "Beta"}}
    out = A._resolve_scope_doc("", ctx)
    assert out.startswith("Error")  # must specify which file
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_agent_scope.py -v`
Expected: FAIL — `_resolve_scope_doc` chưa tồn tại.

- [ ] **Step 3a: Thêm `_resolve_scope_doc`** vào `backend/services/agent.py` (ngay trước `@tool def retrieve_context`, dòng 203)

```python
def _resolve_scope_doc(file: str, ctx: dict[str, Any]) -> str:
    """Resolve a file label (title shown in [File: ...]) or raw doc_id to a
    doc_id within the current scope. Returns the doc_id, or an error string
    for the agent listing the available files.
    """
    doc_ids: list[str] = ctx.get("doc_ids") or []
    titles: dict[str, str] = ctx.get("doc_titles") or {}
    available = ", ".join(titles.get(d, d) for d in doc_ids) or "(none)"

    f = (file or "").strip()
    if not f:
        if len(doc_ids) == 1:
            return doc_ids[0]
        return (
            "Error: multiple files are in scope — specify which file to edit "
            f"(by its [File: ...] name). Available: {available}"
        )
    if f in doc_ids:
        return f
    matches = [d for d in doc_ids if titles.get(d, "").strip().lower() == f.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return f"Error: '{file}' matches multiple files; use the exact file name."
    return f"Error: '{file}' is not in the selected scope. Available files: {available}"
```

- [ ] **Step 3b: Thay `set_tool_context`** (dòng 438-444)

```python
def set_tool_context(
    doc_ids: list[str],
    settings: Settings,
    doc_titles: dict[str, str] | None = None,
) -> None:
    """Set per-request tool context (selected doc scope + settings) via
    ContextVar. ``doc_titles`` maps doc_id → display title so edit tools can
    resolve a file the agent names. Initialises the ``edited`` flag to False.
    """
    _tool_context_var.set({
        "doc_ids": list(doc_ids),
        "settings": settings,
        "doc_titles": doc_titles or {},
        "edited": False,
    })
```

- [ ] **Step 3c: Thay thân `retrieve_context`** (dòng 214-224)

```python
    ctx = _tool_context_var.get()
    doc_ids = ctx.get("doc_ids") or []
    settings = ctx.get("settings")

    if not doc_ids or not settings:
        return "Error: Document context not configured."

    context = retrieval.agentic_retrieve_context_multi(doc_ids, question, settings)
    if context:
        return context
    return "No relevant sections found in the selected documents for this question."
```

Đồng thời cập nhật docstring `retrieve_context` (dòng 204-213) — đổi "the document" → "the selected documents".

- [ ] **Step 3d: Thay `preview_edit`** (dòng 286-319) — thêm tham số `file`

```python
@tool
def preview_edit(file: str, old_string: str, new_string: str) -> str:
    """Preview an edit to ONE of the in-scope documents BEFORE applying it.

    ALWAYS call this before `apply_edit`. Locates `old_string` in the named
    file and returns the exact snippet that would change, plus the replacement.

    Args:
        file: Which document to edit — the file name exactly as shown in the
            "[File: ...]" label from retrieve_context. Must be one of the
            files currently in scope.
        old_string: The exact text to find (from retrieve_context). Do NOT
            include the "[File: ... | Section: ...]" header line.
        new_string: The replacement text. Pass an empty string to DELETE.
    """
    ctx = _tool_context_var.get()
    doc_id = _resolve_scope_doc(file, ctx)
    if doc_id.startswith("Error"):
        return doc_id

    content = _get_document_content(doc_id)
    if not content:
        return "Error: Document content is empty or not yet processed."

    located = _locate_in_content(content, old_string)
    if isinstance(located, str):
        return located

    start, end = located
    matched = content[start:end]
    action = "DELETE" if new_string == "" else "REPLACE"
    title = (ctx.get("doc_titles") or {}).get(doc_id, doc_id)
    return (
        f"Preview ({action}) on file '{title}' — ask the user to confirm before "
        f"calling apply_edit.\n\n"
        f"--- Text that will be removed (exact) ---\n{matched}\n"
        f"--- Replaced with ---\n{new_string if new_string else '(deleted)'}\n"
    )
```

- [ ] **Step 3e: Thay `apply_edit`** (dòng 322-355) — thêm tham số `file`

```python
@tool
async def apply_edit(file: str, old_string: str, new_string: str) -> str:
    """Apply a previewed, USER-CONFIRMED edit to one in-scope document.

    Only call AFTER `preview_edit` and AFTER the user explicitly approved.

    Args:
        file: Same file you previewed (name as in the "[File: ...]" label).
        old_string: The exact text to replace (same as previewed).
        new_string: The replacement text. Empty string deletes the match.
    """
    ctx = _tool_context_var.get()
    doc_id = _resolve_scope_doc(file, ctx)
    if doc_id.startswith("Error"):
        return doc_id

    content = _get_document_content(doc_id)
    if not content:
        return "Error: Document content is empty or not yet processed."

    located = _locate_in_content(content, old_string)
    if isinstance(located, str):
        return located

    start, end = located
    new_content = content[:start] + new_string + content[end:]
    try:
        await persist_document_content(doc_id, new_content)
    except Exception as exc:
        return f"Error applying edit: {exc}"

    ctx["edited"] = True
    ctx["edited_doc_id"] = doc_id
    title = (ctx.get("doc_titles") or {}).get(doc_id, doc_id)
    return f"Edit applied successfully to '{title}'. The document has been updated."
```

- [ ] **Step 3f: Cập nhật SOUL prompt** — thay đoạn `DOCUMENT_SOUL_PROMPT` (dòng 57-100). Các thay đổi tối thiểu cần thiết:

Thay phần đầu (Identity + rule 1,3,5) để dùng số nhiều và yêu cầu cite tên file; thay mục "Editing" để dùng tham số `file`. Nội dung mới:

```python
DOCUMENT_SOUL_PROMPT = """\
You are a Document Assistant - a faithful, precise reader of the user's selected documents.

## Your Identity
You exist to help users understand THEIR documents. You are not a general-purpose AI.
The user has selected one or more documents as the current scope; you may read and
operate ONLY within those files.

## Core Rules (NON-NEGOTIABLE)
1. **Document-grounded ONLY**: Every claim MUST come from context retrieved by your \
tools across the selected documents. If you cannot find the answer, say so honestly.
2. **No fabrication**: NEVER invent or assume information not present in retrieved \
context. "I don't see this in the selected documents" is always a valid answer.
3. **Cite file and section**: When answering, name the FILE (as shown in the \
"[File: ...]" label) and the section title where you found the information. For a \
figure or table, cite it explicitly and read the relevant cells directly.
4. **Retrieval first**: Your FIRST tool call for ANY content question MUST be \
`retrieve_context`. Never answer from memory alone.
5. **Documents are NOT files you can browse**: Your only way to access content is \
`retrieve_context`. NEVER claim "no document exists" — call `retrieve_context` instead.

## Reading Images
- Context may reference images as `![Image N](/assets/...)`.
- When the question concerns such an image, call `read_image` with the EXACT path \
from the context. Only read images that appear in retrieved context.

## Editing a Document
You CAN edit a document — but ONLY when the user explicitly asks you to change it.

1. **Infer the target file**: From the user's request and the retrieved context, \
determine WHICH in-scope file to edit. Pass it as the `file` argument (the name shown \
in the "[File: ...]" label). If it is ambiguous which file, ASK the user first.
2. **Preview first**: ALWAYS call `preview_edit` (with `file`, `old_string`, \
`new_string`) and show the result. NEVER apply without user approval.
3. **Apply**: Only after explicit approval, call `apply_edit` with the SAME `file` and \
strings you previewed.
- `old_string` must come from `retrieve_context` or `preview_edit` — never invent text.
- **Delete** = empty `new_string`. **Add** = use an existing passage as anchor.

## Response Style
- Be concise and well-structured (headers, bullets, bold for key terms)
- Match the user's language (Vietnamese in → Vietnamese out)
- When documents disagree or are ambiguous, present the interpretations clearly
"""
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_agent_scope.py -v`
Expected: 4 passed.

- [ ] **Step 5: Xác minh import agent không gãy**

Run: `python -c "import backend.services.agent"`
Expected: không lỗi.

- [ ] **Step 6: Commit**

```bash
git add backend/services/agent.py tests/test_agent_scope.py
git commit -m "feat(agent): multi-doc scope context + edit-by-intent file arg"
```

---

## Phase 6 — chat.py API: doc_ids + session toàn cục

### Task 8: Cập nhật endpoints chat sang scope + session toàn cục

**Files:**
- Modify: `backend/api/chat.py` (toàn bộ)

- [ ] **Step 1: Thay nội dung `backend/api/chat.py`**

```python
"""Chat API — multi-document Q&A via a LangGraph ReAct agent.

Endpoints:
  POST   /api/chat/stream              - Stream answer over selected doc scope (SSE)
  GET    /api/chat/history             - Load all display messages (all sessions)
  POST   /api/chat/new-session         - Start a fresh global session
  DELETE /api/chat/history             - Clear all history
  DELETE /api/chat/session/{session_id} - Delete one session
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.config import get_settings
from ..core.database import get_db
from ..services.agent import (
    get_document_agent,
    set_tool_context,
    reset_agent,
    document_was_edited,
)
from ..services.chat_history import (
    get_current_session_id,
    get_messages,
    get_messages_for_session,
    save_message,
    start_new_session,
    delete_messages,
    delete_session,
)
from ..services.compactor import compact_if_needed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    doc_ids: list[str]
    question: str
    session_id: int | None = None  # If None, use current global session


class HistoryResponse(BaseModel):
    messages: list[dict]


class SessionResponse(BaseModel):
    session_id: int


def _doc_titles(doc_ids: list[str]) -> dict[str, str]:
    """Map each scoped doc_id → display title (title → filename → id)."""
    if not doc_ids:
        return {}
    placeholders = ",".join("?" * len(doc_ids))
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT id, COALESCE(title, filename, id) FROM documents "
            f"WHERE id IN ({placeholders})",
            tuple(doc_ids),
        ).fetchall()
    return {r[0]: r[1] for r in rows}


@router.post("/stream")
async def chat_stream(body: ChatRequest):
    """Ask a question grounded in the selected documents (SSE stream)."""
    settings = get_settings()
    from ..services.llm import is_llm_configured
    if not is_llm_configured(settings.active_llm):
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured. Please set the LLM endpoint in Settings.",
        )
    if not body.doc_ids:
        raise HTTPException(status_code=400, detail="No documents selected for chat scope.")

    session_id = body.session_id or get_current_session_id()

    try:
        await compact_if_needed(session_id, settings)
    except Exception:
        logger.exception("compact_if_needed failed; continuing without compact")

    titles = _doc_titles(body.doc_ids)
    set_tool_context(body.doc_ids, settings, titles)

    from ..core.telemetry import track_event_sync
    track_event_sync("chat_sent", {"doc_ids": body.doc_ids})

    async def _event_generator():
        full_response = ""
        try:
            agent = await get_document_agent()
            config = {
                "configurable": {
                    # thread_id is per global session → AsyncSqliteSaver replays
                    # the full conversation on resume, independent of doc scope.
                    "thread_id": f"session-{session_id}",
                },
                "run_name": "document-chat",
                "metadata": {
                    "doc_ids": body.doc_ids,
                    "session_id": session_id,
                },
                "tags": ["ai-agent-chatbot"],
            }

            prior = get_messages_for_session(session_id)
            stream_input = {
                "messages": [
                    *[{"role": m["role"], "content": m["content"]} for m in prior],
                    {"role": "user", "content": body.question},
                ],
            }

            async for chunk in agent.astream_events(
                stream_input, version="v2", config=config,
            ):
                if chunk.get("event") != "on_chat_model_stream":
                    continue
                node = chunk.get("metadata", {}).get("langgraph_node", "")
                if node == "tools":
                    continue
                message_obj = chunk.get("data", {}).get("chunk")
                if not message_obj:
                    continue
                content = getattr(message_obj, "content", "")
                if not content or getattr(message_obj, "tool_call_chunks", None):
                    continue
                if isinstance(content, list):
                    token = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                else:
                    token = content
                full_response += token
                escaped = token.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"

        except Exception as exc:
            logger.exception("Chat stream error")
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            if full_response:
                try:
                    save_message(session_id, "user", body.question)
                    save_message(session_id, "assistant", full_response)
                except Exception:
                    logger.exception("Failed to save chat history")
            if document_was_edited():
                yield "data: [EDITED]\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history")
async def get_chat_history() -> HistoryResponse:
    """Load all display messages across all sessions."""
    return HistoryResponse(messages=get_messages())


@router.post("/new-session")
async def new_chat_session() -> SessionResponse:
    """Start a new global conversation session."""
    return SessionResponse(session_id=start_new_session())


@router.delete("/history")
async def clear_chat_history() -> dict:
    """Clear all chat history and reset the agent's in-memory state."""
    delete_messages()
    reset_agent()
    return {"status": "ok"}


@router.delete("/session/{session_id}")
async def delete_chat_session(session_id: int) -> dict:
    """Delete a single conversation session and reset the agent."""
    delete_session(session_id)
    reset_agent()
    return {"status": "ok", "session_id": session_id}
```

- [ ] **Step 2: Xác minh app import được**

Run: `python -c "import backend.api.chat"`
Expected: không lỗi.

- [ ] **Step 3: Chạy toàn bộ test backend (đảm bảo không hồi quy)**

Run: `pytest tests/ -v`
Expected: tất cả test mới PASS; test cũ không gãy.

- [ ] **Step 4: Commit**

```bash
git add backend/api/chat.py
git commit -m "feat(api): chat scope via doc_ids + global session endpoints"
```

---

## Phase 7 — Frontend API client

### Task 9: Cập nhật `src/lib/sidecar.ts`

**Files:**
- Modify: `src/lib/sidecar.ts` (dòng 52-129)

- [ ] **Step 1: Thay khối streaming + history (dòng 52-129)** bằng:

```typescript
// ── Streaming chat (SSE) ──────────────────────────────────────────

export async function streamChat(
  docIds: string[],
  question: string,
  onChunk: (text: string) => void,
  sessionId?: number,
  onEdited?: () => void,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_ids: docIds, question, session_id: sessionId ?? null }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`Chat request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  try {
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;
        const payload = trimmed.slice(6);
        if (payload === "[DONE]") return;
        if (payload === "[EDITED]") { onEdited?.(); continue; }
        if (payload.startsWith("[ERROR]")) throw new Error(payload.slice(8));
        onChunk(payload.replace(/\\n/g, "\n"));
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// ── Document list (for the chat scope picker) ─────────────────────

export interface DocSummary {
  id: string;
  title?: string;
  filename: string;
  folder: string;
}

export async function listDocuments(): Promise<DocSummary[]> {
  return apiGet<DocSummary[]>(`/api/documents/`);
}

// ── Chat history & session management (global sessions) ───────────

export interface ChatMessage {
  id: number;
  session_id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export async function getChatHistory(): Promise<ChatMessage[]> {
  const res = await apiGet<{ messages: ChatMessage[] }>(`/api/chat/history`);
  return res.messages;
}

export async function startNewSession(): Promise<number> {
  const res = await apiPost<{ session_id: number }>(`/api/chat/new-session`, {});
  return res.session_id;
}

export async function clearChatHistory(): Promise<void> {
  await apiDelete(`/api/chat/history`);
}

export async function deleteSession(sessionId: number): Promise<void> {
  await apiDelete(`/api/chat/session/${sessionId}`);
}
```

- [ ] **Step 2: Kiểm tra TypeScript biên dịch**

Run: `pnpm exec tsc --noEmit`
Expected: lỗi còn lại CHỈ ở `ChatPanel.tsx`/`DocumentEditor.tsx` (sẽ sửa ở Phase 8-9) do đổi chữ ký. Không có lỗi trong `sidecar.ts`.

- [ ] **Step 3: Commit**

```bash
git add src/lib/sidecar.ts
git commit -m "feat(ui): chat API client for doc scope + global sessions"
```

---

## Phase 8 — ChatPanel: bộ chọn phạm vi + history toàn cục

### Task 10: Thêm scope picker và nạp history toàn cục

**Files:**
- Modify: `src/components/ChatPanel.tsx`

- [ ] **Step 1: Đổi props + state + import** — thay khối import (dòng 1-2) và interface/đầu component (dòng 116-145)

Thay dòng 2:

```typescript
import { streamChat, getChatHistory, startNewSession, clearChatHistory, deleteSession, listDocuments, DocSummary } from "../lib/sidecar";
```

Thay `interface ChatPanelProps` + đầu hàm (dòng 116-131):

```typescript
interface ChatPanelProps {
  initialDocId: string;            // file the chat was opened from (seeds scope)
  onClose: () => void;
  onDocumentEdited?: () => void;
}

export default function ChatPanel({ initialDocId, onClose, onDocumentEdited }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<number>(1);
  const [sessionMenuOpen, setSessionMenuOpen] = useState(false);

  // Scope: which documents this turn may read/edit. Transient (not persisted).
  const [scopeDocIds, setScopeDocIds] = useState<string[]>([initialDocId]);
  const [allDocs, setAllDocs] = useState<DocSummary[]>([]);
  const [scopeMenuOpen, setScopeMenuOpen] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const sessionMenuRef = useRef<HTMLDivElement>(null);
  const scopeMenuRef = useRef<HTMLDivElement>(null);

  const docTitle = useCallback(
    (id: string) => {
      const d = allDocs.find((x) => x.id === id);
      return d ? (d.title || d.filename) : id;
    },
    [allDocs],
  );
```

- [ ] **Step 2: Nạp danh sách tài liệu + reset scope khi đổi file mở** — thay `useEffect` nạp history (dòng 200-219)

```typescript
  // Load the document list once (for the scope picker).
  useEffect(() => {
    listDocuments().then(setAllDocs).catch(() => { /* ignore */ });
  }, []);

  // Seed scope with the file the panel was opened from.
  useEffect(() => {
    setScopeDocIds([initialDocId]);
  }, [initialDocId]);

  // Load GLOBAL chat history on mount (sessions are not tied to a doc).
  useEffect(() => {
    setMessages([]);
    setSessionId(1);
    setError(null);
    getChatHistory().then((history) => {
      if (history.length > 0) {
        const msgs: Message[] = history.map((h) => ({
          id: String(h.id),
          role: h.role,
          content: h.content,
          sessionId: h.session_id,
        }));
        setMessages(msgs);
        setSessionId(Math.max(...history.map((h) => h.session_id)));
      }
    }).catch(() => { /* ignore load errors */ });
  }, []);
```

- [ ] **Step 3: Đóng menu scope khi click ngoài** — thêm sau `useEffect` đóng session menu (sau dòng 168)

```typescript
  useEffect(() => {
    if (!scopeMenuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (scopeMenuRef.current && !scopeMenuRef.current.contains(e.target as Node)) {
        setScopeMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [scopeMenuOpen]);

  const toggleScopeDoc = useCallback((id: string) => {
    setScopeDocIds((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id],
    );
  }, []);
```

- [ ] **Step 4: Sửa các lời gọi API trong handlers** (đã đổi chữ ký)

`handleDeleteSession` (dòng 171-185): đổi `await deleteSession(docId, sid);` → `await deleteSession(sid);` và bỏ `docId` khỏi mảng phụ thuộc useCallback (đổi `[docId, sessionId, messages]` → `[sessionId, messages]`).

`sendMessage` (dòng 230-257): thay lời gọi và guard:

```typescript
  const sendMessage = useCallback(async () => {
    const question = input.trim();
    if (!question || streaming) return;
    if (scopeDocIds.length === 0) { setError("Hãy chọn ít nhất một tài liệu."); return; }
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    setError(null);

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: question, sessionId };
    const assistantMsg: Message = { id: crypto.randomUUID(), role: "assistant", content: "", streaming: true, sessionId };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    try {
      await streamChat(scopeDocIds, question, (token) => {
        setMessages((prev) =>
          prev.map((m) => m.id === assistantMsg.id ? { ...m, content: m.content + token } : m)
        );
      }, sessionId, onDocumentEdited);
    } catch (err) {
      setError(String(err));
      setMessages((prev) => prev.filter((m) => m.id !== assistantMsg.id));
    } finally {
      setMessages((prev) =>
        prev.map((m) => m.id === assistantMsg.id ? { ...m, streaming: false } : m)
      );
      setStreaming(false);
    }
  }, [scopeDocIds, input, streaming, sessionId, onDocumentEdited]);
```

`startNewSession` trong nút New Topic (dòng 402-409): đổi `await startNewSession(docId)` → `await startNewSession()`.

- [ ] **Step 5: Thêm UI scope picker** — chèn ngay sau khối Session switcher (sau `</div>` đóng session switcher, dòng 366), trước `{/* Messages */}`

```tsx
      {/* Scope picker — which documents are in chat scope */}
      <div ref={scopeMenuRef} className="relative flex items-center gap-2 px-4 py-2 border-b border-[var(--border)] shrink-0 bg-[var(--surface-glass)] z-10">
        <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-faint)] shrink-0">
          Scope
        </span>
        <div className="flex flex-wrap items-center gap-1.5 flex-1 min-w-0">
          {scopeDocIds.map((id) => (
            <span key={id} className="inline-flex items-center gap-1 max-w-[160px] px-2 py-0.5 rounded-full bg-[var(--accent-subtle)] border border-[var(--border-glow)] text-[11px] text-[var(--accent-text)]">
              <span className="truncate">{docTitle(id)}</span>
              {scopeDocIds.length > 1 && (
                <button
                  type="button"
                  onClick={() => toggleScopeDoc(id)}
                  disabled={streaming}
                  className="shrink-0 hover:text-[var(--error)] disabled:opacity-40"
                  title="Remove from scope"
                >
                  <IconX />
                </button>
              )}
            </span>
          ))}
          <button
            type="button"
            onClick={() => setScopeMenuOpen((v) => !v)}
            disabled={streaming}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border border-[var(--border-strong)] text-[11px] text-[var(--text-muted)] hover:border-[var(--border-hover)] hover:text-[var(--text-primary)] disabled:opacity-40"
            title="Add files to scope"
          >
            <IconPlus /> File
          </button>
        </div>

        {scopeMenuOpen && (
          <div role="listbox" className="absolute left-4 right-4 top-[calc(100%+4px)] z-30 max-h-72 overflow-y-auto rounded-xl border border-[var(--border-strong)] bg-[var(--surface)] shadow-2xl shadow-black/40 p-1.5 scale-in origin-top">
            {allDocs.length === 0 && (
              <div className="px-2.5 py-2 text-[12px] text-[var(--text-faint)]">No documents</div>
            )}
            {allDocs.map((d) => {
              const checked = scopeDocIds.includes(d.id);
              return (
                <div
                  key={d.id}
                  role="option"
                  aria-selected={checked}
                  onClick={() => toggleScopeDoc(d.id)}
                  className={`flex items-center gap-2 rounded-lg px-2.5 py-2 cursor-pointer transition-colors ${
                    checked ? "bg-[var(--accent-subtle)] text-[var(--text-primary)]" : "text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]"
                  }`}
                >
                  <input type="checkbox" readOnly checked={checked} className="accent-[var(--accent)]" />
                  <span className="flex-1 min-w-0 truncate text-[12px]">{d.title || d.filename}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
```

- [ ] **Step 6: Cập nhật văn bản tĩnh** — đổi 2 chỗ "this document only":
  - dòng ~379: "Answers are grounded in this document only." → "Answers are grounded in the selected documents."
  - dòng ~452: "Grounded in this document only · ..." → "Grounded in selected documents · Shift+Enter for new line"

- [ ] **Step 7: Build kiểm tra biên dịch (cùng Task 11 cho DocumentEditor)**

Run: `pnpm exec tsc --noEmit`
Expected: lỗi còn lại chỉ ở `DocumentEditor.tsx` (prop `docId` → `initialDocId`), sẽ sửa ở Task 11.

- [ ] **Step 8: Commit**

```bash
git add src/components/ChatPanel.tsx
git commit -m "feat(ui): multi-file scope picker + global session history in ChatPanel"
```

---

## Phase 9 — Wiring + verification

### Task 11: DocumentEditor truyền `initialDocId`

**Files:**
- Modify: `src/pages/DocumentEditor.tsx` (dòng 542)

- [ ] **Step 1: Sửa render ChatPanel** (dòng 542)

```tsx
              <ChatPanel key={id} initialDocId={id} onDocumentEdited={reloadDocument} onClose={() => { chatHasAnimated.current = false; setShowChat(false); }} />
```

- [ ] **Step 2: Biên dịch sạch**

Run: `pnpm exec tsc --noEmit`
Expected: 0 lỗi.

- [ ] **Step 3: Commit**

```bash
git add src/pages/DocumentEditor.tsx
git commit -m "feat(ui): pass initialDocId to ChatPanel"
```

### Task 12: Kiểm thử end-to-end thủ công

**Files:** (không sửa code — kiểm chứng)

- [ ] **Step 1: Khởi động backend**

Run: `python3 backend/main.py --dev`
Expected: log "ready"; `GET http://localhost:8008/api/health` trả 200.

- [ ] **Step 2: Khởi động frontend**

Run: `pnpm dev`
Mở app, vào một tài liệu, bấm **Chat**.

- [ ] **Step 3: Kiểm tra phạm vi nhiều file**
  - Trong vùng **Scope**, bấm **+ File**, tích thêm 1-2 tài liệu khác.
  - Hỏi một câu mà câu trả lời nằm ở file **được thêm vào** (không phải file đang mở).
  - Expected: câu trả lời đúng và **cite đúng tên file nguồn** (theo nhãn `[File: ...]`).

- [ ] **Step 4: Kiểm tra session toàn cục**
  - Bấm **New Topic** → session_id tăng.
  - Mở Chat từ một tài liệu **khác**: danh sách session hiển thị **giống nhau** (toàn cục).
  - Expected: lịch sử dùng chung, không bị bó theo doc.

- [ ] **Step 5: Kiểm tra edit theo intent**
  - Với ≥2 file trong scope, yêu cầu: "Sửa trong file <Tên file B>: đổi 'X' thành 'Y'".
  - Expected: agent gọi `preview_edit` với đúng `file` = file B, hiển thị preview kèm tên file; sau khi bạn xác nhận, `apply_edit` ghi đúng file B. Nếu yêu cầu mơ hồ không rõ file, agent hỏi lại.
  - Lưu ý: editor đang mở file A sẽ không tự đổi nếu agent sửa file B (chỉ reload file đang mở). Đây là hành vi chấp nhận được; mở file B để xem thay đổi.

- [ ] **Step 6: Kiểm tra out-of-scope (chống bịa)**
  - Hỏi một chủ đề không có trong bất kỳ file nào đang chọn.
  - Expected: agent trả lời "không thấy trong các tài liệu đã chọn", không bịa.

- [ ] **Step 7: Commit (nếu có tinh chỉnh nhỏ phát sinh)**

```bash
git add -A
git commit -m "test(e2e): verify multi-file chat scope, global sessions, edit-by-intent"
```

---

## Notes / Tradeoffs đã chấp nhận (theo yêu cầu đơn giản hóa)

- **Không lưu phạm vi (`session_documents`)**: `doc_ids` là state tạm của frontend. Chuyển session cũ sẽ **không** khôi phục lại tập file đã dùng — scope reset về file đang mở. (Có thể nâng cấp sau bằng cách lưu scope vào `localStorage` theo `session_id`.)
- **`agentic_retrieve_context_multi` lặp lại vòng lặp của bản single-doc**: cố ý, để **không đụng** cấu trúc hàm retrieve single-doc hiện có. Có thể refactor gộp về sau nếu muốn.
- **Editor chỉ reload file đang mở** sau khi agent sửa: nếu agent sửa file khác trong scope, mở file đó để thấy thay đổi.
- **Migration `0003`**: gộp id session theo (doc_id, session_id) lịch sử; chạy một lần qua `schema_migrations`. Dữ liệu chat cũ được giữ lại, chỉ đánh số lại session toàn cục.
```
