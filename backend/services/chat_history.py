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
