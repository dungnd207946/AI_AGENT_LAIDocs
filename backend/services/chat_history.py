"""Chat history service for display-layer message persistence.

Stores ALL messages across ALL sessions for UI display purposes.
Separate from the agent's conversation memory (LangGraph MemorySaver
which is intentionally in-memory only — see docs/plans/e2e-issues.md).

Context injection:
  - get_messages_for_session() — load a session's messages to inject into
    the agent on resume, replacing the lost MemorySaver state.

Compact support:
  - Rows with role='summary' are synthetic summaries produced by the compactor.
  - get_messages()             — UI display (all rows including summaries)
  - get_messages_for_compact() — compactor input (latest summary + messages after it)
  - save_compact_summary()     — insert a summary row
  - delete_compacted_messages() — remove rows that were folded into a summary
"""

from __future__ import annotations

from ..core.database import get_db

from datetime import datetime, timezone
import re
import unicodedata
from pathlib import Path

from ..core.config import LAIDOCS_HOME

DOWNLOADS_DIR = LAIDOCS_HOME / "downloads"


def _ensure_downloads_dir() -> Path:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return DOWNLOADS_DIR


def _sanitize_export_filename(filename: str) -> str:
    filename = Path(filename).name
    if filename.lower().endswith(".md"):
        stem = filename[:-3]
    else:
        stem = filename
    stem = stem.strip()
    stem = re.sub(r"[\\/:*?\"<>|]+", "-", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        stem = "report"
    return f"{stem}.md"


def create_markdown_export(doc_id: str, filename: str | None = None, content: str | None = None) -> Path:
    """Build a Markdown file from the latest assistant reply or provided content."""
    if content is None:
        messages = get_messages(doc_id)
        if not messages:
            raise ValueError(f"No chat history found for doc_id {doc_id}")

        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        if not assistant_messages:
            raise ValueError(f"No assistant reply found for doc_id {doc_id}")

        content = assistant_messages[-1].get("content", "")

    if not content:
        raise ValueError("Cannot export empty content.")

    if filename:
        filename = _sanitize_export_filename(filename)
    else:
        filename = f"report-{doc_id}.md"

    export_dir = _ensure_downloads_dir()
    export_path = export_dir / filename
    base_name = export_path.stem
    suffix = export_path.suffix
    counter = 1
    while export_path.exists():
        export_path = export_dir / f"{base_name}-{counter}{suffix}"
        counter += 1

    export_path.write_text(_build_export_content(doc_id, content), encoding="utf-8")
    return export_path


def _build_export_content(doc_id: str, assistant_content: str) -> str:
    lines = [
        f"# Chat report for document {doc_id}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Assistant",
        "",
        assistant_content.rstrip(),
        "",
    ]
    return "\n".join(lines)

def get_current_session_id(doc_id: str) -> int:
    """Get the current (latest) session ID for a document."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT MAX(session_id) FROM chat_messages WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
    return row[0] if row and row[0] else 1


def start_new_session(doc_id: str) -> int:
    """Increment session counter and return the new session ID."""
    current = get_current_session_id(doc_id)
    return current + 1


def save_message(doc_id: str, session_id: int, role: str, content: str) -> None:
    """Save a single message to the display history."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO chat_messages (doc_id, session_id, role, content)
               VALUES (?, ?, ?, ?)""",
            (doc_id, session_id, role, content),
        )


def get_messages(doc_id: str) -> list[dict]:
    """Load all messages for a document, ordered by creation time.

    Includes summary rows (role='summary') so the UI can show them.
    """
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, session_id, role, content, created_at
               FROM chat_messages
               WHERE doc_id = ?
               ORDER BY created_at ASC""",
            (doc_id,),
        ).fetchall()
    return [
        {
            "id": row[0],
            "session_id": row[1],
            "role": row[2],
            "content": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]


def get_messages_for_session(doc_id: str, session_id: int) -> list[dict]:
    """Load messages for agent context injection on session resume.

    Returns 'user' and 'assistant' rows for this session only.
    If a compact summary exists for this doc, it is prepended as a synthetic
    user/assistant exchange so the agent has prior context without seeing the
    raw role='summary' marker (which LangGraph does not understand).
    """
    with get_db() as conn:
        # Latest compact summary across all sessions
        summary_row = conn.execute(
            """SELECT content FROM chat_messages
               WHERE doc_id = ? AND role = 'summary'
               ORDER BY created_at DESC LIMIT 1""",
            (doc_id,),
        ).fetchone()

        rows = conn.execute(
            """SELECT role, content FROM chat_messages
               WHERE doc_id = ? AND session_id = ?
                 AND role IN ('user', 'assistant')
               ORDER BY created_at ASC""",
            (doc_id, session_id),
        ).fetchall()

    result: list[dict] = []
    if summary_row:
        # Inject as a synthetic assistant turn so the agent treats it as prior context
        result.append({"role": "user", "content": "Summarize our conversation so far."})
        result.append({"role": "assistant", "content": summary_row[0]})
    result += [{"role": row[0], "content": row[1]} for row in rows]
    return result


def get_messages_for_compact(doc_id: str) -> list[dict]:
    """Load messages for the compactor.

    Returns the latest summary row (if any) followed by all subsequent
    regular messages, so the compactor always works on a flat sequence
    without re-processing already-summarised content.
    """
    with get_db() as conn:
        # Find the most recent summary row
        summary_row = conn.execute(
            """SELECT id, session_id, role, content, created_at
               FROM chat_messages
               WHERE doc_id = ? AND role = 'summary'
               ORDER BY created_at DESC
               LIMIT 1""",
            (doc_id,),
        ).fetchone()

        if summary_row:
            # Messages AFTER the last summary (regular chat only)
            rows = conn.execute(
                """SELECT id, session_id, role, content, created_at
                   FROM chat_messages
                   WHERE doc_id = ?
                     AND role IN ('user', 'assistant')
                     AND created_at > ?
                   ORDER BY created_at ASC""",
                (doc_id, summary_row[4]),
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
                """SELECT id, session_id, role, content, created_at
                   FROM chat_messages
                   WHERE doc_id = ? AND role IN ('user', 'assistant')
                   ORDER BY created_at ASC""",
                (doc_id,),
            ).fetchall()
            result = []

    result += [
        {
            "id": row[0],
            "session_id": row[1],
            "role": row[2],
            "content": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]
    return result


def save_compact_summary(doc_id: str, session_id: int, summary: str) -> None:
    """Insert a compacted summary row (role='summary')."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO chat_messages (doc_id, session_id, role, content)
               VALUES (?, ?, 'summary', ?)""",
            (doc_id, session_id, summary),
        )


def delete_compacted_messages(doc_id: str, message_ids: list[int]) -> None:
    """Delete rows that have been folded into a summary."""
    if not message_ids:
        return
    placeholders = ",".join("?" * len(message_ids))
    with get_db() as conn:
        conn.execute(
            f"DELETE FROM chat_messages WHERE doc_id = ? AND id IN ({placeholders})",
            [doc_id, *message_ids],
        )


def delete_messages(doc_id: str) -> None:
    """Delete all messages for a document (including summaries)."""
    with get_db() as conn:
        conn.execute("DELETE FROM chat_messages WHERE doc_id = ?", (doc_id,))

def delete_session(doc_id: str, session_id: int) -> None:
    """Delete all messages belonging to a single session of a document."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM chat_messages WHERE doc_id = ? AND session_id = ?",
            (doc_id, session_id),
        )