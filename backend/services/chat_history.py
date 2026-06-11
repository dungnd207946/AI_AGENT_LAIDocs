"""Chat history service for display-layer message persistence.

This stores ALL messages across ALL sessions for UI display purposes.
Separate from the agent's conversation memory (LangGraph checkpointer)
which only holds the current session.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
import unicodedata
from pathlib import Path

from ..core.config import LAIDOCS_HOME
from ..core.database import get_db

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

    content = (content or "").strip()
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
    """Load all messages for a document, ordered by creation time."""
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


def delete_messages(doc_id: str) -> None:
    """Delete all messages for a document."""
    with get_db() as conn:
        conn.execute("DELETE FROM chat_messages WHERE doc_id = ?", (doc_id,))
