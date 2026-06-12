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
from . import retrieval

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


def save_message(doc_id: str, session_id: int, role: str, content: str) -> int:
    """Save a single message to the display history."""
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO chat_messages (doc_id, session_id, role, content)
               VALUES (?, ?, ?, ?)""",
            (doc_id, session_id, role, content),
        )
        return int(cursor.lastrowid)


def save_message_evidence(message_id: int, doc_id: str, evidence: list[dict]) -> None:
    """Save retrieval-unit evidence used by an assistant message."""
    if not message_id or not evidence:
        return

    rows = []
    seen: set[str] = set()
    for item in evidence:
        unit_id = str(item.get("unit_id") or "")
        unit_hash = str(item.get("unit_hash") or "")
        if not unit_id or not unit_hash or unit_id in seen:
            continue
        seen.add(unit_id)
        rows.append(
            (
                message_id,
                doc_id,
                unit_id,
                unit_hash,
                str(item.get("title") or ""),
                str(item.get("kind") or "text"),
            )
        )

    if not rows:
        return

    with get_db() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO chat_message_evidence
               (message_id, doc_id, unit_id, unit_hash, title, kind)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )


def save_message_chain(message_id: int, doc_id: str, chain: str) -> None:
    """Save the graph-of-thought reasoning chain used by an assistant message."""
    if not message_id or not chain or not chain.strip():
        return
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO chat_message_chains (message_id, doc_id, chain)
               VALUES (?, ?, ?)""",
            (message_id, doc_id, chain),
        )


def _load_chains_by_message_id(doc_id: str, message_ids: list[int]) -> dict[int, str]:
    if not message_ids:
        return {}
    placeholders = ",".join("?" for _ in message_ids)
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT message_id, chain FROM chat_message_chains
                WHERE doc_id = ? AND message_id IN ({placeholders})""",
            [doc_id, *message_ids],
        ).fetchall()
    return {int(row[0]): row[1] for row in rows if row[1]}


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


def get_display_messages(doc_id: str) -> list[dict]:
    """Load all messages for the UI, enriched with citation evidence + chains.

    Each assistant message gains an ``evidence`` list (with ``heading_path`` and
    a ``preview`` snippet reconstructed from the *current* document units, so
    stale chunks are simply dropped) and a ``chain`` string when one was saved.
    """
    messages = get_messages(doc_id)
    assistant_ids = [int(m["id"]) for m in messages if m["role"] == "assistant"]
    evidence_by_message = _load_evidence_by_message_id(doc_id, assistant_ids)
    chains_by_message = _load_chains_by_message_id(doc_id, assistant_ids)

    # Reconstruct heading_path + preview from current units (one corpus build).
    units_by_id: dict[str, dict] = {}
    try:
        units_by_id = {
            str(u.get("unit_id") or ""): u
            for u in retrieval.get_retrieval_units(doc_id)
            if str(u.get("unit_id") or "")
        }
    except Exception:
        units_by_id = {}

    for msg in messages:
        if msg["role"] != "assistant":
            continue
        mid = int(msg["id"])
        enriched: list[dict] = []
        for item in evidence_by_message.get(mid, []):
            unit_id = str(item.get("unit_id") or "")
            unit = units_by_id.get(unit_id)
            enriched.append(
                {
                    "unit_id": unit_id,
                    "title": item.get("title") or (unit.get("title") if unit else "") or "",
                    "kind": item.get("kind") or "text",
                    "heading_path": retrieval._evidence_heading_path(unit) if unit else [],
                    "preview": retrieval._evidence_preview(unit) if unit else "",
                }
            )
        msg["evidence"] = enriched
        chain = chains_by_message.get(mid)
        if chain:
            msg["chain"] = chain
    return messages


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
            """SELECT id, role, content FROM chat_messages
               WHERE doc_id = ? AND session_id = ?
                 AND role IN ('user', 'assistant')
               ORDER BY created_at ASC""",
            (doc_id, session_id),
        ).fetchall()

    result: list[dict] = []
    if summary_row:
        # Inject as a synthetic assistant turn so the agent treats it as prior context.
        # Summaries are not chunk-level evidence and must not override fresh retrieval.
        result.append({"role": "user", "content": "Summarize our conversation so far."})
        result.append({
            "role": "assistant",
            "content": (
                f"{summary_row[0]}\n\n"
                "[UNVERIFIED HISTORY - conversation summary only; do not use as document evidence.]"
            ),
        })

    current_hashes = retrieval.get_current_unit_hashes(doc_id)
    evidence_by_message = _load_evidence_by_message_id(doc_id, [int(row[0]) for row in rows])
    for row in rows:
        message_id = int(row[0])
        role = row[1]
        content = row[2]
        if role == "assistant":
            content = _annotate_assistant_history(content, evidence_by_message.get(message_id, []), current_hashes)
        result.append({"role": role, "content": content})
    return result


def _load_evidence_by_message_id(doc_id: str, message_ids: list[int]) -> dict[int, list[dict]]:
    if not message_ids:
        return {}

    placeholders = ",".join("?" for _ in message_ids)
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT message_id, unit_id, unit_hash, title, kind
                FROM chat_message_evidence
                WHERE doc_id = ? AND message_id IN ({placeholders})
                ORDER BY message_id ASC, unit_id ASC""",
            [doc_id, *message_ids],
        ).fetchall()

    evidence_by_message: dict[int, list[dict]] = {}
    for row in rows:
        message_id = int(row[0])
        evidence_by_message.setdefault(message_id, []).append(
            {
                "unit_id": row[1],
                "unit_hash": row[2],
                "title": row[3],
                "kind": row[4],
            }
        )
    return evidence_by_message


def _annotate_assistant_history(
    content: str,
    evidence: list[dict],
    current_hashes: dict[str, str],
) -> str:
    if not evidence:
        return (
            f"{content}\n\n"
            "[UNVERIFIED HISTORY - this older answer has no chunk evidence metadata; "
            "do not use it as evidence for document facts.]"
        )

    stale_units = []
    for item in evidence:
        unit_id = str(item.get("unit_id") or "")
        saved_hash = str(item.get("unit_hash") or "")
        if current_hashes.get(unit_id) != saved_hash:
            stale_units.append(unit_id)

    if stale_units:
        return (
            f"{content}\n\n"
            "[STALE HISTORY - this answer referenced document chunks that have changed "
            f"or no longer exist: {', '.join(stale_units)}. Do not use it as evidence; "
            "call retrieve_context and trust the latest retrieved context.]"
        )

    valid_units = [str(item.get("unit_id") or "") for item in evidence if item.get("unit_id")]
    return (
        f"{content}\n\n"
        "[VALID HISTORY - referenced document chunks are unchanged "
        f"({', '.join(valid_units)}). You may use this for conversation continuity, "
        "but current retrieve_context output remains authoritative for document facts.]"
    )


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
