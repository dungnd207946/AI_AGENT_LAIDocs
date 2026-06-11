"""SQLite database layer for LAIDocs.
Stores document metadata, folder tree, and tree index JSON.
"""
from __future__ import annotations
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable
from .config import LAIDOCS_HOME

DB_PATH = LAIDOCS_HOME / "data" / "laidocs.db"

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    path TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    folder TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT,
    source_type TEXT NOT NULL CHECK(source_type IN ('file', 'url')),
    original_path TEXT,
    content TEXT,
    tree_index TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (folder) REFERENCES folders(path)
);
"""

# Migration for existing databases.
# IMPORTANT: append-only — never edit or reorder existing entries.
_MIGRATIONS = [
    "ALTER TABLE documents ADD COLUMN tree_index TEXT",
    """CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    session_id INTEGER NOT NULL DEFAULT 1,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""",
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_doc_id ON chat_messages(doc_id)",
    # Dense-retrieval vectors, one row per retrieval unit (tree node or chunk).
    # vector is raw float32 bytes; model identifies which embedding produced it
    # so a provider/model change can invalidate stale rows.
    """CREATE TABLE IF NOT EXISTS document_embeddings (
    doc_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    title TEXT,
    chunk TEXT NOT NULL,
    model TEXT NOT NULL,
    corpus_hash TEXT NOT NULL DEFAULT '',
    unit_hash TEXT NOT NULL DEFAULT '',
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (doc_id, unit_id)
)""",
    "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_doc ON document_embeddings(doc_id)",
    "ALTER TABLE document_embeddings ADD COLUMN corpus_hash TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE document_embeddings ADD COLUMN unit_hash TEXT NOT NULL DEFAULT ''",
    "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_doc_model_hash ON document_embeddings(doc_id, model, corpus_hash)",
    "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_doc_model_unit_hash ON document_embeddings(doc_id, model, unit_id, unit_hash)",
    """CREATE TABLE IF NOT EXISTS chat_message_evidence (
    message_id INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    unit_hash TEXT NOT NULL,
    title TEXT,
    kind TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (message_id, unit_id),
    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE
)""",
    "CREATE INDEX IF NOT EXISTS idx_chat_message_evidence_doc ON chat_message_evidence(doc_id)",
    # Graph-of-thought reasoning chain produced by reason_over_graph for an
    # assistant message, so the UI can re-render the chain on history reload.
    """CREATE TABLE IF NOT EXISTS chat_message_chains (
    message_id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chain TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE
)""",
    "CREATE INDEX IF NOT EXISTS idx_chat_message_chains_doc ON chat_message_chains(doc_id)",
    # Knowledge-graph triple cache, one row per retrieval unit (mirrors
    # document_embeddings). ``triples`` is a JSON list of [subject, relation,
    # object]; ``unit_hash`` + ``model`` invalidate stale rows when the unit's
    # content or the extractor LLM changes, so only changed units are re-extracted.
    """CREATE TABLE IF NOT EXISTS document_graph_units (
    doc_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    unit_hash TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    triples TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (doc_id, unit_id)
)""",
    "CREATE INDEX IF NOT EXISTS idx_doc_graph_units_doc_model ON document_graph_units(doc_id, model)",
]


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
                pass  # column already exists
        _migrate_chat_messages_schema(conn)
        _migrate_document_embeddings_schema(conn)
        _migrate_chat_message_evidence_schema(conn)


def _migrate_chat_messages_schema(conn: sqlite3.Connection) -> None:
    """Allow role='summary' rows for compacted chat history."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_messages'"
    ).fetchone()
    if not row:
        return

    table_sql = row[0] or ""
    if "'summary'" in table_sql:
        return

    conn.execute("ALTER TABLE chat_messages RENAME TO chat_messages_old")
    conn.execute(
        """CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    session_id INTEGER NOT NULL DEFAULT 1,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'summary')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""
    )
    conn.execute(
        """INSERT INTO chat_messages (id, doc_id, session_id, role, content, created_at)
           SELECT id, doc_id, session_id, role, content, created_at
           FROM chat_messages_old
           WHERE role IN ('user', 'assistant', 'summary')"""
    )
    conn.execute("DROP TABLE chat_messages_old")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_doc_id ON chat_messages(doc_id)")


def _migrate_document_embeddings_schema(conn: sqlite3.Connection) -> None:
    """Backfill document_embeddings schema additions for older databases."""
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='document_embeddings'"
    ).fetchone()
    if not table:
        return

    columns = {
        row[1]: row
        for row in conn.execute("PRAGMA table_info(document_embeddings)").fetchall()
    }
    if "corpus_hash" not in columns:
        conn.execute(
            "ALTER TABLE document_embeddings ADD COLUMN corpus_hash TEXT NOT NULL DEFAULT ''"
        )
    if "unit_hash" not in columns:
        conn.execute(
            "ALTER TABLE document_embeddings ADD COLUMN unit_hash TEXT NOT NULL DEFAULT ''"
        )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_doc_model_hash "
        "ON document_embeddings(doc_id, model, corpus_hash)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_doc_model_unit_hash "
        "ON document_embeddings(doc_id, model, unit_id, unit_hash)"
    )


def _migrate_chat_message_evidence_schema(conn: sqlite3.Connection) -> None:
    """Store which retrieval units an assistant answer used."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chat_message_evidence (
    message_id INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    unit_hash TEXT NOT NULL,
    title TEXT,
    kind TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (message_id, unit_id),
    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE
)"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_message_evidence_doc "
        "ON chat_message_evidence(doc_id)"
    )


# ---------------------------------------------------------------------------
# SQLite connection helpers
# ---------------------------------------------------------------------------


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection suitable for a single request."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def invalidate_document_embeddings(doc_id: str) -> None:
    """Delete all cached dense vectors for a document."""
    with get_db() as conn:
        conn.execute("DELETE FROM document_embeddings WHERE doc_id=?", (doc_id,))


def invalidate_documents_embeddings(doc_ids: Iterable[str]) -> None:
    """Delete cached dense vectors for multiple documents."""
    ids = [doc_id for doc_id in doc_ids if doc_id]
    if not ids:
        return

    placeholders = ",".join("?" for _ in ids)
    with get_db() as conn:
        conn.execute(
            f"DELETE FROM document_embeddings WHERE doc_id IN ({placeholders})",
            ids,
        )


def cleanup_orphan_embeddings() -> None:
    """Remove embedding rows whose document no longer exists."""
    with get_db() as conn:
        conn.execute(
            """DELETE FROM document_embeddings
               WHERE doc_id NOT IN (SELECT id FROM documents)"""
        )
