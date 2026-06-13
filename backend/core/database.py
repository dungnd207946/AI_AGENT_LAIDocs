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
    # NOTE: the 'summary' role + nullable-doc_id table rebuild used to live here
    # as four destructive statements (CREATE _v2 / INSERT / DROP / RENAME). That
    # was a bug: _MIGRATIONS runs on EVERY startup, so on each restart it
    # recreated chat_messages_v2 (with doc_id NOT NULL) and renamed it over the
    # nullable table produced by guarded migration 0003 — reverting the schema
    # and breaking save_message() (which inserts without doc_id). The rebuild
    # now lives entirely in _GUARDED_MIGRATIONS (0003 + 0004), which run once.
]

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
    (
        # Repair DBs corrupted by the old destructive _MIGRATIONS rebuild loop:
        # 0003 had already run (and is marked applied, so it never re-runs), yet
        # a later restart reverted doc_id back to NOT NULL. Rebuild once more
        # with doc_id nullable, preserving rows and ids. Idempotent for healthy
        # DBs (they are simply rebuilt to the identical schema).
        "0004_repair_doc_id_nullable",
        [
            """CREATE TABLE chat_messages_v4 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                session_id INTEGER NOT NULL DEFAULT 1,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'summary')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """INSERT INTO chat_messages_v4 (id, doc_id, session_id, role, content, created_at)
               SELECT id, doc_id, session_id, role, content, created_at FROM chat_messages""",
            "DROP TABLE chat_messages",
            "ALTER TABLE chat_messages_v4 RENAME TO chat_messages",
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)",
        ],
    ),
]


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
        _repair_dangling_chat_fk(conn)
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


def _repair_dangling_chat_fk(conn: sqlite3.Connection) -> None:
    """Rebuild chat_message_* tables whose FK was rewritten to chat_messages_old.

    A historical RENAME-based migration of chat_messages (run with the default
    legacy_alter_table=OFF) silently rewrote the foreign keys of dependent
    tables to point at chat_messages_old, which was then dropped — leaving a
    dangling reference that makes every INSERT raise
    'no such table: main.chat_messages_old'. Recreate the affected table with
    the FK pointing back at chat_messages, preserving existing rows.
    """
    targets = {
        "chat_message_evidence": """CREATE TABLE chat_message_evidence (
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
        "chat_message_chains": """CREATE TABLE chat_message_chains (
    message_id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chain TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE
)""",
    }

    needs_repair = []
    for name in targets:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        if row and "chat_messages_old" in (row[0] or ""):
            needs_repair.append(name)
    if not needs_repair:
        return

    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA legacy_alter_table=ON")
    try:
        for name in needs_repair:
            cols = [
                r[1] for r in conn.execute(f"PRAGMA table_info({name})").fetchall()
            ]
            col_list = ", ".join(cols)
            conn.execute(f"ALTER TABLE {name} RENAME TO {name}_fix_old")
            conn.execute(targets[name])
            conn.execute(
                f"INSERT INTO {name} ({col_list}) SELECT {col_list} FROM {name}_fix_old"
            )
            conn.execute(f"DROP TABLE {name}_fix_old")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_message_evidence_doc "
            "ON chat_message_evidence(doc_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_message_chains_doc "
            "ON chat_message_chains(doc_id)"
        )
    finally:
        conn.execute("PRAGMA legacy_alter_table=OFF")
        conn.execute("PRAGMA foreign_keys=ON")


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
