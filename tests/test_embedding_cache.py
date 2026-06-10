"""Regression tests for dense embedding cache invalidation and cleanup."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

# Isolate ~/.laidocs before importing backend modules.
_TMP = tempfile.mkdtemp(prefix="laidocs-embeddings-")
os.environ["HOME"] = _TMP
os.environ["USERPROFILE"] = _TMP

from fastapi import BackgroundTasks  # noqa: E402

from backend.api import documents as documents_api  # noqa: E402
from backend.api import folders as folders_api  # noqa: E402
from backend.api import settings as settings_api  # noqa: E402
from backend.core import config as core_config  # noqa: E402
from backend.core import database as core_database  # noqa: E402
from backend.core import vault as core_vault  # noqa: E402
from backend.core.config import LLMConfig, Settings  # noqa: E402
from backend.core.database import DB_PATH, get_db, init_db  # noqa: E402
from backend.core.vault import VAULT_DIR, vault  # noqa: E402
from backend.services import backup as backup_service  # noqa: E402
from backend.services import retrieval  # noqa: E402


def _repoint_state_root() -> None:
    global DB_PATH, VAULT_DIR

    home = Path(tempfile.mkdtemp(prefix="laidocs-embeddings-home-")) / ".laidocs"
    core_config.LAIDOCS_HOME = home
    core_config.CONFIG_PATH = home / "config.json"

    core_database.DB_PATH = home / "data" / "laidocs.db"
    core_vault.VAULT_DIR = home / "vault"
    core_vault.ASSETS_DIR = core_vault.VAULT_DIR / "assets"

    backup_service.LAIDOCS_HOME = home
    backup_service.DB_PATH = core_database.DB_PATH
    backup_service.VAULT_DIR = core_vault.VAULT_DIR

    documents_api.ASSETS_DIR = core_vault.ASSETS_DIR

    DB_PATH = core_database.DB_PATH
    VAULT_DIR = core_vault.VAULT_DIR


def _reset_state() -> None:
    _repoint_state_root()
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    (VAULT_DIR / "unsorted").mkdir(parents=True, exist_ok=True)
    init_db()
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO folders (path, name) VALUES (?, ?)",
            ("unsorted", "unsorted"),
        )


def _settings() -> Settings:
    return Settings(
        llm=LLMConfig(
            provider="openai",
            base_url="http://example.invalid/v1",
            api_key="test-key",
            model="gpt-test",
            embed_model="embed-test",
        )
    )


def _insert_document(
    doc_id: str,
    *,
    content: str,
    tree: dict | None = None,
    folder: str = "unsorted",
    filename: str = "doc.md",
    title: str = "Doc",
) -> None:
    meta = vault.save_document(
        folder=folder,
        filename=filename,
        content=content,
        title=title,
        source_type="file",
        original_path=filename,
        doc_id=doc_id,
    )
    with get_db() as conn:
        parts = Path(folder).parts
        for idx in range(1, len(parts) + 1):
            folder_path = str(Path(*parts[:idx]))
            conn.execute(
                "INSERT OR IGNORE INTO folders (path, name) VALUES (?, ?)",
                (folder_path, Path(folder_path).name or folder_path),
            )
        conn.execute(
            """INSERT OR REPLACE INTO documents
               (id, folder, filename, title, source_type, original_path, content, tree_index)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                meta.folder,
                meta.filename,
                meta.title,
                meta.source_type,
                meta.original_path,
                content,
                json.dumps(tree, ensure_ascii=False) if tree is not None else None,
            ),
        )


def _insert_embedding(
    doc_id: str,
    unit_id: str,
    *,
    model: str = "embed-test",
    corpus_hash: str = "",
    unit_hash: str = "",
    vector: bytes | None = None,
) -> None:
    vector = vector or sqlite3.Binary(b"\x00\x00\x80?\x00\x00\x00\x00")
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO document_embeddings
               (doc_id, unit_id, title, chunk, model, corpus_hash, unit_hash, dim, vector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, unit_id, unit_id, unit_id, model, corpus_hash, unit_hash, 2, vector),
        )


def _row_count(table: str) -> int:
    with get_db() as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


class _DummyEmbedder:
    def __init__(self, *, on_embed=None):
        self._on_embed = on_embed

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._on_embed is not None:
            self._on_embed(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, question: str) -> list[float]:
        return [1.0, 0.0]


def _patch_embeddings(monkeypatch, *, on_embed=None) -> None:
    monkeypatch.setattr(retrieval, "embeddings_supported", lambda cfg: True)
    monkeypatch.setattr(retrieval, "embed_model_name", lambda cfg: "embed-test")
    monkeypatch.setattr(
        retrieval,
        "create_embeddings",
        lambda cfg: _DummyEmbedder(on_embed=on_embed),
    )


def _build_backup_zip(target: Path, *, docs: list[dict], embeddings: list[dict]) -> None:
    tmp_db_dir = Path(tempfile.mkdtemp(prefix="laidocs-backup-db-"))
    try:
        backup_db = tmp_db_dir / "laidocs.db"
        conn = sqlite3.connect(str(backup_db))
        try:
            conn.executescript(
                """
                CREATE TABLE folders (
                    path TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    parent_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    folder TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    title TEXT,
                    source_type TEXT NOT NULL,
                    original_path TEXT,
                    content TEXT,
                    tree_index TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE document_embeddings (
                    doc_id TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    title TEXT,
                    chunk TEXT NOT NULL,
                    model TEXT NOT NULL,
                    corpus_hash TEXT NOT NULL DEFAULT '',
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (doc_id, unit_id)
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO folders (path, name) VALUES (?, ?)",
                ("unsorted", "unsorted"),
            )
            for doc in docs:
                conn.execute(
                    """INSERT INTO documents
                       (id, folder, filename, title, source_type, original_path, content, tree_index)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        doc["id"],
                        doc["folder"],
                        doc["filename"],
                        doc["title"],
                        "file",
                        doc["filename"],
                        doc["content"],
                        doc.get("tree_index"),
                    ),
                )
            for emb in embeddings:
                conn.execute(
                    """INSERT INTO document_embeddings
                       (doc_id, unit_id, title, chunk, model, corpus_hash, dim, vector)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        emb["doc_id"],
                        emb["unit_id"],
                        emb["unit_id"],
                        emb["unit_id"],
                        emb.get("model", "embed-test"),
                        emb.get("corpus_hash", ""),
                        2,
                        sqlite3.Binary(b"\x00\x00\x80?\x00\x00\x00\x00"),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "format_version": 1,
                        "app_version": "1.0.0",
                        "stats": {"documents": len(docs)},
                    }
                ),
            )
            for doc in docs:
                folder = doc["folder"]
                zf.writestr(f"vault/{folder}/{doc['filename']}", doc["content"])
                zf.writestr(
                    f"vault/{folder}/{doc['filename']}.meta.json",
                    json.dumps(
                        {
                            "doc_id": doc["id"],
                            "folder": folder,
                            "filename": doc["filename"],
                            "title": doc["title"],
                            "source_type": "file",
                            "original_path": doc["filename"],
                        }
                    ),
                )
            zf.write(backup_db, "data/laidocs.db")
    finally:
        shutil.rmtree(tmp_db_dir, ignore_errors=True)


def test_ensure_embedding_index_rebuilds_blank_hash_and_replaces_fallback_units(monkeypatch):
    _reset_state()
    doc_id = "doc-stale"
    content = "# Heading\n\nCurrent body"
    tree = {"structure": [{"node_id": "0001", "title": "Heading", "text": content, "nodes": []}]}
    _insert_document(doc_id, content=content, tree=tree)
    _insert_embedding(doc_id, "c0001", corpus_hash="")
    _patch_embeddings(monkeypatch)

    assert retrieval.ensure_embedding_index(doc_id, _settings()) is True

    with get_db() as conn:
        rows = conn.execute(
            "SELECT unit_id, model, unit_hash FROM document_embeddings WHERE doc_id=?",
            (doc_id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["unit_id"] == "0001"
    assert rows[0]["model"] == "embed-test"
    assert rows[0]["unit_hash"]


def test_dense_search_reads_only_current_unit_hash(monkeypatch):
    _reset_state()
    doc_id = "doc-dense"
    content = "# Intro\n\nAlpha"
    tree = {"structure": [{"node_id": "0001", "title": "Intro", "text": content, "nodes": []}]}
    _insert_document(doc_id, content=content, tree=tree)
    _content, units, _corpus_hash = retrieval._get_current_corpus(doc_id)
    current_hash = retrieval._compute_unit_hash(units[0])
    _insert_embedding(doc_id, "0001", unit_hash=current_hash)
    _insert_embedding(doc_id, "9999", unit_hash="old-hash")

    _patch_embeddings(monkeypatch)
    monkeypatch.setattr(retrieval, "ensure_embedding_index", lambda *args, **kwargs: True)

    ranked = retrieval.dense_search(doc_id, "alpha", _settings())
    assert ranked == ["0001"]


def test_ensure_embedding_index_reembeds_only_changed_unit(monkeypatch):
    _reset_state()
    doc_id = "doc-partial"
    tree = {
        "structure": [
            {"node_id": "0001", "title": "One", "text": "Stable body", "nodes": []},
            {"node_id": "0002", "title": "Two", "text": "Old body", "nodes": []},
        ]
    }
    _insert_document(doc_id, content="# One\n\nStable body\n\n# Two\n\nOld body", tree=tree)

    embedded_batches: list[list[str]] = []
    _patch_embeddings(monkeypatch, on_embed=lambda texts: embedded_batches.append(texts))

    assert retrieval.ensure_embedding_index(doc_id, _settings()) is True
    assert len(embedded_batches[-1]) == 2

    changed_tree = {
        "structure": [
            {"node_id": "0001", "title": "One", "text": "Stable body", "nodes": []},
            {"node_id": "0002", "title": "Two", "text": "New body", "nodes": []},
        ]
    }
    with get_db() as conn:
        conn.execute(
            "UPDATE documents SET content=?, tree_index=? WHERE id=?",
            (
                "# One\n\nStable body\n\n# Two\n\nNew body",
                json.dumps(changed_tree, ensure_ascii=False),
                doc_id,
            ),
        )

    assert retrieval.ensure_embedding_index(doc_id, _settings()) is True
    assert embedded_batches[-1] == ["Two\nNew body"]

    with get_db() as conn:
        rows = conn.execute(
            "SELECT unit_id, unit_hash FROM document_embeddings WHERE doc_id=? ORDER BY unit_id",
            (doc_id,),
        ).fetchall()
    assert [row["unit_id"] for row in rows] == ["0001", "0002"]
    assert all(row["unit_hash"] for row in rows)


def test_race_rebuild_drops_old_vectors_when_corpus_changes_during_embedding(monkeypatch):
    _reset_state()
    doc_id = "doc-race"
    _insert_document(doc_id, content="Old content", tree=None)

    def _mutate_corpus(_texts: list[str]) -> None:
        with get_db() as conn:
            conn.execute(
                "UPDATE documents SET content=?, tree_index=? WHERE id=?",
                (
                    "# New heading\n\nFresh body",
                    json.dumps(
                        {
                            "structure": [
                                {
                                    "node_id": "0001",
                                    "title": "New heading",
                                    "text": "# New heading\n\nFresh body",
                                    "nodes": [],
                                }
                            ]
                        }
                    ),
                    doc_id,
                ),
            )

    _patch_embeddings(monkeypatch, on_embed=_mutate_corpus)

    assert retrieval.ensure_embedding_index(doc_id, _settings()) is False
    assert _row_count("document_embeddings") == 0


def test_delete_document_removes_embeddings():
    _reset_state()
    doc_id = "doc-delete"
    _insert_document(doc_id, content="Delete me")
    _insert_embedding(doc_id, "c0001", corpus_hash="hash")

    asyncio.run(documents_api.delete_document(doc_id))

    with get_db() as conn:
        doc = conn.execute("SELECT id FROM documents WHERE id=?", (doc_id,)).fetchone()
        emb = conn.execute(
            "SELECT unit_id FROM document_embeddings WHERE doc_id=?",
            (doc_id,),
        ).fetchone()
    assert doc is None
    assert emb is None


def test_delete_folder_subtree_removes_embeddings():
    _reset_state()
    folder = "team/sub"
    _insert_document("doc-a", content="A", folder=folder, filename="a.md", title="A")
    _insert_document("doc-b", content="B", folder="team/sub/child", filename="b.md", title="B")
    _insert_embedding("doc-a", "c0001", corpus_hash="hash-a")
    _insert_embedding("doc-b", "c0001", corpus_hash="hash-b")
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO folders (path, name) VALUES (?, ?)", ("team", "team"))
        conn.execute("INSERT OR IGNORE INTO folders (path, name) VALUES (?, ?)", ("team/sub", "sub"))
        conn.execute("INSERT OR IGNORE INTO folders (path, name) VALUES (?, ?)", ("team/sub/child", "child"))

    folders_api.delete_folder("team")

    assert _row_count("documents") == 0
    assert _row_count("document_embeddings") == 0
    assert not (VAULT_DIR / "team").exists()


def test_backup_replace_clears_restored_embeddings(tmp_path):
    _reset_state()
    backup_path = tmp_path / "replace.laidocs-backup"
    _build_backup_zip(
        backup_path,
        docs=[
            {
                "id": "doc-backup",
                "folder": "unsorted",
                "filename": "backup.md",
                "title": "Backup",
                "content": "# Backup\n\nBody",
            }
        ],
        embeddings=[{"doc_id": "doc-backup", "unit_id": "0001", "corpus_hash": "stale"}],
    )

    result = backup_service.import_backup(str(backup_path), "replace")

    assert result["success"] is True
    assert _row_count("documents") == 1
    assert _row_count("document_embeddings") == 0


def test_backup_merge_invalidates_only_new_docs(tmp_path):
    _reset_state()
    _insert_document("keep-doc", content="Keep local", filename="keep.md", title="Keep")
    _insert_embedding("keep-doc", "c0001", corpus_hash="keep-hash")

    backup_path = tmp_path / "merge.laidocs-backup"
    _build_backup_zip(
        backup_path,
        docs=[
            {
                "id": "keep-doc",
                "folder": "unsorted",
                "filename": "keep.md",
                "title": "Keep from backup",
                "content": "Should be skipped",
            },
            {
                "id": "new-doc",
                "folder": "unsorted",
                "filename": "new.md",
                "title": "New from backup",
                "content": "# Imported\n\nFresh",
            },
        ],
        embeddings=[
            {"doc_id": "keep-doc", "unit_id": "c0001", "corpus_hash": "backup-keep"},
            {"doc_id": "new-doc", "unit_id": "0001", "corpus_hash": "backup-new"},
        ],
    )

    result = backup_service.import_backup(str(backup_path), "merge")

    assert result["success"] is True
    with get_db() as conn:
        keep_rows = conn.execute(
            "SELECT unit_id FROM document_embeddings WHERE doc_id=?",
            ("keep-doc",),
        ).fetchall()
        new_rows = conn.execute(
            "SELECT unit_id FROM document_embeddings WHERE doc_id=?",
            ("new-doc",),
        ).fetchall()
        new_doc = conn.execute(
            "SELECT id FROM documents WHERE id=?",
            ("new-doc",),
        ).fetchone()
    assert len(keep_rows) == 1
    assert new_rows == []
    assert new_doc is not None


def test_update_document_does_not_delete_embeddings_on_content_change():
    _reset_state()
    doc_id = "doc-update"
    _insert_document(doc_id, content="Same body", filename="same.md", title="Old title")
    _insert_embedding(doc_id, "c0001", corpus_hash="hash")

    asyncio.run(
        documents_api.update_document(
            doc_id,
            {"title": "New title", "filename": "same.md"},
            BackgroundTasks(),
        )
    )
    assert _row_count("document_embeddings") == 1

    asyncio.run(
        documents_api.update_document(
            doc_id,
            {"title": "Newest title", "filename": "same.md", "content": "Changed body"},
            BackgroundTasks(),
        )
    )
    assert _row_count("document_embeddings") == 1


def test_hybrid_rank_keeps_rrf_order_when_reranker_disabled(monkeypatch):
    _reset_state()
    settings = _settings()
    settings.reranker.enabled = False

    monkeypatch.setattr(retrieval, "get_retrieval_units", lambda _doc_id: [
        {"unit_id": "a", "title": "A", "text": "Alpha", "kind": "text", "heading_path": [], "path": ""},
        {"unit_id": "b", "title": "B", "text": "Beta", "kind": "text", "heading_path": [], "path": ""},
    ])
    monkeypatch.setattr(retrieval, "get_tree_index", lambda _doc_id: None)
    monkeypatch.setattr(retrieval, "bm25_search", lambda *args, **kwargs: ["a", "b"])
    monkeypatch.setattr(retrieval, "dense_search", lambda *args, **kwargs: ["b", "a"])
    monkeypatch.setattr(retrieval, "rerank_units", lambda *args, **kwargs: ["b", "a"])

    fused, tree_selected = retrieval.hybrid_rank("doc", "question", settings)
    assert fused == ["a", "b"]
    assert tree_selected is None


def test_hybrid_rank_reranks_rrf_candidates(monkeypatch):
    _reset_state()
    settings = _settings()
    settings.reranker.enabled = True
    settings.reranker.candidate_k = 3
    settings.reranker.top_n = 2

    monkeypatch.setattr(retrieval, "get_retrieval_units", lambda _doc_id: [
        {"unit_id": "a", "title": "A", "text": "Alpha", "kind": "text", "heading_path": [], "path": ""},
        {"unit_id": "b", "title": "B", "text": "Beta", "kind": "text", "heading_path": [], "path": ""},
        {"unit_id": "c", "title": "C", "text": "Gamma", "kind": "text", "heading_path": [], "path": ""},
    ])
    monkeypatch.setattr(retrieval, "get_tree_index", lambda _doc_id: None)
    monkeypatch.setattr(retrieval, "bm25_search", lambda *args, **kwargs: ["a", "b", "c"])
    monkeypatch.setattr(retrieval, "dense_search", lambda *args, **kwargs: ["b", "c", "a"])

    captured = {}

    def _rerank(question, candidate_units, _settings):
        captured["question"] = question
        captured["unit_ids"] = [unit["unit_id"] for unit in candidate_units]
        return ["c", "b", "a"]

    monkeypatch.setattr(retrieval, "rerank_units", _rerank)

    fused, _tree_selected = retrieval.hybrid_rank("doc", "question", settings)
    assert captured["question"] == "question"
    assert captured["unit_ids"] == ["b", "a", "c"]
    assert fused == ["c", "b"]


def test_hybrid_rank_falls_back_to_rrf_on_reranker_error(monkeypatch):
    _reset_state()
    settings = _settings()
    settings.reranker.enabled = True
    settings.reranker.candidate_k = 3

    monkeypatch.setattr(retrieval, "get_retrieval_units", lambda _doc_id: [
        {"unit_id": "a", "title": "A", "text": "Alpha", "kind": "text", "heading_path": [], "path": ""},
        {"unit_id": "b", "title": "B", "text": "Beta", "kind": "text", "heading_path": [], "path": ""},
    ])
    monkeypatch.setattr(retrieval, "get_tree_index", lambda _doc_id: None)
    monkeypatch.setattr(retrieval, "bm25_search", lambda *args, **kwargs: ["a", "b"])
    monkeypatch.setattr(retrieval, "dense_search", lambda *args, **kwargs: ["b", "a"])
    monkeypatch.setattr(retrieval, "rerank_units", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")))

    fused, _tree_selected = retrieval.hybrid_rank("doc", "question", settings)
    assert fused == ["a", "b"]


def test_settings_api_masks_and_updates_reranker():
    _reset_state()
    settings = core_config.get_settings()
    settings.reranker.api_key = "secret-reranker-key"
    settings.save_to_file()
    core_config.reload_settings()

    masked = asyncio.run(settings_api.read_settings())
    assert masked.reranker["api_key"] == "secr***"

    updated = asyncio.run(
        settings_api.update_settings(
            settings_api._SettingsUpdate(
                reranker={
                    "enabled": True,
                    "base_url": "https://example.test/rerank",
                    "model": "rerank-test",
                    "api_key": "abcd1234",
                    "top_n": 5,
                    "candidate_k": 12,
                    "timeout_s": 9.5,
                }
            )
        )
    )
    assert updated.reranker["api_key"] == "abcd***"

    fresh = core_config.get_settings()
    assert fresh.reranker.enabled is True
    assert fresh.reranker.base_url == "https://example.test/rerank"
    assert fresh.reranker.model == "rerank-test"
    assert fresh.reranker.top_n == 5
    assert fresh.reranker.candidate_k == 12
    assert fresh.reranker.timeout_s == 9.5


def test_test_reranker_endpoint_reports_success_and_error(monkeypatch):
    _reset_state()

    class _AsyncResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _AsyncClientOK:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            assert url == "https://example.test/rerank"
            assert json["model"] == "rerank-test"
            assert json["query"]
            assert headers["Authorization"] == "Bearer secret"
            return _AsyncResponse({"results": [{"index": 1, "relevance_score": 0.9}]})

    monkeypatch.setattr(settings_api.httpx, "AsyncClient", _AsyncClientOK)
    success = asyncio.run(
        settings_api.test_reranker(
            settings_api._TestRerankerRequest(
                base_url="https://example.test/rerank",
                api_key="secret",
                model="rerank-test",
                documents=["a", "b"],
            )
        )
    )
    assert success["success"] is True
    assert success["results"][0]["index"] == 1

    class _AsyncClientBad:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            return _AsyncResponse({"unexpected": []})

    monkeypatch.setattr(settings_api.httpx, "AsyncClient", _AsyncClientBad)
    failure = asyncio.run(
        settings_api.test_reranker(
            settings_api._TestRerankerRequest(
                base_url="https://example.test/rerank",
                api_key="secret",
                model="rerank-test",
                documents=["a", "b"],
            )
        )
    )
    assert failure["success"] is False
