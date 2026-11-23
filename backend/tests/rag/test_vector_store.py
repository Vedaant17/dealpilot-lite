import numpy as np
import pytest

from backend.app.rag.chunking import chunk_extraction
from backend.app.rag.extraction import extract_txt
from backend.app.rag.ingestion import ingest_document_bytes
from backend.app.rag.persistence import persist_document_chunks
from backend.app.rag.vector_store import VectorStore
from backend.app.storage.connection import get_connection, init_db
from backend.scripts.seed_db import seed_database


def _persist_one(conn, tmp_path):
    data = b"Hello RAG document.\nSecond line."
    result = ingest_document_bytes(
        conn, "aurora-retail", "note.txt", data, storage_root=tmp_path / "docs"
    )
    extraction = extract_txt(data)
    chunks = chunk_extraction(
        extraction, result.document_id, "aurora-retail", chunk_size=20, chunk_overlap=2
    )
    vectors = [np.array([1.0, 0.0], dtype=np.float32) for _ in chunks]
    persisted = persist_document_chunks(
        conn,
        result.document_id,
        "aurora-retail",
        chunks,
        vectors,
        embedding_model="m",
        embedding_dim=2,
    )
    return result.document_id, persisted["vector_ids"]


@pytest.fixture
def seeded_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    seed_database(db_path)
    return db_path, tmp_path


def test_rebuild_and_verify(seeded_db):
    db_path, tmp_path = seeded_db
    conn = get_connection(db_path)
    _, vector_ids = _persist_one(conn, tmp_path)
    expected_count = len(vector_ids)
    conn.close()

    index_path = tmp_path / "test.index"
    store = VectorStore(
        db_path=db_path, index_path=str(index_path), embedding_dim=2, embedding_model="m"
    )
    index = store.rebuild_index()
    assert index.ntotal == expected_count
    store.verify_index()
    assert index_path.is_file()
    reopened = store.load_index()
    assert reopened.ntotal == expected_count


def test_missing_index_rebuilds(seeded_db):
    db_path, tmp_path = seeded_db
    conn = get_connection(db_path)
    _, vector_ids = _persist_one(conn, tmp_path)
    expected_count = len(vector_ids)
    conn.close()

    index_path = tmp_path / "missing.index"
    store = VectorStore(
        db_path=db_path, index_path=str(index_path), embedding_dim=2, embedding_model="m"
    )
    index = store.ensure_index()
    assert index.ntotal == expected_count
    assert index_path.is_file()


def test_corrupt_index_rebuilds(seeded_db):
    db_path, tmp_path = seeded_db
    conn = get_connection(db_path)
    _, vector_ids = _persist_one(conn, tmp_path)
    expected_count = len(vector_ids)
    conn.close()

    index_path = tmp_path / "corrupt.index"
    index_path.write_bytes(b"not a faiss index")
    store = VectorStore(
        db_path=db_path, index_path=str(index_path), embedding_dim=2, embedding_model="m"
    )
    index = store.ensure_index()
    assert index.ntotal == expected_count
    assert index_path.is_file()


def test_dirty_state_cleared(seeded_db):
    db_path, tmp_path = seeded_db
    conn = get_connection(db_path)
    _persist_one(conn, tmp_path)
    state = conn.execute("SELECT index_dirty FROM rag_index_state WHERE id = 1").fetchone()
    assert state["index_dirty"] == 1
    conn.close()

    index_path = tmp_path / "dirty.index"
    store = VectorStore(
        db_path=db_path, index_path=str(index_path), embedding_dim=2, embedding_model="m"
    )
    store.rebuild_index()
    conn = get_connection(db_path)
    state = conn.execute("SELECT index_dirty FROM rag_index_state WHERE id = 1").fetchone()
    assert state["index_dirty"] == 0
    conn.close()


def test_empty_index(seeded_db):
    db_path, tmp_path = seeded_db
    index_path = tmp_path / "empty.index"
    store = VectorStore(
        db_path=db_path, index_path=str(index_path), embedding_dim=2, embedding_model="m"
    )
    index = store.rebuild_index()
    assert index.ntotal == 0
    assert index_path.is_file()
