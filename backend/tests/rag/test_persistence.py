import numpy as np
import pytest

from backend.app.rag.chunking import chunk_extraction
from backend.app.rag.extraction import extract_txt
from backend.app.rag.ingestion import ingest_document, ingest_document_bytes
from backend.app.rag.persistence import (
    deserialize_vector,
    get_active_embeddings,
    persist_document_chunks,
    reserve_vector_ids,
    serialize_vector,
)
from backend.app.storage.connection import get_connection, init_db
from backend.scripts.seed_db import seed_database


@pytest.fixture
def seeded_conn(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    seed_database(db_path)
    conn = get_connection(db_path)
    yield conn, tmp_path
    conn.close()


def _ready_document(conn, tmp_path):
    data = b"Hello RAG document.\nSecond line."
    result = ingest_document_bytes(
        conn, "aurora-retail", "note.txt", data, storage_root=tmp_path / "docs"
    )
    assert result.ingestion_status == "ready"
    return result.document_id


def test_reserve_vector_ids_monotonic(seeded_conn):
    conn, _ = seeded_conn
    first = reserve_vector_ids(conn, 2)
    second = reserve_vector_ids(conn, 1)
    assert first == [1, 2]
    assert second == [3]
    # Rolled-back reservation must not be reused.
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE embedding_sequences SET next_value = next_value + 1 WHERE sequence_name = 'vector_id'"
        )
        conn.rollback()
    except Exception:
        conn.rollback()
    third = reserve_vector_ids(conn, 1)
    assert third == [4]


def test_persist_and_round_trip(seeded_conn):
    conn, tmp_path = seeded_conn
    document_id = _ready_document(conn, tmp_path)
    extraction = extract_txt(b"Hello RAG document.\nSecond line.")
    chunks = chunk_extraction(
        extraction, document_id, "aurora-retail", chunk_size=20, chunk_overlap=2
    )
    vectors = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
    # Normalize already; vectors are unit length.
    result = persist_document_chunks(
        conn,
        document_id,
        "aurora-retail",
        chunks,
        vectors,
        embedding_model="text-embedding-3-small",
        embedding_dim=2,
    )
    assert result["chunk_count"] == len(chunks)
    assert result["vector_ids"] == [1, 2]

    active = get_active_embeddings(conn, embedding_model="text-embedding-3-small", embedding_dim=2)
    assert len(active) == len(chunks)
    assert active[0].vector.dtype == np.float32
    assert active[0].vector_id == 1


def test_serialization_validation():
    vector = np.array([3.0, 4.0], dtype=np.float32)
    normalized = (vector / np.linalg.norm(vector)).astype(np.float32)
    blob = serialize_vector(normalized, 2)
    restored = deserialize_vector(blob, 2)
    assert restored.shape == (2,)
    assert abs(np.linalg.norm(restored) - 1.0) < 1e-6

    try:
        serialize_vector(np.array([1.0, 2.0], dtype=np.float32), 2)
        raise AssertionError("expected validation error")
    except ValueError:
        pass

    try:
        deserialize_vector(b"\x00" * 7, 2)
        raise AssertionError("expected validation error")
    except ValueError:
        pass


def test_superseded_document_excluded(seeded_conn):
    conn, tmp_path = seeded_conn
    storage = tmp_path / "docs"
    source = tmp_path / "source.txt"
    source.write_bytes(b"version one")
    first = ingest_document(
        conn, "aurora-retail", source, storage_root=storage, source_reference=str(source.resolve())
    )
    source.write_bytes(b"version two")
    second = ingest_document(
        conn, "aurora-retail", source, storage_root=storage, source_reference=str(source.resolve())
    )
    assert first.ingestion_status == "ready"
    assert second.ingestion_status == "ready"

    # Persist chunks for the first document only.
    extraction = extract_txt(b"version one")
    chunks = chunk_extraction(
        extraction, first.document_id, "aurora-retail", chunk_size=20, chunk_overlap=2
    )
    vectors = [np.array([1.0, 0.0], dtype=np.float32) for _ in chunks]
    persist_document_chunks(
        conn,
        first.document_id,
        "aurora-retail",
        chunks,
        vectors,
        embedding_model="m",
        embedding_dim=2,
    )

    # Now activate the replacement and persist its chunks.
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "UPDATE documents SET ingestion_status = 'superseded', superseded_by = ? WHERE id = ?",
        (second.document_id, first.document_id),
    )
    conn.commit()
    extraction2 = extract_txt(b"version two")
    chunks2 = chunk_extraction(
        extraction2, second.document_id, "aurora-retail", chunk_size=20, chunk_overlap=2
    )
    vectors2 = [np.array([0.0, 1.0], dtype=np.float32) for _ in chunks2]
    persist_document_chunks(
        conn,
        second.document_id,
        "aurora-retail",
        chunks2,
        vectors2,
        embedding_model="m",
        embedding_dim=2,
    )

    active = get_active_embeddings(conn, embedding_model="m", embedding_dim=2)
    assert {item.document_id for item in active} == {second.document_id}
