from unittest.mock import patch

import numpy as np
import pytest

from backend.app.rag.chunking import chunk_extraction
from backend.app.rag.embeddings import EmbeddingResult
from backend.app.rag.extraction import extract_txt
from backend.app.rag.ingestion import ingest_document_bytes
from backend.app.rag.persistence import persist_document_chunks
from backend.app.rag.retrieval import RetrievalError, retrieve
from backend.app.rag.vector_store import VectorStore
from backend.app.storage.connection import get_connection, init_db
from backend.scripts.seed_db import seed_database


@pytest.fixture
def seeded_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    seed_database(db_path)
    return db_path, tmp_path


async def _ingest_and_index(db_path, tmp_path, company_id, text, vector):
    conn = get_connection(db_path)
    storage = tmp_path / "docs"
    result = ingest_document_bytes(conn, company_id, "doc.txt", text.encode(), storage_root=storage)
    extraction = extract_txt(text.encode())
    chunks = chunk_extraction(
        extraction, result.document_id, company_id, chunk_size=100, chunk_overlap=10
    )
    vectors = [vector for _ in chunks]
    persisted = persist_document_chunks(
        conn,
        result.document_id,
        company_id,
        chunks,
        vectors,
        embedding_model="text-embedding-3-small",
        embedding_dim=2,
    )
    conn.close()
    store = VectorStore(
        db_path=db_path,
        index_path=str(tmp_path / "test.index"),
        embedding_dim=2,
        embedding_model="text-embedding-3-small",
    )
    store.rebuild_index()
    return result.document_id, persisted["vector_ids"]


@pytest.mark.asyncio
async def test_retrieval_returns_ranked_results(seeded_db, monkeypatch):
    db_path, tmp_path = seeded_db
    await _ingest_and_index(
        db_path,
        tmp_path,
        "aurora-retail",
        "alpha beta gamma",
        np.array([1.0, 0.0], dtype=np.float32),
    )
    vector = np.array([1.0, 0.0], dtype=np.float32)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[vector], model="text-embedding-3-small", dimension=2
        )
        results = await retrieve(db_path, "aurora-retail", "alpha", top_k=5, embedding_dim=2)
    assert len(results) == 1
    assert results[0].company_id == "aurora-retail"
    assert results[0].similarity_score == 1.0
    assert results[0].filename == "doc.txt"


@pytest.mark.asyncio
async def test_company_isolation(seeded_db, monkeypatch):
    db_path, tmp_path = seeded_db
    await _ingest_and_index(
        db_path, tmp_path, "aurora-retail", "aurora content", np.array([1.0, 0.0], dtype=np.float32)
    )
    await _ingest_and_index(
        db_path, tmp_path, "cedar-health", "cedar content", np.array([0.0, 1.0], dtype=np.float32)
    )
    # Rebuild once with both companies in the same index.
    store = VectorStore(
        db_path=db_path,
        index_path=str(tmp_path / "test.index"),
        embedding_dim=2,
        embedding_model="text-embedding-3-small",
    )
    store.rebuild_index()
    vector = np.array([1.0, 0.0], dtype=np.float32)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[vector], model="text-embedding-3-small", dimension=2
        )
        results = await retrieve(db_path, "aurora-retail", "query", top_k=5, embedding_dim=2)
    assert all(result.company_id == "aurora-retail" for result in results)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_exact_top_k_lower_ranked_company(seeded_db, monkeypatch):
    db_path, tmp_path = seeded_db
    await _ingest_and_index(
        db_path,
        tmp_path,
        "aurora-retail",
        "aurora content",
        np.array([0.2, 0.9797958971], dtype=np.float32),
    )
    await _ingest_and_index(
        db_path, tmp_path, "cedar-health", "cedar content", np.array([1.0, 0.0], dtype=np.float32)
    )
    store = VectorStore(
        db_path=db_path,
        index_path=str(tmp_path / "test.index"),
        embedding_dim=2,
        embedding_model="text-embedding-3-small",
    )
    store.rebuild_index()
    vector = np.array([1.0, 0.0], dtype=np.float32)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[vector], model="text-embedding-3-small", dimension=2
        )
        results = await retrieve(db_path, "aurora-retail", "query", top_k=1, embedding_dim=2)
    assert len(results) == 1
    assert results[0].company_id == "aurora-retail"


@pytest.mark.asyncio
async def test_empty_index_returns_empty(seeded_db, monkeypatch):
    db_path, tmp_path = seeded_db
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[np.array([1.0, 0.0], dtype=np.float32)],
            model="text-embedding-3-small",
            dimension=2,
        )
        # Force no active embeddings by using an unknown company.
        with pytest.raises(RetrievalError):
            await retrieve(db_path, "missing-company", "query", top_k=1)


@pytest.mark.asyncio
async def test_invalid_inputs(seeded_db):
    db_path, tmp_path = seeded_db
    with pytest.raises(RetrievalError):
        await retrieve(db_path, "", "query", top_k=1)
    with pytest.raises(RetrievalError):
        await retrieve(db_path, "aurora-retail", "   ", top_k=1)
    with pytest.raises(RetrievalError):
        await retrieve(db_path, "aurora-retail", "query", top_k=0)
    with pytest.raises(RetrievalError):
        await retrieve(db_path, "aurora-retail", "query", top_k=21)


@pytest.mark.asyncio
async def test_known_company_no_embeddings_returns_empty(seeded_db, monkeypatch):
    db_path, tmp_path = seeded_db
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        results = await retrieve(db_path, "borealis-industrial", "query", top_k=1)
        assert results == []
        mocked.assert_not_called()


@pytest.mark.asyncio
async def test_top_k_and_provenance(seeded_db, monkeypatch):
    db_path, tmp_path = seeded_db
    await _ingest_and_index(
        db_path,
        tmp_path,
        "aurora-retail",
        "alpha beta gamma",
        np.array([1.0, 0.0], dtype=np.float32),
    )
    vector = np.array([1.0, 0.0], dtype=np.float32)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[vector], model="text-embedding-3-small", dimension=2
        )
        results = await retrieve(db_path, "aurora-retail", "alpha", top_k=1, embedding_dim=2)
    assert len(results) == 1
    result = results[0]
    assert result.page_start == 1
    assert result.page_end == 1
    assert result.embedding_model == "text-embedding-3-small"
    assert result.filename == "doc.txt"
    assert result.source.endswith("doc.txt")
