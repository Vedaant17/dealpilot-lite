from unittest.mock import patch

import numpy as np
import pytest

from backend.app.rag.chunking import chunk_extraction
from backend.app.rag.embeddings import EmbeddingResult
from backend.app.rag.extraction import extract_txt
from backend.app.rag.ingestion import ingest_document_bytes
from backend.app.rag.persistence import persist_document_chunks
from backend.app.rag.retrieval import retrieve
from backend.app.storage.connection import get_connection, init_db
from backend.scripts.seed_db import seed_database
from backend.tests.rag.corpus import (
    build_corpus,
    corpus_documents,
    corpus_vector,
)


def _seeded_db(tmp_path):
    db_path = str(tmp_path / "corpus.db")
    init_db(db_path)
    seed_database(db_path)
    return db_path


def _query_vector(components):
    vector = np.asarray(components, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("query vector must not be zero")
    return (vector / np.float32(norm)).astype(np.float32)


@pytest.mark.asyncio
async def test_corpus_inventory_and_ingestion(tmp_path):
    db_path = _seeded_db(tmp_path)
    storage = tmp_path / "documents"
    conn = get_connection(db_path)
    ingested, store = build_corpus(conn, storage, tmp_path / "corpus.index")
    assert len(ingested) == 15
    assert len({item[0].filename for item in ingested}) == 15
    for document, result in ingested:
        assert result.ingestion_status == "ready"
        assert result.duplicate is False
    assert store is not None
    conn.close()


@pytest.mark.asyncio
async def test_corpus_retrieval_relevance(tmp_path):
    db_path = _seeded_db(tmp_path)
    storage = tmp_path / "documents"
    conn = get_connection(db_path)
    ingested, _store = build_corpus(conn, storage, tmp_path / "corpus.index")
    conn.close()

    query_vector = _query_vector((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        results = await retrieve(
            db_path, "aurora-retail", "financial performance", top_k=3, embedding_dim=8
        )
    assert len(results) >= 1
    assert all(result.company_id == "aurora-retail" for result in results)
    assert any(result.filename == "aurora_financials.txt" for result in results)


@pytest.mark.asyncio
async def test_corpus_company_isolation(tmp_path):
    db_path = _seeded_db(tmp_path)
    storage = tmp_path / "documents"
    conn = get_connection(db_path)
    build_corpus(conn, storage, tmp_path / "corpus.index")
    conn.close()

    # The query is intentionally aligned with the financial topic. A Cedar financial
    # document has a stronger global topic component, but company filtering must
    # still return only Aurora evidence.
    query_vector = _query_vector((0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        results = await retrieve(
            db_path, "aurora-retail", "financial performance", top_k=5, embedding_dim=8
        )
    assert len(results) >= 1
    assert all(result.company_id == "aurora-retail" for result in results)


@pytest.mark.asyncio
async def test_corpus_provenance(tmp_path):
    db_path = _seeded_db(tmp_path)
    storage = tmp_path / "documents"
    conn = get_connection(db_path)
    ingested, _store = build_corpus(conn, storage, tmp_path / "corpus.index")
    conn.close()

    document, _result = ingested[0]
    query_vector = np.asarray(document.vector, dtype=np.float32)
    norm = float(np.linalg.norm(query_vector))
    query_vector = (query_vector / np.float32(norm)).astype(np.float32)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        results = await retrieve(
            db_path, document.company_id, "company overview", top_k=3, embedding_dim=8
        )
    assert len(results) >= 1
    result = results[0]
    assert result.company_id == document.company_id
    assert result.document_id == _result.document_id
    assert isinstance(result.chunk_id, int)
    assert isinstance(result.chunk_index, int)
    assert result.text
    assert result.source.endswith(document.filename)
    assert result.filename == document.filename
    assert result.page_start == 1
    assert result.page_end == 1
    assert result.embedding_model == "text-embedding-3-small"
    assert isinstance(result.similarity_score, float)


@pytest.mark.asyncio
async def test_corpus_conflicting_evidence(tmp_path):
    db_path = _seeded_db(tmp_path)
    storage = tmp_path / "documents"
    conn = get_connection(db_path)
    ingested, _store = build_corpus(conn, storage, tmp_path / "corpus.index")
    conn.close()

    financial_doc = next(
        item[0]
        for item in ingested
        if item[0].doc_type == "financial_summary" and item[0].company_id == "aurora-retail"
    )
    supplemental_doc = next(
        item[0]
        for item in ingested
        if item[0].doc_type == "supplemental" and item[0].company_id == "aurora-retail"
    )
    query_vector = _query_vector(financial_doc.vector)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        results = await retrieve(
            db_path, "aurora-retail", "revenue discrepancy", top_k=5, embedding_dim=8
        )
    filenames = {result.filename for result in results}
    assert financial_doc.filename in filenames
    assert supplemental_doc.filename in filenames
    assert any("53.0 million" in result.text for result in results)
    assert any("54.0 million" in result.text for result in results)


@pytest.mark.asyncio
async def test_corpus_missing_information(tmp_path):
    db_path = _seeded_db(tmp_path)
    storage = tmp_path / "documents"
    conn = get_connection(db_path)
    build_corpus(conn, storage, tmp_path / "corpus.index")
    conn.close()

    query_vector = _query_vector((0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        results = await retrieve(
            db_path, "aurora-retail", "quantum computing revenue", top_k=5, embedding_dim=8
        )
    assert all("quantum" not in result.text.lower() for result in results)


@pytest.mark.asyncio
async def test_corpus_multi_chunk_document(tmp_path):
    db_path = _seeded_db(tmp_path)
    storage = tmp_path / "documents"
    conn = get_connection(db_path)
    ingested, _store = build_corpus(conn, storage, tmp_path / "corpus.index")
    conn.close()

    management_doc = next(
        item[0]
        for item in ingested
        if item[0].doc_type == "management_discussion" and item[0].company_id == "aurora-retail"
    )
    query_vector = _query_vector(management_doc.vector)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        results = await retrieve(
            db_path, "aurora-retail", "management discussion", top_k=5, embedding_dim=8
        )
    management_results = [
        result for result in results if result.document_id == management_doc and False
    ]
    # Document IDs are generated during ingestion; compare by filename instead.
    management_results = [
        result for result in results if result.filename == management_doc.filename
    ]
    assert len(management_results) >= 2
    assert {result.chunk_index for result in management_results} >= {0, 1}


@pytest.mark.asyncio
async def test_corpus_empty_company_returns_empty(tmp_path):
    db_path = str(tmp_path / "empty.db")
    init_db(db_path)
    seed_database(db_path)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        results = await retrieve(db_path, "borealis-industrial", "risk", top_k=5, embedding_dim=8)
        assert results == []
        mocked.assert_not_called()


@pytest.mark.asyncio
async def test_corpus_repeatable(tmp_path):
    first_db = _seeded_db(tmp_path / "first")
    second_db = _seeded_db(tmp_path / "second")
    first_storage = tmp_path / "first-docs"
    second_storage = tmp_path / "second-docs"
    conn = get_connection(first_db)
    first_ingested, _ = build_corpus(conn, first_storage, tmp_path / "first.index")
    conn.close()
    conn = get_connection(second_db)
    second_ingested, _ = build_corpus(conn, second_storage, tmp_path / "second.index")
    conn.close()

    assert [item[1].document_id for item in first_ingested] == [
        item[1].document_id for item in second_ingested
    ]

    query_vector = _query_vector(corpus_documents()[0].vector)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        first_results = await retrieve(
            first_db, "aurora-retail", "overview", top_k=3, embedding_dim=8
        )
        second_results = await retrieve(
            second_db, "aurora-retail", "overview", top_k=3, embedding_dim=8
        )
    assert [result.document_id for result in first_results] == [
        result.document_id for result in second_results
    ]


@pytest.mark.asyncio
async def test_corpus_superseded_document_excluded(tmp_path):
    db_path = _seeded_db(tmp_path)
    storage = tmp_path / "docs"
    conn = get_connection(db_path)
    doc = corpus_documents()[0]
    first = ingest_document_bytes(
        conn,
        doc.company_id,
        doc.filename,
        doc.text.encode("utf-8"),
        storage_root=storage,
        title=doc.title,
        doc_type=doc.doc_type,
        source_reference=doc.source_reference,
    )
    second_text = doc.text + "\nUpdated supplemental sentence."
    second = ingest_document_bytes(
        conn,
        doc.company_id,
        doc.filename,
        second_text.encode("utf-8"),
        storage_root=storage,
        title=doc.title,
        doc_type=doc.doc_type,
        source_reference=doc.source_reference,
    )
    first_extraction = extract_txt(doc.text.encode("utf-8"))
    first_chunks = chunk_extraction(
        first_extraction, first.document_id, doc.company_id, chunk_size=3200, chunk_overlap=400
    )
    first_vectors = [corpus_vector(doc) for _ in first_chunks]
    persist_document_chunks(
        conn,
        first.document_id,
        doc.company_id,
        first_chunks,
        first_vectors,
        embedding_model="text-embedding-3-small",
        embedding_dim=8,
    )
    second_extraction = extract_txt(second_text.encode("utf-8"))
    second_chunks = chunk_extraction(
        second_extraction, second.document_id, doc.company_id, chunk_size=3200, chunk_overlap=400
    )
    second_vectors = [corpus_vector(doc) for _ in second_chunks]
    persist_document_chunks(
        conn,
        second.document_id,
        doc.company_id,
        second_chunks,
        second_vectors,
        embedding_model="text-embedding-3-small",
        embedding_dim=8,
    )
    conn.close()

    query_vector = corpus_vector(doc)
    with patch("backend.app.rag.retrieval.embed_texts") as mocked:
        mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        results = await retrieve(
            db_path, doc.company_id, "company overview", top_k=5, embedding_dim=8
        )
    result_document_ids = {result.document_id for result in results}
    assert second.document_id in result_document_ids
    assert first.document_id not in result_document_ids
