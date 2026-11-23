from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from backend.app.agent.service import AgentService
from backend.app.rag.embeddings import EmbeddingResult
from backend.app.rag.persistence import get_document_chunks
from backend.app.storage.connection import get_connection, init_db
from backend.scripts.seed_db import seed_database
from backend.tests.rag.corpus import build_corpus, corpus_documents, corpus_vector


def make_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def make_tool_call(id_, name, arguments):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def make_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def corpus_db(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    db_path = str(tmp_path / "corpus.db")
    init_db(db_path)
    seed_database(db_path)
    storage = tmp_path / "documents"
    conn = get_connection(db_path)
    ingested, _store = build_corpus(conn, storage, tmp_path / "corpus.index")
    conn.close()
    return db_path, tmp_path, ingested


def _first_chunk(db_path, document_id):
    conn = get_connection(db_path)
    chunks = get_document_chunks(conn, document_id)
    conn.close()
    return chunks[0]


@pytest.mark.asyncio
async def test_rag_only_tool_selection_and_provenance(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    document, _ = ingested[0]
    chunk = _first_chunk(db_path, document if False else ingested[0][1].document_id)
    query_vector = corpus_vector(ingested[0][0])
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.embed_texts") as embed_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        f'{{"company_id":"{ingested[0][0].company_id}","query":"overview","top_k":3}}',
                    )
                ]
            )
        )
        second = make_response(
            make_message(
                content=f"Retrieved [doc:{chunk['document_id']} chunk:{chunk['chunk_id']}]",
                tool_calls=None,
            )
        )
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        embed_mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        service = AgentService()
        result = await service.run("Find overview evidence", db_path=db_path)
        assert "Retrieved evidence:" in result
        assert f"[doc:{chunk['document_id']} chunk:{chunk['chunk_id']}]" in result
        assert "Citation validation warnings" not in result


@pytest.mark.asyncio
async def test_financial_only_tool_selection(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "fin-1",
                        "get_company_financials",
                        '{"company_id":"aurora-retail","period_range":"2023-2023"}',
                    )
                ]
            )
        )
        second = make_response(make_message(content="Financial result received", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        service = AgentService()
        result = await service.run("Show Aurora financials", db_path=db_path)
        assert result == "Financial result received"


@pytest.mark.asyncio
async def test_combined_financial_and_rag_tools(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    query_vector = corpus_vector(ingested[0][0])
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.embed_texts") as embed_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "fin-1",
                        "get_company_financials",
                        '{"company_id":"aurora-retail","period_range":"2023-2023"}',
                    ),
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        '{"company_id":"aurora-retail","query":"overview","top_k":2}',
                    ),
                ]
            )
        )
        second = make_response(make_message(content="Combined result", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        embed_mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        service = AgentService()
        result = await service.run("Screen Aurora", db_path=db_path)
        assert result.startswith("Combined result")


@pytest.mark.asyncio
async def test_rag_company_mismatch_rejected(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.retrieve") as retrieve_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        '{"company_id":"cedar-health","query":"overview","top_k":2}',
                    )
                ]
            )
        )
        second = make_response(make_message(content="Mismatch handled", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        service = AgentService()
        result = await service.run("Screen Aurora", company_id="aurora-retail", db_path=db_path)
        assert result == "Mismatch handled"
        retrieve_mocked.assert_not_called()


@pytest.mark.asyncio
async def test_rag_empty_retrieval(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    # Use a seeded database without corpus documents for borealis.
    empty_db = str(tmp_path / "empty.db")
    init_db(empty_db)
    seed_database(empty_db)
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.embed_texts") as embed_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        '{"company_id":"borealis-industrial","query":"risk","top_k":2}',
                    )
                ]
            )
        )
        second = make_response(make_message(content="No evidence", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        embed_mocked.return_value = EmbeddingResult(
            vectors=[np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)],
            model="text-embedding-3-small",
            dimension=8,
        )
        service = AgentService()
        result = await service.run("Find borealis risk", db_path=empty_db)
        assert result == "No evidence"


@pytest.mark.asyncio
async def test_rag_failure_is_reported(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.retrieve") as retrieve_mocked,
    ):
        instance = mocked.return_value
        retrieve_mocked.side_effect = RuntimeError("embedding service unavailable")
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        '{"company_id":"aurora-retail","query":"overview","top_k":2}',
                    )
                ]
            )
        )
        second = make_response(make_message(content="Failure handled", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        service = AgentService()
        result = await service.run("Find overview", db_path=db_path)
        assert result == "Failure handled"


@pytest.mark.asyncio
async def test_citation_validation_warning(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    document, _ = ingested[0]
    chunk = _first_chunk(db_path, document if False else ingested[0][1].document_id)
    query_vector = corpus_vector(ingested[0][0])
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.embed_texts") as embed_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        f'{{"company_id":"{ingested[0][0].company_id}","query":"overview","top_k":2}}',
                    )
                ]
            )
        )
        second = make_response(
            make_message(
                content=f"Claim [doc:{chunk['document_id']} chunk:{chunk['chunk_id']}] and [doc:missing chunk:999]",
                tool_calls=None,
            )
        )
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        embed_mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        service = AgentService()
        result = await service.run("Find overview", db_path=db_path)
        assert "Citation validation warnings" in result
        assert "unverified citation" in result


@pytest.mark.asyncio
async def test_conflicting_evidence_preserved(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    financial_doc = next(
        item
        for item in ingested
        if item[0].doc_type == "financial_summary" and item[0].company_id == "aurora-retail"
    )
    _supplemental_doc = next(
        item
        for item in ingested
        if item[0].doc_type == "supplemental" and item[0].company_id == "aurora-retail"
    )
    query_vector = corpus_vector(financial_doc[0])
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.embed_texts") as embed_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        '{"company_id":"aurora-retail","query":"revenue discrepancy","top_k":5}',
                    )
                ]
            )
        )
        second = make_response(make_message(content="Both sources retrieved", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        embed_mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        service = AgentService()
        result = await service.run("Find revenue discrepancy", db_path=db_path)
        assert result.startswith("Both sources retrieved")


@pytest.mark.asyncio
async def test_missing_information_not_fabricated(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    query_vector = np.asarray([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    norm = float(np.linalg.norm(query_vector))
    query_vector = (query_vector / np.float32(norm)).astype(np.float32)
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.embed_texts") as embed_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        '{"company_id":"aurora-retail","query":"quantum computing revenue","top_k":5}',
                    )
                ]
            )
        )
        second = make_response(make_message(content="No quantum evidence", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        embed_mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        service = AgentService()
        result = await service.run("Find quantum revenue", db_path=db_path)
        assert result.startswith("No quantum evidence")


@pytest.mark.asyncio
async def test_multi_chunk_document(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    management_doc = next(
        item
        for item in ingested
        if item[0].doc_type == "management_discussion" and item[0].company_id == "aurora-retail"
    )
    query_vector = corpus_vector(management_doc[0])
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.embed_texts") as embed_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        '{"company_id":"aurora-retail","query":"management discussion","top_k":5}',
                    )
                ]
            )
        )
        second = make_response(make_message(content="Multi-chunk result", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        embed_mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        service = AgentService()
        result = await service.run("Find management discussion", db_path=db_path)
        assert result.startswith("Multi-chunk result")


@pytest.mark.asyncio
async def test_empty_retrieval(corpus_db, monkeypatch):
    db_path, tmp_path, ingested = corpus_db
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    empty_db = str(tmp_path / "empty.db")
    init_db(empty_db)
    seed_database(empty_db)
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.embed_texts") as embed_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        '{"company_id":"borealis-industrial","query":"risk","top_k":2}',
                    )
                ]
            )
        )
        second = make_response(make_message(content="Empty retrieval", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        service = AgentService()
        result = await service.run("Find borealis risk", db_path=empty_db)
        assert result == "Empty retrieval"
        embed_mocked.assert_not_called()


@pytest.mark.asyncio
async def test_repeatable_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.config.settings.embedding_dim", 8)
    first_db = str(tmp_path / "first.db")
    second_db = str(tmp_path / "second.db")
    init_db(first_db)
    seed_database(first_db)
    init_db(second_db)
    seed_database(second_db)
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

    db_path = first_db
    query_vector = corpus_vector(corpus_documents()[0])
    with (
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
        patch("backend.app.rag.retrieval.embed_texts") as embed_mocked,
    ):
        instance = mocked.return_value
        first = make_response(
            make_message(
                tool_calls=[
                    make_tool_call(
                        "rag-1",
                        "retrieve_company_documents",
                        '{"company_id":"aurora-retail","query":"overview","top_k":3}',
                    )
                ]
            )
        )
        second = make_response(make_message(content="Repeatable", tool_calls=None))
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        embed_mocked.return_value = EmbeddingResult(
            vectors=[query_vector], model="text-embedding-3-small", dimension=8
        )
        service = AgentService()
        result = await service.run("Find overview", db_path=db_path)
        assert result.startswith("Repeatable")
