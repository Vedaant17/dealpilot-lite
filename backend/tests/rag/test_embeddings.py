from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from backend.app.rag.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    embed_chunks,
    embed_texts,
)
from backend.app.rag.schemas import Chunk


def make_response(vectors, indices=None):
    if indices is None:
        indices = list(range(len(vectors)))
    data = [
        SimpleNamespace(index=index, embedding=vector) for index, vector in zip(indices, vectors)
    ]
    return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_empty_input_no_request():
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        result = await embed_texts([], api_key="test-key")
        assert result.vectors == []
        mocked.assert_not_called()


@pytest.mark.asyncio
async def test_successful_embedding_and_normalization():
    vector = [3.0, 4.0]
    response = make_response([vector])
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.embeddings.create = AsyncMock(return_value=response)
        result = await embed_texts(["hello"], api_key="test-key", dimensions=2)
        assert len(result.vectors) == 1
        assert result.vectors[0].dtype == np.float32
        assert result.vectors[0].shape == (2,)
        assert abs(np.linalg.norm(result.vectors[0]) - 1.0) < 1e-6
        assert result.normalized is True


@pytest.mark.asyncio
async def test_batch_splitting_and_ordering():
    responses = [make_response([[1.0, 0.0], [0.0, 1.0]]), make_response([[1.0, 1.0]])]
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.embeddings.create = AsyncMock(side_effect=responses)
        result = await embed_texts(["a", "b", "c"], api_key="test-key", dimensions=2, batch_size=2)
        assert len(result.vectors) == 3
        assert instance.embeddings.create.await_count == 2


@pytest.mark.asyncio
async def test_response_index_validation():
    response = make_response([[1.0, 0.0], [0.0, 1.0]], indices=[0, 0])
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.embeddings.create = AsyncMock(return_value=response)
        try:
            await embed_texts(["a", "b"], api_key="test-key", dimensions=2)
            raise AssertionError("expected response error")
        except EmbeddingResponseError:
            pass


@pytest.mark.asyncio
async def test_wrong_dimensions_and_nonfinite():
    response = make_response([[1.0, 2.0, 3.0]])
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.embeddings.create = AsyncMock(return_value=response)
        try:
            await embed_texts(["a"], api_key="test-key", dimensions=2)
            raise AssertionError("expected response error")
        except EmbeddingResponseError:
            pass

    response = make_response([[float("nan"), 1.0]])
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.embeddings.create = AsyncMock(return_value=response)
        try:
            await embed_texts(["a"], api_key="test-key", dimensions=2)
            raise AssertionError("expected response error")
        except EmbeddingResponseError:
            pass


@pytest.mark.asyncio
async def test_zero_vector_rejected():
    response = make_response([[0.0, 0.0]])
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.embeddings.create = AsyncMock(return_value=response)
        try:
            await embed_texts(["a"], api_key="test-key", dimensions=2)
            raise AssertionError("expected response error")
        except EmbeddingResponseError:
            pass


@pytest.mark.asyncio
async def test_missing_api_key():
    try:
        await embed_texts(["a"], api_key="")
        raise AssertionError("expected configuration error")
    except EmbeddingConfigurationError:
        pass


@pytest.mark.asyncio
async def test_empty_string_rejected():
    try:
        await embed_texts([""], api_key="test-key")
        raise AssertionError("expected response error")
    except EmbeddingResponseError:
        pass


@pytest.mark.asyncio
async def test_retryable_failure():
    response = make_response([[1.0, 0.0]])
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.embeddings.create = AsyncMock(side_effect=[Exception("temporary"), response])
        # Force the generic exception to be treated as retryable by monkeypatching retry tuple.
        from backend.app.rag import embeddings as module

        original = module.RETRYABLE_EXCEPTIONS
        module.RETRYABLE_EXCEPTIONS = (Exception,)
        try:
            result = await embed_texts(["a"], api_key="test-key", dimensions=2, max_retries=1)
            assert len(result.vectors) == 1
            assert instance.embeddings.create.await_count == 2
        finally:
            module.RETRYABLE_EXCEPTIONS = original


@pytest.mark.asyncio
async def test_nonretryable_failure():
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.embeddings.create = AsyncMock(side_effect=ValueError("bad request"))
        try:
            await embed_texts(["a"], api_key="test-key", dimensions=2, max_retries=1)
            raise AssertionError("expected request error")
        except EmbeddingRequestError:
            pass
        assert instance.embeddings.create.await_count == 1


@pytest.mark.asyncio
async def test_embed_chunks():
    chunk = Chunk(
        document_id="doc-1",
        company_id="aurora-retail",
        chunk_index=0,
        text="hello",
        text_hash="abc",
        page_start=1,
        page_end=1,
        char_count=5,
        token_count_estimate=2,
        chunking_version="char-v1-3200-400",
    )
    response = make_response([[1.0, 0.0]])
    with patch("backend.app.rag.embeddings.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.embeddings.create = AsyncMock(return_value=response)
        result = await embed_chunks([chunk], api_key="test-key", dimensions=2)
        assert len(result) == 1
        assert result[0].chunk == chunk
        assert result[0].normalized is True
