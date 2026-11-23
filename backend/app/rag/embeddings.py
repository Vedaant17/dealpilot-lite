"""Asynchronous OpenAI embedding adapter for M6-B."""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any, Optional, Sequence

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)

from backend.app.config import settings
from backend.app.rag.schemas import EmbeddedChunk, EmbeddingResult


class EmbeddingConfigurationError(ValueError):
    """Raised when embedding configuration is invalid."""


class EmbeddingRequestError(RuntimeError):
    """Raised when an embedding API request fails."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class EmbeddingResponseError(ValueError):
    """Raised when an embedding API response is invalid."""


RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    asyncio.TimeoutError,
)


def _validate_settings(
    model: str,
    dimension: int,
    batch_size: int,
    timeout: float,
    max_retries: int,
    api_key: str,
) -> None:
    if not api_key:
        raise EmbeddingConfigurationError("OpenAI API key is required for embeddings")
    if not model:
        raise EmbeddingConfigurationError("embedding model is required")
    if dimension <= 0:
        raise EmbeddingConfigurationError("embedding dimension must be positive")
    if batch_size <= 0:
        raise EmbeddingConfigurationError("embedding batch size must be positive")
    if timeout <= 0:
        raise EmbeddingConfigurationError("embedding timeout must be positive")
    if max_retries < 0:
        raise EmbeddingConfigurationError("embedding max retries must be non-negative")


async def _embed_batch(
    client: AsyncOpenAI,
    batch: list[str],
    *,
    model: str,
    dimension: int,
    timeout: float,
) -> list[np.ndarray]:
    try:
        response = await asyncio.wait_for(
            client.embeddings.create(
                model=model,
                input=batch,
                dimensions=dimension,
                timeout=timeout,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise EmbeddingRequestError("embedding request timed out", retryable=True) from exc
    except RETRYABLE_EXCEPTIONS as exc:
        raise EmbeddingRequestError(str(exc)[:500], retryable=True) from exc
    except (AuthenticationError, BadRequestError) as exc:
        raise EmbeddingRequestError(str(exc)[:500], retryable=False) from exc
    except OpenAIError as exc:
        raise EmbeddingRequestError(str(exc)[:500], retryable=False) from exc
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingRequestError(str(exc)[:500], retryable=False) from exc

    if len(response.data) != len(batch):
        raise EmbeddingResponseError(
            f"expected {len(batch)} embeddings, received {len(response.data)}"
        )

    vectors: list[Optional[np.ndarray]] = [None] * len(batch)
    seen_indices: set[int] = set()

    for item in response.data:
        index = item.index
        if not isinstance(index, int) or index < 0 or index >= len(batch):
            raise EmbeddingResponseError(f"invalid embedding index: {index!r}")
        if index in seen_indices:
            raise EmbeddingResponseError(f"duplicate embedding index: {index}")
        seen_indices.add(index)

        vector = np.asarray(item.embedding, dtype=np.float32)
        if vector.shape != (dimension,):
            raise EmbeddingResponseError(
                f"embedding {index} has dimension {vector.shape[0]}, expected {dimension}"
            )
        if not np.all(np.isfinite(vector)):
            raise EmbeddingResponseError(f"embedding {index} contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0 or not math.isfinite(norm):
            raise EmbeddingResponseError(f"embedding {index} has zero or invalid norm")
        vectors[index] = (vector / np.float32(norm)).astype(np.float32)

    if any(vector is None for vector in vectors):
        missing = sorted(set(range(len(batch))) - seen_indices)
        raise EmbeddingResponseError(f"missing embedding indices: {missing}")

    return [vector for vector in vectors if vector is not None]


async def embed_texts(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    api_key: Optional[str] = None,
    batch_size: Optional[int] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    client: Optional[AsyncOpenAI] = None,
) -> EmbeddingResult:
    resolved_model = model if model is not None else settings.openai_embedding_model
    resolved_dimensions = dimensions if dimensions is not None else settings.embedding_dim
    resolved_api_key = api_key if api_key is not None else settings.openai_api_key
    resolved_batch_size = (
        batch_size if batch_size is not None else settings.rag_embedding_batch_size
    )
    resolved_timeout = timeout if timeout is not None else settings.rag_embedding_timeout_seconds
    resolved_retries = (
        max_retries if max_retries is not None else settings.rag_embedding_max_retries
    )

    _validate_settings(
        resolved_model,
        resolved_dimensions,
        resolved_batch_size,
        resolved_timeout,
        resolved_retries,
        resolved_api_key,
    )

    inputs = list(texts)
    if not inputs:
        return EmbeddingResult(
            vectors=[], model=resolved_model, dimension=resolved_dimensions, normalized=True
        )

    for text in inputs:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingResponseError("embedding input must be a non-empty string")

    owned_client = client is None
    if owned_client:
        async_client = AsyncOpenAI(
            api_key=resolved_api_key, timeout=resolved_timeout, max_retries=0
        )
    else:
        async_client = client

    try:
        all_vectors: list[np.ndarray] = []
        for start in range(0, len(inputs), resolved_batch_size):
            batch = inputs[start : start + resolved_batch_size]
            vectors = None
            last_error: Exception | None = None
            for attempt in range(resolved_retries + 1):
                try:
                    vectors = await _embed_batch(
                        async_client,
                        batch,
                        model=resolved_model,
                        dimension=resolved_dimensions,
                        timeout=resolved_timeout,
                    )
                    break
                except EmbeddingResponseError:
                    raise
                except EmbeddingRequestError as exc:
                    last_error = exc
                    if attempt >= resolved_retries or not exc.retryable:
                        raise
                    await asyncio.sleep(min(2**attempt, 8) + random.uniform(0.0, 0.25))
            if vectors is None:
                raise last_error or EmbeddingRequestError("embedding request failed")
            all_vectors.extend(vectors)
        return EmbeddingResult(
            vectors=all_vectors,
            model=resolved_model,
            dimension=resolved_dimensions,
            normalized=True,
        )
    finally:
        if owned_client:
            close_result = async_client.aclose()
            if hasattr(close_result, "__await__"):
                await close_result


async def embed_chunks(chunks: list[Any], **kwargs: Any) -> list["EmbeddedChunk"]:
    from backend.app.rag.schemas import EmbeddedChunk

    texts = [chunk.text for chunk in chunks]
    result = await embed_texts(texts, **kwargs)
    embedded: list[EmbeddedChunk] = []
    for chunk, vector in zip(chunks, result.vectors):
        embedded.append(
            EmbeddedChunk(
                chunk=chunk,
                vector=vector,
                model=result.model,
                dimension=result.dimension,
                normalized=result.normalized,
            )
        )
    return embedded


# Compatibility aliases.
generate_embeddings = embed_texts
