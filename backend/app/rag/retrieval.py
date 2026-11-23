"""Company-filtered semantic retrieval for M6-D."""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Optional

import numpy as np

from backend.app.config import settings
from backend.app.rag.embeddings import embed_texts
from backend.app.rag.persistence import get_active_embeddings
from backend.app.rag.schemas import RagRetrievalToolInput, RetrievalResult
from backend.app.rag.vector_store import VectorStore
from backend.app.storage import repositories as repo
from backend.app.storage.connection import get_connection


class RetrievalError(ValueError):
    """Raised when retrieval input or state is invalid."""


async def embed_query(
    query: str,
    *,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    client: Any = None,
) -> np.ndarray:
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("query must be a non-empty string")
    result = await embed_texts(
        [query],
        model=model,
        dimensions=dimensions,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        client=client,
    )
    if not result.vectors:
        raise RetrievalError("query embedding was not returned")
    return result.vectors[0]


def _database_path(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None or not row["file"]:
        raise RetrievalError("unable to resolve database path from connection")
    return str(row["file"])


def _validate_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise RetrievalError("top_k must be an integer")
    if top_k < 1:
        raise RetrievalError("top_k must be positive")
    maximum = getattr(settings, "rag_top_k_max", 20)
    if top_k > maximum:
        raise RetrievalError(f"top_k must not exceed {maximum}")
    return top_k


async def retrieve(
    *args: Any,
    company_id: Optional[str] = None,
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    db_path: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
    vector_store: Optional[VectorStore] = None,
    embedding_model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> list[RetrievalResult]:
    positional = list(args)
    if len(positional) >= 3 and isinstance(positional[0], sqlite3.Connection):
        conn = positional.pop(0)
        company_id = positional.pop(0) if company_id is None else company_id
        query = positional.pop(0) if query is None else query
    elif len(positional) >= 3 and all(isinstance(item, str) for item in positional[:3]):
        db_path = positional.pop(0)
        company_id = positional.pop(0) if company_id is None else company_id
        query = positional.pop(0) if query is None else query
    elif len(positional) >= 2 and isinstance(positional[0], sqlite3.Connection):
        conn = positional.pop(0)
        if company_id is None:
            company_id = positional.pop(0)
        if query is None and positional:
            query = positional.pop(0)
    elif len(positional) >= 2 and all(isinstance(item, str) for item in positional[:2]):
        if company_id is None:
            company_id = positional.pop(0)
        if query is None:
            query = positional.pop(0)
    if (
        positional
        and top_k is None
        and isinstance(positional[0], int)
        and not isinstance(positional[0], bool)
    ):
        top_k = positional.pop(0)

    if not isinstance(company_id, str) or not company_id.strip():
        raise RetrievalError("company_id must be a non-empty string")
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("query must be a non-empty string")

    resolved_top_k = settings.rag_top_k_default if top_k is None else top_k
    resolved_top_k = _validate_top_k(resolved_top_k)

    owns_connection = conn is None
    if conn is None:
        resolved_db_path = db_path if db_path is not None else settings.sqlite_db_path
        conn = get_connection(resolved_db_path)
    else:
        resolved_db_path = _database_path(conn)

    try:
        company = repo.get_company(conn, company_id)
        if company is None:
            raise RetrievalError(f"unknown company: {company_id}")

        resolved_model = (
            embedding_model
            or (vector_store.embedding_model if vector_store else None)
            or settings.openai_embedding_model
        )
        resolved_dim = (
            embedding_dim
            or (vector_store.embedding_dim if vector_store else None)
            or settings.embedding_dim
        )

        active = get_active_embeddings(
            conn,
            company_id=company_id,
            embedding_model=resolved_model,
            embedding_dim=resolved_dim,
        )
        if not active:
            return []

        query_vector = await embed_query(
            query,
            model=resolved_model,
            dimensions=resolved_dim,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        store = (
            vector_store
            if vector_store is not None
            else VectorStore(
                db_path=resolved_db_path,
                embedding_dim=resolved_dim,
                embedding_model=resolved_model,
            )
        )
        return await asyncio.to_thread(store.retrieve, company_id, query_vector, resolved_top_k)
    finally:
        if owns_connection:
            conn.close()


def retrieve_sync(*args, **kwargs) -> list[RetrievalResult]:
    return asyncio.run(retrieve(*args, **kwargs))


def rag_tool_definition() -> dict[str, Any]:
    schema = RagRetrievalToolInput.model_json_schema()
    schema.pop("title", None)
    return {
        "name": "retrieve_company_documents",
        "description": (
            "Retrieve relevant company-document passages and provenance. "
            "This is not a financial calculation tool."
        ),
        "inputSchema": schema,
        "source": "rag",
    }
