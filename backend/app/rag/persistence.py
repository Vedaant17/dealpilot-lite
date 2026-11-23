"""SQLite persistence for document chunks and embeddings."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from backend.app.rag.chunking import Chunk
from backend.app.rag.embeddings import EmbeddedChunk, EmbeddingResult
from backend.app.rag.schemas import StoredEmbedding

VECTOR_SEQUENCE_NAME = "vector_id"


class EmbeddingModelError(Exception):
    """Raised when stored or supplied embeddings use incompatible models."""


class EmbeddingDimensionError(Exception):
    """Raised when stored or supplied embeddings use incompatible dimensions."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunk_value(chunk: Any, name: str) -> Any:
    if isinstance(chunk, Chunk):
        return getattr(chunk, name)
    if isinstance(chunk, dict):
        return chunk[name]
    return getattr(chunk, name)


def _embedding_vector(embedding: Any) -> np.ndarray:
    if isinstance(embedding, EmbeddedChunk):
        return embedding.vector
    if isinstance(embedding, np.ndarray):
        return embedding
    if isinstance(embedding, dict):
        if "vector" in embedding:
            return embedding["vector"]
        if "embedding" in embedding:
            return embedding["embedding"]
    return getattr(embedding, "vector", getattr(embedding, "embedding", None))


def _embedding_model(embedding: Any, fallback: Optional[str]) -> str:
    if isinstance(embedding, EmbeddedChunk):
        return embedding.model
    if isinstance(embedding, dict):
        if "model" in embedding:
            return embedding["model"]
    value = getattr(embedding, "model", None)
    return value if value is not None else fallback


def _embedding_dimension(embedding: Any, fallback: Optional[int]) -> int:
    if isinstance(embedding, EmbeddedChunk):
        return embedding.dimension
    if isinstance(embedding, dict):
        if "dimension" in embedding:
            return int(embedding["dimension"])
    value = getattr(embedding, "dimension", None)
    if value is not None:
        return int(value)
    vector = _embedding_vector(embedding)
    return int(np.asarray(vector).shape[0])


def serialize_vector(vector: Any, dimension: int) -> bytes:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.shape[0] != dimension:
        raise ValueError(f"vector dimension must be {dimension}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("vector contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("vector has zero or invalid norm")
    if abs(norm - 1.0) > 1e-5:
        raise ValueError("vector must be unit-normalized")
    return array.astype("<f4").tobytes()


def deserialize_vector(blob: bytes, dimension: int) -> np.ndarray:
    expected = dimension * 4
    if len(blob) != expected:
        raise ValueError(f"vector BLOB has {len(blob)} bytes, expected {expected}")
    array = np.frombuffer(blob, dtype="<f4").astype(np.float32)
    if array.shape != (dimension,):
        raise ValueError(f"vector dimension must be {dimension}, got {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise ValueError("stored vector contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("stored vector has zero or invalid norm")
    if abs(norm - 1.0) > 1e-5:
        raise ValueError("stored vector is not unit-normalized")
    return array.copy()


def reserve_vector_ids(conn: sqlite3.Connection, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("reservation count must be positive")
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT next_value FROM embedding_sequences WHERE sequence_name = ?",
            (VECTOR_SEQUENCE_NAME,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO embedding_sequences (sequence_name, next_value) VALUES (?, ?)",
                (VECTOR_SEQUENCE_NAME, 1),
            )
            current = 1
        else:
            current = int(row["next_value"])
        allocated = list(range(current, current + count))
        conn.execute(
            "UPDATE embedding_sequences SET next_value = ? WHERE sequence_name = ?",
            (current + count, VECTOR_SEQUENCE_NAME),
        )
        conn.commit()
        return allocated
    except Exception:
        conn.rollback()
        raise


def insert_chunks(conn: sqlite3.Connection, chunks: list[Any]) -> list[int]:
    chunk_ids: list[int] = []
    for chunk in chunks:
        cursor = conn.execute(
            """INSERT INTO document_chunks (
                document_id, company_id, chunk_index, text, text_hash,
                page_start, page_end, char_count, token_count_estimate,
                chunking_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _chunk_value(chunk, "document_id"),
                _chunk_value(chunk, "company_id"),
                _chunk_value(chunk, "chunk_index"),
                _chunk_value(chunk, "text"),
                _chunk_value(chunk, "text_hash"),
                _chunk_value(chunk, "page_start"),
                _chunk_value(chunk, "page_end"),
                _chunk_value(chunk, "char_count"),
                _chunk_value(chunk, "token_count_estimate"),
                _chunk_value(chunk, "chunking_version"),
                _now(),
            ),
        )
        chunk_ids.append(int(cursor.lastrowid))
    return chunk_ids


def insert_embeddings(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
    company_id: str,
    embeddings: list[Any],
    vector_ids: list[int],
    embedding_model: str,
    embedding_dim: int,
) -> None:
    if not (len(chunk_ids) == len(embeddings) == len(vector_ids)):
        raise ValueError("chunk, embedding, and vector ID counts must match")
    rows = []
    for chunk_id, embedding, vector_id in zip(chunk_ids, embeddings, vector_ids):
        vector = _embedding_vector(embedding)
        serialized = serialize_vector(vector, embedding_dim)
        rows.append(
            (
                chunk_id,
                company_id,
                vector_id,
                embedding_model,
                embedding_dim,
                serialized,
                1,
                _now(),
            )
        )
    conn.executemany(
        """INSERT INTO chunk_embeddings (
            chunk_id, company_id, vector_id, embedding_model, embedding_dim,
            embedding, normalized, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def delete_document_chunks(conn: sqlite3.Connection, document_id: str) -> None:
    conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))


def get_document_chunks(conn: sqlite3.Connection, document_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM document_chunks WHERE document_id = ? ORDER BY chunk_index",
        (document_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_active_embeddings(
    conn: sqlite3.Connection,
    *,
    company_id: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
) -> list[StoredEmbedding]:
    query = """SELECT dc.chunk_id, dc.document_id, dc.company_id, dc.chunk_index, dc.text,
                  dc.text_hash, dc.page_start, dc.page_end, dc.char_count,
                  dc.token_count_estimate, dc.chunking_version, dc.created_at AS chunk_created_at,
                  ce.vector_id, ce.embedding_model, ce.embedding_dim, ce.embedding,
                  ce.normalized, ce.created_at AS embedding_created_at
           FROM chunk_embeddings AS ce
           JOIN document_chunks AS dc
             ON ce.chunk_id = dc.chunk_id AND ce.company_id = dc.company_id
           JOIN documents AS d
             ON dc.document_id = d.id AND dc.company_id = d.company_id
           WHERE d.ingestion_status = 'ready'"""
    params: list[Any] = []
    if company_id is not None:
        query += " AND dc.company_id = ?"
        params.append(company_id)
    query += " ORDER BY ce.vector_id"
    rows = conn.execute(query, params).fetchall()

    if not rows:
        return []

    models = {row["embedding_model"] for row in rows}
    dimensions = {row["embedding_dim"] for row in rows}
    if embedding_model is not None and models != {embedding_model}:
        raise EmbeddingModelError(
            f"stored embedding models {sorted(models)} do not match {embedding_model}"
        )
    if embedding_dim is not None and dimensions != {embedding_dim}:
        raise EmbeddingDimensionError(
            f"stored embedding dimensions {sorted(dimensions)} do not match {embedding_dim}"
        )
    if len(models) != 1 or len(dimensions) != 1:
        raise EmbeddingModelError("stored embeddings use mixed models or dimensions")

    result: list[StoredEmbedding] = []
    for row in rows:
        vector = deserialize_vector(row["embedding"], int(row["embedding_dim"]))
        result.append(
            StoredEmbedding(
                chunk_id=int(row["chunk_id"]),
                document_id=row["document_id"],
                company_id=row["company_id"],
                vector_id=int(row["vector_id"]),
                embedding_model=row["embedding_model"],
                embedding_dim=int(row["embedding_dim"]),
                vector=vector,
                normalized=bool(row["normalized"]),
                chunk_index=int(row["chunk_index"]),
                text=row["text"],
                text_hash=row["text_hash"],
                page_start=int(row["page_start"]),
                page_end=int(row["page_end"]),
                char_count=int(row["char_count"]),
                token_count_estimate=int(row["token_count_estimate"]),
                chunking_version=row["chunking_version"],
                created_at=row["embedding_created_at"],
            )
        )
    return result


def get_index_state(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id, index_dirty, last_verified_at, last_error, updated_at FROM rag_index_state WHERE id = 1"
    ).fetchone()
    if row is None:
        conn.execute(
            """INSERT INTO rag_index_state (id, index_dirty, last_verified_at, last_error, updated_at)
               VALUES (1, 1, NULL, NULL, ?)""",
            (_now(),),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, index_dirty, last_verified_at, last_error, updated_at FROM rag_index_state WHERE id = 1"
        ).fetchone()
    return dict(row)


def set_index_dirty(conn: sqlite3.Connection, dirty: bool, error: Optional[str] = None) -> None:
    conn.execute(
        """UPDATE rag_index_state
           SET index_dirty = ?, last_error = ?, updated_at = ?
           WHERE id = 1""",
        (1 if dirty else 0, (error[:500] if error else None), _now()),
    )


def mark_index_verified(conn: sqlite3.Connection) -> None:
    now = _now()
    conn.execute(
        """UPDATE rag_index_state
           SET index_dirty = 0, last_verified_at = ?, last_error = NULL, updated_at = ?
           WHERE id = 1""",
        (now, now),
    )


def persist_document_chunks(
    conn: sqlite3.Connection,
    document_id: str,
    company_id: str,
    chunks: list[Any],
    embeddings: Any,
    *,
    embedding_model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    chunking_version: Optional[str] = None,
) -> dict[str, Any]:
    if not chunks:
        raise ValueError("at least one chunk is required")
    if len(chunks) != len(getattr(embeddings, "vectors", embeddings)):
        raise ValueError("chunk and embedding counts must match")

    document = conn.execute(
        "SELECT * FROM documents WHERE id = ? AND company_id = ?",
        (document_id, company_id),
    ).fetchone()
    if document is None:
        raise ValueError("document not found")
    if document["ingestion_status"] != "ready":
        raise ValueError("only ready documents can be indexed")

    if isinstance(embeddings, EmbeddingResult):
        supplied_model = embeddings.model
        supplied_dim = embeddings.dimension
        vector_list = list(embeddings.vectors)
    elif isinstance(embeddings, (list, tuple)):
        vector_list = [_embedding_vector(item) for item in embeddings]
        supplied_model = _embedding_model(embeddings[0] if embeddings else None, embedding_model)
        supplied_dim = _embedding_dimension(embeddings[0] if embeddings else None, embedding_dim)
    else:
        vector_list = [_embedding_vector(embeddings)]
        supplied_model = _embedding_model(embeddings, embedding_model)
        supplied_dim = _embedding_dimension(embeddings, embedding_dim)

    resolved_model = embedding_model or supplied_model
    resolved_dim = embedding_dim or supplied_dim
    if not resolved_model:
        raise ValueError("embedding model is required")
    if not resolved_dim:
        raise ValueError("embedding dimension is required")

    existing_active = conn.execute(
        """SELECT ce.embedding_model, ce.embedding_dim
           FROM chunk_embeddings AS ce
           JOIN document_chunks AS dc
                 ON ce.chunk_id = dc.chunk_id AND ce.company_id = dc.company_id
               JOIN documents AS d
                 ON dc.document_id = d.id AND dc.company_id = d.company_id
               WHERE d.ingestion_status = 'ready' AND dc.document_id != ?""",
        (document_id,),
    ).fetchall()
    if existing_active:
        models = {row["embedding_model"] for row in existing_active}
        dims = {row["embedding_dim"] for row in existing_active}
        if models != {resolved_model} or dims != {resolved_dim}:
            raise EmbeddingModelError("stored embeddings use an incompatible model or dimension")

    # Reserve IDs outside the main persistence transaction.
    vector_ids = reserve_vector_ids(conn, len(chunks))

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "DELETE FROM document_chunks WHERE document_id = ? AND company_id = ?",
            (document_id, company_id),
        )
        chunk_ids = insert_chunks(conn, chunks)
        insert_embeddings(
            conn,
            chunk_ids,
            company_id,
            vector_list,
            vector_ids,
            resolved_model,
            resolved_dim,
        )
        if document["source_reference"] is not None:
            prior_documents = conn.execute(
                """SELECT id, ingested_at FROM documents
                   WHERE company_id = ? AND source_reference = ? AND ingestion_status = 'ready'
                     AND id != ?""",
                (company_id, document["source_reference"], document_id),
            ).fetchall()
            for prior in prior_documents:
                if prior["ingested_at"] < document["ingested_at"] or (
                    prior["ingested_at"] == document["ingested_at"] and prior["id"] < document_id
                ):
                    conn.execute(
                        "UPDATE documents SET ingestion_status = 'superseded', superseded_by = ? WHERE id = ?",
                        (document_id, prior["id"]),
                    )
        set_index_dirty(conn, True)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "document_id": document_id,
        "company_id": company_id,
        "chunk_count": len(chunks),
        "vector_ids": vector_ids,
        "embedding_model": resolved_model,
        "embedding_dim": resolved_dim,
    }
