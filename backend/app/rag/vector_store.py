"""FAISS lifecycle and SQLite synchronization for M6-C."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from backend.app.config import settings
from backend.app.rag.persistence import (
    get_active_embeddings,
    get_index_state,
    mark_index_verified,
    set_index_dirty,
)
from backend.app.rag.schemas import RetrievalResult

# Re-export persistence helpers for callers that import from this module.


class IndexVerificationError(RuntimeError):
    """Raised when FAISS and SQLite are not consistent."""


class VectorStore:
    def __init__(
        self,
        db_path: str,
        index_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_model: Optional[str] = None,
        lock: Optional[threading.RLock] = None,
    ) -> None:
        self.db_path = db_path
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.embedding_dim = embedding_dim or settings.embedding_dim
        self.embedding_model = embedding_model or settings.openai_embedding_model
        self._lock = lock if lock is not None else threading.RLock()
        self._index: faiss.IndexIDMap2 | None = None

    def _connect(self) -> sqlite3.Connection:
        from backend.app.storage.connection import get_connection

        return get_connection(self.db_path)

    def _index_file_exists(self) -> bool:
        return self.index_path.is_file()

    def load_index(self) -> faiss.IndexIDMap2:
        with self._lock:
            if self._index is not None:
                return self._index
            if not self._index_file_exists():
                raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
            index = faiss.read_index(str(self.index_path))
            if not isinstance(index, faiss.IndexIDMap2):
                raise ValueError("FAISS index is not an ID-mapped index")
            if index.d != self.embedding_dim:
                raise ValueError(
                    f"FAISS dimension {index.d} does not match configured dimension {self.embedding_dim}"
                )
            self._index = index
            return index

    def get_index(self) -> faiss.IndexIDMap2:
        if self._index is None:
            return self.load_index()
        return self._index

    def _faiss_ids(self, index: faiss.IndexIDMap2) -> set[int]:
        return set(int(value) for value in faiss.vector_to_array(index.id_map).tolist())

    def verify_index(self) -> None:
        with self._lock:
            index = self.get_index()
            conn = self._connect()
            try:
                active = get_active_embeddings(
                    conn,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dim,
                )
                active_ids = {item.vector_id for item in active}
                faiss_ids = self._faiss_ids(index)
                if index.d != self.embedding_dim:
                    raise IndexVerificationError("FAISS dimension mismatch")
                if len(active_ids) != index.ntotal or active_ids != faiss_ids:
                    raise IndexVerificationError("FAISS and SQLite ID sets differ")
                for item in active:
                    reconstructed = index.reconstruct(item.vector_id)
                    reconstructed_array = np.asarray(reconstructed, dtype=np.float32)
                    if not np.allclose(reconstructed_array, item.vector, atol=1e-6):
                        raise IndexVerificationError("FAISS and SQLite vectors differ")
            finally:
                conn.close()

    def rebuild_index(self) -> faiss.IndexIDMap2:
        with self._lock:
            conn = self._connect()
            temp_path = self.index_path.with_name(self.index_path.name + ".tmp")
            try:
                active = get_active_embeddings(
                    conn,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dim,
                )
                index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))
                if active:
                    vectors = np.stack([item.vector for item in active]).astype(np.float32)
                    ids = np.asarray([item.vector_id for item in active], dtype=np.int64)
                    index.add_with_ids(vectors, ids)

                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(index, str(temp_path))
                os.replace(temp_path, self.index_path)

                reopened = faiss.read_index(str(self.index_path))
                if not isinstance(reopened, faiss.IndexIDMap2) or reopened.d != self.embedding_dim:
                    raise IndexVerificationError("rebuilt FAISS index failed validation")
                reopened_ids = set(
                    int(value) for value in faiss.vector_to_array(reopened.id_map).tolist()
                )
                active_ids = {item.vector_id for item in active}
                if reopened_ids != active_ids or reopened.ntotal != len(active_ids):
                    raise IndexVerificationError("rebuilt FAISS index ID set mismatch")

                self._index = reopened
                mark_index_verified(conn)
                conn.commit()
                return reopened
            except Exception as exc:
                try:
                    set_index_dirty(conn, True, str(exc))
                    conn.commit()
                except Exception:
                    conn.rollback()
                temp_path.unlink(missing_ok=True)
                raise
            finally:
                conn.close()

    def ensure_index(self) -> faiss.IndexIDMap2:
        with self._lock:
            conn = self._connect()
            try:
                state = get_index_state(conn)
                dirty = bool(state["index_dirty"])
                conn.close()
            except Exception:
                conn.close()
                raise

            if dirty or not self._index_file_exists():
                return self.rebuild_index()

            try:
                index = self.load_index()
                self.verify_index()
                return index
            except Exception:
                return self.rebuild_index()

    def retrieve(self, company_id: str, query_vector: Any, top_k: int) -> list[RetrievalResult]:
        import json

        if not company_id:
            raise ValueError("company_id is required")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        with self._lock:
            index = self.ensure_index()
            vector = np.asarray(query_vector, dtype=np.float32)
            if vector.ndim != 1 or vector.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"query vector dimension must be {self.embedding_dim}, got {vector.shape[0]}"
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError("query vector contains non-finite values")
            norm = float(np.linalg.norm(vector))
            if norm == 0.0 or not np.isfinite(norm):
                raise ValueError("query vector has zero or invalid norm")

            conn = self._connect()
            try:
                active = get_active_embeddings(
                    conn,
                    company_id=company_id,
                    embedding_model=self.embedding_model,
                    embedding_dim=self.embedding_dim,
                )
                active_ids = {item.vector_id for item in active}
                if index.ntotal == 0 or not active_ids:
                    return []

                distances, ids = index.search(vector.reshape(1, -1), index.ntotal)
                selected: list[tuple[int, float]] = []
                for score, vector_id in zip(distances[0], ids[0]):
                    candidate = int(vector_id)
                    if candidate == -1 or candidate not in active_ids:
                        continue
                    selected.append((candidate, float(score)))
                    if len(selected) >= top_k:
                        break

                if not selected:
                    return []

                placeholders = ",".join("?" for _ in selected)
                rows = conn.execute(
                    f"""SELECT ce.vector_id, ce.embedding_model, dc.document_id, dc.chunk_id,
                               dc.chunk_index, dc.text, dc.page_start, dc.page_end,
                               d.file_path_or_id, d.source_reference, d.title, d.metadata
                        FROM chunk_embeddings AS ce
                        JOIN document_chunks AS dc
                          ON ce.chunk_id = dc.chunk_id AND ce.company_id = dc.company_id
                        JOIN documents AS d
                          ON dc.document_id = d.id AND dc.company_id = d.company_id
                        WHERE ce.company_id = ? AND ce.vector_id IN ({placeholders})""",
                    [company_id] + [vector_id for vector_id, _ in selected],
                ).fetchall()
                by_id = {int(row["vector_id"]): row for row in rows}

                results: list[RetrievalResult] = []
                for vector_id, score in selected:
                    row = by_id.get(vector_id)
                    if row is None:
                        continue
                    metadata = {}
                    if row["metadata"]:
                        try:
                            metadata = json.loads(row["metadata"])
                        except (TypeError, ValueError):
                            metadata = {}
                    filename = metadata.get("original_filename")
                    if not filename:
                        filename = Path(row["file_path_or_id"]).name
                    source = row["source_reference"] or row["file_path_or_id"] or row["title"]
                    results.append(
                        RetrievalResult(
                            company_id=company_id,
                            document_id=row["document_id"],
                            chunk_id=int(row["chunk_id"]),
                            chunk_index=int(row["chunk_index"]),
                            text=row["text"],
                            source=source,
                            filename=filename,
                            page_start=int(row["page_start"]),
                            page_end=int(row["page_end"]),
                            similarity_score=score,
                            embedding_model=row["embedding_model"],
                        )
                    )
                return results
            finally:
                conn.close()

    def close(self) -> None:
        self._index = None
