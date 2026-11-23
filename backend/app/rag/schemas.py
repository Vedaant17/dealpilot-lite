"""Typed structures for M6-A document ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


@dataclass
class PageExtraction:
    page_number: int
    text: str


@dataclass
class ExtractionResult:
    file_format: str
    page_count: int
    pages: list[PageExtraction] = field(default_factory=list)
    text: str = ""
    extraction_error: Optional[str] = None

    @property
    def has_usable_text(self) -> bool:
        return any(page.text.strip() for page in self.pages)


@dataclass
class IngestionResult:
    document_id: str
    company_id: str
    duplicate: bool
    storage_path: Path
    ingestion_status: str
    extraction: Optional[ExtractionResult] = None
    superseded_document_ids: list[str] = field(default_factory=list)


class DocumentIngestionError(Exception):
    """Base error for document ingestion."""


class DocumentValidationError(DocumentIngestionError, ValueError):
    """Raised when document input or policy validation fails."""


class DocumentExtractionError(DocumentIngestionError, ValueError):
    """Raised when a document cannot be extracted."""


@dataclass(frozen=True)
class Chunk:
    document_id: str
    company_id: str
    chunk_index: int
    text: str
    text_hash: str
    page_start: int
    page_end: int
    char_count: int
    token_count_estimate: int
    chunking_version: str


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: np.ndarray
    model: str
    dimension: int
    normalized: bool = True


@dataclass
class EmbeddingResult:
    vectors: list[np.ndarray]
    model: str
    dimension: int
    normalized: bool = True


@dataclass
class RetrievalResult:
    company_id: str
    document_id: str
    chunk_id: int
    chunk_index: int
    text: str
    source: str
    filename: str
    page_start: int
    page_end: int
    similarity_score: float
    embedding_model: str


@dataclass
class StoredEmbedding:
    chunk_id: int
    document_id: str
    company_id: str
    vector_id: int
    embedding_model: str
    embedding_dim: int
    vector: np.ndarray
    normalized: bool
    chunk_index: int
    text: str
    text_hash: str
    page_start: int
    page_end: int
    char_count: int
    token_count_estimate: int
    chunking_version: str
    created_at: str


class RagRetrievalToolInput(BaseModel):
    company_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
