"""Deterministic page-aware document chunking for M6-B."""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from backend.app.config import settings
from backend.app.rag.extraction import ExtractionResult
from backend.app.rag.schemas import Chunk


def validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if chunk_size > 100000:
        raise ValueError("chunk_size exceeds the supported maximum")
    if chunk_overlap > 10000:
        raise ValueError("chunk_overlap exceeds the supported maximum")


def chunking_version(chunk_size: int, chunk_overlap: int) -> str:
    return f"char-v1-{chunk_size}-{chunk_overlap}"


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def _page_segments(extraction: ExtractionResult) -> list[tuple[int, str, int, int]]:
    segments: list[tuple[int, str, int, int]] = []
    offset = 0
    for page in extraction.pages:
        text = _normalize_text(page.text).strip()
        if not text:
            continue
        if segments:
            offset += 2
        start = offset
        end = start + len(text)
        segments.append((page.page_number, text, start, end))
        offset = end
    return segments


def _pages_for_range(
    segments: list[tuple[int, str, int, int]], start: int, end: int
) -> tuple[int, int]:
    contributing = [segment for segment in segments if segment[2] < end and segment[3] > start]
    if not contributing:
        # This should not occur for a non-empty chunk, but keep a deterministic fallback.
        nearest = min(segments, key=lambda segment: abs(segment[2] - start))
        return nearest[0], nearest[0]
    return min(segment[0] for segment in contributing), max(segment[0] for segment in contributing)


def chunk_extraction(
    extraction: ExtractionResult,
    document_id: str,
    company_id: str,
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    chunking_version_override: Optional[str] = None,
) -> list[Chunk]:
    if extraction is None:
        raise ValueError("extraction is required")
    if not document_id:
        raise ValueError("document_id is required")
    if not company_id:
        raise ValueError("company_id is required")

    effective_size = settings.rag_chunk_chars if chunk_size is None else chunk_size
    effective_overlap = settings.rag_chunk_overlap_chars if chunk_overlap is None else chunk_overlap
    validate_chunk_config(effective_size, effective_overlap)

    segments = _page_segments(extraction)
    if not segments:
        return []

    full_text = "\n\n".join(segment[1] for segment in segments)
    version = chunking_version_override or chunking_version(effective_size, effective_overlap)

    chunks: list[Chunk] = []
    start = 0
    length = len(full_text)
    while start < length:
        end = min(start + effective_size, length)
        if end < length:
            boundary = full_text.rfind(" ", start, end)
            minimum_boundary = start + max(1, effective_size // 2)
            if boundary >= minimum_boundary:
                end = boundary

        chunk_text = full_text[start:end].strip()
        if chunk_text:
            page_start, page_end = _pages_for_range(segments, start, end)
            chunks.append(
                Chunk(
                    document_id=document_id,
                    company_id=company_id,
                    chunk_index=len(chunks),
                    text=chunk_text,
                    text_hash=hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                    page_start=page_start,
                    page_end=page_end,
                    char_count=len(chunk_text),
                    token_count_estimate=max(1, (len(chunk_text) + 3) // 4),
                    chunking_version=version,
                )
            )

        if end >= length:
            break

        next_start = end - effective_overlap
        if next_start <= start:
            next_start = start + 1
        while next_start < length and full_text[next_start].isspace():
            next_start += 1
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def chunk_document(*args, **kwargs) -> list[Chunk]:
    return chunk_extraction(*args, **kwargs)


def chunk_pages(pages, document_id: str, company_id: str, **kwargs) -> list[Chunk]:
    extraction = ExtractionResult(
        file_format="txt",
        page_count=len(pages),
        pages=pages,
        text="\n\n".join(getattr(page, "text", str(page)) for page in pages),
    )
    return chunk_extraction(extraction, document_id, company_id, **kwargs)


def chunk_text(
    text: str,
    *,
    document_id: str = "",
    company_id: str = "",
    page_number: int = 1,
    **kwargs,
) -> list[Chunk]:
    extraction = ExtractionResult(
        file_format="txt",
        page_count=1,
        pages=[
            __import__("backend.app.rag.schemas", fromlist=["PageExtraction"]).PageExtraction(
                page_number=page_number, text=text
            )
        ],
        text=text,
    )
    return chunk_extraction(extraction, document_id, company_id, **kwargs)
