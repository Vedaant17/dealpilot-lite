"""Document ingestion and validation for M6-A."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from backend.app.config import settings
from backend.app.storage import repositories as repo
from backend.app.storage.models import Document

from .extraction import extract_document
from .schemas import DocumentValidationError, IngestionResult

SUPPORTED_FORMATS = {"pdf", "txt"}


def _storage_root(storage_root: Optional[Path]) -> Path:
    root = Path(storage_root) if storage_root is not None else Path(settings.document_storage_path)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _source_reference(source_path: Path, explicit: Optional[str]) -> str:
    if explicit is not None:
        return explicit
    return str(source_path.resolve())


def _safe_destination(storage_root: Path, document_id: str, suffix: str) -> Path:
    candidate = storage_root / f"{document_id}{suffix}"
    resolved_root = storage_root.resolve()
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_root not in resolved_candidate.parents and resolved_candidate != resolved_root:
        raise DocumentValidationError("unsafe storage path")
    return candidate


def _write_atomic(destination: Path, data: bytes) -> bool:
    existed = destination.exists()
    temp_path = destination.with_name(destination.name + ".tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return not existed


def ingest_document_bytes(
    conn,
    company_id: str,
    filename: str,
    data: bytes,
    *,
    storage_root: Optional[Path] = None,
    title: Optional[str] = None,
    doc_type: str = "uploaded",
    source_reference: Optional[str] = None,
    max_bytes: Optional[int] = None,
    max_pdf_pages: Optional[int] = None,
) -> IngestionResult:
    if not data:
        raise DocumentValidationError("document is empty")

    company = repo.get_company(conn, company_id)
    if company is None:
        raise DocumentValidationError(f"unknown company: {company_id}")

    suffix = Path(filename).suffix.lower()
    file_format = suffix.lstrip(".")
    if file_format not in SUPPORTED_FORMATS:
        raise DocumentValidationError(f"unsupported document format: {file_format or 'none'}")

    effective_max_bytes = max_bytes if max_bytes is not None else settings.rag_max_document_bytes
    effective_max_pages = max_pdf_pages if max_pdf_pages is not None else settings.rag_max_pdf_pages
    if len(data) > effective_max_bytes:
        raise DocumentValidationError(
            f"document exceeds maximum size of {effective_max_bytes} bytes"
        )

    content_hash = hashlib.sha256(data).hexdigest()
    existing = repo.get_document_by_content_hash(conn, company_id, content_hash)
    if existing is not None:
        return IngestionResult(
            document_id=existing.id,
            company_id=company_id,
            duplicate=True,
            storage_path=Path(existing.storage_path)
            if existing.storage_path
            else Path(existing.file_path_or_id),
            ingestion_status=existing.ingestion_status,
            extraction=None,
            superseded_document_ids=[],
        )

    resolved_source_reference = source_reference if source_reference is not None else filename

    document_id = f"doc-{company_id}-{content_hash[:16]}"
    root = _storage_root(storage_root)
    destination = _safe_destination(root, document_id, suffix)
    created_file = _write_atomic(destination, data)

    try:
        extraction = extract_document(data, file_format)
    except Exception:
        if created_file:
            destination.unlink(missing_ok=True)
        raise

    if file_format == "pdf" and extraction.page_count > effective_max_pages:
        if created_file:
            destination.unlink(missing_ok=True)
        raise DocumentValidationError(
            f"PDF has {extraction.page_count} pages, exceeding maximum of {effective_max_pages}"
        )

    storage_path = destination.relative_to(root).as_posix()
    ingestion_status = "ready" if extraction.has_usable_text else "failed"
    extraction_error = extraction.extraction_error

    document = Document(
        id=document_id,
        company_id=company_id,
        title=title or Path(filename).name,
        doc_type=doc_type,
        file_path_or_id=storage_path,
        page_count=extraction.page_count,
        metadata={
            "original_filename": Path(filename).name,
            "source_reference": resolved_source_reference,
        },
        file_format=file_format,
        storage_path=storage_path,
        file_size_bytes=len(data),
        content_hash=content_hash,
        source_reference=resolved_source_reference,
        ingestion_status=ingestion_status,
        extraction_error=extraction_error,
    )

    conn.execute("BEGIN IMMEDIATE")
    try:
        repo.insert_document(conn, document)
        conn.commit()
    except Exception:
        conn.rollback()
        if created_file:
            destination.unlink(missing_ok=True)
        raise

    return IngestionResult(
        document_id=document.id,
        company_id=company_id,
        duplicate=False,
        storage_path=destination,
        ingestion_status=ingestion_status,
        extraction=extraction,
        superseded_document_ids=[],
    )


def ingest_document(
    conn,
    company_id: str,
    source_path: str | Path,
    *,
    storage_root: Optional[Path] = None,
    title: Optional[str] = None,
    doc_type: str = "uploaded",
    source_reference: Optional[str] = None,
    max_bytes: Optional[int] = None,
    max_pdf_pages: Optional[int] = None,
) -> IngestionResult:
    path = Path(source_path)
    if not path.is_file():
        raise DocumentValidationError(f"document not found: {path}")
    data = path.read_bytes()
    resolved_source_reference = (
        source_reference if source_reference is not None else str(path.resolve())
    )
    return ingest_document_bytes(
        conn,
        company_id,
        path.name,
        data,
        storage_root=storage_root,
        title=title,
        doc_type=doc_type,
        source_reference=resolved_source_reference,
        max_bytes=max_bytes,
        max_pdf_pages=max_pdf_pages,
    )
