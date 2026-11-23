"""PDF and TXT extraction for M6-A."""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader

from backend.app.rag.schemas import DocumentExtractionError, ExtractionResult, PageExtraction


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_pdf(data: bytes) -> ExtractionResult:
    if not data:
        raise DocumentExtractionError("PDF data is empty")

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise DocumentExtractionError(f"corrupt or unreadable PDF: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        raise DocumentExtractionError("encrypted PDFs are not supported")

    pages: list[PageExtraction] = []
    try:
        page_count = len(reader.pages)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageExtraction(page_number=page_number, text=_normalize_text(text)))
    except Exception as exc:  # noqa: BLE001
        raise DocumentExtractionError(f"PDF extraction failed: {exc}") from exc

    full_text = "\n\n".join(page.text for page in pages)
    error = None if any(page.text.strip() for page in pages) else "no extractable text"
    return ExtractionResult(
        file_format="pdf",
        page_count=page_count,
        pages=pages,
        text=full_text,
        extraction_error=error,
    )


def extract_txt(data: bytes) -> ExtractionResult:
    if not data:
        raise DocumentExtractionError("TXT data is empty")

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentExtractionError(f"invalid UTF-8 text: {exc}") from exc

    text = _normalize_text(text)
    pages = [PageExtraction(page_number=1, text=text)]
    error = None if text.strip() else "no extractable text"
    return ExtractionResult(
        file_format="txt",
        page_count=1,
        pages=pages,
        text=text,
        extraction_error=error,
    )


def extract_document(data: bytes, file_format: str) -> ExtractionResult:
    normalized = file_format.lower().lstrip(".")
    if normalized == "pdf":
        return extract_pdf(data)
    if normalized == "txt":
        return extract_txt(data)
    raise DocumentExtractionError(f"unsupported document format: {file_format}")


def extract_from_path(path: str | Path) -> ExtractionResult:
    source = Path(path)
    data = source.read_bytes()
    return extract_document(data, source.suffix)
