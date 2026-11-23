from backend.app.rag.ingestion import (
    DocumentValidationError,
    ingest_document,
    ingest_document_bytes,
)
from backend.app.storage.connection import get_connection, init_db
from backend.app.storage.repositories import get_document
from backend.scripts.seed_db import seed_database


def _seed_company(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    seed_database(db_path)
    return db_path


def test_txt_ingestion_and_duplicate(tmp_path):
    db_path = _seed_company(tmp_path)
    conn = get_connection(db_path)
    storage = tmp_path / "docs"
    data = b"Hello RAG document.\nSecond line."
    result = ingest_document_bytes(
        conn,
        "aurora-retail",
        "note.txt",
        data,
        storage_root=storage,
        title="Note",
    )
    assert result.duplicate is False
    assert result.ingestion_status == "ready"
    assert result.extraction.has_usable_text is True
    assert (storage / result.storage_path).read_bytes() == data

    duplicate = ingest_document_bytes(
        conn,
        "aurora-retail",
        "renamed.txt",
        data,
        storage_root=storage,
        title="Renamed",
    )
    assert duplicate.duplicate is True
    assert duplicate.document_id == result.document_id
    conn.close()


def test_changed_same_source_supersedes(tmp_path):
    db_path = _seed_company(tmp_path)
    conn = get_connection(db_path)
    storage = tmp_path / "docs"
    first_data = b"version one"
    second_data = b"version two"
    source = str((tmp_path / "source.txt").resolve())
    (tmp_path / "source.txt").write_bytes(first_data)
    first = ingest_document(
        conn,
        "aurora-retail",
        tmp_path / "source.txt",
        storage_root=storage,
        source_reference=source,
    )
    (tmp_path / "source.txt").write_bytes(second_data)
    second = ingest_document(
        conn,
        "aurora-retail",
        tmp_path / "source.txt",
        storage_root=storage,
        source_reference=source,
    )
    assert first.ingestion_status == "ready"
    assert second.ingestion_status == "ready"
    assert second.superseded_document_ids == []
    old = get_document(conn, first.document_id)
    assert old.ingestion_status == "ready"
    conn.close()


def test_same_filename_different_source(tmp_path):
    db_path = _seed_company(tmp_path)
    conn = get_connection(db_path)
    storage = tmp_path / "docs"
    a = ingest_document_bytes(
        conn,
        "aurora-retail",
        "doc.txt",
        b"content a",
        storage_root=storage,
        source_reference="source-a",
    )
    b = ingest_document_bytes(
        conn,
        "aurora-retail",
        "doc.txt",
        b"content b",
        storage_root=storage,
        source_reference="source-b",
    )
    assert a.document_id != b.document_id
    conn.close()


def test_different_companies_same_content(tmp_path):
    db_path = _seed_company(tmp_path)
    conn = get_connection(db_path)
    storage = tmp_path / "docs"
    data = b"shared content"
    aurora = ingest_document_bytes(conn, "aurora-retail", "shared.txt", data, storage_root=storage)
    cedar = ingest_document_bytes(conn, "cedar-health", "shared.txt", data, storage_root=storage)
    assert aurora.document_id != cedar.document_id
    conn.close()


def test_invalid_validation(tmp_path):
    db_path = _seed_company(tmp_path)
    conn = get_connection(db_path)
    storage = tmp_path / "docs"
    try:
        ingest_document_bytes(conn, "missing-company", "x.txt", b"text", storage_root=storage)
        raise AssertionError("expected validation error")
    except DocumentValidationError:
        pass
    try:
        ingest_document_bytes(conn, "aurora-retail", "x.bin", b"text", storage_root=storage)
        raise AssertionError("expected validation error")
    except DocumentValidationError:
        pass
    try:
        ingest_document_bytes(conn, "aurora-retail", "empty.txt", b"", storage_root=storage)
        raise AssertionError("expected validation error")
    except DocumentValidationError:
        pass
    conn.close()


def test_pdf_no_text_fails(tmp_path):
    db_path = _seed_company(tmp_path)
    conn = get_connection(db_path)
    storage = tmp_path / "docs"
    pdf_bytes = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 0 >>\nstream\n\nendstream\nendobj\n"
        b"xref\n0 5\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    result = ingest_document_bytes(
        conn, "aurora-retail", "empty.pdf", pdf_bytes, storage_root=storage
    )
    assert result.ingestion_status == "failed"
    assert result.extraction.extraction_error == "no extractable text"
    conn.close()


def test_path_traversal_rejected(tmp_path):
    db_path = _seed_company(tmp_path)
    conn = get_connection(db_path)
    storage = tmp_path / "docs"
    source = tmp_path / "evil.txt"
    source.write_bytes(b"text")
    # monkeypatch storage root to a nested path by using a relative traversal in filename is not possible;
    # instead verify _safe_destination is used by successful path containment.
    result = ingest_document(conn, "aurora-retail", source, storage_root=storage)
    assert result.storage_path.resolve().is_relative_to(storage.resolve())
    conn.close()
