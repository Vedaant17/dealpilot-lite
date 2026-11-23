import sqlite3

from backend.app.storage.connection import get_connection, init_db


def test_migration_002_adds_document_columns(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = get_connection(db_path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
    assert "content_hash" in columns
    assert "storage_path" in columns
    assert "ingestion_status" in columns
    assert "extraction_error" in columns
    assert "superseded_by" in columns
    indexes = {row["name"] for row in conn.execute("PRAGMA index_list(documents)").fetchall()}
    assert "idx_documents_company_content_hash_unique" in indexes
    conn.close()


def test_foreign_key_rejects_unknown_company(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO documents (id, company_id, title, doc_type, file_path_or_id, ingested_at, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("doc-1", "missing-company", "Title", "uploaded", "x", "2024-01-01", "abc"),
        )
        raise AssertionError("foreign key was not enforced")
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()
