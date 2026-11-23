-- Migration 002: Document ingestion metadata for M6-A.

ALTER TABLE documents ADD COLUMN file_format TEXT;
ALTER TABLE documents ADD COLUMN storage_path TEXT;
ALTER TABLE documents ADD COLUMN file_size_bytes INTEGER;
ALTER TABLE documents ADD COLUMN content_hash TEXT;
ALTER TABLE documents ADD COLUMN source_reference TEXT;
ALTER TABLE documents ADD COLUMN ingestion_status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE documents ADD COLUMN extraction_error TEXT;
ALTER TABLE documents ADD COLUMN superseded_by TEXT REFERENCES documents(id);

CREATE INDEX IF NOT EXISTS idx_documents_company_content_hash
    ON documents(company_id, content_hash)
    WHERE content_hash IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_company_content_hash_unique
    ON documents(company_id, content_hash)
    WHERE content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_documents_company_source
    ON documents(company_id, source_reference);

CREATE INDEX IF NOT EXISTS idx_documents_company_status
    ON documents(company_id, ingestion_status);