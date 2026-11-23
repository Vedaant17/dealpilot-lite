-- Migration 003: RAG chunk and embedding persistence.

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    page_start INTEGER NOT NULL CHECK (page_start >= 1),
    page_end INTEGER NOT NULL CHECK (page_end >= page_start),
    char_count INTEGER NOT NULL CHECK (char_count >= 0),
    token_count_estimate INTEGER NOT NULL CHECK (token_count_estimate >= 0),
    chunking_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (document_id, chunk_index),
    UNIQUE (chunk_id, company_id),
    FOREIGN KEY (document_id, company_id) REFERENCES documents (id, company_id)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document
    ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_company
    ON document_chunks (company_id);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY,
    company_id TEXT NOT NULL,
    vector_id INTEGER NOT NULL UNIQUE CHECK (vector_id > 0),
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
    embedding BLOB NOT NULL,
    normalized INTEGER NOT NULL DEFAULT 1 CHECK (normalized IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE (chunk_id, company_id),
    FOREIGN KEY (chunk_id, company_id) REFERENCES document_chunks (chunk_id, company_id),
    FOREIGN KEY (company_id) REFERENCES companies (id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_company
    ON chunk_embeddings (company_id);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_vector_id
    ON chunk_embeddings (vector_id);

CREATE TABLE IF NOT EXISTS embedding_sequences (
    sequence_name TEXT PRIMARY KEY,
    next_value INTEGER NOT NULL CHECK (next_value > 0)
);

INSERT INTO embedding_sequences (sequence_name, next_value)
VALUES ('vector_id', 1)
ON CONFLICT(sequence_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS rag_index_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    index_dirty INTEGER NOT NULL DEFAULT 1 CHECK (index_dirty IN (0, 1)),
    last_verified_at TEXT,
    last_error TEXT,
    updated_at TEXT NOT NULL
);

INSERT INTO rag_index_state (id, index_dirty, last_verified_at, last_error, updated_at)
VALUES (1, 1, NULL, NULL, datetime('now'))
ON CONFLICT(id) DO NOTHING;