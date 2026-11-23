"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    llm_model: str = ""
    gemini_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/v1/"
    ollama_num_ctx: int = 4096
    ollama_base_url: str = "http://localhost:11434/v1/"
    ollama_num_ctx: int = 4096
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    sqlite_db_path: str = "data/dealpilot.db"
    faiss_index_path: str = "data/faiss.index"
    document_storage_path: str = "data/documents"
    rag_max_document_bytes: int = 10 * 1024 * 1024
    rag_max_pdf_pages: int = 100
    rag_chunk_chars: int = 3200
    rag_chunk_overlap_chars: int = 400
    rag_embedding_batch_size: int = 32
    rag_embedding_timeout_seconds: float = 30.0
    rag_embedding_max_retries: int = 3
    rag_top_k_default: int = 5
    rag_top_k_max: int = 20
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    agent_max_rounds: int = 3
    agent_max_tool_calls: int = 15
    agent_timeout_seconds: float = 90.0
    agent_max_tool_result_chars: int = 12000
    cors_allowed_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
