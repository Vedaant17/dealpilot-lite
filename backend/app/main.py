"""FastAPI application entry point."""

from __future__ import annotations

import importlib.util
import logging
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.screenings import router as screenings_router
from backend.app.config import settings
from backend.app.storage.connection import get_connection
from backend.app.storage.schema import get_schema_version

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(title="DealPilot Lite", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(screenings_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/ready")
def ready():
    db_path = Path(settings.sqlite_db_path)
    if not db_path.is_file():
        raise HTTPException(status_code=503, detail="Database is not available")
    conn = get_connection(str(db_path))
    try:
        version = get_schema_version(conn)
    finally:
        conn.close()
    if version < 3:
        raise HTTPException(status_code=503, detail="Database schema is not ready")
    provider = settings.llm_provider.strip().lower()
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise HTTPException(status_code=503, detail="Gemini API key is not configured")
    elif provider == "openai":
        if not settings.openai_api_key:
            raise HTTPException(status_code=503, detail="OpenAI API key is not configured")
    elif provider == "ollama":
        try:
            with urllib.request.urlopen(
                settings.ollama_base_url.rstrip("/") + "/models", timeout=2.0
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Ollama is not available")
        except (urllib.error.URLError, TimeoutError, ValueError):
            raise HTTPException(status_code=503, detail="Ollama is not available")
    else:
        raise HTTPException(status_code=503, detail="Unsupported LLM provider")
    if not Path(settings.document_storage_path).exists():
        raise HTTPException(status_code=503, detail="Document storage is not available")
    if importlib.util.find_spec("backend.app.mcp_server.server") is None:
        raise HTTPException(status_code=503, detail="MCP server module is not available")
    return {
        "status": "ready",
        "database": "available",
        "schema_version": version,
        "openai": "configured",
    }
