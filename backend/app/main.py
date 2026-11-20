"""FastAPI application entry point."""

import logging

from fastapi import FastAPI

from backend.app.config import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(title="DealPilot Lite", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}