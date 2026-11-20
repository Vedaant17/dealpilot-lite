# DealPilot Lite

Agentic AI Private Equity Screening Assistant.

## Tech Stack
- Python 3.12, FastAPI, LangGraph, MCP SDK v2
- SQLite + FAISS for local storage
- OpenAI API for LLM and embeddings
- Nuxt 4, Vue 3, TypeScript frontend

## Setup
1. Create venv: `python -m venv .venv`
2. Install deps: `.venv\Scripts\pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add OpenAI API key
4. Run backend: `.venv\Scripts\uvicorn backend.app.main:app --reload`
5. Run frontend: `cd frontend && npm run dev`

## Note
All company data is synthetic and fictional. This is a decision-support tool, not an investment recommendation.