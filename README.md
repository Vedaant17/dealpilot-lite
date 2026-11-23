# DealPilot Lite

Agentic AI Private Equity Screening Assistant.

DealPilot Lite is a local-first MVP for screening synthetic private-equity targets. It combines deterministic financial calculations, SQLite-backed company data, company-filtered RAG retrieval, an MCP tool layer, a LangGraph agent, and a Nuxt screening workspace.

## Tech Stack
- Python 3.12, FastAPI, LangGraph, MCP SDK v2
- SQLite for authoritative structured storage
- FAISS CPU for local vector search
- OpenAI API for embeddings and optional LLM inference
- Optional local Ollama/Qwen3 LLM inference
- Nuxt 4, Vue 3, TypeScript frontend

## Setup

### Backend
1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Install the project:
   ```powershell
   python -m pip install -e .
   ```
3. Copy environment configuration:
   ```powershell
   Copy-Item .env.example .env
   ```
4. Add an OpenAI API key to `.env` for embeddings and OpenAI LLM inference. OpenAI embeddings remain required for the existing RAG pipeline.
5. Initialize and seed the local database:
   ```powershell
   python backend/scripts/seed_db.py
   ```
   This creates `data/dealpilot.db`.
6. Start the backend:
   ```powershell
   .venv\Scripts\uvicorn backend.app.main:app --reload
   ```

### Frontend
```powershell
cd frontend
pnpm install
pnpm dev
```

The frontend runs on `http://localhost:3000` and connects to the backend at `http://127.0.0.1:8000` by default.

## Local LLM Provider
The agent supports OpenAI, Gemini, and optional local Ollama.

For Ollama:
```powershell
$env:LLM_PROVIDER = "ollama"
$env:LLM_MODEL = "qwen3:4b"
$env:OLLAMA_BASE_URL = "http://localhost:11434/v1/"
```

Ollama runs locally and does not require an API key for LLM inference. OpenAI embeddings are still required for RAG retrieval.

Qwen3 4B is suitable as a local candidate, but inference on an 8 GB CPU-only machine can be slow. The default model context is large; use a bounded context where supported.

## Tests

Backend:
```powershell
python -m pytest backend/tests/ -q
python -m ruff check backend
python -m ruff format --check backend
```

Frontend:
```powershell
cd frontend
pnpm install --frozen-lockfile
pnpm typecheck
pnpm test
pnpm build
```

## Data and Safety Notes
- All company and document data is synthetic and fictional.
- Screening and valuation output is decision support, not investment, legal, tax, or financial advice.
- RAG embeddings currently depend on OpenAI.
- No authentication or multi-user support is implemented in the MVP.

## Note
This is a local-first MVP. Do not commit real API keys, databases, FAISS indexes, generated documents, or virtual environments.