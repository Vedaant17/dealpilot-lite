import pytest
from fastapi.testclient import TestClient

from backend.app.agent.service import AgentRunResult
from backend.app.main import app
from backend.app.storage.connection import init_db
from backend.scripts.seed_db import seed_database


class FakeAgentService:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc

    async def run_result(self, user_input, company_id=None, db_path=None, top_k=None):
        if self.exc is not None:
            raise self.exc
        return self.result


def _seeded_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    seed_database(db_path)
    return db_path


def _client_with_service(result=None, exc=None, db_path=None):
    service = FakeAgentService(result=result, exc=exc)
    app.dependency_overrides[get_agent_service_dependency()] = lambda: service
    client = TestClient(app)
    if db_path is not None:
        import backend.app.api.v1.screenings as screenings

        screenings.settings.sqlite_db_path = db_path
    return client, service


def get_agent_service_dependency():
    from backend.app.api.v1.screenings import get_agent_service

    return get_agent_service


@pytest.fixture
def seeded_db(tmp_path):
    return _seeded_db(tmp_path)


def test_valid_screening(seeded_db):
    result = AgentRunResult(
        status="completed",
        narrative="Screening narrative",
        retrieved_evidence=[
            {
                "company_id": "aurora-retail",
                "document_id": "doc-1",
                "chunk_id": 1,
                "chunk_index": 0,
                "text_preview": "Evidence text",
                "source": "source.txt",
                "filename": "source.txt",
                "page_start": 1,
                "page_end": 1,
                "similarity_score": 0.9,
                "embedding_model": "text-embedding-3-small",
            }
        ],
        citation_warnings=[],
        tool_errors=[],
        tool_call_count=2,
        round_count=1,
        termination_reason=None,
        duration_ms=10,
        tools_used=["get_company_financials", "retrieve_company_documents"],
    )
    client, _ = _client_with_service(result=result, db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings",
        json={"company_id": "aurora-retail", "query": "Screen Aurora"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["company_id"] == "aurora-retail"
    assert body["company_name"] == "Aurora Retail Solutions"
    assert body["status"] == "completed"
    assert body["narrative"] == "Screening narrative"
    assert body["evidence"][0]["document_id"] == "doc-1"
    assert body["execution"]["tool_call_count"] == 2
    assert "synthetic" in " ".join(body["disclaimers"]).lower()


def test_unknown_company(seeded_db):
    client, _ = _client_with_service(db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "missing-company", "query": "Screen"}
    )
    assert response.status_code == 404


def test_invalid_body(seeded_db):
    client, _ = _client_with_service(db_path=seeded_db)
    response = client.post("/api/v1/screenings", json={"query": "Screen"})
    assert response.status_code == 422


def test_empty_query(seeded_db):
    client, _ = _client_with_service(db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "   "}
    )
    assert response.status_code == 422


def test_oversized_query(seeded_db):
    client, _ = _client_with_service(db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "x" * 4001}
    )
    assert response.status_code == 422


def test_invalid_top_k(seeded_db):
    client, _ = _client_with_service(db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "Screen", "top_k": 21}
    )
    assert response.status_code == 422


def test_citation_warnings(seeded_db):
    result = AgentRunResult(
        status="completed",
        narrative="Narrative",
        retrieved_evidence=[],
        citation_warnings=["unverified citation: [doc:missing chunk:999]"],
        tool_errors=[],
        tool_call_count=1,
        round_count=1,
        termination_reason=None,
        duration_ms=5,
        tools_used=["retrieve_company_documents"],
    )
    client, _ = _client_with_service(result=result, db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "Screen"}
    )
    assert response.status_code == 200
    assert response.json()["citation_warnings"] == ["unverified citation: [doc:missing chunk:999]"]


def test_rag_failure_partial(seeded_db):
    result = AgentRunResult(
        status="partial",
        narrative="Partial narrative",
        retrieved_evidence=[],
        citation_warnings=[],
        tool_errors=["retrieve_company_documents: retrieval failed"],
        tool_call_count=1,
        round_count=1,
        termination_reason=None,
        duration_ms=5,
        tools_used=["retrieve_company_documents"],
    )
    client, _ = _client_with_service(result=result, db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "Screen"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert "Tool error: retrieve_company_documents: retrieval failed" in body["warnings"]


def test_financial_failure_partial(seeded_db):
    result = AgentRunResult(
        status="partial",
        narrative="Partial narrative",
        retrieved_evidence=[],
        citation_warnings=[],
        tool_errors=["get_company_financials: tool failed"],
        tool_call_count=1,
        round_count=1,
        termination_reason=None,
        duration_ms=5,
        tools_used=["get_company_financials"],
    )
    client, _ = _client_with_service(result=result, db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "Screen"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "partial"


def test_model_failure(seeded_db):
    client, _ = _client_with_service(exc=RuntimeError("OpenAI model failure"), db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "Screen"}
    )
    assert response.status_code == 502


def test_mcp_failure(seeded_db):
    client, _ = _client_with_service(exc=RuntimeError("MCP server failed"), db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "Screen"}
    )
    assert response.status_code == 503


def test_timeout(seeded_db):
    client, _ = _client_with_service(exc=RuntimeError("agent timeout"), db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "Screen"}
    )
    assert response.status_code == 504


def test_unexpected_internal_exception(seeded_db):
    client, _ = _client_with_service(exc=ValueError("unexpected"), db_path=seeded_db)
    response = client.post(
        "/api/v1/screenings", json={"company_id": "aurora-retail", "query": "Screen"}
    )
    assert response.status_code == 500


def test_health():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready(seeded_db, monkeypatch):
    import backend.app.api.v1.screenings as screenings
    import backend.app.main as main

    screenings.settings.sqlite_db_path = seeded_db
    main.settings.sqlite_db_path = seeded_db
    monkeypatch.setattr("backend.app.config.settings.openai_api_key", "test-key")
    from fastapi.testclient import TestClient

    from backend.app.main import app

    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_cors(seeded_db):
    client, _ = _client_with_service(db_path=seeded_db)
    response = client.options(
        "/api/v1/screenings",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
