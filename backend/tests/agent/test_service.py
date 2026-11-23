from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agent.service import AgentService


class FakeMCPClient:
    def __init__(self, db_path=None, timeout=30.0):
        self.db_path = db_path

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def list_tools(self):
        return [
            {
                "name": "get_company_financials",
                "description": "d",
                "inputSchema": {
                    "type": "object",
                    "properties": {"company_id": {"type": "string"}},
                    "required": ["company_id"],
                },
            }
        ]

    async def call_tool(self, name, arguments):
        return {"value": "1", "currency": "USD"}


@pytest.mark.asyncio
async def test_service_returns_final_response(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    message = SimpleNamespace(content="final", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    with (
        patch("backend.app.agent.service.MCPToolClient", FakeMCPClient),
        patch("backend.app.agent.nodes.AsyncOpenAI") as mocked,
    ):
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(return_value=response)
        service = AgentService()
        result = await service.run("hello", db_path=db_path)
        assert result == "final"
