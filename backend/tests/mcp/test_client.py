import pytest

from backend.app.mcp_client.client import MCPToolClient
from backend.scripts.seed_db import seed_database


@pytest.mark.asyncio
async def test_client_discovers_five_tools(tmp_path):
    db_path = str(tmp_path / "test.db")
    seed_database(db_path)
    async with MCPToolClient(db_path=db_path) as client:
        tools = await client.list_tools()
        assert len(tools) == 5


@pytest.mark.asyncio
async def test_client_invokes_metrics(tmp_path):
    db_path = str(tmp_path / "test.db")
    seed_database(db_path)
    async with MCPToolClient(db_path=db_path) as client:
        result = await client.call_tool(
            "calculate_financial_metrics",
            {"company_id": "cedar-health", "period_range": "2023-2023"},
        )
        assert result["periods"][0]["reporting_currency"] == "EUR"
        assert result["periods"][0]["metrics"]["operating_nwc"]["value"] is not None


@pytest.mark.asyncio
async def test_client_malformed_arguments(tmp_path):
    db_path = str(tmp_path / "test.db")
    seed_database(db_path)
    async with MCPToolClient(db_path=db_path) as client:
        with pytest.raises(Exception):
            await client.call_tool(
                "calculate_financial_metrics", {"company_id": "cedar-health", "period_range": "bad"}
            )
