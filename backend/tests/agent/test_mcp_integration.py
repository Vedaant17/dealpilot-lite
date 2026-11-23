import pytest

from backend.app.mcp_client.client import MCPToolClient, MCPToolError
from backend.scripts.seed_db import seed_database


@pytest.fixture
def seeded_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    seed_database(db_path)
    return db_path


@pytest.mark.asyncio
async def test_all_five_tools_over_stdio(seeded_db):
    async with MCPToolClient(db_path=seeded_db) as client:
        tools = await client.list_tools()
        names = {tool["name"] for tool in tools}
        assert names == {
            "get_company_financials",
            "calculate_financial_metrics",
            "calculate_valuation",
            "calculate_comparable_multiples",
            "run_sensitivity_analysis",
        }

        financials = await client.call_tool(
            "get_company_financials", {"company_id": "aurora-retail", "period_range": "2023-2023"}
        )
        assert financials["periods"][0]["reporting_currency"] == "USD"
        assert financials["periods"][0]["income_statement"]["revenue"] == "60000000"

        metrics = await client.call_tool(
            "calculate_financial_metrics",
            {"company_id": "borealis-industrial", "period_range": "2022-2022"},
        )
        assert metrics["periods"][0]["metrics"]["ebitda"]["value"] == "5760000"

        valuation = await client.call_tool(
            "calculate_valuation",
            {
                "company_id": "aurora-retail",
                "base_period": "2023",
                "assumptions": {"revenue_growth_rate": "0.05", "cost_of_equity": "0.10"},
            },
        )
        assert valuation["available"] is True
        assert valuation["currency"] == "USD"
        assert valuation["enterprise_value"] is not None

        comparables = await client.call_tool(
            "calculate_comparable_multiples", {"company_id": "cedar-health", "peer_filter": None}
        )
        assert comparables["peer_count"] == 3
        assert comparables["peers"][0]["currency"] == "EUR"

        sensitivity = await client.call_tool(
            "run_sensitivity_analysis",
            {
                "company_id": "aurora-retail",
                "base_period": "2023",
                "assumptions": {"revenue_growth_rate": "0.05", "cost_of_equity": "0.10"},
                "ranges": {
                    "wacc_min": "0.10",
                    "wacc_max": "0.10",
                    "wacc_step": "0.01",
                    "terminal_growth_min": "0.02",
                    "terminal_growth_max": "0.02",
                    "terminal_growth_step": "0.01",
                },
            },
        )
        assert sensitivity["scenario_count"] == 1


@pytest.mark.asyncio
async def test_invalid_period_range_is_tool_error(seeded_db):
    async with MCPToolClient(db_path=seeded_db) as client:
        with pytest.raises(MCPToolError):
            await client.call_tool(
                "get_company_financials", {"company_id": "aurora-retail", "period_range": "bad"}
            )


@pytest.mark.asyncio
async def test_missing_base_nwc_does_not_create_false_success(seeded_db):
    async with MCPToolClient(db_path=seeded_db) as client:
        result = await client.call_tool(
            "calculate_valuation",
            {
                "company_id": "aurora-retail",
                "base_period": "2021",
                "assumptions": {"revenue_growth_rate": "0.05", "cost_of_equity": "0.10"},
            },
        )
        assert result["available"] is False
        assert any("NWC" in reason or "nwc" in reason.lower() for reason in result["reasons"])
