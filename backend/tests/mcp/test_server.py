import pytest

from backend.app.mcp_server.server import create_mcp_server
from backend.scripts.seed_db import seed_database


@pytest.fixture
def seeded_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    seed_database(db_path)
    return db_path


def test_server_registers_exactly_five_tools(seeded_db):
    mcp = create_mcp_server(seeded_db)
    tools = mcp._tool_manager.list_tools()
    assert len(tools) == 5
    assert {tool.name for tool in tools} == {
        "get_company_financials",
        "calculate_financial_metrics",
        "calculate_valuation",
        "calculate_comparable_multiples",
        "run_sensitivity_analysis",
    }
