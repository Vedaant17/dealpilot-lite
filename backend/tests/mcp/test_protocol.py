import pytest

from backend.app.mcp_client.client import MCPToolError


@pytest.mark.asyncio
async def test_unknown_tool_raises(tmp_path):
    from backend.app.mcp_client.client import MCPToolClient
    from backend.scripts.seed_db import seed_database

    db_path = str(tmp_path / "test.db")
    seed_database(db_path)
    async with MCPToolClient(db_path=db_path) as client:
        with pytest.raises(MCPToolError):
            await client.call_tool("not_a_tool", {})
