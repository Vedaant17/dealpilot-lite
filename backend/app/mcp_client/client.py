"""MCP client for DealPilot Lite."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters, get_default_environment


class MCPToolError(RuntimeError):
    pass


class MCPToolClient:
    def __init__(self, db_path: str | None = None, timeout: float = 30.0):
        self.timeout = timeout
        project_root = Path(__file__).resolve().parents[3]
        env = get_default_environment()
        env["PYTHONPATH"] = str(project_root)
        if db_path is not None:
            env["DEALPILOT_SQLITE_DB_PATH"] = str(Path(db_path).resolve())
        self._server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "backend.app.mcp_server.server"],
            env=env,
            cwd=project_root,
        )
        self._client: Client | None = None

    async def __aenter__(self) -> "MCPToolClient":
        self._client = Client(self._server_params)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("client is not connected")
        result = await asyncio.wait_for(self._client.list_tools(), timeout=self.timeout)
        return [tool.model_dump(mode="json") for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if self._client is None:
            raise RuntimeError("client is not connected")
        try:
            result = await asyncio.wait_for(
                self._client.call_tool(name, arguments or {}),
                timeout=self.timeout,
            )
        except Exception as exc:
            raise MCPToolError(f"tool call failed: {exc}") from exc

        if result.is_error:
            texts = [block.text for block in result.content if hasattr(block, "text")]
            raise MCPToolError(" ".join(texts) if texts else "tool returned an error")

        if result.structured_content is not None:
            return result.structured_content
        texts = [block.text for block in result.content if hasattr(block, "text")]
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return {"text": texts[0]}
        return {"content": texts}
