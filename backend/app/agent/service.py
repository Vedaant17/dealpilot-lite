"""Public service interface for the DealPilot Lite agent."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from backend.app.agent.graph import build_agent_graph
from backend.app.agent.state import AgentState
from backend.app.agent.tools import (
    SYSTEM_PROMPT,
    build_tool_registry,
)
from backend.app.config import settings
from backend.app.mcp_client.client import MCPToolClient
from backend.app.rag.retrieval import rag_tool_definition
from backend.app.rag.vector_store import VectorStore


@dataclass
class AgentRunResult:
    status: str
    narrative: str
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    citation_warnings: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    tool_call_count: int = 0
    round_count: int = 0
    termination_reason: str | None = None
    duration_ms: int = 0
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None


_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service


class AgentService:
    def __init__(self, settings_obj=settings):
        self.settings = settings_obj
        self._rag_lock = threading.RLock()
        self._vector_stores: dict[str, VectorStore] = {}

    def _vector_store(self, db_path: str) -> VectorStore:
        with self._rag_lock:
            store = self._vector_stores.get(db_path)
            if store is None:
                store = VectorStore(
                    db_path=db_path,
                    lock=self._rag_lock,
                    embedding_dim=self.settings.embedding_dim,
                    embedding_model=self.settings.openai_embedding_model,
                )
                self._vector_stores[db_path] = store
            return store

    async def run(
        self, user_input: str, company_id: str | None = None, db_path: str | None = None
    ) -> str:
        result = await self.run_result(user_input, company_id=company_id, db_path=db_path)
        return result.narrative

    async def run_result(
        self,
        user_input: str,
        *,
        company_id: str | None = None,
        db_path: str | None = None,
        top_k: int | None = None,
    ) -> AgentRunResult:
        start = time.perf_counter()
        deadline = start + self.settings.agent_timeout_seconds
        resolved_db_path = db_path if db_path is not None else self.settings.sqlite_db_path
        try:
            async with MCPToolClient(db_path=resolved_db_path, timeout=30.0) as client:
                mcp_tools = await client.list_tools()
                rag_tool = rag_tool_definition()
                combined_tools = [dict(tool) for tool in mcp_tools] + [rag_tool]
                for tool in combined_tools:
                    tool.setdefault(
                        "source", "mcp" if tool.get("name") != rag_tool["name"] else "rag"
                    )
                tool_registry = build_tool_registry(combined_tools)
                graph = build_agent_graph(client, deadline, resolved_db_path)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
                if company_id is not None:
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Selected company ID: {company_id}. Use this company unless the user explicitly names another company.",
                        }
                    )
                messages.append({"role": "user", "content": user_input})
                initial_state: AgentState = {
                    "messages": messages,
                    "user_input": user_input,
                    "company_id": company_id,
                    "db_path": resolved_db_path,
                    "vector_store": self._vector_store(resolved_db_path),
                    "tool_registry": tool_registry,
                    "tool_call_count": 0,
                    "round_count": 0,
                    "termination_reason": None,
                    "final_response": None,
                    "retrieved_evidence": [],
                    "citation_warnings": [],
                    "tool_errors": [],
                }
                result = await graph.ainvoke(initial_state)
                final_response = result.get("final_response")
                if not final_response:
                    for message in reversed(result.get("messages", [])):
                        if message.get("role") == "assistant" and message.get("content"):
                            final_response = str(message["content"])
                            break
                if not final_response:
                    final_response = "The agent did not produce a final response."

                termination_reason = result.get("termination_reason")
                tool_errors = result.get("tool_errors", [])
                status = "completed"
                if termination_reason or tool_errors:
                    status = "partial"

                tools_used = []
                for message in result.get("messages", []):
                    if message.get("role") == "tool" and message.get("name"):
                        if message["name"] not in tools_used:
                            tools_used.append(message["name"])

                return AgentRunResult(
                    status=status,
                    narrative=str(final_response),
                    retrieved_evidence=result.get("retrieved_evidence", []),
                    citation_warnings=result.get("citation_warnings", []),
                    tool_errors=tool_errors,
                    tool_call_count=result.get("tool_call_count", 0),
                    round_count=result.get("round_count", 0),
                    termination_reason=termination_reason,
                    duration_ms=int((time.perf_counter() - start) * 1000),
                    tools_used=tools_used,
                    error=None,
                )
        except Exception as exc:  # noqa: BLE001
            return AgentRunResult(
                status="failed",
                narrative=f"Agent invocation failed: {str(exc)[:500]}",
                error=str(exc)[:500],
                duration_ms=int((time.perf_counter() - start) * 1000),
            )
