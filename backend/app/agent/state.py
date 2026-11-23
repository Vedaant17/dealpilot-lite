"""LangGraph state for the DealPilot Lite agent."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    user_input: str
    company_id: str | None
    db_path: str | None
    vector_store: Any
    tool_registry: dict[str, dict[str, Any]]
    tool_call_count: int
    round_count: int
    termination_reason: str | None
    final_response: str | None
    error: str | None
    retrieved_evidence: list[dict[str, Any]]
    citation_warnings: list[str]
    tool_errors: list[str]
