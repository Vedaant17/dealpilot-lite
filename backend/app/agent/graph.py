"""LangGraph construction for the DealPilot Lite agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.app.agent.nodes import finalize_node, model_node, tool_node
from backend.app.agent.state import AgentState


def _route_after_model(state: AgentState) -> str:
    messages = state.get("messages", [])
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
        return "tools"
    return "finalize"


def _route_after_tools(state: AgentState) -> str:
    if state.get("termination_reason") or state.get("final_response"):
        return "finalize"
    return "model"


def build_agent_graph(client: Any, deadline: float, db_path: str | None = None):
    graph = StateGraph(AgentState)

    async def model_with_dependencies(state: AgentState) -> AgentState:
        return await model_node(state, client, deadline)

    async def tools_with_dependencies(state: AgentState) -> AgentState:
        if db_path is not None:
            state = {**state, "db_path": db_path}
        return await tool_node(state, client, deadline)

    async def finalize_with_dependencies(state: AgentState) -> AgentState:
        return await finalize_node(state)

    graph.add_node("model", model_with_dependencies)
    graph.add_node("tools", tools_with_dependencies)
    graph.add_node("finalize", finalize_with_dependencies)
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", _route_after_model, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "tools", _route_after_tools, {"model": "model", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    return graph.compile()
