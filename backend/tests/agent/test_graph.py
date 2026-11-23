from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agent.graph import build_agent_graph
from backend.app.agent.state import AgentState


def make_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def make_tool_call(id_, name, arguments):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def make_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.asyncio
async def test_graph_single_tool_then_final():
    client = AsyncMock()
    client.call_tool.return_value = {"value": "10", "currency": "USD"}
    registry = {
        "get_company_financials": {
            "name": "get_company_financials",
            "inputSchema": {
                "type": "object",
                "properties": {"company_id": {"type": "string"}},
                "required": ["company_id"],
            },
        }
    }
    first = make_response(
        make_message(
            tool_calls=[
                make_tool_call("c1", "get_company_financials", '{"company_id":"aurora-retail"}')
            ]
        )
    )
    second = make_response(make_message(content="done", tool_calls=None))
    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(side_effect=[first, second])
        graph = build_agent_graph(client, 9999999999.0)
        state: AgentState = {
            "messages": [{"role": "user", "content": "show financials"}],
            "user_input": "show financials",
            "company_id": "aurora-retail",
            "tool_registry": registry,
            "tool_call_count": 0,
            "round_count": 0,
            "termination_reason": None,
            "final_response": None,
        }
        result = await graph.ainvoke(state)
        assert result["final_response"] == "done"
        assert result["tool_call_count"] == 1
        assert any(m.get("role") == "tool" for m in result["messages"])


@pytest.mark.asyncio
async def test_graph_no_tool():
    client = AsyncMock()
    response = make_response(make_message(content="no tool needed", tool_calls=None))
    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(return_value=response)
        graph = build_agent_graph(client, 9999999999.0)
        state: AgentState = {
            "messages": [{"role": "user", "content": "hello"}],
            "user_input": "hello",
            "company_id": None,
            "tool_registry": {},
            "tool_call_count": 0,
            "round_count": 0,
            "termination_reason": None,
            "final_response": None,
        }
        result = await graph.ainvoke(state)
        assert result["final_response"] == "no tool needed"
        assert result["tool_call_count"] == 0
