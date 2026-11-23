from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agent.nodes import model_node, tool_node
from backend.app.agent.state import AgentState


class FakeTool:
    def __init__(self, id_, name, arguments):
        self.id = id_
        self.function = SimpleNamespace(name=name, arguments=arguments)


def make_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def make_response(message):
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


@pytest.mark.asyncio
async def test_model_node_no_tool(monkeypatch):
    fake_client = object()
    message = make_message(content="final answer", tool_calls=None)
    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(return_value=make_response(message))
        state: AgentState = {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_registry": {},
            "round_count": 0,
            "tool_call_count": 0,
            "termination_reason": None,
            "final_response": None,
        }
        result = await model_node(state, fake_client, 9999999999.0)
        assert result["final_response"] == "final answer"
        assert result["messages"][-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_model_node_tool_call(monkeypatch):
    fake_client = object()
    message = make_message(
        content=None,
        tool_calls=[FakeTool("call_1", "get_company_financials", '{"company_id":"aurora-retail"}')],
    )
    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(return_value=make_response(message))
        state: AgentState = {
            "messages": [{"role": "user", "content": "metrics"}],
            "tool_registry": {
                "get_company_financials": {
                    "name": "get_company_financials",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"company_id": {"type": "string"}},
                        "required": ["company_id"],
                    },
                }
            },
            "round_count": 0,
            "tool_call_count": 0,
            "termination_reason": None,
            "final_response": None,
        }
        result = await model_node(state, fake_client, 9999999999.0)
        assert result["messages"][-1]["tool_calls"][0]["id"] == "call_1"
        assert result["final_response"] is None


@pytest.mark.asyncio
async def test_tool_node_executes_and_returns_result():
    client = AsyncMock()
    client.call_tool.return_value = {"value": "1", "currency": "USD"}
    state: AgentState = {
        "messages": [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_company_financials",
                            "arguments": '{"company_id":"aurora-retail"}',
                        },
                    }
                ],
            },
        ],
        "tool_registry": {
            "get_company_financials": {
                "name": "get_company_financials",
                "inputSchema": {
                    "type": "object",
                    "properties": {"company_id": {"type": "string"}},
                    "required": ["company_id"],
                },
            }
        },
        "tool_call_count": 0,
        "round_count": 1,
        "termination_reason": None,
        "final_response": None,
    }
    result = await tool_node(state, client, 9999999999.0)
    assert result["tool_call_count"] == 1
    assert result["messages"][-1]["role"] == "tool"
    assert result["messages"][-1]["tool_call_id"] == "call_1"
    assert '"value": "1"' in result["messages"][-1]["content"]
