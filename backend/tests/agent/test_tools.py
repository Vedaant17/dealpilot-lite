from backend.app.agent.tools import (
    build_tool_registry,
    invoke_mcp_tool,
    mcp_tools_to_openai_tools,
    parse_tool_arguments,
    validate_tool_call,
)


def test_tool_conversion_preserves_schema():
    tools = [
        {
            "name": "t",
            "description": "desc",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        }
    ]
    converted = mcp_tools_to_openai_tools(tools)
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "t"
    assert converted[0]["function"]["parameters"]["properties"]["x"]["type"] == "integer"


def test_parse_and_validate():
    assert parse_tool_arguments(None) == ({}, None)
    assert parse_tool_arguments("") == ({}, None)
    assert parse_tool_arguments("{bad")[1] is not None
    registry = build_tool_registry(
        [
            {
                "name": "t",
                "inputSchema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            }
        ]
    )
    assert validate_tool_call("missing", {}, registry) is not None
    assert validate_tool_call("t", {}, registry) is not None
    assert validate_tool_call("t", {"x": 1}, registry) is None


class FakeClient:
    async def call_tool(self, name, arguments):
        return {"value": "1", "currency": "USD"}


async def test_invoke_mcp_tool_success():
    result = await invoke_mcp_tool(
        FakeClient(),
        "t",
        {},
        per_tool_timeout=1.0,
        max_result_chars=100,
        deadline=9999999999.0,
    )
    assert result["value"] == "1"
