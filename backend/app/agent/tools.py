"""MCP tool discovery, validation, and safe invocation for the agent."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict
from typing import Any

import jsonschema

from backend.app.mcp_client.client import MCPToolClient, MCPToolError

SYSTEM_PROMPT = """You are DealPilot Lite, an agentic private-equity screening assistant.
Use the available deterministic financial tools for financial calculations.
Use retrieve_company_documents when documentary evidence, risks, management discussion, or source support is needed.
Never invent financial figures, tool results, currencies, units, or unavailable values.
Preserve currency and units. Treat unavailable values as unavailable, never as zero.
Explain missing inputs and calculation limitations.
Identify assumptions used in valuations and describe DCF results as illustrative, not transaction prices.
Distinguish historical figures from forecast assumptions and documentary claims.
All company data is synthetic. Do not claim to have reviewed documents beyond retrieved passages.
Cite retrieved passages as [doc:<document_id> chunk:<chunk_id>] and do not invent citations.
Similarity scores are retrieval relevance, not proof that a statement is true.
Provide decision-support information only, not investment advice."""


def mcp_tools_to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
        )
    return result


def build_tool_registry(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {tool["name"]: tool for tool in tools}


def parse_tool_arguments(arguments: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if arguments is None or arguments == "":
        return {}, None
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return None, f"malformed tool arguments JSON: {exc}"
    if not isinstance(value, dict):
        return None, "tool arguments must be a JSON object"
    return value, None


def validate_tool_call(
    name: str,
    arguments: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> str | None:
    if name not in registry:
        return f"unknown tool: {name}"
    schema = registry[name].get("inputSchema") or {"type": "object", "properties": {}}
    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as exc:
        return f"invalid arguments for {name}: {exc.message}"
    return None


def bound_tool_result(result: Any, max_chars: int) -> Any:
    serialized = json.dumps(result, ensure_ascii=False, default=str)
    if len(serialized) <= max_chars:
        return result
    return {
        "status": "oversized_result",
        "original_size_chars": len(serialized),
        "limit_chars": max_chars,
        "message": "Tool result exceeded the configured size limit; essential values were not truncated.",
    }


async def invoke_mcp_tool(
    client: MCPToolClient,
    name: str,
    arguments: dict[str, Any],
    *,
    per_tool_timeout: float,
    max_result_chars: int,
    deadline: float,
) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {
            "status": "error",
            "tool": name,
            "error": "agent deadline expired before tool invocation",
        }
    timeout = min(per_tool_timeout, remaining)
    try:
        result = await asyncio.wait_for(client.call_tool(name, arguments), timeout=timeout)
    except asyncio.TimeoutError:
        return {"status": "error", "tool": name, "error": "tool invocation timed out"}
    except MCPToolError as exc:
        return {"status": "error", "tool": name, "error": str(exc)[:500]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "tool": name, "error": str(exc)[:500]}
    return bound_tool_result(result, max_result_chars)


def rag_result_to_evidence(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, list):
        return []
    evidence = []
    for item in result:
        record = asdict(item) if hasattr(item, "__dataclass_fields__") else item
        if not isinstance(record, dict):
            continue
        text = str(record.get("text", ""))
        evidence.append(
            {
                "company_id": record.get("company_id"),
                "document_id": record.get("document_id"),
                "chunk_id": record.get("chunk_id"),
                "source": record.get("source"),
                "filename": record.get("filename"),
                "page_start": record.get("page_start"),
                "page_end": record.get("page_end"),
                "similarity_score": record.get("similarity_score"),
                "embedding_model": record.get("embedding_model"),
                "text_preview": text[:300],
            }
        )
    return evidence


CITATION_PATTERN = re.compile(r"\[doc:([^\]\s]+)\s+chunk:([^\]]+)\]")


def extract_citation_keys(text: str) -> list[tuple[str, str]]:
    return [(match.group(1), match.group(2).strip()) for match in CITATION_PATTERN.finditer(text)]


def validate_citations(final_text: str, evidence: list[dict[str, Any]]) -> tuple[list[str], str]:
    known = {(str(item.get("document_id")), str(item.get("chunk_id"))) for item in evidence}
    warnings: list[str] = []
    for document_id, chunk_id in extract_citation_keys(final_text):
        if (document_id, chunk_id) not in known:
            warnings.append(f"unverified citation: [doc:{document_id} chunk:{chunk_id}]")
    if not evidence:
        return warnings, ""
    lines = ["", "Retrieved evidence:"]
    for item in evidence:
        lines.append(
            f"- [doc:{item['document_id']} chunk:{item['chunk_id']}] "
            f"{item['filename']} (pages {item['page_start']}-{item['page_end']}, "
            f"source: {item['source']})"
        )
    appendix = "\n".join(lines)
    if warnings:
        appendix += "\n\nCitation validation warnings:\n" + "\n".join(
            f"- {warning}" for warning in warnings
        )
    return warnings, appendix
