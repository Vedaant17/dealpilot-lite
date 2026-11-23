"""LangGraph nodes for the DealPilot Lite agent."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from backend.app.agent.state import AgentState
from backend.app.agent.tools import (
    bound_tool_result,
    extract_citation_keys,
    invoke_mcp_tool,
    mcp_tools_to_openai_tools,
    parse_tool_arguments,
    rag_result_to_evidence,
    validate_tool_call,
)
from backend.app.config import settings
from backend.app.rag.retrieval import RetrievalError, retrieve


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"role": "assistant", "content": message.content}
    if getattr(message, "tool_calls", None):
        result["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return result


class LLMConfigurationError(RuntimeError):
    """Raised when the configured LLM provider cannot be used."""


def _resolve_llm_configuration() -> tuple[str, str, str | None]:
    provider = settings.llm_provider.strip().lower()
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        return (
            settings.gemini_api_key,
            settings.llm_model or "gemini-3.8-flash",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return (
            settings.openai_api_key,
            settings.llm_model or settings.openai_chat_model,
            None,
        )
    if provider == "ollama":
        if not settings.ollama_base_url:
            raise LLMConfigurationError("OLLAMA_BASE_URL is required when LLM_PROVIDER=ollama")
        if settings.ollama_num_ctx < 2048 or settings.ollama_num_ctx > 4096:
            raise LLMConfigurationError("OLLAMA_NUM_CTX must be between 2048 and 4096")
        return (
            "ollama",
            settings.llm_model or "qwen3:4b",
            settings.ollama_base_url,
        )
    raise LLMConfigurationError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


async def model_node(state: AgentState, client: Any, deadline: float) -> AgentState:
    messages = list(state.get("messages", []))
    round_count = state.get("round_count", 0)
    if state.get("termination_reason"):
        return state
    if round_count >= settings.agent_max_rounds:
        return {
            **state,
            "termination_reason": "maximum model/tool rounds reached",
            "final_response": "The request reached the maximum number of model and tool rounds before completion.",
        }
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {
            **state,
            "termination_reason": "agent deadline expired",
            "final_response": "The agent deadline expired before a model response could be completed.",
        }
    tool_defs = mcp_tools_to_openai_tools(list(state.get("tool_registry", {}).values()))
    try:
        api_key, model, base_url = _resolve_llm_configuration()
    except LLMConfigurationError as exc:
        return {
            **state,
            "termination_reason": "llm configuration failed",
            "final_response": f"LLM configuration failed: {str(exc)[:500]}",
            "error": str(exc)[:500],
        }
    client_kwargs = {"api_key": api_key, "timeout": min(60.0, remaining)}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    openai_client = AsyncOpenAI(**client_kwargs)
    try:
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_defs or None,
                tool_choice="auto",
                temperature=0,
            ),
            timeout=remaining,
        )
    except (OpenAIError, asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        return {
            **state,
            "termination_reason": "openai request failed",
            "final_response": f"The language model request failed: {str(exc)[:500]}",
            "error": str(exc)[:500],
        }

    message = response.choices[0].message
    assistant_message = _assistant_message_to_dict(message)
    new_messages = messages + [assistant_message]
    if getattr(message, "tool_calls", None):
        return {
            **state,
            "messages": new_messages,
            "round_count": round_count + 1,
        }
    return {
        **state,
        "messages": new_messages,
        "round_count": round_count + 1,
        "final_response": message.content or "",
    }


async def tool_node(state: AgentState, client: Any, deadline: float) -> AgentState:
    messages = list(state.get("messages", []))
    if (
        not messages
        or messages[-1].get("role") != "assistant"
        or not messages[-1].get("tool_calls")
    ):
        return state
    if state.get("termination_reason"):
        return state

    assistant_message = messages[-1]
    tool_calls = assistant_message["tool_calls"]
    current_count = state.get("tool_call_count", 0)
    if current_count + len(tool_calls) > settings.agent_max_tool_calls:
        return {
            **state,
            "termination_reason": "maximum tool invocations reached",
            "final_response": "The request would exceed the maximum number of tool invocations.",
        }

    registry = state.get("tool_registry", {})
    db_path = state.get("db_path")
    selected_company = state.get("company_id")
    new_messages = list(messages)
    evidence = list(state.get("retrieved_evidence", []))
    tool_errors = list(state.get("tool_errors", []))
    invoked = 0
    for tool_call in tool_calls:
        name = tool_call["function"]["name"]
        arguments, parse_error = parse_tool_arguments(tool_call["function"].get("arguments"))
        if parse_error:
            result = {"status": "error", "tool": name, "error": parse_error}
            tool_errors.append(f"{name}: {parse_error}")
        else:
            validation_error = validate_tool_call(name, arguments or {}, registry)
            if validation_error:
                result = {"status": "error", "tool": name, "error": validation_error}
                tool_errors.append(f"{name}: {validation_error}")
            elif registry.get(name, {}).get("source") == "rag":
                if selected_company is not None and arguments.get("company_id") != selected_company:
                    error = "company_id does not match the selected company"
                    result = {"status": "error", "tool": name, "error": error}
                    tool_errors.append(f"{name}: {error}")
                else:
                    try:
                        result = await retrieve(
                            db_path,
                            arguments.get("company_id", selected_company),
                            arguments.get("query", ""),
                            top_k=arguments.get("top_k"),
                            embedding_dim=settings.embedding_dim,
                            vector_store=state.get("vector_store"),
                        )
                        evidence.extend(rag_result_to_evidence(result))
                        invoked += 1
                    except (RetrievalError, ValueError, Exception) as exc:  # noqa: BLE001
                        error = str(exc)[:500]
                        result = {"status": "error", "tool": name, "error": error}
                        tool_errors.append(f"{name}: {error}")
            else:
                result = await invoke_mcp_tool(
                    client,
                    name,
                    arguments or {},
                    per_tool_timeout=30.0,
                    max_result_chars=settings.agent_max_tool_result_chars,
                    deadline=deadline,
                )
                invoked += 1
                if isinstance(result, dict) and result.get("status") == "error":
                    tool_errors.append(f"{name}: {result.get('error', 'tool failed')}")
        new_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": name,
                "content": json.dumps(
                    bound_tool_result(result, settings.agent_max_tool_result_chars),
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )
    return {
        **state,
        "messages": new_messages,
        "tool_call_count": current_count + invoked,
        "retrieved_evidence": evidence,
        "tool_errors": tool_errors,
    }


async def finalize_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    final_response = state.get("final_response")
    if not final_response:
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                final_response = str(message["content"])
                break
    if not final_response:
        final_response = "The agent did not produce a final response."

    evidence = state.get("retrieved_evidence", [])
    warnings = list(state.get("citation_warnings", []))
    for document_id, chunk_id in extract_citation_keys(final_response):
        try:
            key = (document_id, int(chunk_id))
        except ValueError:
            warnings.append(f"malformed citation: [doc:{document_id} chunk:{chunk_id}]")
            continue
        known = {(item.get("document_id"), item.get("chunk_id")) for item in evidence}
        if key not in known:
            warnings.append(f"unverified citation: [doc:{document_id} chunk:{chunk_id}]")
    warnings = list(dict.fromkeys(warnings))

    if evidence:
        final_response += "\n\nRetrieved evidence:"
        for item in evidence:
            final_response += (
                f"\n- [doc:{item['document_id']} chunk:{item['chunk_id']}] "
                f"{item['filename']} (pages {item['page_start']}-{item['page_end']}, "
                f"source: {item['source']})"
            )
    if warnings:
        final_response += "\n\nCitation validation warnings:\n" + "\n".join(
            f"- {warning}" for warning in warnings
        )

    return {
        **state,
        "final_response": final_response,
        "citation_warnings": warnings,
    }
