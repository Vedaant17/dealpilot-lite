from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agent.nodes import LLMConfigurationError, model_node
from backend.app.config import settings


def make_message(content="ok", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def make_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.asyncio
async def test_openai_provider_uses_existing_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "openai-key")
    monkeypatch.setattr(settings, "openai_chat_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "llm_model", "")

    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(return_value=make_response(make_message()))
        state = {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_registry": {},
            "round_count": 0,
            "tool_call_count": 0,
            "termination_reason": None,
            "final_response": None,
        }
        await model_node(state, object(), 9999999999.0)
        mocked.assert_called_once()
        kwargs = mocked.call_args.kwargs
        assert kwargs["api_key"] == "openai-key"
        assert "base_url" not in kwargs
        assert instance.chat.completions.create.await_args.kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_gemini_provider_uses_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "gemini-key")
    monkeypatch.setattr(settings, "llm_model", "gemini-3.8-flash")

    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(return_value=make_response(make_message()))
        state = {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_registry": {},
            "round_count": 0,
            "tool_call_count": 0,
            "termination_reason": None,
            "final_response": None,
        }
        await model_node(state, object(), 9999999999.0)
        kwargs = mocked.call_args.kwargs
        assert kwargs["api_key"] == "gemini-key"
        assert kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
        assert instance.chat.completions.create.await_args.kwargs["model"] == "gemini-3.8-flash"


@pytest.mark.asyncio
async def test_missing_gemini_key_fails_clearly(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    state = {
        "messages": [{"role": "user", "content": "hello"}],
        "tool_registry": {},
        "round_count": 0,
        "tool_call_count": 0,
        "termination_reason": None,
        "final_response": None,
    }
    result = await model_node(state, object(), 9999999999.0)
    assert result["termination_reason"] == "llm configuration failed"
    assert "GEMINI_API_KEY" in result["final_response"]


@pytest.mark.asyncio
async def test_unsupported_provider_fails_clearly(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    state = {
        "messages": [{"role": "user", "content": "hello"}],
        "tool_registry": {},
        "round_count": 0,
        "tool_call_count": 0,
        "termination_reason": None,
        "final_response": None,
    }
    result = await model_node(state, object(), 9999999999.0)
    assert result["termination_reason"] == "llm configuration failed"
    assert "Unsupported LLM_PROVIDER" in result["final_response"]


def test_llm_configuration_error_type():
    assert issubclass(LLMConfigurationError, RuntimeError)


@pytest.mark.asyncio
async def test_ollama_provider_uses_local_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_model", "qwen3:4b")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434/v1/")
    monkeypatch.setattr(settings, "ollama_num_ctx", 4096)

    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(return_value=make_response(make_message()))
        state = {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_registry": {},
            "round_count": 0,
            "tool_call_count": 0,
            "termination_reason": None,
            "final_response": None,
        }
        await model_node(state, object(), 9999999999.0)
        kwargs = mocked.call_args.kwargs
        assert kwargs["api_key"] == "ollama"
        assert kwargs["base_url"] == "http://localhost:11434/v1/"
        assert instance.chat.completions.create.await_args.kwargs["model"] == "qwen3:4b"


@pytest.mark.asyncio
async def test_ollama_missing_service_fails_clearly(monkeypatch):
    from openai import APIConnectionError

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_model", "qwen3:4b")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434/v1/")
    monkeypatch.setattr(settings, "ollama_num_ctx", 4096)

    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(side_effect=APIConnectionError(request=None))
        state = {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_registry": {},
            "round_count": 0,
            "tool_call_count": 0,
            "termination_reason": None,
            "final_response": None,
        }
        result = await model_node(state, object(), 9999999999.0)
        assert result["termination_reason"] == "openai request failed"
        assert (
            "Connection error" in result["final_response"]
            or "connection" in result["final_response"].lower()
        )


@pytest.mark.asyncio
async def test_ollama_invalid_context_fails_clearly(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_model", "qwen3:4b")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434/v1/")
    monkeypatch.setattr(settings, "ollama_num_ctx", 1024)
    state = {
        "messages": [{"role": "user", "content": "hello"}],
        "tool_registry": {},
        "round_count": 0,
        "tool_call_count": 0,
        "termination_reason": None,
        "final_response": None,
    }
    result = await model_node(state, object(), 9999999999.0)
    assert result["termination_reason"] == "llm configuration failed"
    assert "OLLAMA_NUM_CTX" in result["final_response"]


@pytest.mark.asyncio
async def test_ollama_no_cloud_key_required(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_model", "qwen3:4b")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434/v1/")
    monkeypatch.setattr(settings, "ollama_num_ctx", 4096)

    with patch("backend.app.agent.nodes.AsyncOpenAI") as mocked:
        instance = mocked.return_value
        instance.chat.completions.create = AsyncMock(return_value=make_response(make_message()))
        state = {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_registry": {},
            "round_count": 0,
            "tool_call_count": 0,
            "termination_reason": None,
            "final_response": None,
        }
        result = await model_node(state, object(), 9999999999.0)
        assert result["final_response"] == "ok"
        assert mocked.call_args.kwargs["api_key"] == "ollama"
        assert mocked.call_args.kwargs["base_url"] == "http://localhost:11434/v1/"
