import pytest

from backend.app.config import settings


@pytest.fixture(autouse=True)
def configure_test_llm(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_chat_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "llm_model", "")
