"""Configuration loading test."""

from backend.app.config import settings


def test_config_defaults():
    assert settings.app_env == "development"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.embedding_dim == 1536
    assert settings.api_port == 8000