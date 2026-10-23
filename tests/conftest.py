import os

import pytest

import config
import llm


@pytest.fixture(autouse=True)
def test_env(monkeypatch):
    """Isolate every test from the developer's real .env values."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-gemini-model")
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-anthropic-model")
    monkeypatch.setenv("OPENAI_MODEL", "test-openai-model")
    if os.environ.get("RUN_LIVE") != "1":  # live checks need the real OpenAI key for query embeddings
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("PORT", raising=False)
    config.get_settings.cache_clear()
    llm.get_llm.cache_clear()
    yield
    config.get_settings.cache_clear()
    llm.get_llm.cache_clear()
