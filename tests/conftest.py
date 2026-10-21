import pytest

import config


@pytest.fixture(autouse=True)
def test_env(monkeypatch):
    """Isolate every test from the developer's real .env values."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-gemini-model")
    monkeypatch.delenv("PORT", raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
