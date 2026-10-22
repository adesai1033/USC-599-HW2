"""Provider clients by tier (fast / strong / fallback)."""

from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings


@lru_cache
def get_llm(tier: Literal["fast"], *, thinking_level: Literal["low", "medium", "high"] | None = None) -> BaseChatModel:
    """Return the chat model for a tier. Phase 1 has only the fast tier (Gemini Flash).

    thinking_level caps Gemini's reasoning before it answers; None keeps the model's default. Gemini
    3.8 Flash rejects "minimal".
    """
    if tier != "fast":
        raise ValueError(f"Unknown LLM tier: {tier!r}")
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        api_key=settings.google_api_key,
        streaming=True,
        timeout=60,
        max_retries=2,
        thinking_level=thinking_level,
    )
