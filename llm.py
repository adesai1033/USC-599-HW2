"""Provider clients by tier (fast / strong / fallback)."""

from collections.abc import Callable
from functools import lru_cache
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import get_settings

Tier = Literal["fast", "strong", "fallback"]
ThinkingLevel = Literal["low", "medium", "high"]

# A model per tier. Graph nodes take one so the router can pick the tier per call; tests pass a fake.
Models = Callable[[Tier], BaseChatModel]

PROVIDERS: dict[str, str] = {"fast": "google", "strong": "anthropic", "fallback": "openai"}
TIMEOUT_S = 60


def model_id(tier: Tier) -> str:
    """The configured model ID for a tier."""
    settings = get_settings()
    return {"fast": settings.gemini_model, "strong": settings.anthropic_model, "fallback": settings.openai_model}[tier]


@lru_cache
def get_llm(tier: Tier, *, thinking_level: ThinkingLevel | None = None) -> BaseChatModel:
    """Return the chat model for a tier: fast = Gemini Flash, strong = Claude Sonnet, fallback = OpenAI.

    thinking_level caps reasoning before the model answers; None keeps the model's default. Gemini takes
    it as thinking_level (3.8 Flash rejects "minimal") and OpenAI as reasoning_effort. Claude runs with
    thinking off, so it is ignored there.
    """
    settings = get_settings()
    # The SDKs retry transient errors themselves. Strong and fallback retry only once, so a failing
    # provider hands over to the fallback quickly.
    if tier == "fast":
        return ChatGoogleGenerativeAI(model=settings.gemini_model, api_key=settings.google_api_key, streaming=True,
                                      timeout=TIMEOUT_S, max_retries=2, thinking_level=thinking_level)
    if tier == "strong":
        # Sonnet 5.5 thinks by default, and its thinking blocks are bound to the exact prompt prefix: the
        # worker's round budget changes the system prompt every call, so a replayed block is rejected (400).
        # "between_tools" is how this model turns thinking off ("disabled" is a 400); default effort is high.
        return ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key, streaming=True,
                             default_request_timeout=TIMEOUT_S, max_retries=1, thinking={"type": "between_tools"})
    if tier == "fallback":
        # Raises at construction when OPENAI_API_KEY is missing; callers build it inside their error handling.
        return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, streaming=True,
                          request_timeout=TIMEOUT_S, max_retries=1, reasoning_effort=thinking_level)
    raise ValueError(f"Unknown LLM tier: {tier!r}")


def resolve_models(llm: BaseChatModel | Models | None, *, thinking_level: ThinkingLevel | None = None) -> Models:
    """A Models resolver: None means the configured providers, one model means that model for every tier.

    Anything that isn't a plain callable (a chat model, or a test's duck-typed fake) counts as one model.
    """
    if llm is None:
        return lambda tier: get_llm(tier, thinking_level=thinking_level)
    if isinstance(llm, BaseChatModel) or not callable(llm):
        return lambda tier: llm
    return llm
