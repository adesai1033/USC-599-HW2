"""Provider clients by tier (fast / strong); each tier is the other's fallback."""

from collections.abc import Callable
from functools import lru_cache
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings

Tier = Literal["fast", "strong"]
ThinkingLevel = Literal["low", "medium", "high"]

# A model per tier. Graph nodes take one so the router can pick the tier per call; tests pass a fake.
Models = Callable[[Tier], BaseChatModel]

PROVIDERS: dict[str, str] = {"fast": "google", "strong": "anthropic"}
# A provider error on one tier retries on the other, so each provider backs up the other.
FALLBACK: dict[Tier, Tier] = {"fast": "strong", "strong": "fast"}
TIMEOUT_S = 60


def model_id(tier: Tier) -> str:
    """The configured model ID for a tier."""
    settings = get_settings()
    return {"fast": settings.gemini_model, "strong": settings.anthropic_model}[tier]


@lru_cache
def get_llm(tier: Tier, *, thinking_level: ThinkingLevel | None = None) -> BaseChatModel:
    """Return the chat model for a tier: fast = Gemini Flash, strong = Claude Sonnet.

    thinking_level caps reasoning before the model answers; None keeps the model's default. Gemini takes
    it as thinking_level (3.8 Flash rejects "minimal"). Claude thinks
    only between tool calls, so it is ignored there.
    """
    settings = get_settings()
    # The SDKs retry transient errors themselves. Strong retries only once, so a failing Claude hands
    # over to Gemini quickly.
    if tier == "fast":
        return ChatGoogleGenerativeAI(model=settings.gemini_model, api_key=settings.google_api_key, streaming=True,
                                      timeout=TIMEOUT_S, max_retries=2, thinking_level=thinking_level)
    if tier == "strong":
        # Sonnet 5.5 thinks by default ("disabled" is a 400). "between_tools" keeps thinking to the gaps
        # between tool calls, for latency. Those blocks are still signed and bound to the exact prompt
        # prefix, which is why worker histories are append-only (graph/workers.py:round_budget).
        return ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key, streaming=True,
                             default_request_timeout=TIMEOUT_S, max_retries=1, thinking={"type": "between_tools"})
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
