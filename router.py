"""Provider routing: state features -> RoutingDecision; provider fallback.

route(call_site, features) is a pure function of inspectable features (task type, SQL errors, papers
in scope, findings, context size), so every decision can be logged, streamed and unit-tested:

  call site   default  escalate (or downgrade) when                                         dimension
  supervisor  fast     never: classification is cheap                                        cost
  data        fast     a run_sql call in this run errored                                    hybrid
  research    fast     task_type is compare or lit_review                                    capability
  summary     strong   downgrade to fast when exactly one paper is in scope                  capability
  synthesize  fast     >= 2 findings, >= 3 distinct papers cited, or context > STRONG_CONTEXT_TOKENS  hybrid

invoke_with_fallback() runs a call on the routed tier and, on a provider error, retries it once on
the fallback tier, returning a second decision with fallback=True and the original error's class.
"""

import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Literal, TypeVar

from langchain_core.messages import AnyMessage, ToolMessage

from config import strong_context_tokens
from graph.guard import is_provider_error
from graph.state import Finding, RoutingDecision
from llm import PROVIDERS, Tier, model_id

logger = logging.getLogger("router")

CallSite = Literal["supervisor", "research", "data", "summary", "synthesize"]
MULTI_PAPER_TASKS = frozenset({"compare", "lit_review"})
ARXIV_ID = re.compile(r"(?<![\d.])\d{4}\.\d{4,5}(?![\d])")
# About four characters per token for English prose; only used to compare against a threshold.
CHARS_PER_TOKEN = 4

T = TypeVar("T")


@dataclass(frozen=True)
class RouteFeatures:
    """The state features routing looks at. Each call site reads only the ones its rule names."""

    task_type: str = "general"
    sql_errored: bool = False
    papers_in_scope: int = 0
    n_findings: int = 0
    n_cited_papers: int = 0
    context_tokens: int = 0


def sql_errored(messages: Iterable[AnyMessage]) -> bool:
    """A run_sql call in these messages returned an error."""
    return any(isinstance(m, ToolMessage) and m.name == "run_sql" and m.status == "error" for m in messages)


def papers_in_scope(instruction: str) -> int:
    """Distinct arXiv IDs named in the supervisor's instruction (0 when it names none)."""
    return len(set(ARXIV_ID.findall(instruction)))


def cited_papers(findings: Iterable[Finding]) -> int:
    """Distinct papers cited across findings, from corpus chunks and paper-database rows."""
    return len({c.arxiv_id for f in findings for c in f.citations if c.arxiv_id})


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _decision(call_site: CallSite, tier: Tier, dimension: str, reason: str) -> RoutingDecision:
    return RoutingDecision(call_site=call_site, agent=call_site, provider=PROVIDERS[tier], model=model_id(tier),
                           tier=tier, dimension=dimension, reason=reason)


def route(call_site: CallSite, features: RouteFeatures) -> RoutingDecision:
    """Pick the tier for one LLM call from inspectable features; the reason names the deciding values."""
    f = features
    if call_site == "supervisor":
        return _decision(call_site, "fast", "cost", "routing decisions are a cheap classification: fast tier")
    if call_site == "data":
        if f.sql_errored:
            return _decision(call_site, "strong", "hybrid", "a run_sql call errored this run: strong tier to fix the query")
        return _decision(call_site, "fast", "hybrid", "no SQL errors so far: fast tier")
    if call_site == "research":
        if f.task_type in MULTI_PAPER_TASKS:
            return _decision(call_site, "strong", "capability",
                             f"task_type={f.task_type}: strong tier for multi-paper reasoning")
        return _decision(call_site, "fast", "capability", f"task_type={f.task_type}: fast tier")
    if call_site == "summary":
        if f.papers_in_scope == 1:
            return _decision(call_site, "fast", "capability", "one paper in scope: fast tier is enough")
        scope = f"{f.papers_in_scope} papers named" if f.papers_in_scope else "papers not named yet"
        return _decision(call_site, "strong", "capability", f"multi-paper summary ({scope}): strong tier")
    if call_site == "synthesize":
        limit = strong_context_tokens()
        reasons = [r for r, hit in [(f"{f.n_findings} findings", f.n_findings >= 2),
                                    (f"{f.n_cited_papers} papers cited", f.n_cited_papers >= 3),
                                    (f"~{f.context_tokens} context tokens > {limit}", f.context_tokens > limit)] if hit]
        if reasons:
            return _decision(call_site, "strong", "hybrid", f"{', '.join(reasons)}: strong tier")
        return _decision(call_site, "fast", "hybrid",
                         f"{f.n_findings} finding(s), {f.n_cited_papers} paper(s) cited, ~{f.context_tokens} "
                         "context tokens: fast tier")
    raise ValueError(f"Unknown call site: {call_site!r}")


def fallback_decision(decision: RoutingDecision, error: BaseException) -> RoutingDecision:
    """The decision recorded when a call moves to the fallback tier after a provider error."""
    return decision.model_copy(update={
        "provider": PROVIDERS["fallback"], "model": model_id("fallback"), "tier": "fallback", "fallback": True,
        "reason": f"{type(error).__name__} from {decision.provider} ({decision.tier} tier): retried on the fallback tier",
    })


async def invoke_with_fallback(
    decision: RoutingDecision,
    primary: Callable[[], Awaitable[T]],
    fallback: Callable[[], Awaitable[T]],
    on_fallback: Callable[[RoutingDecision], None] | None = None,
) -> tuple[T, list[RoutingDecision]]:
    """Run primary(); on a provider error, run fallback() once. Returns the result and any fallback decision.

    on_fallback(decision) runs before the fallback call, so the switch is recorded even if the fallback
    fails too. Only provider errors fall back: tool errors, bugs and cancellation propagate unchanged,
    and so does a failure of the fallback itself. A call already routed to the fallback tier isn't retried.
    """
    try:
        return await primary(), []
    except Exception as exc:
        if decision.tier == "fallback" or not is_provider_error(exc):
            raise
        switched = fallback_decision(decision, exc)
        logger.warning("%s: %s: %s", decision.call_site, switched.reason, str(exc)[:500])
        if on_fallback is not None:
            on_fallback(switched)
        return await fallback(), [switched]
