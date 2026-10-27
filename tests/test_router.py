"""Phase 5a: provider tiers, the routing rules and provider fallback. No network calls."""

import asyncio

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import ModelError
from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import router
from graph.guard import classify, is_provider_error
from graph.state import Citation, Finding
from llm import get_llm, model_id
from router import RouteFeatures, invoke_with_fallback, route


def test_each_tier_is_its_own_provider():
    assert isinstance(get_llm("fast"), ChatGoogleGenerativeAI)
    assert isinstance(get_llm("strong"), ChatAnthropic)
    assert get_llm("strong").thinking == {"type": "between_tools"}  # less thinking, for latency
    assert [model_id(t) for t in ("fast", "strong")] == ["test-gemini-model", "test-anthropic-model"]
    with pytest.raises(ValueError):
        get_llm("fallback")  # OpenAI is not a chat provider; each tier falls back to the other


def test_model_ids_default_to_env_example_values(monkeypatch):
    import config
    monkeypatch.delenv("ANTHROPIC_MODEL")
    config.get_settings.cache_clear()
    assert model_id("strong") == config.DEFAULT_ANTHROPIC_MODEL
    assert "key" not in repr(config.get_settings()).replace("api keys=<redacted>", "")


def tiers(call_site, cases):
    return [route(call_site, RouteFeatures(**features)).tier for features in cases]


def test_supervisor_is_always_fast():
    d = route("supervisor", RouteFeatures(task_type="compare", n_findings=5, context_tokens=10**6))
    assert (d.tier, d.provider, d.dimension, d.fallback) == ("fast", "google", "cost", False)


def test_data_escalates_after_a_sql_error():
    assert tiers("data", [{}, {"sql_errored": True}]) == ["fast", "strong"]
    assert route("data", RouteFeatures(sql_errored=True)).dimension == "hybrid"


def test_research_escalates_for_multi_paper_tasks():
    assert tiers("research", [{"task_type": t} for t in ("lookup", "current_events", "compare", "lit_review")]) == [
        "fast", "fast", "strong", "strong"]
    d = route("research", RouteFeatures(task_type="compare"))
    assert (d.provider, d.model, d.dimension) == ("anthropic", "test-anthropic-model", "capability")
    assert "task_type=compare" in d.reason


def test_summary_is_strong_unless_one_paper_is_in_scope():
    assert tiers("summary", [{"papers_in_scope": n} for n in (0, 1, 2, 10)]) == ["strong", "fast", "strong", "strong"]


def test_synthesize_escalates_on_findings_papers_or_context(monkeypatch):
    monkeypatch.setenv("STRONG_CONTEXT_TOKENS", "1000")
    cases = [{"n_findings": 1, "n_cited_papers": 2, "context_tokens": 1000},
             {"n_findings": 2}, {"n_cited_papers": 3}, {"context_tokens": 1001}]
    assert tiers("synthesize", cases) == ["fast", "strong", "strong", "strong"]
    reason = route("synthesize", RouteFeatures(n_findings=2, n_cited_papers=4)).reason
    assert "2 findings" in reason and "4 papers cited" in reason


def test_unknown_call_site_is_rejected():
    with pytest.raises(ValueError):
        route("planner", RouteFeatures())


def test_feature_helpers():
    ok = ToolMessage('{"sql": "SELECT 1", "rows": [[1]]}', name="run_sql", tool_call_id="a")
    bad = ToolMessage("SQL error: no such column", name="run_sql", tool_call_id="b", status="error")
    other = ToolMessage("Unknown arXiv ID", name="get_paper_metadata", tool_call_id="c", status="error")
    assert (router.sql_errored([AIMessage(""), ok]), router.sql_errored([ok, other]), router.sql_errored([ok, bad])) == (
        False, False, True)
    assert router.papers_in_scope("Summarize 2403.17888, 2406.09756 and 2403.17888 (again)") == 2
    assert router.papers_in_scope("Review anti-aliasing in radiance fields since 2021.") == 0
    assert router.papers_in_scope("What does 2308.04079 propose? See 2308.04079#method-03") == 1
    findings = [Finding(agent="data", content="", citations=[Citation(id="db:2308.04079", source="db", arxiv_id="2308.04079")]),
                Finding(agent="research", content="", citations=[
                    Citation(id="2308.04079#method-03", source="corpus", arxiv_id="2308.04079"),
                    Citation(id="2003.08934#method-00", source="corpus", arxiv_id="2003.08934"),
                    Citation(id="web:1a2b3c4d", source="web", url="https://example.org")])]
    assert router.cited_papers(findings) == 2
    assert router.estimate_tokens("x" * 400) == 100


class AuthenticationError(Exception):
    """Shaped like an SDK error: classified by its module, as the real anthropic errors are."""


AuthenticationError.__module__ = "anthropic"


def calls(primary_error=None, fallback_error=None):
    log = []

    async def primary():
        log.append("primary")
        if primary_error:
            raise primary_error
        return "primary answer"

    async def fallback():
        log.append("fallback")
        if fallback_error:
            raise fallback_error
        return "fallback answer"

    return log, primary, fallback


@pytest.mark.asyncio
async def test_no_fallback_when_the_primary_succeeds():
    log, primary, fallback = calls()
    assert await invoke_with_fallback(route("summary", RouteFeatures()), primary, fallback) == ("primary answer", [])
    assert log == ["primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [AuthenticationError("invalid x-api-key"), ModelError("429")])
async def test_provider_error_falls_back_once_to_the_other_provider_and_is_recorded(error):
    log, primary, fallback = calls(primary_error=error)
    result, [switched] = await invoke_with_fallback(route("summary", RouteFeatures()), primary, fallback)
    assert result == "fallback answer" and log == ["primary", "fallback"]
    assert (switched.call_site, switched.tier, switched.provider, switched.model, switched.fallback) == (
        "summary", "fast", "google", "test-gemini-model", True)
    assert type(error).__name__ in switched.reason and "anthropic" in switched.reason


@pytest.mark.asyncio
async def test_a_failing_fast_call_falls_back_to_the_strong_provider():
    _, primary, fallback = calls(primary_error=ModelError("429"))
    _, [switched] = await invoke_with_fallback(route("research", RouteFeatures()), primary, fallback)
    assert (switched.tier, switched.provider, switched.model) == ("strong", "anthropic", "test-anthropic-model")
    assert "from google (fast tier): retried on anthropic (strong tier)" in switched.reason


@pytest.mark.asyncio
async def test_a_wrapped_provider_error_also_falls_back():
    try:
        try:
            raise AuthenticationError("bad key")
        except AuthenticationError as inner:
            raise RuntimeError("model call failed") from inner
    except RuntimeError as wrapped:
        error = wrapped
    _, primary, fallback = calls(primary_error=error)
    result, [switched] = await invoke_with_fallback(route("research", RouteFeatures()), primary, fallback)
    assert result == "fallback answer" and switched.fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("bad output"), asyncio.CancelledError()])
async def test_non_provider_errors_do_not_fall_back(error):
    log, primary, fallback = calls(primary_error=error)
    with pytest.raises(type(error)):
        await invoke_with_fallback(route("data", RouteFeatures()), primary, fallback)
    assert log == ["primary"]


@pytest.mark.asyncio
async def test_a_failing_fallback_propagates_as_a_provider_error():
    _, primary, fallback = calls(primary_error=AuthenticationError("anthropic"), fallback_error=ModelError("google down"))
    with pytest.raises(ModelError) as raised:
        await invoke_with_fallback(route("synthesize", RouteFeatures()), primary, fallback)
    assert classify(raised.value) == "provider_error"


@pytest.mark.asyncio
async def test_a_call_that_is_already_a_fallback_is_not_retried():
    decision = route("research", RouteFeatures()).model_copy(update={"tier": "strong", "fallback": True})
    log, primary, fallback = calls(primary_error=ModelError("down"))
    with pytest.raises(ModelError):
        await invoke_with_fallback(decision, primary, fallback)
    assert log == ["primary"]


def test_is_provider_error_matches_guard_classification():
    assert is_provider_error(AuthenticationError("x")) and is_provider_error(ModelError("x"))
    assert not is_provider_error(ValueError("x")) and not is_provider_error(TimeoutError())
    assert classify(AuthenticationError("x")) == "provider_error"


@pytest.mark.asyncio
async def test_a_primary_call_past_its_timeout_falls_back():
    log = []

    async def stalled():
        log.append("primary")
        await asyncio.sleep(5)

    async def fallback():
        log.append("fallback")
        return "fallback answer"

    result, [switched] = await invoke_with_fallback(route("supervisor", RouteFeatures()), stalled, fallback, timeout=0.05)
    assert result == "fallback answer" and log == ["primary", "fallback"]
    assert switched.fallback and "no reply within 0.05 s from google" in switched.reason


@pytest.mark.asyncio
async def test_a_timeout_without_a_limit_set_is_not_a_provider_error():
    _, primary, fallback = calls(primary_error=TimeoutError())
    with pytest.raises(TimeoutError):
        await invoke_with_fallback(route("research", RouteFeatures()), primary, fallback)
