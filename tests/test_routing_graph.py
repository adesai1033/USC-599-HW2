"""Phase 5b: every LLM call in the graph is routed, recorded in state and streamed, with fallback.

Fakes only: each test maps tiers to scripted models, so a call on the wrong tier fails loudly.
"""

import itertools

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents.registry import WORKERS
from graph.build import build_graph
from graph.synthesize import ANSWER_FAILED_NOTICE
from graph.workers import make_worker
from tests.test_data_worker import DATA_TOOLS, TOP_CITED
from tests.test_graph_skeleton import ANSWER, FakeWorker, ScriptedSupervisor, answer_llm, decision, run
from tests.test_robustness import BreakingStream
from tests.test_router import AuthenticationError
from tests.test_workers import ScriptedChatModel, call, search_corpus, worker_input

pytestmark = pytest.mark.asyncio

RESEARCH, DATA, SUMMARY = WORKERS["research"], WORKERS["data"], WORKERS["summary"]
CITED = "Gaussians are rasterized per tile [2308.04079#method-03]."


class Unused(BaseChatModel):
    """A tier that must not be called in this test."""

    @property
    def _llm_type(self) -> str:
        return "unused"

    def _generate(self, *args, **kwargs) -> ChatResult:
        raise AssertionError("called a tier the router should not have picked")

    def bind_tools(self, tools, **kwargs):
        return self


class Failing(Unused):
    """A provider whose every call fails, like a bad API key; counts the attempts."""

    calls: int = 0

    def _generate(self, *args, **kwargs) -> ChatResult:
        self.calls += 1
        raise AuthenticationError("invalid x-api-key")


def tiers(fast=None, strong=None, fallback=None):
    models = {"fast": fast or Unused(), "strong": strong or Unused(), "fallback": fallback or Unused()}
    return lambda tier: models[tier]


def routed(final: dict) -> list[tuple[str, str, bool]]:
    return [(d.call_site, d.tier, d.fallback) for d in final["routing"]]


def streamed(traces: list[dict]) -> list[tuple[str, str, bool]]:
    return [(t["routing"]["call_site"], t["routing"]["tier"], t["routing"]["fallback"]) for t in traces
            if t["kind"] == "routing"]


async def test_compare_routes_research_to_strong_and_each_call_site_emits_once():
    strong = ScriptedChatModel(script=[call("search_corpus", {"query": "speed vs quality"}), AIMessage(CITED)])
    graph = build_graph(
        workers={"research": make_worker(RESEARCH, [search_corpus], tiers(strong=strong))},
        supervisor_llm=tiers(fast=ScriptedSupervisor([decision("research", "compare 3DGS and NeRF", "compare"),
                                                      decision("synthesize", task_type="compare")])),
        synthesize_llm=tiers(fast=answer_llm()),
    )
    final, traces = await run(graph, "Compare 3DGS and NeRF")
    expected = [("supervisor", "fast", False), ("research", "strong", False), ("supervisor", "fast", False),
                ("synthesize", "fast", False)]
    assert routed(final) == expected and streamed(traces) == expected  # two research model calls, one decision
    research = final["routing"][1]
    assert (research.provider, research.model, research.dimension) == ("anthropic", "test-anthropic-model", "capability")
    assert len(strong.seen) == 2


async def test_lookup_stays_on_fast():
    fast = ScriptedChatModel(script=[call("search_corpus", {"query": "3DGS rendering"}), AIMessage(CITED)])
    graph = build_graph(workers={"research": make_worker(RESEARCH, [search_corpus], tiers(fast=fast))},
                        supervisor_llm=ScriptedSupervisor([decision("research", "explain 3DGS", "lookup"),
                                                           decision("synthesize")]),
                        synthesize_llm=answer_llm())
    final, _ = await run(graph)
    assert {tier for _, tier, _ in routed(final)} == {"fast"}


async def test_two_findings_route_synthesize_to_strong():
    graph = build_graph(
        workers={"data": FakeWorker("data"), "research": FakeWorker("research")},
        supervisor_llm=ScriptedSupervisor([decision("data", "top cited"), decision("research", "what each proposes"),
                                           decision("synthesize")]),
        synthesize_llm=tiers(strong=answer_llm("strong answer")),
    )
    final, _ = await run(graph)
    assert routed(final)[-1] == ("synthesize", "strong", False) and final["messages"][-1].text == "strong answer"
    assert "2 findings" in final["routing"][-1].reason


async def test_data_escalates_to_strong_after_a_sql_error():
    fast = ScriptedChatModel(script=[call("describe_schema", {}, "c1"),
                                     call("run_sql", {"sql": "SELECT citations FROM papers"}, "c2")])
    strong = ScriptedChatModel(script=[call("run_sql", {"sql": TOP_CITED}, "c3"),
                                       AIMessage("3DGS is the most cited [db:2308.04079].")])
    out = await make_worker(DATA, DATA_TOOLS, tiers(fast=fast, strong=strong))(worker_input("most cited papers"))
    assert [(d.tier, d.dimension) for d in out["routing"]] == [("fast", "hybrid"), ("strong", "hybrid")]
    assert "run_sql call errored" in out["routing"][1].reason
    assert out["findings"][0].grounded and len(strong.seen) == 2


async def test_summary_downgrades_to_fast_for_one_paper():
    fast = ScriptedChatModel(script=[AIMessage("Not in the corpus.")])
    out = await make_worker(SUMMARY, [search_corpus], tiers(fast=fast))(worker_input("Summarize 2308.04079"))
    assert [(d.tier, d.reason) for d in out["routing"]] == [("fast", "one paper in scope: fast tier is enough")]


async def test_a_failing_strong_provider_falls_back_and_stays_on_the_fallback():
    strong = Failing()
    fallback = ScriptedChatModel(script=[call("search_corpus", {"query": "speed"}), AIMessage(CITED)])
    graph = build_graph(
        workers={"research": make_worker(RESEARCH, [search_corpus], tiers(strong=strong, fallback=fallback))},
        supervisor_llm=ScriptedSupervisor([decision("research", "compare 3DGS and NeRF", "compare"),
                                           decision("synthesize")]),
        synthesize_llm=answer_llm(),
    )
    final, traces = await run(graph)
    assert routed(final)[1:3] == [("research", "strong", False), ("research", "fallback", True)]
    assert streamed(traces)[1:3] == routed(final)[1:3]
    switched = final["routing"][2]
    assert (switched.provider, switched.model) == ("openai", "test-openai-model")
    assert "AuthenticationError from anthropic" in switched.reason
    assert strong.calls == 1 and len(fallback.seen) == 2  # the second call went straight to the fallback
    assert final.get("errors", []) == [] and final["findings"][0].grounded


async def test_supervisor_falls_back_when_its_provider_fails():
    graph = build_graph(workers={}, synthesize_llm=answer_llm(),
                        supervisor_llm=tiers(fast=ScriptedSupervisor([AuthenticationError("bad key")]),
                                             fallback=ScriptedSupervisor([decision("synthesize")])))
    final, _ = await run(graph)
    assert routed(final)[:2] == [("supervisor", "fast", False), ("supervisor", "fallback", True)]
    assert final.get("errors", []) == [] and final["messages"][-1].text == ANSWER


async def test_synthesize_falls_back_only_before_the_first_token():
    fallback = GenericFakeChatModel(messages=itertools.cycle([AIMessage("fallback answer")]))
    graph = build_graph(workers={}, supervisor_llm=ScriptedSupervisor([decision("synthesize")]),
                        synthesize_llm=tiers(fast=BreakingStream(tokens=[]), fallback=fallback))
    final, _ = await run(graph)
    assert routed(final)[-2:] == [("synthesize", "fast", False), ("synthesize", "fallback", True)]
    assert final["messages"][-1].text == "fallback answer" and final.get("errors", []) == []

    graph = build_graph(workers={}, supervisor_llm=ScriptedSupervisor([decision("synthesize")]),
                        synthesize_llm=tiers(fast=BreakingStream(tokens=["NeRF uses ", "an MLP."]), fallback=fallback))
    final, _ = await run(graph)
    assert routed(final)[-1] == ("synthesize", "fast", False)  # tokens had streamed: no second answer
    assert final["messages"][-1].text == f"NeRF uses an MLP.\n\n{ANSWER_FAILED_NOTICE}"
    assert [e.kind for e in final["errors"]] == ["synthesize_error"]


async def test_a_failing_fallback_is_a_provider_error_and_the_graph_finishes():
    graph = build_graph(
        workers={"research": make_worker(RESEARCH, [search_corpus], tiers(strong=Failing(), fallback=Failing()))},
        supervisor_llm=ScriptedSupervisor([decision("research", "compare 3DGS and NeRF", "compare"),
                                           decision("synthesize")]),
        synthesize_llm=answer_llm(),
    )
    final, traces = await run(graph)
    assert [e.kind for e in final["errors"]] == ["provider_error"] and final["messages"][-1].text == ANSWER
    assert ("research", "fallback", True) in streamed(traces)  # the switch was streamed before the fallback failed
