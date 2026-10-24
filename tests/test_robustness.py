"""Phase 4c: the graph can't crash and can't cite what wasn't retrieved.

Fakes only: scripted chat models, fake worker callables, and the test stdio MCP server in
tests/fixtures/fake_server.py. No network.
"""

import asyncio
import logging

import anyio
import pytest
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

import mcp_client
from agents.registry import WORKERS
from graph.build import build_graph
from graph.guard import guarded
from graph.state import Citation, Finding
from graph.synthesize import ANSWER_FAILED_NOTICE
from graph.workers import FIX_CITATIONS, make_worker
from rag.citations import validate
from tests.test_graph_skeleton import ANSWER, ScriptedSupervisor, answer_llm, decision, run
from tests.test_phase3 import FAKE_SERVER, stdio_manifest
from tests.test_workers import ScriptedChatModel, call, search_corpus, worker_input

pytestmark = pytest.mark.asyncio

FAKE_KEY = "tvly-FAKE-SECRET-0123456789"
RESEARCH = WORKERS["research"]


def failing_worker(exc: BaseException):
    calls = []

    async def worker(state):
        calls.append(state["instruction"])
        raise exc

    worker.calls = calls
    return worker


def kinds(final: dict) -> list[str]:
    return [e.kind for e in final.get("errors", [])]


# ---------------------------------------------------------------- the worker failure wrapper

async def test_worker_exception_becomes_an_error_and_the_graph_answers():
    supervisor = ScriptedSupervisor([decision("research", "a", "lookup"), decision("synthesize")])
    graph = build_graph(workers={"research": failing_worker(ValueError("boom"))},
                        supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph)
    assert kinds(final) == ["worker_exception"]
    assert final.get("findings", []) == [] and final["messages"][-1].text == ANSWER
    error = next(t for t in traces if t["kind"] == "error" and t["node"] == "research")
    assert error["detail"] == "worker_exception" and error["agent"] == "research" and error["ms"] is not None
    # The worker's node still emits its exit event, and control went back to the supervisor.
    assert [t["node"] for t in traces if t["kind"] == "enter"] == ["load_history", "supervisor", "research", "supervisor", "synthesize", "save_history"]


async def test_slow_worker_times_out(monkeypatch):
    monkeypatch.setenv("WORKER_TIMEOUT_S", "0.05")

    async def slow(state):
        await asyncio.sleep(5)
        return {}

    supervisor = ScriptedSupervisor([decision("research", "a", "lookup"), decision("synthesize")])
    graph = build_graph(workers={"research": slow}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, _ = await run(graph)
    assert kinds(final) == ["worker_timeout"] and final["messages"][-1].text == ANSWER


async def test_cancellation_propagates_and_is_not_an_agent_error():
    node = guarded("research", failing_worker(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await node({"instruction": "a", "step": 1})


@pytest.mark.parametrize("exc, kind", [
    (McpError(ErrorData(code=-32000, message=f"Connection closed at https://mcp.tavily.com/mcp/?tavilyApiKey={FAKE_KEY}")),
     "transport"),
    (RuntimeError("tool failed"), "worker_exception"),
    (ModelRateLimitError(f"429 Too Many Requests; key={FAKE_KEY}"), "provider_error"),
])
async def test_failures_are_classified_and_messages_carry_no_secrets(exc, kind, caplog):
    supervisor = ScriptedSupervisor([decision("research", "a", "lookup"), decision("synthesize")])
    graph = build_graph(workers={"research": failing_worker(exc)}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph)
    assert kinds(final) == [kind]
    assert FAKE_KEY not in final["errors"][0].message
    assert all(FAKE_KEY not in (t["detail"] or "") for t in traces)


async def test_failure_wrapped_in_another_exception_is_classified_by_its_cause():
    try:
        raise anyio.ClosedResourceError()
    except anyio.ClosedResourceError as inner:
        outer = RuntimeError("worker gave up")
        outer.__cause__ = inner
    supervisor = ScriptedSupervisor([decision("research", "a", "lookup"), decision("synthesize")])
    graph = build_graph(workers={"research": failing_worker(outer)}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, _ = await run(graph)
    assert kinds(final) == ["transport"]


# ---------------------------------------------------------------- supervisor hardening

async def test_failing_worker_is_retried_at_most_once_then_synthesize():
    worker = failing_worker(ValueError("boom"))
    supervisor = ScriptedSupervisor(decision("research", f"attempt {i}", "lookup") for i in range(10))
    graph = build_graph(workers={"research": worker}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph)
    assert worker.calls == ["attempt 0", "attempt 1"]  # one try plus MAX_WORKER_RETRIES=1
    assert kinds(final) == ["worker_exception", "worker_exception"]
    assert supervisor.calls == 2  # every worker exhausted: the third decision skips the model
    assert any("every worker failed" in t["detail"] for t in traces if t["kind"] == "decision")
    assert final["messages"][-1].text == ANSWER


async def test_retry_past_the_limit_or_with_the_same_instruction_is_refused(monkeypatch):
    # Over the limit while another worker is still available.
    research = failing_worker(ValueError("boom"))
    supervisor = ScriptedSupervisor([decision("research", "a", "lookup"), decision("research", "b", "lookup"),
                                     decision("research", "c", "lookup")])
    graph = build_graph(workers={"research": research, "data": failing_worker(ValueError("unused"))},
                        supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, _ = await run(graph)
    assert research.calls == ["a", "b"] and kinds(final)[-1] == "retry_limit"

    # Within the limit, but the same instruction that already failed.
    research = failing_worker(ValueError("boom"))
    supervisor = ScriptedSupervisor([decision("research", "Explain 3DGS.", "lookup"),
                                     decision("research", "explain 3dgs", "lookup")])
    graph = build_graph(workers={"research": research}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, _ = await run(graph)
    assert research.calls == ["Explain 3DGS."] and kinds(final) == ["worker_exception", "retry_limit"]


async def test_all_workers_failed_gives_an_honest_answer_from_the_errors():
    synth = ScriptedChatModel(script=[AIMessage("Corpus search is unavailable right now, so I can't answer from the papers.")])
    supervisor = ScriptedSupervisor([decision("research", "a", "lookup"), decision("research", "b", "lookup")])
    graph = build_graph(workers={"research": failing_worker(ConnectionError("broken pipe"))},
                        supervisor_llm=supervisor, synthesize_llm=synth)
    final, _ = await run(graph)
    assert kinds(final) == ["transport", "transport"]
    prompt = synth.seen[0][-1].text
    assert "Findings:\n(none)" in prompt and "lost its connection to a tool server" in prompt
    assert final["messages"][-1].text.startswith("Corpus search is unavailable")


async def test_supervisor_model_error_goes_to_synthesize():
    graph = build_graph(workers={}, supervisor_llm=ScriptedSupervisor([RuntimeError("503 Service Unavailable")]),
                        synthesize_llm=answer_llm())
    final, _ = await run(graph)
    assert kinds(final) == ["supervisor_error"] and final["messages"][-1].text == ANSWER


# ---------------------------------------------------------------- synthesize hardening

class BreakingStream(BaseChatModel):
    """Streams the given tokens, then raises as a provider would mid-stream."""

    tokens: list[str]

    @property
    def _llm_type(self) -> str:
        return "breaking"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("".join(self.tokens)))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for token in self.tokens:
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        raise ModelRateLimitError("429 Too Many Requests")


@pytest.mark.parametrize("tokens, expected", [
    ([], ANSWER_FAILED_NOTICE),
    (["NeRF uses ", "an MLP."], f"NeRF uses an MLP.\n\n{ANSWER_FAILED_NOTICE}"),
])
async def test_synthesize_failure_ends_with_a_fixed_notice(tokens, expected):
    graph = build_graph(workers={}, supervisor_llm=ScriptedSupervisor([decision("synthesize")]),
                        synthesize_llm=BreakingStream(tokens=tokens))
    final, traces = await run(graph)
    assert final["messages"][-1].text == expected
    assert kinds(final) == ["synthesize_error"]
    assert any(t["kind"] == "error" and t["node"] == "synthesize" for t in traces)


# ---------------------------------------------------------------- citation validation

@pytest.mark.parametrize("text, grounded", [
    ("Tiles [2308.04079#method-03]. Invented [2308.04079#method-99].", True),
    ("Only invented [2308.04079#method-99].", False),
])
async def test_worker_marks_unretrieved_citations_unverified(text, grounded):
    # The worker asks once for a fix; this model repeats itself, so the invented ID stays.
    model = ScriptedChatModel(script=[call("search_corpus", {"query": "x"}), AIMessage(text), AIMessage(text)])
    supervisor = ScriptedSupervisor([decision("research", "a", "lookup"), decision("synthesize")])
    graph = build_graph(workers={"research": make_worker(RESEARCH, [search_corpus], model)},
                        supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph)
    finding = final["findings"][0]
    assert "2308.04079#method-99" not in finding.content and "[unverified]" in finding.content
    assert finding.grounded is grounded
    assert [c.id for c in finding.citations] == (["2308.04079#method-03"] if grounded else [])
    error = next(e for e in final["errors"] if e.kind == "invalid_citation")
    assert "2308.04079#method-99" in error.message
    assert any("invalid citations" in t["detail"] for t in traces if t["kind"] == "error" and t["agent"] == "research")


async def test_worker_gets_one_chance_to_replace_an_unretrieved_citation():
    model = ScriptedChatModel(script=[call("search_corpus", {"query": "x"}),
                                      AIMessage("Tiles [2308.04079#method-99]."),
                                      AIMessage("Tiles [2308.04079#method-03].")])
    out = await make_worker(RESEARCH, [search_corpus], model)(worker_input())
    finding = out["findings"][0]
    assert finding.content == "Tiles [2308.04079#method-03]." and out.get("errors") is None
    assert [c.id for c in finding.citations] == ["2308.04079#method-03"]
    fix = model.seen[2][-1].text
    assert fix.startswith(FIX_CITATIONS.split(":")[0]) and "2308.04079#method-99" in fix
    assert model.seen[2][-2].text == "Tiles [2308.04079#method-99]."  # appended after the reply: history stays append-only


async def test_answer_citations_split_valid_and_invalid_in_order():
    cite = lambda cid: Citation(id=cid, source="corpus", arxiv_id=cid.split("#")[0])  # noqa: E731
    finding = Finding(agent="research", content="x", citations=[cite("2003.08934#method-00"), cite("2308.04079#method-03")],
                      grounded=True)

    async def worker(state):
        return {"findings": [finding]}

    answer = ("3DGS sorts tiles [2308.04079#method-03]. NeRF uses an MLP [2003.08934#method-00]. "
              "Again [2308.04079#method-03]. Invented [2201.05989#intro-00][web:deadbeef].")
    supervisor = ScriptedSupervisor([decision("research", "a", "compare"), decision("synthesize")])
    graph = build_graph(workers={"research": worker}, supervisor_llm=supervisor, synthesize_llm=answer_llm(answer))
    final, traces = await run(graph)
    report = final["answer_citations"]
    assert [c.id for c in report.valid] == ["2308.04079#method-03", "2003.08934#method-00"]
    assert report.invalid == ["2201.05989#intro-00", "web:deadbeef"]
    assert final["messages"][-1].text == answer  # already streamed: reported, not rewritten
    assert any("unverified citations in the answer" in t["detail"] for t in traces if t["kind"] == "error")


async def test_validate_keeps_first_appearance_order_and_collapses_duplicates():
    allowed = {i: Citation(id=i, source="corpus") for i in ["2308.04079#method-03", "2003.08934#method-00"]}
    report = validate("[2003.08934#method-00] [2308.04079#method-03][2003.08934#method-00] [web:00000000]", allowed)
    assert [c.id for c in report.valid] == ["2003.08934#method-00", "2308.04079#method-03"]
    assert report.invalid == ["web:00000000"]


# ---------------------------------------------------------------- MCP resilience

def fake_manifest() -> dict:
    return stdio_manifest(FAKE_SERVER, ["echo", "crash", "sleep"], "Fake server is unavailable.")


async def tool_call(tools, name, **args):
    message = await tools._tools[name].ainvoke({"type": "tool_call", "id": "t1", "name": name, "args": args})
    text = message.content if isinstance(message.content, str) else "".join(b.get("text", "") for b in message.content)
    return message.status, text


async def test_tool_timeout_is_a_tool_error_and_the_worker_finishes(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_S", "0.5")
    tools = mcp_client.MCPTools({"fake": fake_manifest()})
    await tools.start()
    try:
        assert (await tool_call(tools, "sleep", seconds=5))[0] == "error"
        assert tools.status["fake"].state == "up"  # a timeout is not a dead server
        model = ScriptedChatModel(script=[call("sleep", {"seconds": 5}), AIMessage("The lookup timed out; not in the corpus.")])
        out = await make_worker(RESEARCH, [tools._tools["sleep"]], model)(worker_input())
        tool_message = model.seen[1][-2]  # then the round budget
        assert tool_message.status == "error" and "timed out after 0.5 s" in tool_message.text
        assert out["findings"][0].content == "The lookup timed out; not in the corpus."
    finally:
        await tools.aclose()


async def test_dead_session_is_marked_unhealthy_and_reconnects_next_turn_with_backoff(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="mcp_client")
    manifest = fake_manifest()
    tools = mcp_client.MCPTools({"fake": manifest})
    await tools.start()
    try:
        # The session dies mid-call: a clean tool error, and the server is unhealthy.
        assert await tool_call(tools, "crash") == ("error", "Fake server is unavailable.")
        assert tools.status["fake"].state == "down"
        assert "is unhealthy" in caplog.text
        # Until it reconnects, calls fail immediately without touching the server.
        assert await tool_call(tools, "echo", text="hi") == ("error", "Fake server is unavailable.")

        # Next turn: one reconnect attempt. Make it fail, to exercise the backoff.
        manifest["command"] = "definitely-not-a-real-command"
        monkeypatch.setenv("MCP_RECONNECT_BACKOFF_S", "0.3")
        tools.refresh()
        assert "trying to reconnect" in caplog.text
        await asyncio.gather(*tools._background)
        assert tools.status["fake"].state == "down"

        # Within the backoff, a new turn does not try again.
        tools.refresh()
        assert not tools._background and tools.status["fake"].state == "down"

        # After the backoff, the next turn reconnects, and the same tool object works again.
        manifest["command"] = "python"
        await asyncio.sleep(0.35)
        tools.refresh()
        await asyncio.gather(*tools._background)
        assert tools.status["fake"].state == "up" and "reconnected" in caplog.text
        assert await tool_call(tools, "echo", text="back") == ("success", "back")
    finally:
        await tools.aclose()


async def test_unhealthy_server_description_reaches_the_workers_prompt():
    tools = mcp_client.MCPTools({"corpus": {**stdio_manifest(FAKE_SERVER, ["search_corpus", "get_chunks"],
                                                             "Corpus search is unavailable."), "command": "nope"}})
    await tools.start()
    try:
        model = ScriptedChatModel(script=[AIMessage("Not in the corpus: corpus search is unavailable.")])
        worker = make_worker(RESEARCH, lambda: tools.tools_for("research"), model,
                             unavailable_notes=lambda: tools.unavailable_notes("research"))
        await worker(worker_input())
        assert "Corpus search is unavailable." in model.seen[0][0].text and model.bound == []
    finally:
        await tools.aclose()

