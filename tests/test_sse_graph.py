"""Phase 6: a graph run streamed through POST /chat/stream follows the SSE contract.

The graph is built from fakes and served by main.app in place of the real MCP servers.
"""

import itertools

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import main
import streaming
from agents.registry import WORKERS
from graph.build import build_graph
from graph.state import RoutingDecision
from graph.workers import make_worker
from tests.sse import parse_sse, serve
from tests.test_graph_skeleton import ScriptedSupervisor, decision
from tests.test_workers import ScriptedChatModel, search_corpus

USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
FINDING = "Gaussians are rasterized per tile [2308.04079#method-03]."
ANSWER = "3DGS rasterizes Gaussians per tile [2308.04079#method-03], unlike NeRF [2003.08934#method-00]."


def research_graph():
    research = ScriptedChatModel(script=[
        AIMessage("", tool_calls=[{"name": "search_corpus", "args": {"query": "3DGS"}, "id": "c1"},
                                  {"name": "search_corpus", "args": {"query": "tiles"}, "id": "c2"}],
                  usage_metadata=USAGE),
        AIMessage(FINDING, usage_metadata=USAGE),
    ])
    return build_graph(
        workers={"research": make_worker(WORKERS["research"], [search_corpus], research)},
        supervisor_llm=ScriptedSupervisor([decision("research", "explain 3DGS rendering", "lookup"),
                                           decision("synthesize", task_type="lookup")]),
        synthesize_llm=GenericFakeChatModel(messages=itertools.cycle([AIMessage(ANSWER)])),
    )


def post(monkeypatch, graph) -> tuple[int, str, list[tuple[str, dict]]]:
    mcp = serve(monkeypatch, graph)
    with TestClient(main.app) as client:
        response = client.post("/chat/stream", json={"session_id": "s1", "message": "How does 3DGS render?"})
    assert mcp.refreshes == 1  # unhealthy servers get their reconnect attempt each turn
    return response.status_code, response.text, parse_sse(response.text)


def names(events) -> list[str]:
    return [n for n, _ in events]


def test_event_order_follows_the_contract(monkeypatch):
    status, _, events = post(monkeypatch, research_graph())
    order = names(events)
    assert status == 200
    assert order.index("routing") < order.index("token")
    last_token = max(i for i, n in enumerate(order) if n == "token")
    assert order.index("citations") > last_token
    assert order.count("done") == 1 and order[-1] == "done" and "error" not in order
    assert "".join(d["text"] for n, d in events if n == "token") == ANSWER


def test_routing_events_are_full_decisions_one_per_call_site(monkeypatch):
    _, _, events = post(monkeypatch, research_graph())
    decisions = [RoutingDecision.model_validate(d) for n, d in events if n == "routing"]
    assert [d.call_site for d in decisions] == ["supervisor", "research", "supervisor", "synthesize"]
    assert all(d.reason for d in decisions)


def test_trace_has_enter_exit_tools_and_a_structured_supervisor_decision(monkeypatch):
    _, _, events = post(monkeypatch, research_graph())
    traces = [d for n, d in events if n == "trace"]
    assert all(t["kind"] != "routing" for t in traces)  # those go out as routing events
    for node in ("supervisor", "research", "synthesize"):
        kinds = {t["kind"] for t in traces if t["node"] == node}
        assert {"enter", "exit"} <= kinds
    assert [t["detail"].split("(")[0] for t in traces if t["kind"] == "tool_call"] == ["search_corpus"] * 2
    first = next(t for t in traces if t["kind"] == "decision")
    assert first["decision"] == {"next": "research", "instruction": "explain 3DGS rendering",
                                 "task_type": "lookup", "reason": "go research"}


def test_citations_split_valid_and_invalid(monkeypatch):
    _, _, events = post(monkeypatch, research_graph())
    citations = dict(events)["citations"]
    # The finding cited only the 3DGS chunk, so the answer's NeRF chunk is not backed by any finding.
    assert [c["id"] for c in citations["valid"]] == ["2308.04079#method-03"]
    valid = citations["valid"][0]
    assert (valid["arxiv_id"], valid["source"], valid["url"]) == ("2308.04079", "corpus", "https://arxiv.org/abs/2308.04079")
    assert citations["invalid"] == ["2003.08934#method-00"]


def test_done_reports_usage_and_counts_a_tool_round_once(monkeypatch):
    _, _, events = post(monkeypatch, research_graph())
    done = events[-1][1]
    tool_results = [d for n, d in events if n == "trace" and d["kind"] == "tool_result"]
    assert len(tool_results) == 2 and tool_results[1]["ms"] is None  # one round, timed once
    assert done["timings"]["retrieval_ms"] == tool_results[0]["ms"]
    assert set(done["timings"]) == {"ttft_ms", "retrieval_ms", "llm_ms", "total_ms"}
    assert done["timings"]["llm_ms"] <= done["timings"]["total_ms"]
    # Both research calls reported usage; they come from the same (fake) provider.
    [(provider, totals)] = [(p, t) for p, t in done["usage"].items() if t["input"] >= 200]
    assert totals == {"input": 200, "output": 40} and provider


def test_a_crashing_graph_still_ends_with_error_then_done(monkeypatch):
    class Exploding:
        async def astream(self, *args, **kwargs):
            yield (), "custom", {"step": 1, "node": "supervisor", "kind": "enter", "agent": None, "detail": "", "ms": None}
            raise RuntimeError("secret internal detail")

    status, body, events = post(monkeypatch, Exploding())
    assert status == 200
    assert names(events) == ["trace", "error", "done"]
    assert events[1][1]["message"] == streaming.RUN_FAILED_MESSAGE
    assert "secret internal detail" not in body and "Traceback" not in body


def test_no_tool_question_streams_without_tool_events(monkeypatch):
    graph = build_graph(workers={}, supervisor_llm=ScriptedSupervisor([decision("synthesize")]),
                        synthesize_llm=GenericFakeChatModel(messages=itertools.cycle([AIMessage("A set of 3D points.")])))
    _, _, events = post(monkeypatch, graph)
    assert not [d for n, d in events if n == "trace" and d["kind"].startswith("tool")]
    assert dict(events)["citations"] == {"valid": [], "invalid": []}
    assert [d["call_site"] for n, d in events if n == "routing"] == ["supervisor", "synthesize"]


def test_supervisor_structured_output_json_is_not_streamed_as_tokens(monkeypatch):
    class JsonSupervisor(GenericFakeChatModel):
        def with_structured_output(self, schema, **_):
            return self | RunnableLambda(lambda message: schema.model_validate_json(message.text))

    supervisor = JsonSupervisor(messages=iter([AIMessage(decision("synthesize").model_dump_json())]))
    graph = build_graph(workers={}, supervisor_llm=supervisor,
                        synthesize_llm=GenericFakeChatModel(messages=itertools.cycle([AIMessage("answer")])))
    _, _, events = post(monkeypatch, graph)
    assert "".join(d["text"] for n, d in events if n == "token") == "answer"

