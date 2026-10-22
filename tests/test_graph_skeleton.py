"""Phase 4a: orchestration skeleton (dispatch, loop-back, limits, graceful degradation, streaming).

The supervisor's model, the synthesize model and the workers are all fakes injected through build_graph.
"""

import itertools

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from graph.build import DEFAULT_CONFIG, build_graph
from graph.state import Finding, SupervisorDecision
from graph.trace import emit_trace

pytestmark = pytest.mark.asyncio

ANSWER = "A point cloud is a set of 3D points."


def decision(next_: str, instruction: str = "", task_type: str = "general") -> SupervisorDecision:
    return SupervisorDecision(next=next_, instruction=instruction, task_type=task_type, reason=f"go {next_}")


class ScriptedSupervisor:
    """Stands in for a chat model: with_structured_output returns a runnable that plays `script`.

    Each item is a SupervisorDecision, a dict (returned raw, to test validation) or an exception (raised).
    """

    def __init__(self, script):
        self.script = iter(script)
        self.calls = 0

    def with_structured_output(self, schema):
        async def decide(_messages):
            self.calls += 1
            item = next(self.script)
            if isinstance(item, Exception):
                raise item
            return item

        return RunnableLambda(decide)


class StreamingJsonSupervisor(GenericFakeChatModel):
    """A fake chat model whose structured output is parsed from streamed JSON, like a real provider."""

    def with_structured_output(self, schema, **_):
        return self | RunnableLambda(lambda message: schema.model_validate_json(message.text))


def answer_llm(text: str = ANSWER) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.cycle([AIMessage(text)]))


class FakeWorker:
    def __init__(self, name: str = "research"):
        self.name = name
        self.instructions: list[str] = []

    async def __call__(self, state) -> dict:
        self.instructions.append(state["instruction"])
        return {"findings": [Finding(agent=self.name, content=f"finding for {state['instruction']}")]}


def inputs(question: str = "What's a point cloud?") -> dict:
    return {"session_id": "s1", "messages": [HumanMessage(question)], "step": 0}


async def run(graph, question: str = "What's a point cloud?") -> tuple[dict, list[dict]]:
    """Run the graph streaming custom events; return the final state and the trace events."""
    traces, final = [], {}
    async for mode, payload in graph.astream(inputs(question), config=DEFAULT_CONFIG, stream_mode=["custom", "values"]):
        if mode == "custom":
            traces.append(payload)
        else:
            final = payload
    return final, traces


def entered(traces: list[dict]) -> list[str]:
    return [t["node"] for t in traces if t["kind"] == "enter"]


async def test_empty_registry_general_question_goes_straight_to_synthesize():
    supervisor = ScriptedSupervisor([decision("synthesize")])
    graph = build_graph(supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph)
    assert entered(traces) == ["supervisor", "synthesize"]
    assert final["messages"][-1].text == ANSWER
    assert final.get("dispatches", []) == [] and final.get("errors", []) == []
    assert final["step"] == 1 and supervisor.calls == 1


async def test_dispatch_follows_next_and_worker_returns_to_supervisor():
    worker = FakeWorker()
    supervisor = ScriptedSupervisor([decision("research", "explain 3DGS rasterization", "lookup"), decision("synthesize")])
    graph = build_graph(workers={"research": worker}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph, "How does 3D Gaussian Splatting render a novel view?")
    assert entered(traces) == ["supervisor", "research", "supervisor", "synthesize"]
    assert worker.instructions == ["explain 3DGS rasterization"]
    assert [d.worker for d in final["dispatches"]] == ["research"]
    assert [f.agent for f in final["findings"]] == ["research"]
    assert final["step"] == 2 and final.get("errors", []) == []


async def test_step_limit_forces_synthesize(monkeypatch):
    monkeypatch.setenv("MAX_SUPERVISOR_STEPS", "3")
    worker = FakeWorker()
    supervisor = ScriptedSupervisor(decision("research", f"look up part {i}", "lookup") for i in itertools.count())
    graph = build_graph(workers={"research": worker}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph)
    assert final["step"] == 3 and len(worker.instructions) == 2
    assert [e.kind for e in final["errors"]] == ["step_limit"]
    assert entered(traces)[-1] == "synthesize" and final["messages"][-1].text == ANSWER


async def test_repeated_dispatch_is_refused():
    worker = FakeWorker()
    supervisor = ScriptedSupervisor([
        decision("research", "Explain 3DGS rasterization.", "lookup"),
        decision("research", "  explain 3dgs   rasterization", "lookup"),  # same after normalization
    ])
    graph = build_graph(workers={"research": worker}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, _ = await run(graph)
    assert len(worker.instructions) == 1
    assert [e.kind for e in final["errors"]] == ["repeated_dispatch"]
    assert final["messages"][-1].text == ANSWER


async def test_unregistered_worker_is_unavailable_and_graph_completes():
    supervisor = ScriptedSupervisor([decision("data", "best PSNR on Mip-NeRF 360", "metadata")])
    graph = build_graph(workers={"research": FakeWorker()}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph)
    assert [(e.agent, e.kind) for e in final["errors"]] == [("data", "worker_unavailable")]
    assert entered(traces) == ["supervisor", "synthesize"]
    assert final.get("dispatches", []) == [] and final["messages"][-1].text == ANSWER


@pytest.mark.parametrize("bad", [
    ValueError("model returned unparseable JSON"),
    {"next": "teleport", "instruction": "", "task_type": "general", "reason": "x"},  # off-schema
    None,  # a provider that returns nothing
])
async def test_bad_structured_output_is_a_parse_error_and_graph_completes(bad):
    graph = build_graph(supervisor_llm=ScriptedSupervisor([bad]), synthesize_llm=answer_llm())
    final, traces = await run(graph)
    assert [e.kind for e in final["errors"]] == ["supervisor_parse_error"]
    assert final["next"] == "synthesize" and final["messages"][-1].text == ANSWER
    assert any(t["kind"] == "error" and "supervisor_parse_error" in t["detail"] for t in traces)


async def test_every_node_that_ran_has_enter_and_exit_traces():
    supervisor = ScriptedSupervisor([decision("research", "a", "lookup"), decision("synthesize")])
    graph = build_graph(workers={"research": FakeWorker()}, supervisor_llm=supervisor, synthesize_llm=answer_llm())
    _, traces = await run(graph)
    enters = [t["node"] for t in traces if t["kind"] == "enter"]
    exits = [t["node"] for t in traces if t["kind"] == "exit"]
    assert enters == exits == ["supervisor", "research", "supervisor", "synthesize"]
    assert all(t["ms"] is not None for t in traces if t["kind"] == "exit")
    assert [t["detail"] for t in traces if t["kind"] == "decision"] == ["research (lookup): go research",
                                                                        "synthesize (general): go synthesize"]


async def test_streamed_answer_tokens_come_only_from_synthesize():
    supervisor = StreamingJsonSupervisor(messages=iter([AIMessage(decision("synthesize").model_dump_json())]))
    graph = build_graph(supervisor_llm=supervisor, synthesize_llm=answer_llm())
    chunks = []
    async for chunk, metadata in graph.astream(inputs(), config=DEFAULT_CONFIG, stream_mode="messages"):
        chunks.append((metadata["langgraph_node"], metadata.get("tags", []), chunk.text))
    # The supervisor's JSON does stream through the same channel, so consumers must filter...
    assert any(node == "supervisor" for node, _, _ in chunks)
    # ...and the "synthesize" tag selects exactly the answer tokens.
    answer = [(node, text) for node, tags, text in chunks if "synthesize" in tags]
    assert {node for node, _ in answer} == {"synthesize"}
    assert "".join(text for _, text in answer) == ANSWER


async def test_emit_trace_is_a_no_op_outside_a_streaming_context():
    emit_trace("supervisor", "enter", step=1)  # no runnable context at all
    graph = build_graph(supervisor_llm=ScriptedSupervisor([decision("synthesize")]), synthesize_llm=answer_llm())
    final = await graph.ainvoke(inputs(), config=DEFAULT_CONFIG)  # a run that doesn't stream "custom"
    assert final["messages"][-1].text == ANSWER
