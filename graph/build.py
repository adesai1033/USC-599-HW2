"""Assembles the StateGraph, conditional edges, recursion limit.

  START -> supervisor -> (worker -> supervisor)* -> synthesize -> END

The conditional edge from the supervisor dispatches on state["next"]. The graph is compiled without
a checkpointer: conversation history is the only cross-request state (Firestore, Phase 7).
"""

import time
from collections.abc import Awaitable, Callable, Mapping

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.registry import available_workers
from config import max_supervisor_steps
from graph.state import OrchestratorState
from graph.supervisor import make_supervisor
from graph.synthesize import make_synthesize
from graph.trace import emit_trace

WorkerNode = Callable[[OrchestratorState], Awaitable[dict]]

# Each supervisor decision is at most two graph steps (supervisor, then a worker), plus synthesize.
DEFAULT_CONFIG = {"recursion_limit": 2 * max_supervisor_steps() + 4}


def _traced(name: str, worker: WorkerNode) -> WorkerNode:
    """Wrap a worker node with enter/exit trace events."""

    async def node(state: OrchestratorState) -> dict:
        start = time.perf_counter()
        emit_trace(name, "enter", step=state.get("step"), agent=name, detail=state.get("instruction", ""))
        update = await worker(state)
        emit_trace(name, "exit", step=state.get("step"), agent=name, ms=round((time.perf_counter() - start) * 1000))
        return update

    return node


def build_graph(
    *,
    workers: Mapping[str, WorkerNode] | None = None,
    supervisor_llm: BaseChatModel | None = None,
    synthesize_llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Compile the orchestration graph. Dependencies are injectable so tests can pass fakes.

    `workers` maps a worker name to its node function. By default it comes from the registry, which
    has no workers yet, so the default graph is supervisor -> synthesize -> END.
    """
    if workers is None:
        if available_workers():
            raise ValueError(f"registry lists workers {available_workers()} but no worker nodes were given")
        workers = {}

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", make_supervisor(supervisor_llm, list(workers)))
    graph.add_node("synthesize", make_synthesize(synthesize_llm))
    for name, worker in workers.items():
        graph.add_node(name, _traced(name, worker))
        graph.add_edge(name, "supervisor")

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", lambda state: state["next"], {**{name: name for name in workers}, "synthesize": "synthesize"}
    )
    graph.add_edge("synthesize", END)
    return graph.compile()
