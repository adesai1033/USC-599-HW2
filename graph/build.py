"""Assembles the StateGraph, conditional edges, recursion limit.

  START -> supervisor -> (worker -> supervisor)* -> synthesize -> END

The conditional edge from the supervisor dispatches on state["next"]. The graph is compiled without
a checkpointer: conversation history is the only cross-request state (Firestore, Phase 7).
"""

import time
from collections.abc import Mapping
from functools import partial

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.registry import WORKERS, available_workers
from config import max_supervisor_steps
from graph.guard import guarded
from graph.state import OrchestratorState
from graph.supervisor import make_supervisor
from graph.synthesize import make_synthesize
from graph.trace import emit_trace
from graph.workers import WorkerNode, make_worker
from llm import get_llm
from mcp_client import MCPTools

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
    mcp: MCPTools | None = None,
) -> CompiledStateGraph:
    """Compile the orchestration graph. Dependencies are injectable so tests can pass fakes.

    `workers` maps a worker name to its node function. By default a worker is built for every agent in
    the registry's WORKERS, with its allowlisted tools from `mcp` (a started MCPTools) and its
    default-tier model. Tools and "unavailable" notes are read from `mcp` at the start of each run, so
    server health changes reach the next run. Every worker node is wrapped by guarded(): a failure
    becomes an AgentError and the graph returns to the supervisor.
    """
    if workers is None:
        if mcp is None:
            raise ValueError("build_graph needs a started MCPTools (mcp=...) to build the registry's workers")
        # Low thinking halved the research worker's time (27 s -> 13 s) and still found the same papers.
        workers = {
            name: make_worker(WORKERS[name], partial(mcp.tools_for, name),
                              get_llm(WORKERS[name].default_tier, thinking_level="low"),
                              unavailable_notes=partial(mcp.unavailable_notes, name))
            for name in available_workers()
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", make_supervisor(supervisor_llm, list(workers)))
    graph.add_node("synthesize", make_synthesize(synthesize_llm))
    for name, worker in workers.items():
        graph.add_node(name, _traced(name, guarded(name, worker)))
        graph.add_edge(name, "supervisor")

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", lambda state: state["next"], {**{name: name for name in workers}, "synthesize": "synthesize"}
    )
    graph.add_edge("synthesize", END)
    return graph.compile()
