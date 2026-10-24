"""Trace events, written to LangGraph's custom stream: {step, node, kind, agent, detail, ms}.

A routing decision is a trace event of kind "routing" that also carries the full RoutingDecision under
"routing"; streaming.py sends it as the `routing` SSE event. The supervisor's choice is a "decision"
event that also carries {next, task_type, reason, instruction} under "decision", for the UI's routing
badge. Inside recording(), each emitted routing decision is also collected, so a worker that fails can
still report the decisions it made (graph/guard.py).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from langgraph.config import get_stream_writer

from graph.state import RoutingDecision

TraceKind = Literal["enter", "exit", "tool_call", "tool_result", "error", "decision", "routing"]

_recorded: ContextVar[list[RoutingDecision] | None] = ContextVar("recorded_routing", default=None)


@contextmanager
def recording() -> Iterator[list[RoutingDecision]]:
    """Collect every routing decision emitted inside the block, including from tasks it starts."""
    decisions: list[RoutingDecision] = []
    token = _recorded.set(decisions)
    try:
        yield decisions
    finally:
        _recorded.reset(token)


def emit_trace(
    node: str,
    kind: TraceKind,
    *,
    step: int | None = None,
    agent: str | None = None,
    detail: str = "",
    ms: int | None = None,
    decision: dict | None = None,
) -> None:
    """Write one trace event. A no-op outside a graph run, or in a run not streaming "custom"."""
    event = {"step": step, "node": node, "kind": kind, "agent": agent, "detail": detail, "ms": ms}
    _write(event | {"decision": decision} if decision is not None else event)


def emit_routing(decision: RoutingDecision, *, step: int | None = None) -> None:
    """Write one routing decision, e.g. detail "strong anthropic claude-sonnet-5-5 (capability): ..."."""
    if (decisions := _recorded.get()) is not None:
        decisions.append(decision)
    detail = f"{decision.tier} {decision.provider} {decision.model} ({decision.dimension}): {decision.reason}"
    _write({"step": step, "node": decision.call_site, "kind": "routing", "agent": decision.agent, "detail": detail,
            "ms": None, "routing": decision.model_dump()})


def _write(event: dict) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:  # not inside a runnable context, e.g. a node function called directly
        return
    writer(event)
