"""Trace events, written to LangGraph's custom stream: {step, node, kind, agent, detail, ms}.

A routing decision is a trace event of kind "routing" that also carries the full RoutingDecision under
"routing"; Phase 6 sends it as the `routing` SSE event.
"""

from typing import Literal

from langgraph.config import get_stream_writer

from graph.state import RoutingDecision

TraceKind = Literal["enter", "exit", "tool_call", "tool_result", "error", "decision", "routing"]


def emit_trace(
    node: str,
    kind: TraceKind,
    *,
    step: int | None = None,
    agent: str | None = None,
    detail: str = "",
    ms: int | None = None,
) -> None:
    """Write one trace event. A no-op outside a graph run, or in a run not streaming "custom"."""
    _write({"step": step, "node": node, "kind": kind, "agent": agent, "detail": detail, "ms": ms})


def emit_routing(decision: RoutingDecision, *, step: int | None = None) -> None:
    """Write one routing decision, e.g. detail "strong anthropic claude-sonnet-5-5 (capability): ..."."""
    detail = f"{decision.tier} {decision.provider} {decision.model} ({decision.dimension}): {decision.reason}"
    _write({"step": step, "node": decision.call_site, "kind": "routing", "agent": decision.agent, "detail": detail,
            "ms": None, "routing": decision.model_dump()})


def _write(event: dict) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:  # not inside a runnable context, e.g. a node function called directly
        return
    writer(event)
