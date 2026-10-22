"""Trace events, written to LangGraph's custom stream: {step, node, kind, agent, detail, ms}."""

from typing import Literal

from langgraph.config import get_stream_writer

TraceKind = Literal["enter", "exit", "tool_call", "tool_result", "error", "decision"]


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
    try:
        writer = get_stream_writer()
    except RuntimeError:  # not inside a runnable context, e.g. a node function called directly
        return
    writer({"step": step, "node": node, "kind": kind, "agent": agent, "detail": detail, "ms": ms})
