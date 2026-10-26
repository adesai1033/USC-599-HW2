"""SSE event encoding, heartbeat, and the mapping from a graph run to the SSE contract.

graph_frames() streams one turn as: routing / trace events while the graph runs, token events from
synthesize, then citations, an error if the answer failed, and always a final done.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.graph.state import CompiledStateGraph

from graph.build import DEFAULT_CONFIG
from graph.state import CitationReport

logger = logging.getLogger("streaming")

HEARTBEAT = ": ping\n\n"
HEARTBEAT_INTERVAL_S = 15.0
RUN_FAILED_MESSAGE = "Sorry, something went wrong while answering. Please try again."

# LangChain's ls_provider names, mapped to the provider names RoutingDecision uses.
_PROVIDER_NAMES = {"google_genai": "google"}


def sse_event(event: str, data: dict) -> str:
    """Encode one SSE frame: `event: <name>\\ndata: <json>\\n\\n`."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def with_heartbeat(
    frames: AsyncIterator[str], interval: float = HEARTBEAT_INTERVAL_S
) -> AsyncIterator[str]:
    """Pass frames through, emitting a heartbeat comment whenever `interval` passes with none."""
    iterator = aiter(frames)
    pending = asyncio.ensure_future(anext(iterator))
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield HEARTBEAT
                continue
            try:
                frame = pending.result()
            except StopAsyncIteration:
                return
            yield frame
            pending = asyncio.ensure_future(anext(iterator))
    finally:
        pending.cancel()


class UsageTimer(BaseCallbackHandler):
    """Token usage per provider and summed model-call time for one run.

    llm_ms adds up each chat model call's wall time; calls rarely overlap, so it is close to the time
    spent waiting on models.
    """

    run_inline = True

    def __init__(self) -> None:
        self.usage: dict[str, dict[str, int]] = {}
        self.llm_ms = 0
        self._started: dict[UUID, tuple[float, str]] = {}

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: UUID,
                            metadata: dict | None = None, **kwargs: Any) -> None:
        provider = (metadata or {}).get("ls_provider") or "unknown"
        self._started[run_id] = (time.perf_counter(), _PROVIDER_NAMES.get(provider, provider))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start, provider = self._started.pop(run_id, (None, "unknown"))
        if start is not None:
            self.llm_ms += round((time.perf_counter() - start) * 1000)
        generation = response.generations[0][0] if response.generations and response.generations[0] else None
        usage = generation.message.usage_metadata if isinstance(generation, ChatGeneration) else None
        if usage:
            totals = self.usage.setdefault(provider, {"input": 0, "output": 0})
            totals["input"] += usage.get("input_tokens", 0)
            totals["output"] += usage.get("output_tokens", 0)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        start, _ = self._started.pop(run_id, (None, ""))
        if start is not None:
            self.llm_ms += round((time.perf_counter() - start) * 1000)


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


async def graph_frames(graph: CompiledStateGraph, inputs: dict, *, session_id: str, start: float,
                       final_state: dict | None = None) -> AsyncIterator[str]:
    """Run one turn of the graph and yield its SSE frames, ending with done whatever happens.

    If `final_state` is given, it is filled with the graph's final state (eval/run_ragas.py reads the
    retrieved chunk IDs from it).
    """
    usage = UsageTimer()
    final: dict = {}
    streamed: list[str] = []
    ttft_ms: int | None = None
    retrieval_ms = 0
    failed = False
    try:
        # subgraphs=True: trace events written inside worker subgraphs reach this stream too.
        async for namespace, mode, payload in graph.astream(
            inputs, config=DEFAULT_CONFIG | {"callbacks": [usage]},
            stream_mode=["custom", "messages", "values"], subgraphs=True,
        ):
            if mode == "custom":
                if payload["kind"] == "routing":
                    yield sse_event("routing", payload["routing"])
                    continue
                if payload["kind"] == "tool_result" and payload.get("ms"):
                    retrieval_ms += payload["ms"]
                yield sse_event("trace", payload)
            elif mode == "messages":
                chunk, metadata = payload
                # Only the answer: supervisor calls stream structured-output JSON through the same channel.
                if metadata.get("langgraph_node") == "synthesize" and "synthesize" in metadata.get("tags", []) and chunk.text:
                    if ttft_ms is None:
                        ttft_ms = _ms_since(start)
                    streamed.append(chunk.text)
                    yield sse_event("token", {"text": chunk.text})
            elif not namespace:  # the root graph's state, not a worker subgraph's
                final = payload
    except Exception:
        logger.exception("graph run failed for session_id=%s", session_id)
        failed = True

    if not failed:
        # Text the stream didn't carry, e.g. the notice synthesize adds when the model fails.
        answer = final["messages"][-1].text if final.get("messages") else ""
        sent = "".join(streamed)
        if answer.strip() != sent.strip():
            rest = answer.removeprefix(sent.rstrip())
            yield sse_event("token", {"text": rest if sent else rest.lstrip()})
        report = final.get("answer_citations") or CitationReport()
        yield sse_event("citations", report.model_dump())
        failure = next((e for e in final.get("errors", []) if e.kind == "synthesize_error"), None)
        if failure is not None:
            yield sse_event("error", {"message": failure.message})
    else:
        yield sse_event("error", {"message": RUN_FAILED_MESSAGE})

    if final_state is not None:
        final_state.update(final)
    yield sse_event("done", {
        "session_id": session_id,
        "timings": {"ttft_ms": ttft_ms, "retrieval_ms": retrieval_ms, "llm_ms": usage.llm_ms,
                    "total_ms": _ms_since(start)},
        "usage": usage.usage,
    })
