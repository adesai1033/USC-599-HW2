"""FastAPI app: GET / (static), POST /chat/stream (SSE), GET /health; reads PORT; lifespan init.

Startup waits for the MCP sessions and the compiled graph before the port opens (bounded: each server
gets MCP_START_TIMEOUT_S, and a server that fails is recorded, not raised). Cloud Run gives an instance
full CPU until its port opens, but almost none between requests after that, so servers started in the
background on an instance that wasn't serving a request yet missed their limit. A runtime that failed
to start is reported by /chat/stream as an error event.
GET /sessions/{session_id}/turns returns a session's stored turns, so the UI can restore a conversation.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Path as PathParam
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, StringConstraints

from config import get_settings, history_turns
from graph.build import build_graph
from mcp_client import MCPTools, install_log_redaction, load_manifests
from state_store import History, Turn, make_history
from streaming import graph_frames, sse_event, with_heartbeat

logger = logging.getLogger("paper_assistant")

STATIC_DIR = Path(__file__).parent / "static"
STARTUP_FAILED_MESSAGE = "Sorry, the assistant failed to start. Please try again later."
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# session_id becomes a Firestore document ID, so it is limited to characters that are safe there.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
RESTORED_TURNS = 50


async def make_runtime(history: History) -> tuple[MCPTools, CompiledStateGraph]:
    """Connect the MCP servers (failures are recorded, not raised) and compile the graph."""
    mcp = MCPTools(load_manifests())
    await mcp.start()
    return mcp, build_graph(mcp=mcp, history=history)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # fail fast on missing configuration
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    install_log_redaction()
    logger.info("starting with %r", settings)
    app.state.history = make_history()
    app.state.runtime = asyncio.create_task(make_runtime(app.state.history))
    await asyncio.wait({app.state.runtime})  # doesn't raise: a failure is reported per request
    try:
        yield
    finally:
        app.state.runtime.cancel()
        with suppress(Exception, asyncio.CancelledError):  # startup failed or was cut short
            mcp, _ = await app.state.runtime
            await mcp.aclose()


app = FastAPI(title="3D Vision Paper Research Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]


async def _chat_frames(runtime: asyncio.Task, request: ChatRequest, start: float) -> AsyncIterator[str]:
    try:
        mcp, graph = await runtime
    except Exception:
        logger.exception("runtime startup failed")
        yield sse_event("error", {"message": STARTUP_FAILED_MESSAGE})
        yield sse_event("done", {"session_id": request.session_id, "timings": {"ttft_ms": None, "retrieval_ms": 0,
                                 "llm_ms": 0, "total_ms": round((time.perf_counter() - start) * 1000)}, "usage": {}})
        return
    mcp.refresh()  # one reconnect attempt per turn for unhealthy servers, in the background
    inputs = {"session_id": request.session_id, "messages": [HumanMessage(request.message)], "step": 0}
    async for frame in graph_frames(graph, inputs, session_id=request.session_id, start=start):
        yield frame


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sessions/{session_id}/turns")
async def session_turns(
    session_id: Annotated[str, PathParam(pattern=SESSION_ID_PATTERN)],
) -> list[Turn]:
    """The session's stored turns, oldest first (at most the last RESTORED_TURNS)."""
    try:
        return await app.state.history.load(session_id, max(RESTORED_TURNS, history_turns()))
    except Exception as exc:
        logger.warning("loading turns failed for session_id=%s: %s: %s", session_id, type(exc).__name__, exc)
        raise HTTPException(status_code=503, detail="Conversation history is unavailable right now.") from None


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    start = time.perf_counter()
    return StreamingResponse(
        with_heartbeat(_chat_frames(app.state.runtime, request, start)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate (a cheap ETag check), so a new deploy never runs old JS."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", RevalidatedStaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=get_settings().port)
