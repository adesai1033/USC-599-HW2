"""FastAPI app: GET / (static), POST /chat/stream (SSE), GET /health; reads PORT; lifespan init.

The MCP sessions and the compiled graph start in a background task, so /health answers during a cold
start while the servers spawn and the Chroma index loads. /chat/stream waits for that task.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, StringConstraints

from config import get_settings
from graph.build import build_graph
from mcp_client import MCPTools, install_log_redaction, load_manifests
from streaming import graph_frames, sse_event, with_heartbeat

logger = logging.getLogger("paper_assistant")

STATIC_DIR = Path(__file__).parent / "static"
STARTUP_FAILED_MESSAGE = "Sorry, the assistant failed to start. Please try again later."
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


async def make_runtime() -> tuple[MCPTools, CompiledStateGraph]:
    """Connect the MCP servers (failures are recorded, not raised) and compile the graph."""
    mcp = MCPTools(load_manifests())
    await mcp.start()
    return mcp, build_graph(mcp=mcp)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # fail fast on missing configuration
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    install_log_redaction()
    logger.info("starting with %r", settings)
    app.state.runtime = asyncio.create_task(make_runtime())
    try:
        yield
    finally:
        app.state.runtime.cancel()
        with suppress(Exception, asyncio.CancelledError):  # startup failed or was cut short
            mcp, _ = await app.state.runtime
            await mcp.aclose()


app = FastAPI(title="3D Vision Paper Research Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    # session_id becomes a Firestore document ID, so it is limited to characters that are safe there.
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
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


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    start = time.perf_counter()
    return StreamingResponse(
        with_heartbeat(_chat_frames(app.state.runtime, request, start)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=get_settings().port)
