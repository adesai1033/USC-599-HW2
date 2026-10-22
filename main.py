"""FastAPI app: GET / (static), POST /chat/stream (SSE), GET /health; reads PORT; lifespan init."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, StringConstraints

from config import get_settings
from llm import get_llm
from mcp_client import install_log_redaction
from streaming import sse_event, with_heartbeat

logger = logging.getLogger("paper_assistant")

STATIC_DIR = Path(__file__).parent / "static"
SYSTEM_PROMPT = "You are a research assistant for 3D computer vision papers."
USER_SAFE_ERROR = "Sorry, the language model is unavailable right now. Please try again."
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # fail fast on missing configuration
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    install_log_redaction()
    logger.info("starting with %r", settings)
    yield


app = FastAPI(title="3D Vision Paper Research Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    # session_id becomes a Firestore document ID, so it is limited to characters that are safe there.
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


async def _chat_frames(request: ChatRequest, start: float) -> AsyncIterator[str]:
    ttft_ms: int | None = None
    try:
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(request.message)]
        async for chunk in get_llm("fast").astream(messages):
            text = chunk.text
            if not text:
                continue
            if ttft_ms is None:
                ttft_ms = _ms_since(start)
            yield sse_event("token", {"text": text})
    except Exception:
        logger.exception("chat stream failed for session_id=%s", request.session_id)
        yield sse_event("error", {"message": USER_SAFE_ERROR})
    yield sse_event(
        "done",
        {
            "session_id": request.session_id,
            "timings": {"ttft_ms": ttft_ms, "total_ms": _ms_since(start)},
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    start = time.perf_counter()
    return StreamingResponse(
        with_heartbeat(_chat_frames(request, start)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=get_settings().port)
