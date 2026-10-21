"""SSE event encoding and heartbeat."""

import asyncio
import json
from collections.abc import AsyncIterator

HEARTBEAT = ": ping\n\n"
HEARTBEAT_INTERVAL_S = 15.0


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
