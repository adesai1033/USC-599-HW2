import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

import config
import main
import streaming


class FakeLLM:
    def __init__(self, pieces, fail_after=None):
        self.pieces = pieces
        self.fail_after = fail_after

    async def astream(self, messages):
        for i, piece in enumerate(self.pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("provider exploded")
            yield AIMessageChunk(content=piece)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.split("\n\n"):
        lines = [l for l in frame.split("\n") if l and not l.startswith(":")]
        if not lines:
            continue
        name = next(l[len("event: "):] for l in lines if l.startswith("event: "))
        data = next(l[len("data: "):] for l in lines if l.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


def post_chat(monkeypatch, llm):
    monkeypatch.setattr(main, "get_llm", lambda tier: llm)
    with TestClient(main.app) as client:
        return client.post("/chat/stream", json={"session_id": "s1", "message": "What is NeRF?"})


def test_health_does_not_call_llm(monkeypatch):
    def boom(tier):
        raise AssertionError("health must not touch the LLM")

    monkeypatch.setattr(main, "get_llm", boom)
    with TestClient(main.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stream_emits_tokens_and_one_done(monkeypatch):
    response = post_chat(monkeypatch, FakeLLM(["Neural ", "radiance ", "fields"]))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names.count("token") >= 1
    assert names.count("done") == 1 and names[-1] == "done"
    assert "".join(d["text"] for n, d in events if n == "token") == "Neural radiance fields"

    done = events[-1][1]
    assert done["session_id"] == "s1"
    assert isinstance(done["timings"]["ttft_ms"], int)
    assert done["timings"]["total_ms"] >= done["timings"]["ttft_ms"]


def test_llm_failure_mid_stream_yields_error_then_done(monkeypatch):
    response = post_chat(monkeypatch, FakeLLM(["partial ", "never"], fail_after=1))
    assert response.status_code == 200

    names = [name for name, _ in parse_sse(response.text)]
    assert names == ["token", "error", "done"]
    error = parse_sse(response.text)[1][1]
    assert "provider exploded" not in error["message"]


def test_sse_event_format():
    frame = streaming.sse_event("token", {"text": "héllo"})
    assert frame == 'event: token\ndata: {"text": "héllo"}\n\n'


@pytest.mark.asyncio
async def test_heartbeat_emitted_while_idle():
    async def slow():
        await asyncio.sleep(0.05)
        yield "frame"

    out = [f async for f in streaming.with_heartbeat(slow(), interval=0.01)]
    assert out[-1] == "frame"
    assert streaming.HEARTBEAT in out


def test_port_read_from_env(monkeypatch):
    monkeypatch.setenv("PORT", "8123")
    assert config.get_settings().port == 8123


def test_port_defaults_to_8080():
    assert config.get_settings().port == 8080


def test_missing_required_key_fails_fast(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        config.get_settings()


def test_settings_repr_redacts_key():
    assert "test-google-key" not in repr(config.get_settings())
