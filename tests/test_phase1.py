import asyncio
import io
import logging

import pytest
from fastapi.testclient import TestClient

import config
import main
import streaming
from graph.build import build_graph
from graph.synthesize import ANSWER_FAILED_NOTICE
from tests.sse import parse_sse, serve
from tests.test_graph_skeleton import ScriptedSupervisor, answer_llm, decision
from tests.test_robustness import BreakingStream


@pytest.fixture(autouse=True)
def no_real_runtime(monkeypatch):
    """No test here starts the real MCP servers; tests that post a message serve their own graph."""
    serve(monkeypatch, make=_never_started)


async def _never_started():
    raise AssertionError("this test must not start the runtime")


def direct_graph(synthesize_llm):
    """A graph whose supervisor goes straight to synthesize, which streams from `synthesize_llm`."""
    return build_graph(workers={}, supervisor_llm=ScriptedSupervisor([decision("synthesize")]),
                       synthesize_llm=synthesize_llm)


def post_chat(monkeypatch, synthesize_llm):
    serve(monkeypatch, direct_graph(synthesize_llm))
    with TestClient(main.app) as client:
        return client.post("/chat/stream", json={"session_id": "s1", "message": "What is NeRF?"})


def test_health_does_not_wait_for_the_runtime(monkeypatch):
    started = asyncio.Event()

    async def slow_start():
        await started.wait()  # never set: the MCP servers are still spawning

    serve(monkeypatch, make=slow_start)
    with TestClient(main.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stream_emits_tokens_and_one_done(monkeypatch):
    response = post_chat(monkeypatch, answer_llm("Neural radiance fields"))
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
    response = post_chat(monkeypatch, BreakingStream(tokens=["partial "]))
    assert response.status_code == 200

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[-2:] == ["error", "done"] and names.count("done") == 1
    assert ANSWER_FAILED_NOTICE in "".join(d["text"] for n, d in events if n == "token")
    error = dict(events)["error"]
    assert "429" not in error["message"] and "Traceback" not in response.text


def test_startup_failure_yields_error_then_done(monkeypatch):
    async def broken():
        raise RuntimeError("chroma index missing")

    serve(monkeypatch, make=broken)
    with TestClient(main.app) as client:
        response = client.post("/chat/stream", json={"session_id": "s1", "message": "What is NeRF?"})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [n for n, _ in events] == ["error", "done"]
    assert "chroma" not in events[0][1]["message"]


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


@pytest.mark.parametrize("body", [
    {"session_id": "s1", "message": "   \n "},       # whitespace only
    {"session_id": "a/b", "message": "hi"},           # not a safe Firestore document ID
    {"session_id": "s1", "message": "x" * 8001},
])
def test_invalid_requests_are_rejected_before_the_graph(body):
    # The autouse fixture's runtime fails if anything awaits it.
    with TestClient(main.app) as client:
        assert client.post("/chat/stream", json=body).status_code == 422


def test_provider_keys_are_redacted_from_every_log_handler(monkeypatch):
    # uvicorn's loggers have their own handlers and don't propagate to root, so those need the filter too.
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    uvicorn_logger = logging.getLogger("uvicorn.error")
    uvicorn_logger.addHandler(handler)
    try:
        with TestClient(main.app):
            uvicorn_logger.error("provider rejected key test-google-key")
    finally:
        uvicorn_logger.removeHandler(handler)
    assert "test-google-key" not in stream.getvalue() and "<redacted>" in stream.getvalue()


def test_static_files_are_revalidated_so_a_new_deploy_serves_new_js():
    with TestClient(main.app) as client:
        first = client.get("/app.js")
        again = client.get("/app.js", headers={"If-None-Match": first.headers["etag"]})
    assert first.headers["cache-control"] == "no-cache" and again.status_code == 304
