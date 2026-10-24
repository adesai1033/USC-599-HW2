"""Helpers for endpoint tests: parse an SSE body, and serve a given graph from main.app without MCP."""

import json

import main


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


class StubMCP:
    """Stands in for MCPTools at the endpoint: the graph under test carries its own (fake) tools."""

    def __init__(self) -> None:
        self.refreshes = 0
        self.closed = False

    def refresh(self) -> None:
        self.refreshes += 1

    async def aclose(self) -> None:
        self.closed = True


def serve(monkeypatch, graph=None, make=None) -> StubMCP:
    """Make main.app's lifespan start `graph` (or run `make`) instead of the real MCP servers and graph."""
    mcp = StubMCP()

    async def make_runtime():
        if make is not None:
            return await make()
        return mcp, graph

    monkeypatch.setattr(main, "make_runtime", make_runtime)
    return mcp
