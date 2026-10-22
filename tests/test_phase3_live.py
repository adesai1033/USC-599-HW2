"""Phase 3 "done when" check against the real servers, index and Tavily.

  RUN_LIVE=1 pytest tests/test_phase3_live.py

Skipped unless RUN_LIVE=1. Needs index/, OPENAI_API_KEY and TAVILY_API_KEY. Costs one query
embedding (a fraction of a cent) and about two Tavily credits.
"""

import json
import os

import pytest

import mcp_client

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="live check; set RUN_LIVE=1"),
]

CALLS = {
    "search_corpus": {"query": "How does 3D Gaussian Splatting render a novel view?", "k": 3},
    "get_chunks": {"chunk_ids": ["2308.04079#abstract-00"]},
    "describe_schema": {},
    "run_sql": {"sql": "SELECT COUNT(*) FROM papers"},
    "get_paper_metadata": {"arxiv_ids": ["2308.04079"]},
    "tavily_search": {"query": "3D Gaussian Splatting", "max_results": 1},
    "tavily_extract": {"urls": ["https://arxiv.org/abs/2308.04079"]},
}


async def call(tool, args):
    msg = await tool.ainvoke({"type": "tool_call", "id": "live", "name": tool.name, "args": args})
    return msg.status, msg.content if isinstance(msg.content, str) else "".join(b.get("text", "") for b in msg.content)


async def test_every_tool_is_listed_and_callable():
    tools = mcp_client.MCPTools()
    await tools.start()
    try:
        assert {n: s.state for n, s in tools.status.items()} == {"corpus": "up", "papers_db": "up", "tavily": "up"}
        assert set(tools._tools) == set(CALLS)
        for name, args in CALLS.items():
            status, text = await call(tools._tools[name], args)
            assert status == "success", (name, text[:300])
            assert text, name
        _, text = await call(tools._tools["search_corpus"], CALLS["search_corpus"])
        assert json.loads(text)["results"][0]["chunk_id"].count("#") == 1
    finally:
        await tools.aclose()


async def test_broken_corpus_manifest_leaves_other_servers_working():
    manifests = mcp_client.load_manifests()
    manifests["corpus"] = {**manifests["corpus"], "command": "definitely-not-a-real-command"}
    tools = mcp_client.MCPTools(manifests)
    await tools.start()
    try:
        assert tools.status["corpus"].state == "down"
        assert tools.status["papers_db"].state == "up" and tools.status["tavily"].state == "up"
        assert [t.name for t in tools.tools_for("research")] == ["get_paper_metadata", "tavily_search"]
        assert tools.unavailable_notes("research") == [manifests["corpus"]["description"]]
        status, _ = await call(tools._tools["tavily_search"], CALLS["tavily_search"])
        assert status == "success"
    finally:
        await tools.aclose()
