"""Phase 4b: the worker factory, citation parsing, Tavily normalization and per-agent tools.

A scripted tool-calling chat model and fake LangChain tools stand in for the LLM and MCP; no network.
"""

import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from mcp.types import CallToolResult, TextContent
from pydantic import Field

import mcp_client
from agents.registry import AGENTS, WORKERS
from graph.build import build_graph
from graph.workers import FIX_ATTRIBUTION, SUMMARIZE_NOW, make_worker
from rag.citations import attribution_mismatches, parse_citation_ids
from tests.test_graph_skeleton import ScriptedSupervisor, answer_llm, decision, run

RESEARCH = WORKERS["research"]
WEB_URL = "https://example.org/papers/new-3dgs"
WEB_ID = mcp_client.web_source_id(WEB_URL)


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted messages in order and records every prompt it was sent."""

    script: list[AIMessage]
    seen: list[list] = Field(default_factory=list)
    bound: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        self.bound = [t.name for t in tools]
        return self


def call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def chunk(chunk_id: str, title: str = "3D Gaussian Splatting", section: str = "method", page: int = 4) -> dict:
    arxiv_id = chunk_id.split("#")[0]
    return {"chunk_id": chunk_id, "arxiv_id": arxiv_id, "title": title, "section": section, "page": page,
            "url": f"https://arxiv.org/abs/{arxiv_id}", "score": 0.7, "text": "Gaussians are rasterized per tile."}


CHUNKS = [chunk("2308.04079#method-03"), chunk("2308.04079#method-04"), chunk("2003.08934#method-00", "NeRF", page=3)]


@tool
async def search_corpus(query: str, k: int = 6) -> str:
    """Fake corpus search."""
    return json.dumps({"query": query, "k": k, "results": CHUNKS})


@tool
async def tavily_search(query: str) -> str:
    """Fake web search, already normalized the way mcp_client returns it."""
    return mcp_client.normalize_tavily("tavily_search", json.dumps(
        {"query": query, "results": [{"url": WEB_URL, "title": "New 3DGS paper", "content": "..."}]}))


@tool
async def get_paper_metadata(arxiv_ids: list[str]) -> str:
    """Fake metadata lookup that always fails."""
    raise ValueError("database is locked")


def worker_input(instruction: str = "explain 3DGS rasterization") -> dict:
    return {"messages": [HumanMessage("How does 3DGS render?")], "instruction": instruction, "task_type": "lookup", "step": 1}


# ---------------------------------------------------------------- the worker loop

@pytest.mark.asyncio
async def test_worker_loops_tool_call_result_model_and_returns_one_finding():
    model = ScriptedChatModel(script=[call("search_corpus", {"query": "3DGS rasterizer"}),
                                      AIMessage("Tiles are rasterized [2308.04079#method-03].")])
    out = await make_worker(RESEARCH, [search_corpus], model)(worker_input())
    [finding] = out["findings"]
    assert finding.agent == "research" and finding.content == "Tiles are rasterized [2308.04079#method-03]."
    assert len(model.seen) == 2 and isinstance(model.seen[1][-1], ToolMessage)  # the model saw the result
    first_prompt = model.seen[0][-1].text
    assert "explain 3DGS rasterization" in first_prompt and "How does 3DGS render?" in first_prompt and "lookup" in first_prompt


@pytest.mark.asyncio
async def test_retrieved_chunk_ids_include_uncited_chunks():
    model = ScriptedChatModel(script=[call("search_corpus", {"query": "x"}), AIMessage("Only one [2308.04079#method-03].")])
    out = await make_worker(RESEARCH, [search_corpus], model)(worker_input())
    assert out["retrieved_chunk_ids"] == [c["chunk_id"] for c in CHUNKS]


@pytest.mark.asyncio
async def test_citations_carry_metadata_from_tool_results():
    model = ScriptedChatModel(script=[
        call("search_corpus", {"query": "x"}, "c1"),
        call("tavily_search", {"query": "new 3dgs"}, "c2"),
        AIMessage(f"**Corpus** NeRF uses an MLP [2003.08934#method-00]. **Web** A new paper [{WEB_ID}]. "
                  "Also [2308.04079#intro-09]."),
    ])
    out = await make_worker(RESEARCH, [search_corpus, tavily_search], model)(worker_input())
    citations = {c.id: c for c in out["findings"][0].citations}
    nerf = citations["2003.08934#method-00"]
    assert (nerf.source, nerf.title, nerf.section, nerf.page, nerf.url) == (
        "corpus", "NeRF", "method", 3, "https://arxiv.org/abs/2003.08934")
    web = citations[WEB_ID]
    assert (web.source, web.title, web.url) == ("web", "New 3DGS paper", WEB_URL)
    # Cited but never retrieved: dropped from the citations and marked in the text (see test_robustness).
    assert "2308.04079#intro-09" not in citations and "Also [unverified]." in out["findings"][0].content


@pytest.mark.asyncio
async def test_tool_round_cap_ends_with_a_summarizing_call(monkeypatch):
    monkeypatch.setenv("WORKER_MAX_TOOL_ROUNDS", "2")
    model = ScriptedChatModel(script=[call("search_corpus", {"query": "a"}, "c1"), call("search_corpus", {"query": "b"}, "c2"),
                                      AIMessage("Summary [2308.04079#method-03].")])
    supervisor = ScriptedSupervisor([decision("research", "explain 3DGS", "lookup"), decision("synthesize")])
    graph = build_graph(workers={"research": make_worker(RESEARCH, [search_corpus], model)},
                        supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph)
    assert len(model.seen) == 3 and model.seen[2][-1].text == SUMMARIZE_NOW
    assert any(t["kind"] == "decision" and t["agent"] == "research" and "cap (2)" in t["detail"] for t in traces)
    assert final["findings"][0].content == "Summary [2308.04079#method-03]."


@pytest.mark.asyncio
async def test_a_raising_tool_becomes_an_error_message_and_the_worker_finishes():
    model = ScriptedChatModel(script=[call("get_paper_metadata", {"arxiv_ids": ["2308.04079"]}),
                                      AIMessage("Metadata lookup failed; not in the corpus.")])
    out = await make_worker(RESEARCH, [get_paper_metadata], model)(worker_input())
    error = model.seen[1][-1]
    assert isinstance(error, ToolMessage) and error.status == "error" and "database is locked" in error.text
    assert out["findings"][0].content == "Metadata lookup failed; not in the corpus."


# ---------------------------------------------------------------- per-agent tools

def manifests() -> dict:
    return {
        "corpus": {"transport": "stdio", "command": "python", "tools": ["search_corpus", "get_chunks"], "description": "Corpus down."},
        "papers_db": {"transport": "stdio", "command": "python", "tools": ["describe_schema", "run_sql", "get_paper_metadata"],
                      "description": "DB down."},
        "tavily": {"transport": "streamable_http", "url": "x", "tools": ["tavily_search", "tavily_extract"], "description": "Web search down."},
    }


def fake_tool(name: str):
    @tool(name)
    def _t(x: str) -> str:
        """Fake."""
        return x

    return _t


def test_worker_gets_only_its_allowlisted_tools():
    tools = mcp_client.MCPTools(manifests())
    every = ["search_corpus", "get_chunks", "describe_schema", "run_sql", "get_paper_metadata", "tavily_search",
             "tavily_extract", "echo"]
    tools._tools = {n: fake_tool(n) for n in every}
    for status in tools.status.values():
        status.state = "up"
    assert [t.name for t in tools.tools_for("research")] == sorted(AGENTS["research"].tools)
    assert tools.unavailable_notes("research") == []


@pytest.mark.asyncio
async def test_missing_allowlisted_tool_adds_the_server_description_to_the_prompt(caplog):
    tools = mcp_client.MCPTools(manifests())
    tools._tools = {n: fake_tool(n) for n in ["search_corpus", "get_chunks", "get_paper_metadata"]}
    tools.status["tavily"].state = "down"
    for name in ("corpus", "papers_db"):
        tools.status[name].state = "up"
    assert [t.name for t in tools.tools_for("research")] == ["get_paper_metadata", "search_corpus"]
    assert "missing allowlisted tools ['tavily_search']" in caplog.text
    notes = tools.unavailable_notes("research")
    assert notes == ["Web search down."]

    model = ScriptedChatModel(script=[AIMessage("Not in the corpus.")])
    await make_worker(RESEARCH, tools.tools_for("research"), model, unavailable_notes=notes)(worker_input())
    assert "Web search down." in model.seen[0][0].text  # the system prompt
    assert model.bound == ["get_paper_metadata", "search_corpus"]


# ---------------------------------------------------------------- Tavily normalization

def test_same_url_gives_the_same_source_id():
    variants = ["https://Example.org/papers/new-3dgs", "https://example.org/papers/new-3dgs/",
                "https://example.org/papers/new-3dgs#results", "https://example.org/papers/new-3dgs?utm_source=x"]
    assert {mcp_client.web_source_id(u) for u in variants} == {WEB_ID}
    assert mcp_client.web_source_id("https://example.org/papers/other") != WEB_ID
    out = json.loads(mcp_client.normalize_tavily("tavily_search", json.dumps(
        {"query": "q", "results": [{"url": WEB_URL, "title": "T", "content": "c", "raw_content": None, "score": 0.9}]})))
    assert out["results"] == [{"source_id": WEB_ID, "url": WEB_URL, "title": "T", "content": "c"}]


def test_tavily_error_payload_is_passed_to_the_model_not_called_malformed():
    text = json.dumps({"error": "Search failed", "status": 400,
                       "detail": {"error": "Query cannot consist only of site: operators. Please provide search terms."}})
    result = mcp_client._normalize_tavily_result("tavily_search", CallToolResult(content=[TextContent(type="text", text=text)]))
    assert result.isError
    assert result.content[0].text == ("tavily_search failed: Query cannot consist only of site: operators. "
                                      "Please provide search terms.")


def test_every_arxiv_page_of_a_paper_gets_one_source_id():
    pages = ["https://arxiv.org/abs/2610.04351", "https://arxiv.org/html/2610.04351v1", "http://www.arxiv.org/pdf/2610.04351v2.pdf",
             "https://export.arxiv.org/abs/2610.04351v3/", "https://ar5iv.labs.arxiv.org/html/2610.04351v1",
             "https://www.alphaxiv.org/abs/2610.04351", "https://www.alphaxiv.org/overview/2610.04351v2",
             "https://huggingface.co/papers/2610.04351", "https://www.semanticscholar.org/arxiv/2610.04351",
             "https://www.emergentmind.com/papers/2610.04351"]
    assert {mcp_client.normalize_url(u) for u in pages} == {"https://arxiv.org/abs/2610.04351"}
    assert mcp_client.web_source_id("https://arxiv.org/abs/2610.04352") != mcp_client.web_source_id(pages[0])
    assert mcp_client.normalize_url("https://arxiv.org/list/cs.CV/recent") == "https://arxiv.org/list/cs.CV/recent"
    assert mcp_client.normalize_url("https://huggingface.co/papers") == "https://huggingface.co/papers"  # not a paper page
    out = json.loads(mcp_client.normalize_tavily("tavily_search", json.dumps({"results": [
        {"url": pages[0], "title": "LoCoSplat", "content": "abs"},
        {"url": pages[1], "title": "LoCoSplat (html)", "content": "html"},
    ]})))
    assert [(r["source_id"], r["content"]) for r in out["results"]] == [(mcp_client.web_source_id(pages[0]), "abs")]


@pytest.mark.parametrize("text", ["<html>rate limited</html>", "[]", '{"results": "nope"}', '{"results": [{"title": "no url"}]}'])
def test_malformed_tavily_payload_becomes_a_tool_error(text):
    with pytest.raises(ValueError):
        mcp_client.normalize_tavily("tavily_search", text)
    result = mcp_client._normalize_tavily_result("tavily_search", CallToolResult(content=[TextContent(type="text", text=text)]))
    assert result.isError and "malformed response" in result.content[0].text


# ---------------------------------------------------------------- citation parsing

def test_citation_parsing_handles_both_formats_adjacency_and_look_alikes():
    text = (
        "Tiles [2308.04079#method-03][web:1a2b3c4d] and [2003.08934#method-00, 2308.04079#method-03; web:deadbeef]. "
        "Look-alikes: [2308.04079] [web:1] [web:XYZ12345] [Figure 3] 2308.04079#method-05 [2308.04079#method] "
        "[hierarchical: 2406.12080#hierarchical-3d-gaussian-representation-00] [2406.12080#hierarchical-3d-gaussian-representation-01]"
    )
    assert parse_citation_ids(text) == [
        "2308.04079#method-03", "web:1a2b3c4d", "2003.08934#method-00", "web:deadbeef",
        "2406.12080#hierarchical-3d-gaussian-representation-01",
    ]


# ---------------------------------------------------------------- graph level

@pytest.mark.asyncio
async def test_graph_runs_research_with_tool_traces_tagged_research():
    model = ScriptedChatModel(script=[call("search_corpus", {"query": "3DGS rasterizer"}),
                                      AIMessage("Tiles [2308.04079#method-03].")])
    supervisor = ScriptedSupervisor([decision("research", "explain 3DGS rasterization", "lookup"), decision("synthesize")])
    graph = build_graph(workers={"research": make_worker(RESEARCH, [search_corpus], model)},
                        supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph, "How does 3D Gaussian Splatting render a novel view?")
    assert [t["node"] for t in traces if t["kind"] == "enter"] == ["supervisor", "research", "supervisor", "synthesize"]
    tool_events = [t for t in traces if t["kind"] in ("tool_call", "tool_result")]
    assert [t["kind"] for t in tool_events] == ["tool_call", "tool_result"]
    assert all(t["agent"] == "research" and t["step"] == 1 for t in tool_events)
    assert tool_events[0]["detail"].startswith("search_corpus(")
    assert tool_events[1]["detail"].startswith("search_corpus: 3 chunks: 2308.04079#method-03")
    assert tool_events[1]["ms"] is not None
    assert final["retrieved_chunk_ids"] == [c["chunk_id"] for c in CHUNKS]
    assert [c.id for c in final["findings"][0].citations] == ["2308.04079#method-03"]


# ---------------------------------------------------------------- attribution check

def test_attribution_mismatch_flags_a_claim_citing_another_papers_source():
    sources = {"web:772d7f0a": "https://mdpi.com/x Comparative assessment of NeRF and 3DGS on UAV imagery",
               "web:14bf7c55": "https://arxiv.org/abs/2503.16964 DroneSplat",
               "2308.04079#method-03": "2308.04079 3D Gaussian Splatting"}
    wrong = "- **DroneSplat** (arXiv:2503.16964): drone reconstruction [web:772d7f0a]."
    assert attribution_mismatches(wrong, sources) == [(wrong, ["web:772d7f0a"], ["2503.16964"])]
    assert attribution_mismatches(wrong.replace("772d7f0a", "14bf7c55"), sources) == []
    # Comparisons that also name the cited paper, claims naming no paper, and unknown IDs are not flagged.
    assert attribution_mismatches("Unlike NeRF (2003.08934), 3DGS (2308.04079) rasterizes [2308.04079#method-03].", sources) == []
    assert attribution_mismatches("Tiles are sorted [2308.04079#method-03].", sources) == []
    assert attribution_mismatches("X (2503.16964) [web:deadbeef].", sources) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fixed", [True, False])
async def test_worker_asks_once_to_fix_a_misattributed_claim(fixed):
    wrong = f"**Web** NovelSplat (arXiv:2610.99999) is new [{WEB_ID}]."
    repaired = "**Web** No paper matching the request was found." if fixed else wrong
    model = ScriptedChatModel(script=[call("tavily_search", {"query": "new 3dgs"}), AIMessage(wrong), AIMessage(repaired)])
    supervisor = ScriptedSupervisor([decision("research", "find new papers", "current_events"), decision("synthesize")])
    graph = build_graph(workers={"research": make_worker(RESEARCH, [tavily_search], model)},
                        supervisor_llm=supervisor, synthesize_llm=answer_llm())
    final, traces = await run(graph)
    assert len(model.seen) == 3 and "NovelSplat" in model.seen[2][-1].text  # the fix request names the claim
    assert model.seen[2][-1].text.startswith(FIX_ATTRIBUTION.split(":")[0])
    assert final["findings"][0].content == repaired
    checks = [t["detail"] for t in traces if t["kind"] == "error" and t["agent"] == "research"]
    assert checks[0].startswith("attribution check: 1 claim(s)")
    if fixed:
        assert len(checks) == 1 and final.get("errors", []) == []
    else:
        assert "still mismatched" in checks[1]
        assert [e.kind for e in final["errors"]] == ["attribution_mismatch"]
