"""Phase 4e: the summary worker (multi-paper evidence grouped by paper) and its hand-offs.

Fakes only: a scripted tool-calling model and fake LangChain tools shaped like the real MCP payloads.
"""

import json
import re

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

import mcp_client
from agents.registry import AGENTS, WORKERS, available_workers
from graph.build import build_graph
from graph.state import SupervisorDecision
from graph.supervisor import _roster
from graph.workers import SUMMARIZE_NOW, make_worker
from tests.test_graph_skeleton import answer_llm, run
from tests.test_workers import ScriptedChatModel, call, worker_input

SUMMARY = WORKERS["summary"]
RESEARCH = WORKERS["research"]
PAPERS = {"2103.13415": "Mip-NeRF", "2111.12077": "Mip-NeRF 360", "2304.06706": "Zip-NeRF"}
BLOG_URL = "https://example.org/blog/anti-aliasing-in-radiance-fields"


def chunk(arxiv_id: str, section: str = "abstract", n: int = 0) -> dict:
    return {"chunk_id": f"{arxiv_id}#{section}-{n:02d}", "arxiv_id": arxiv_id, "title": PAPERS.get(arxiv_id, "Paper"),
            "section": section, "page": 1, "url": f"https://arxiv.org/abs/{arxiv_id}", "score": 0.7, "text": "..."}


@tool
async def search_corpus(query: str, k: int = 6, arxiv_ids: list[str] | None = None, sections: list[str] | None = None) -> str:
    """Fake corpus search: with arxiv_ids, that paper's chunks; without, one chunk per known paper."""
    ids = arxiv_ids or list(PAPERS)
    return json.dumps({"query": query, "k": k, "results": [chunk(i, (sections or ["method"])[0]) for i in ids]})


@tool
async def get_chunks(chunk_ids: list[str]) -> str:
    """Fake chunk lookup."""
    found = [chunk(i.split("#")[0], i.split("#")[1].rsplit("-", 1)[0], int(i.rsplit("-", 1)[1])) for i in chunk_ids]
    return json.dumps({"chunks": found, "missing": []})


@tool
async def get_paper_metadata(arxiv_ids: list[str]) -> str:
    """Fake metadata."""
    return json.dumps({"papers": [{"arxiv_id": i, "title": PAPERS.get(i), "year": 2021} for i in arxiv_ids], "missing": []})


@tool
async def tavily_extract(urls: list[str]) -> str:
    """Fake extract, normalized the way mcp_client returns it."""
    return mcp_client.normalize_tavily("tavily_extract", json.dumps(
        {"results": [{"url": u, "raw_content": "A survey of anti-aliasing in radiance fields."} for u in urls]}))


@tool
async def tavily_search(query: str) -> str:
    """Fake web search that finds one blog post."""
    return mcp_client.normalize_tavily("tavily_search", json.dumps(
        {"query": query, "results": [{"url": BLOG_URL, "title": "Anti-aliasing in radiance fields", "content": "..."}]}))


SUMMARY_TOOLS = [search_corpus, get_chunks, get_paper_metadata, tavily_extract]


def parallel(*calls: tuple[str, dict]) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": n, "args": a, "id": f"p{i}"} for i, (n, a) in enumerate(calls)])


@pytest.mark.asyncio
async def test_summary_gathers_evidence_per_paper_and_cites_every_paper():
    model = ScriptedChatModel(script=[
        call("search_corpus", {"query": "anti-aliasing in radiance fields"}, "c1"),
        parallel(*(("search_corpus", {"query": "how it handles aliasing", "arxiv_ids": [i]}) for i in PAPERS)),
        call("get_chunks", {"chunk_ids": ["2304.06706#method-01"]}, "c3"),
        AIMessage("**Mip-NeRF (2021)**: cones instead of rays [2103.13415#method-00].\n"
                  "**Mip-NeRF 360 (2022)**: unbounded scenes [2111.12077#method-00].\n"
                  "**Zip-NeRF (2023)**: grid anti-aliasing [2304.06706#method-00][2304.06706#method-01].\n"
                  "**Across papers**: Zip-NeRF builds on both [2304.06706#method-01]; no disagreement found."),
    ])
    out = await make_worker(SUMMARY, SUMMARY_TOOLS, model)(worker_input("review anti-aliasing in radiance fields"))
    finding = out["findings"][0]
    assert {c.arxiv_id for c in finding.citations} == set(PAPERS)
    assert finding.grounded is True and out.get("errors") is None
    assert "2304.06706#method-01" in out["retrieved_chunk_ids"]  # from get_chunks, cited and allowed


@pytest.mark.asyncio
async def test_summary_can_cite_a_web_source_it_extracted():
    model = ScriptedChatModel(script=[
        call("tavily_extract", {"urls": [BLOG_URL]}),
        AIMessage(f"A blog surveys the area [{mcp_client.web_source_id(BLOG_URL)}]."),
    ])
    out = await make_worker(SUMMARY, SUMMARY_TOOLS, model)(worker_input())
    [cite] = out["findings"][0].citations
    assert (cite.source, cite.url) == ("web", BLOG_URL)


@pytest.mark.asyncio
async def test_summary_tool_rounds_are_capped_at_four(monkeypatch):
    monkeypatch.setenv("WORKER_MAX_TOOL_ROUNDS", "9")  # the per-agent value wins over the global setting
    model = ScriptedChatModel(script=[call("search_corpus", {"query": f"q{i}"}, f"c{i}") for i in range(4)] + [AIMessage("done")])
    await make_worker(SUMMARY, SUMMARY_TOOLS, model)(worker_input())
    assert len(model.seen) == 5 and model.seen[-1][-1].text == SUMMARIZE_NOW
    budgets = [m[0].text.rsplit("## Tool budget", 1)[1] for m in model.seen]  # the system prompt's last section
    assert "used: 0 of 4 (4 left)" in budgets[0] and "used: 3 of 4 (1 left)" in budgets[3]
    assert "This is your last round" in budgets[3] and "used all 4 rounds" in budgets[4]


def test_summary_is_registered_with_its_allowlist():
    assert available_workers() == ["research", "data", "summary"]
    assert SUMMARY.tools == frozenset({"search_corpus", "get_chunks", "get_paper_metadata", "tavily_extract"})
    assert SUMMARY.max_tool_rounds == 4 and AGENTS["summary"].prompt_file.read_text().startswith("You are the summary worker")
    assert "(unavailable right now)" not in _roster(available_workers())


class WebHandoffSupervisor:
    """research first; then summary, with the web URL it reads from its own view of the findings."""

    def __init__(self):
        self.summary_instruction = None

    def with_structured_output(self, schema):
        async def decide(messages):
            context = messages[-1].text
            if "Dispatches so far:\n(none)" in context:
                return SupervisorDecision(next="research", instruction="Find a web overview of anti-aliasing in NeRFs.",
                                          task_type="lit_review", reason="find sources")
            if "- summary:" not in context:
                url = re.search(r"Web sources cited: web:[0-9a-f]{8} (\S+)", context).group(1)
                self.summary_instruction = f"Review anti-aliasing across Mip-NeRF, Mip-NeRF 360 and Zip-NeRF; also read {url}"
                return SupervisorDecision(next="summary", instruction=self.summary_instruction, task_type="lit_review",
                                          reason="review with the source")
            return SupervisorDecision(next="synthesize", instruction="", task_type="lit_review", reason="done")

        return RunnableLambda(decide)


@pytest.mark.asyncio
async def test_research_hands_a_web_url_to_summary_and_citations_survive_to_the_answer():
    web_id = mcp_client.web_source_id(BLOG_URL)
    research_model = ScriptedChatModel(script=[
        call("tavily_search", {"query": "anti-aliasing NeRF overview"}),
        AIMessage(f"**Web** An overview exists [{web_id}]."),
    ])
    summary_model = ScriptedChatModel(script=[
        parallel(("tavily_extract", {"urls": [BLOG_URL]}),
                 *(("search_corpus", {"query": "aliasing", "arxiv_ids": [i]}) for i in PAPERS)),
        AIMessage(f"Mip-NeRF [2103.13415#method-00]; Mip-NeRF 360 [2111.12077#method-00]; Zip-NeRF [2304.06706#method-00]. "
                  f"Overview [{web_id}]."),
    ])
    supervisor = WebHandoffSupervisor()
    answer = (f"Mip-NeRF uses cones [2103.13415#method-00], Mip-NeRF 360 extends it [2111.12077#method-00], and "
              f"Zip-NeRF adds grids [2304.06706#method-00]; an overview agrees [{web_id}].")
    graph = build_graph(
        workers={"research": make_worker(RESEARCH, [tavily_search], research_model),
                 "summary": make_worker(SUMMARY, SUMMARY_TOOLS, summary_model)},
        supervisor_llm=supervisor, synthesize_llm=answer_llm(answer),
    )
    final, traces = await run(graph, "Review how anti-aliasing has been addressed in radiance fields")
    assert [t["node"] for t in traces if t["kind"] == "enter"] == [
        "supervisor", "research", "supervisor", "summary", "supervisor", "synthesize"]
    assert BLOG_URL in supervisor.summary_instruction and BLOG_URL in summary_model.seen[0][-1].text
    assert any(t["agent"] == "summary" and t["detail"].startswith("tavily_extract(") for t in traces if t["kind"] == "tool_call")
    report = final["answer_citations"]
    assert report.invalid == [] and {c.arxiv_id for c in report.valid if c.source == "corpus"} == set(PAPERS)
    assert web_id in [c.id for c in report.valid]
