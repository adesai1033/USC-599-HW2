"""Phase 4d: the data worker, [db:...] citations, hand-offs to research, and the results seed script.

Fakes only: a scripted tool-calling model and fake LangChain tools shaped like the real papers_db
MCP payloads. The seed script runs against a temporary SQLite database.
"""

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from agents.registry import AGENTS, WORKERS
from graph.build import build_graph
from graph.state import Citation, SupervisorDecision
from graph.workers import SUMMARIZE_NOW, make_worker
from rag.citations import mark_unverified, parse_citation_ids, validate
from tests.test_graph_skeleton import answer_llm, run
from tests.test_workers import ScriptedChatModel, call, worker_input

ROOT = Path(__file__).resolve().parent.parent
DATA = WORKERS["data"]
RESEARCH = WORKERS["research"]

TOP_CITED = "SELECT arxiv_id, title, citation_count FROM papers ORDER BY citation_count DESC LIMIT 2"
PER_YEAR = "SELECT year, COUNT(*) FROM papers GROUP BY year ORDER BY year LIMIT 50"
NOTHING = "SELECT arxiv_id FROM papers WHERE year = 1999 LIMIT 5"
ROWS = {
    TOP_CITED: (["arxiv_id", "title", "citation_count"],
                [["2308.04079", "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 11022],
                 ["2201.05989", "Instant Neural Graphics Primitives", 6490]]),
    PER_YEAR: (["year", "COUNT(*)"], [[2023, 43], [2024, 61]]),
    NOTHING: (["arxiv_id"], []),
}


@tool
async def describe_schema() -> str:
    """Fake schema."""
    return json.dumps({"tables": [{"name": "papers", "row_count": 220, "columns": []}], "notes": []})


@tool
async def run_sql(sql: str) -> str:
    """Fake run_sql: known queries return rows; anything else is a SQL error, as the real tool reports it."""
    if sql not in ROWS:
        raise ValueError("SQL error: no such column: citations. Check table and column names with describe_schema.")
    columns, rows = ROWS[sql]
    return json.dumps({"sql": sql, "columns": columns, "rows": rows, "row_count": len(rows), "truncated": False})


@tool
async def get_paper_metadata(arxiv_ids: list[str]) -> str:
    """Fake metadata."""
    return json.dumps({"papers": [{"arxiv_id": i, "title": f"Paper {i}", "authors": []} for i in arxiv_ids], "missing": []})


DATA_TOOLS = [describe_schema, run_sql, get_paper_metadata]


@pytest.mark.asyncio
async def test_data_worker_loop_captures_queries_and_builds_db_citations():
    model = ScriptedChatModel(script=[
        call("describe_schema", {}, "c1"),
        call("run_sql", {"sql": TOP_CITED}, "c2"),
        AIMessage(f"3DGS has 11022 citations [db:2308.04079].\n\n```sql\n{TOP_CITED}\n```"),
    ])
    out = await make_worker(DATA, DATA_TOOLS, model)(worker_input("most cited papers"))
    [finding] = out["findings"]
    assert finding.queries == [TOP_CITED] and finding.grounded is True
    [cite] = finding.citations
    assert (cite.id, cite.source, cite.arxiv_id, cite.url) == (
        "db:2308.04079", "db", "2308.04079", "https://arxiv.org/abs/2308.04079")
    assert cite.title == "3D Gaussian Splatting for Real-Time Radiance Field Rendering"  # from the row's title column
    assert out.get("errors") is None and out["retrieved_chunk_ids"] == []


@pytest.mark.asyncio
async def test_db_citation_not_in_this_runs_rows_is_unverified():
    model = ScriptedChatModel(script=[
        call("run_sql", {"sql": TOP_CITED}),
        AIMessage("3DGS leads [db:2308.04079]; NeRF has 5324 citations [db:2003.08934]."),
        AIMessage("3DGS leads [db:2308.04079]; NeRF has 5324 citations [db:2003.08934]."),  # the fix request, ignored
    ])
    out = await make_worker(DATA, DATA_TOOLS, model)(worker_input())
    finding = out["findings"][0]
    assert "[db:2003.08934]" not in finding.content and "[unverified]" in finding.content
    assert [c.id for c in finding.citations] == ["db:2308.04079"]
    assert [(e.kind, "db:2003.08934" in e.message) for e in out["errors"]] == [("invalid_citation", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("sql, grounded", [(PER_YEAR, True), (NOTHING, False)])
async def test_aggregate_rows_ground_a_finding_without_citations(sql, grounded):
    model = ScriptedChatModel(script=[call("run_sql", {"sql": sql}), AIMessage(f"Counts by year: 2023 43, 2024 61.\n\n```sql\n{sql}\n```")])
    out = await make_worker(DATA, DATA_TOOLS, model)(worker_input())
    finding = out["findings"][0]
    assert finding.citations == [] and finding.queries == [sql] and finding.grounded is grounded


@pytest.mark.asyncio
async def test_sql_error_is_retried_with_a_different_query():
    bad = "SELECT arxiv_id, citations FROM papers ORDER BY citations DESC LIMIT 2"
    model = ScriptedChatModel(script=[
        call("run_sql", {"sql": bad}, "c1"),
        call("run_sql", {"sql": TOP_CITED}, "c2"),
        AIMessage("3DGS leads [db:2308.04079]."),
    ])
    out = await make_worker(DATA, DATA_TOOLS, model)(worker_input())
    error = model.seen[1][-2]  # then the round budget
    assert error.status == "error" and "no such column" in error.text  # the model saw the error and fixed the query
    assert out["findings"][0].queries == [TOP_CITED]  # only SQL that ran successfully
    assert out["findings"][0].grounded is True


@pytest.mark.asyncio
@pytest.mark.parametrize("spec, tools, rounds", [(DATA, DATA_TOOLS, 5), (RESEARCH, [run_sql], 4)])
async def test_max_tool_rounds_is_per_agent(spec, tools, rounds, monkeypatch):
    monkeypatch.delenv("WORKER_MAX_TOOL_ROUNDS", raising=False)
    model = ScriptedChatModel(script=[call("run_sql", {"sql": PER_YEAR}, f"c{i}") for i in range(rounds)] + [AIMessage("done")])
    out = await make_worker(spec, tools, model)(worker_input())
    assert len(model.seen) == rounds + 1 and model.seen[-1][-1].text == SUMMARIZE_NOW
    assert out["findings"][0].content == "done"
    assert AGENTS["data"].max_tool_rounds == 5 and AGENTS["research"].max_tool_rounds is None


def test_parser_and_validator_handle_all_three_citation_types():
    text = ("3DGS [2308.04079#method-03][db:2308.04079] has a web write-up [web:1a2b3c4d]; "
            "grouped [db:2201.05989, 2003.08934#method-00]. Look-alikes: [db:2308] [db:abc] [DB:2308.04079] db:2308.04079")
    assert parse_citation_ids(text) == ["2308.04079#method-03", "db:2308.04079", "web:1a2b3c4d", "db:2201.05989",
                                        "2003.08934#method-00"]
    allowed = {i: Citation(id=i, source=s) for i, s in [("2308.04079#method-03", "corpus"), ("db:2308.04079", "db"),
                                                         ("web:1a2b3c4d", "web")]}
    report = validate(text, allowed)
    assert [c.id for c in report.valid] == ["2308.04079#method-03", "db:2308.04079", "web:1a2b3c4d"]
    assert report.invalid == ["db:2201.05989", "2003.08934#method-00"]
    assert "[unverified]" in mark_unverified(text, set(report.invalid))


# ---------------------------------------------------------------- graph-level hand-off

class HandoffSupervisor:
    """data first; then research, with the paper IDs it reads from its own view of the findings."""

    def __init__(self):
        self.research_instruction = None

    def with_structured_output(self, schema):
        async def decide(messages):
            context = messages[-1].text
            if "Dispatches so far:\n(none)" in context:
                return SupervisorDecision(next="data", instruction="Find the two most-cited papers.", task_type="metadata",
                                          reason="metadata first")
            if "- research:" not in context:
                ids = re.search(r"Papers cited: (.+)", context).group(1)
                self.research_instruction = f"Explain what each of these papers proposes: {ids}"
                return SupervisorDecision(next="research", instruction=self.research_instruction, task_type="lookup",
                                          reason="hand off the IDs")
            return SupervisorDecision(next="synthesize", instruction="", task_type="lookup", reason="done")

        return RunnableLambda(decide)


@tool
async def search_corpus(query: str, arxiv_ids: list[str] | None = None, sections: list[str] | None = None) -> str:
    """Fake corpus search that honours the arxiv_ids filter."""
    chunks = [{"chunk_id": f"{i}#abstract-00", "arxiv_id": i, "title": f"Paper {i}", "section": "abstract", "page": 1,
               "url": f"https://arxiv.org/abs/{i}", "score": 0.8, "text": "We propose..."} for i in arxiv_ids or []]
    return json.dumps({"query": query, "k": 6, "results": chunks})


@pytest.mark.asyncio
async def test_data_hands_ids_to_research_and_the_answer_cites_both():
    data_model = ScriptedChatModel(script=[
        call("describe_schema", {}, "c1"), call("run_sql", {"sql": TOP_CITED}, "c2"),
        AIMessage("Most cited: 3DGS [db:2308.04079], Instant-NGP [db:2201.05989]."),
    ])
    research_model = ScriptedChatModel(script=[
        call("search_corpus", {"query": "what each proposes", "arxiv_ids": ["2308.04079", "2201.05989"],
                               "sections": ["abstract"]}),
        AIMessage("3DGS proposes Gaussians [2308.04079#abstract-00]; Instant-NGP a hash grid [2201.05989#abstract-00]."),
    ])
    supervisor = HandoffSupervisor()
    answer = ("3DGS is the most cited [db:2308.04079] and proposes Gaussians [2308.04079#abstract-00]; "
              "Instant-NGP [db:2201.05989] proposes a hash grid [2201.05989#abstract-00].")
    graph = build_graph(
        workers={"data": make_worker(DATA, DATA_TOOLS, data_model), "research": make_worker(RESEARCH, [search_corpus], research_model)},
        supervisor_llm=supervisor, synthesize_llm=answer_llm(answer),
    )
    final, traces = await run(graph, "Most-cited papers and what each proposes")
    assert [t["node"] for t in traces if t["kind"] == "enter"] == ["load_history", "supervisor", "data", "supervisor", "research", "supervisor", "synthesize", "save_history"]
    assert supervisor.research_instruction == "Explain what each of these papers proposes: 2308.04079, 2201.05989"
    assert "2308.04079, 2201.05989" in research_model.seen[0][-1].text  # the research worker got the IDs
    report = final["answer_citations"]
    assert {c.source for c in report.valid} == {"db", "corpus"} and report.invalid == []
    assert [c.id for c in report.valid] == ["db:2308.04079", "2308.04079#abstract-00", "db:2201.05989", "2201.05989#abstract-00"]


# ---------------------------------------------------------------- results seed

@pytest.fixture
def seed_db(tmp_path):
    sys.path.insert(0, str(ROOT / "ingest"))
    from build_db import SCHEMA

    path = tmp_path / "papers.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?)", [
        ("2308.04079", "3DGS", 2023, "TOG", 11022, "cs.GR", "u"), ("2201.05989", "Instant-NGP", 2022, "TOG", 6490, "cs.CV", "u")])
    conn.commit()
    conn.close()
    return path


def write_csv(path: Path, rows: list[list]) -> Path:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arxiv_id", "benchmark", "scene", "metric", "value", "higher_is_better", "setting", "source_note"])
        writer.writerows(rows)
    return path


def test_seed_results_validates_upserts_and_is_idempotent(seed_db, tmp_path):
    sys.path.insert(0, str(ROOT / "ingest"))
    import seed_results

    csv_path = write_csv(tmp_path / "seed.csv", [
        ["2308.04079", "Mip-NeRF 360", "", "PSNR", "27.21", "1", "", "Tab. 1"],
        ["2308.04079", "Mip-NeRF 360", "", "lpips", "0.214", "0", "", "Tab. 1"],      # metric matched case-insensitively
        ["2201.05989", "Mip-NeRF 360", "garden", "PSNR", "26.0", "1", "full res", ""],
        ["9999.99999", "Mip-NeRF 360", "", "PSNR", "30", "1", "", ""],              # unknown paper
        ["2308.04079", "Tanks and Temples", "", "PSNR", "twenty", "1", "", ""],     # not a number
        ["2308.04079", "Tanks and Temples", "", "PSNR", "23.14", "yes", "", ""],    # bad higher_is_better
        ["2308.04079", "Tanks and Temples", "", "LPIPS", "0.18", "1", "", ""],      # contradicts LPIPS direction
        ["2308.04079", "Tanks and Temples", "", "FID", "10", "0", "", ""],          # unknown metric
    ])
    counts, rejections = seed_results.seed(csv_path, seed_db)
    assert counts == {"inserted": 3, "updated": 0, "unchanged": 0, "rejected": 5}
    reasons = " | ".join(rejections)
    for expected in ["line 5: arXiv ID '9999.99999' is not in papers", "line 6: value 'twenty' is not a number",
                     "line 7: higher_is_better 'yes' must be 0 or 1", "line 8: higher_is_better=1 contradicts LPIPS",
                     "line 9: metric 'FID' is not one of"]:
        assert expected in reasons
    with sqlite3.connect(seed_db) as conn:
        rows = conn.execute("SELECT arxiv_id, metric, scene, value, verified FROM results ORDER BY value").fetchall()
    assert rows == [("2308.04079", "LPIPS", None, 0.214, 1), ("2201.05989", "PSNR", "garden", 26.0, 1),
                    ("2308.04079", "PSNR", None, 27.21, 1)]

    # Running again changes nothing; a corrected value updates the row in place.
    assert seed_results.seed(csv_path, seed_db)[0] == {"inserted": 0, "updated": 0, "unchanged": 3, "rejected": 5}
    write_csv(csv_path, [["2308.04079", "Mip-NeRF 360", "", "PSNR", "27.43", "1", "", "Tab. 1, corrected"]])
    assert seed_results.seed(csv_path, seed_db)[0] == {"inserted": 0, "updated": 1, "unchanged": 0, "rejected": 0}
    with sqlite3.connect(seed_db) as conn:
        assert conn.execute("SELECT COUNT(*), MAX(value) FROM results WHERE metric = 'PSNR' AND scene IS NULL").fetchone() == (1, 27.43)


def test_shipped_seed_csv_is_header_only():
    assert (ROOT / "ingest" / "results_seed.csv").read_text().strip() == (
        "arxiv_id,benchmark,scene,metric,value,higher_is_better,setting,source_note")
