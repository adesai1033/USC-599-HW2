"""Phase 3: MCP servers (SQL guard, corpus validation) and the MCP client (failures, filtering, redaction).

Client tests spawn real stdio servers: papers_db over a temporary database, and
tests/fixtures/fake_server.py standing in for corpus and Tavily. No network access is needed.
"""

import asyncio
import io
import os
import json
import logging
import sqlite3
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from mcp.types import CallToolResult, TextContent
from langchain_chroma import Chroma
from mcp.server.fastmcp.exceptions import ToolError

import mcp_client
from agents.registry import AGENTS
from mcp_servers import corpus, papers_db
from rag import retriever

ROOT = Path(__file__).resolve().parent.parent
FAKE_SERVER = str(ROOT / "tests" / "fixtures" / "fake_server.py")
SECRET = "tvly-test-SECRET-0123456789"


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def papers_sqlite(tmp_path, monkeypatch):
    """A small papers.sqlite built with the real ingest schema, with results rows for ranking tests."""
    sys.path.insert(0, str(ROOT / "ingest"))
    from build_db import SCHEMA

    path = tmp_path / "papers.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?)", [
        ("2003.08934", "NeRF", 2020, "ECCV", 5324, "cs.CV", "https://arxiv.org/abs/2003.08934"),
        ("2201.05989", "Instant-NGP", 2022, "TOG", 6490, "cs.CV", "https://arxiv.org/abs/2201.05989"),
        ("2308.04079", "3DGS", 2023, "TOG", 11022, "cs.GR", "https://arxiv.org/abs/2308.04079"),
    ])
    conn.executemany("INSERT INTO authors (name) VALUES (?)", [("Ben Mildenhall",), ("Pratul Srinivasan",)])
    conn.executemany("INSERT INTO paper_authors VALUES (?, ?)", [("2003.08934", 2), ("2003.08934", 1)])
    conn.executemany("INSERT INTO results VALUES (?, ?, NULL, ?, ?, ?, NULL, 0)", [
        ("2003.08934", "Mip-NeRF 360", "PSNR", 23.85, 1),
        ("2201.05989", "Mip-NeRF 360", "PSNR", 25.59, 1),
        ("2308.04079", "Mip-NeRF 360", "PSNR", 27.21, 1),
        ("2003.08934", "Mip-NeRF 360", "LPIPS", 0.451, 0),
        ("2201.05989", "Mip-NeRF 360", "LPIPS", 0.331, 0),
        ("2308.04079", "Mip-NeRF 360", "LPIPS", 0.214, 0),
    ])
    conn.commit()
    conn.close()
    monkeypatch.setenv("PAPERS_DB_PATH", str(path))
    return path


@pytest.fixture
def corpus_index(tmp_path, monkeypatch):
    """A tiny Chroma index with deterministic fake embeddings, and the corpus server pointed at it."""
    embeddings = DeterministicFakeEmbedding(size=16)
    chunks = [
        ("2308.04079#abstract-00", "2308.04079", "3DGS", "abstract", 1, "Gaussians are splatted with a tile-based rasterizer."),
        ("2308.04079#method-00", "2308.04079", "3DGS", "method", 3, "Adaptive density control clones and splits Gaussians."),
        ("2003.08934#abstract-00", "2003.08934", "NeRF", "abstract", 1, "An MLP maps 5D coordinates to density and color."),
    ]
    store = Chroma(collection_name="papers", embedding_function=embeddings, persist_directory=str(tmp_path / "chroma"),
                   collection_configuration={"hnsw": {"space": "cosine"}})
    store.add_texts(
        ids=[c[0] for c in chunks],
        texts=[c[5] for c in chunks],
        metadatas=[{"chunk_id": c[0], "arxiv_id": c[1], "title": c[2], "year": 2023, "section": c[3], "page": c[4],
                    "url": f"https://arxiv.org/abs/{c[1]}"} for c in chunks],
    )
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(corpus, "OpenAIEmbeddings", lambda **_: embeddings)
    corpus.corpus.cache_clear()
    yield
    corpus.corpus.cache_clear()


def stdio_manifest(script: str, tools: list[str], description: str, env: list[str] = ()) -> dict:
    return {"transport": "stdio", "command": "python", "args": [script], "env_passthrough": list(env),
            "tools": tools, "description": description}


def papers_db_manifest() -> dict:
    return stdio_manifest("mcp_servers/papers_db.py", ["describe_schema", "run_sql", "get_paper_metadata"],
                          "papers_db is unavailable.", env=["PAPERS_DB_PATH"])


def run_sql(sql: str) -> str:
    return asyncio.run(papers_db.run_sql(sql))


async def call(tool, **args):
    msg = await tool.ainvoke({"type": "tool_call", "id": "t1", "name": tool.name, "args": args})
    text = msg.content if isinstance(msg.content, str) else "".join(b.get("text", "") for b in msg.content)
    return msg.status, text


# ---------------------------------------------------------------- papers_db: SQL guard and ranking

@pytest.mark.parametrize("sql", [
    "DROP TABLE papers",
    "INSERT INTO papers VALUES ('x', 't', 2020, NULL, 0, 'cs.CV', 'u')",
    "UPDATE papers SET year = 1999",
    "DELETE FROM papers",
    "REPLACE INTO authors (name) VALUES ('x')",
    "WITH t AS (SELECT 1) DELETE FROM papers",
    "PRAGMA table_info(papers)",
    "ATTACH DATABASE '/tmp/x.db' AS x",
    "SELECT 1; SELECT 2",
    "SELECT 1; DROP TABLE papers",
    "SELECT * FROM papers WHERE 1 = 1; PRAGMA writable_schema = 1",
    "",
])
def test_run_sql_rejects_non_select(papers_sqlite, sql):
    with pytest.raises(ToolError):
        run_sql(sql)
    with sqlite3.connect(papers_sqlite) as conn:
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 3


def test_run_sql_allows_keywords_inside_strings_and_functions(papers_sqlite):
    out = json.loads(run_sql("SELECT replace(title, 'NeRF', 'DROP TABLE') AS t FROM papers WHERE arxiv_id = '2003.08934';"))
    assert out["rows"] == [["DROP TABLE"]]


def test_run_sql_rejects_extension_loading(papers_sqlite):
    # Passes the text check (a plain SELECT), but SQLite refuses load_extension on this connection.
    with pytest.raises(ToolError):
        run_sql("SELECT load_extension('x')")


def test_run_sql_caps_rows(papers_sqlite):
    out = json.loads(run_sql("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 500) SELECT i FROM n"))
    assert out["row_count"] == papers_db.MAX_ROWS and out["truncated"] is True


def test_run_sql_aborts_long_queries(papers_sqlite, monkeypatch):
    monkeypatch.setattr(papers_db, "QUERY_TIME_LIMIT_S", 0.2)
    with pytest.raises(ToolError, match="stopped"):
        run_sql("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n")


def test_run_sql_returns_json_safe_values(papers_sqlite):
    out = json.loads(run_sql("SELECT randomblob(4), 1e999, printf('%.*c', 5000, 'x')"))
    blob, inf, text = out["rows"][0]
    assert blob == "<blob, 4 bytes>" and inf == "inf"
    assert text.startswith("x" * papers_db.MAX_CELL_CHARS) and "more chars" in text


def test_run_sql_bounds_memory_and_output_size(papers_sqlite):
    with pytest.raises(ToolError, match="too big"):
        run_sql("SELECT printf('%.*c', 400000000, 'x')")  # 400 MB string, refused by SQLite
    out = run_sql("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 200) "
                  "SELECT printf('%.*c', 900, 'x'), printf('%.*c', 900, 'y') FROM n")
    assert len(out) < papers_db.MAX_RESPONSE_CHARS + 1000
    assert json.loads(out)["truncated"] is True


@pytest.mark.asyncio
async def test_slow_query_does_not_block_other_calls(papers_sqlite, monkeypatch):
    monkeypatch.setattr(papers_db, "QUERY_TIME_LIMIT_S", 1.0)
    slow = asyncio.create_task(papers_db.run_sql("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n"))
    await asyncio.sleep(0.1)
    fast = await asyncio.wait_for(papers_db.run_sql("SELECT COUNT(*) FROM papers"), 0.5)
    assert json.loads(fast)["rows"] == [[3]] and not slow.done()
    with pytest.raises(ToolError, match="stopped"):
        await slow


@pytest.mark.parametrize("metric, expected_order", [
    ("PSNR", ["2308.04079", "2201.05989", "2003.08934"]),   # higher is better: descending
    ("LPIPS", ["2308.04079", "2201.05989", "2003.08934"]),  # lower is better: ascending
])
def test_metric_ranking_respects_higher_is_better(papers_sqlite, metric, expected_order):
    sql = ("SELECT arxiv_id FROM results WHERE benchmark = 'Mip-NeRF 360' AND metric = '%s' AND scene IS NULL "
           "ORDER BY CASE WHEN higher_is_better = 1 THEN -value ELSE value END" % metric)
    assert [r[0] for r in json.loads(run_sql(sql))["rows"]] == expected_order


def test_describe_schema_lists_tables_and_ranking_notes(papers_sqlite):
    out = json.loads(papers_db.describe_schema())
    assert {t["name"] for t in out["tables"]} == {"papers", "authors", "paper_authors", "results"}
    assert any("higher_is_better" in n for n in out["notes"])
    assert out["results_metrics"] == ["LPIPS", "PSNR"]


def test_get_paper_metadata_resolves_ids_and_reports_missing(papers_sqlite):
    out = json.loads(papers_db.get_paper_metadata(["2003.08934v2", "9999.99999"]))
    assert [p["arxiv_id"] for p in out["papers"]] == ["2003.08934"]
    assert out["papers"][0]["authors"] == ["Pratul Srinivasan", "Ben Mildenhall"]  # byline order, not id order
    assert out["missing"] == ["9999.99999"]
    assert json.loads(papers_db.get_paper_metadata(["ARXIV:2003.08934"]))["papers"][0]["arxiv_id"] == "2003.08934"
    with pytest.raises(ToolError):
        papers_db.get_paper_metadata(["not-an-id"])


# ---------------------------------------------------------------- corpus: validation and metadata

@pytest.mark.asyncio
@pytest.mark.parametrize("k", [0, 13, -1, True])
async def test_search_corpus_bounds_k(corpus_index, k):
    with pytest.raises(ToolError, match="k must be"):
        await corpus.search_corpus("gaussians", k=k)


@pytest.mark.asyncio
async def test_search_corpus_returns_chunk_metadata(corpus_index):
    out = json.loads(await corpus.search_corpus("tile-based rasterizer", k=2))
    assert len(out["results"]) == 2
    for hit in out["results"]:
        assert set(hit) == {"chunk_id", "arxiv_id", "title", "section", "page", "url", "score", "text"}


@pytest.mark.asyncio
async def test_search_corpus_filters_and_validates(corpus_index):
    out = json.loads(await corpus.search_corpus("density", k=12, arxiv_ids=["arXiv:2308.04079v1"], sections=["method"]))
    assert [h["chunk_id"] for h in out["results"]] == ["2308.04079#method-00"]
    with pytest.raises(ToolError, match="Not in the corpus"):
        await corpus.search_corpus("x", arxiv_ids=["9999.99999"])
    with pytest.raises(ToolError, match="Malformed"):
        await corpus.search_corpus("x", arxiv_ids=["nerf"])
    with pytest.raises(ToolError, match="Unknown sections"):
        await corpus.search_corpus("x", sections=["appendix-z"])
    with pytest.raises(ToolError, match="empty"):
        await corpus.search_corpus("   ")


@pytest.mark.asyncio
async def test_get_chunks_returns_text_and_missing(corpus_index):
    out = json.loads(await corpus.get_chunks(["2003.08934#abstract-00", "2003.08934#method-07"]))
    assert [c["chunk_id"] for c in out["chunks"]] == ["2003.08934#abstract-00"]
    assert out["missing"] == ["2003.08934#method-07"]
    with pytest.raises(ToolError, match="Malformed"):
        await corpus.get_chunks(["2003.08934"])


# ---------------------------------------------------------------- client: filtering and failure modes

@pytest.mark.asyncio
async def test_each_agent_receives_exactly_its_allowlisted_tools(papers_sqlite):
    fake_tools = ["search_corpus", "get_chunks", "tavily_search", "tavily_extract"]
    tools = mcp_client.MCPTools({
        "fake": stdio_manifest(FAKE_SERVER, fake_tools, "fake is unavailable."),
        "papers_db": papers_db_manifest(),
    })
    await tools.start()
    try:
        for agent, spec in AGENTS.items():
            assert {t.name for t in tools.tools_for(agent)} == spec.tools
            assert tools.unavailable_notes(agent) == []
        assert "echo" not in tools._tools  # discovered but not in the manifest's tools list
    finally:
        await tools.aclose()


@pytest.mark.asyncio
async def test_server_that_wont_start_does_not_stop_the_others(papers_sqlite):
    tools = mcp_client.MCPTools({
        "corpus": {**stdio_manifest("x.py", ["search_corpus", "get_chunks"], "Corpus search is unavailable."),
                   "command": "definitely-not-a-real-command"},
        "papers_db": papers_db_manifest(),
    })
    await tools.start()
    try:
        assert tools.status["corpus"].state == "down" and tools.status["papers_db"].state == "up"
        assert [t.name for t in tools.tools_for("research")] == ["get_paper_metadata"]
        assert tools.unavailable_notes("research") == ["Corpus search is unavailable."]
        assert tools.unavailable_notes("data") == []
        status, text = await call(tools.tools_for("data")[2], sql="SELECT COUNT(*) FROM papers")  # run_sql
        assert status == "success" and json.loads(text)["rows"] == [[3]]
    finally:
        await tools.aclose()


@pytest.mark.asyncio
async def test_missing_env_var_in_manifest_marks_server_down(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tools = mcp_client.MCPTools({"tavily": {"transport": "streamable_http", "url": "https://x/?tavilyApiKey=${TAVILY_API_KEY}",
                                            "tools": ["tavily_search"], "description": "Web search is unavailable."}})
    await tools.start()
    assert tools.status["tavily"].state == "down"
    assert "TAVILY_API_KEY is not set" in tools.status["tavily"].error
    assert tools.unavailable_notes("research") == ["Web search is unavailable."]
    await tools.aclose()


@pytest.mark.asyncio
async def test_tool_error_is_returned_to_the_model_and_server_stays_up(papers_sqlite):
    tools = mcp_client.MCPTools({"papers_db": papers_db_manifest()})
    await tools.start()
    try:
        status, text = await call(tools._tools["run_sql"], sql="DROP TABLE papers")
        assert status == "error" and "read-only" in text
        status, text = await call(tools._tools["get_paper_metadata"], arxiv_ids=["9999.99999"])
        assert status == "success" and json.loads(text)["missing"] == ["9999.99999"]
        assert tools.status["papers_db"].state == "up"
    finally:
        await tools.aclose()


@pytest.mark.asyncio
async def test_server_crash_mid_call_is_graceful_and_restarts(monkeypatch):
    tools = mcp_client.MCPTools({"fake": stdio_manifest(FAKE_SERVER, ["echo", "crash"], "Fake server is unavailable.")})
    await tools.start()
    try:
        assert await call(tools._tools["echo"], text="hi") == ("success", "hi")
        status, text = await call(tools._tools["crash"])
        assert status == "error" and text == "Fake server is unavailable."
        assert tools.status["fake"].state == "down"
        # Within the cooldown the server is not restarted, and calls fail gracefully.
        assert await call(tools._tools["echo"], text="hi") == ("error", "Fake server is unavailable.")
        # After the cooldown, the next call restarts it and the same tool object works again.
        monkeypatch.setattr(mcp_client, "RESTART_COOLDOWN_S", 0.0)
        assert await call(tools._tools["echo"], text="back") == ("success", "back")
        assert tools.status["fake"].state == "up"
    finally:
        await tools.aclose()


@pytest.mark.asyncio
async def test_concurrent_calls_after_a_crash_wait_for_one_restart(monkeypatch):
    monkeypatch.setattr(mcp_client, "RESTART_COOLDOWN_S", 0.0)
    tools = mcp_client.MCPTools({"fake": stdio_manifest(FAKE_SERVER, ["echo", "crash"], "Fake server is unavailable.")})
    await tools.start()
    try:
        await call(tools._tools["crash"])
        results = await asyncio.gather(*(call(tools._tools["echo"], text=str(i)) for i in range(5)))
        assert results == [("success", str(i)) for i in range(5)]
        assert len(tools._tasks) == 1
    finally:
        await tools.aclose()


@pytest.mark.asyncio
async def test_refresh_retries_a_server_that_never_started(monkeypatch):
    monkeypatch.setattr(mcp_client, "RESTART_COOLDOWN_S", 0.0)
    manifest = {**stdio_manifest(FAKE_SERVER, ["echo"], "Fake server is unavailable."), "command": "definitely-not-a-real-command"}
    tools = mcp_client.MCPTools({"fake": manifest})
    await tools.start()
    try:
        assert tools.status["fake"].state == "down" and "echo" not in tools._tools
        manifest["command"] = "python"  # the failure was transient
        tools.refresh()
        await asyncio.gather(*tools._background)
        assert tools.status["fake"].state == "up"
        assert await call(tools._tools["echo"], text="hi") == ("success", "hi")
    finally:
        await tools.aclose()


def test_oversized_tool_output_is_truncated():
    result = mcp_client._cap(CallToolResult(content=[TextContent(type="text", text="x" * (mcp_client.MAX_RESULT_CHARS + 50))]))
    assert len(result.content[0].text) < mcp_client.MAX_RESULT_CHARS + 100
    assert result.content[0].text.endswith("[truncated 50 chars]")


# ---------------------------------------------------------------- retriever

def test_orphaned_index_copies_are_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever.tempfile, "gettempdir", lambda: str(tmp_path))
    dead = tmp_path / "chroma-999999999-abc"   # no such process
    alive = tmp_path / f"chroma-{os.getppid()}-abc"
    for d in (dead, alive):
        d.mkdir()
    retriever._remove_orphaned_copies()
    assert not dead.exists() and alive.exists()


# ---------------------------------------------------------------- redaction

def test_redact_removes_key_from_urls_and_bare_text(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", SECRET)
    text = f"GET https://mcp.tavily.com/mcp/?tavilyApiKey={SECRET}&x=1 failed; key was {SECRET}"
    clean = mcp_client.redact(text)
    assert SECRET not in clean and "tavilyApiKey=<redacted>&x=1" in clean


def test_redact_covers_every_provider_key(monkeypatch):
    for i, name in enumerate(mcp_client.SECRET_ENV_VARS):
        monkeypatch.setenv(name, f"secret-value-{i}-0123456789")
    text = " ".join(f"secret-value-{i}-0123456789" for i in range(len(mcp_client.SECRET_ENV_VARS)))
    assert "secret-value" not in mcp_client.redact(text)


@pytest.mark.asyncio
async def test_tavily_key_never_appears_in_log_output(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", SECRET)
    monkeypatch.setattr(mcp_client, "START_TIMEOUT_S", 10.0)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        mcp_client.install_log_redaction()
        # Nothing listens on port 9, so the connection fails and the URL ends up in errors and logs.
        tools = mcp_client.MCPTools({"tavily": {
            "transport": "streamable_http", "url": "http://127.0.0.1:9/mcp/?tavilyApiKey=${TAVILY_API_KEY}",
            "tools": ["tavily_search"], "description": "Web search is unavailable."}})
        await tools.start()
        await tools.aclose()
        try:
            raise RuntimeError(f"boom at https://mcp.tavily.com/mcp/?tavilyApiKey={SECRET}")
        except RuntimeError:
            logging.getLogger("test").exception("request to %s failed", f"https://x/?tavilyApiKey={SECRET}")
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    output = stream.getvalue()
    assert tools.status["tavily"].state == "down"
    assert "tavilyApiKey=<redacted>" in output
    assert SECRET not in output and SECRET not in (tools.status["tavily"].error or "")


# ---------------------------------------------------------------- retriever: citation prior

class FakeStore:
    """Returns fixed (document, cosine distance) hits, as Chroma's similarity search does."""

    def __init__(self, hits):
        self.hits = hits
        self.asked_k = None

    async def asimilarity_search_with_score(self, query, k, filter=None):
        self.asked_k = k
        return self.hits[:k]


def hit(arxiv_id: str, similarity: float):
    from langchain_core.documents import Document

    meta = {"chunk_id": f"{arxiv_id}#method-00", "arxiv_id": arxiv_id, "title": "t", "section": "method", "page": 1,
            "url": f"https://arxiv.org/abs/{arxiv_id}"}
    return Document(page_content="x", metadata=meta), 1 - similarity


@pytest.mark.asyncio
async def test_citation_prior_lets_the_original_paper_outrank_close_follow_ups():
    store = FakeStore([hit("2404.00001", 0.78), hit("2404.00002", 0.77), hit("2308.04079", 0.73)])
    counts = {"2404.00001": 30, "2404.00002": 12, "2308.04079": 11000}
    plain = await retriever.build_retriever(store, counts, 0.0).ainvoke(retriever.SearchRequest("q", k=2))
    assert [r["arxiv_id"] for r in plain] == ["2404.00001", "2404.00002"] and store.asked_k == 2
    boosted = await retriever.build_retriever(store, counts, 0.02).ainvoke(retriever.SearchRequest("q", k=2))
    assert [r["arxiv_id"] for r in boosted] == ["2308.04079", "2404.00001"] and store.asked_k == 40
    assert boosted[0]["score"] == 0.73  # the reported score stays the plain similarity
