"""MCP server (stdio) over index/papers.sqlite: describe_schema, run_sql, get_paper_metadata.

  python mcp_servers/papers_db.py

Reads PAPERS_DB_PATH (default index/papers.sqlite). Every query runs on a read-only connection
behind several independent guards: a single SELECT/WITH statement, a keyword check, an authorizer
that permits only reads, a row cap and a time limit.
"""

import asyncio
import json
import math
import logging
import os
import re
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

ROOT = Path(__file__).resolve().parent.parent
MAX_ROWS = 200
MAX_SQL_CHARS = 5000
MAX_VALUE_BYTES = 100_000  # SQLite refuses to build a longer string or blob, so a query can't exhaust memory
MAX_CELL_CHARS = 1000
MAX_RESPONSE_CHARS = 100_000
QUERY_TIME_LIMIT_S = 2.0
MAX_METADATA_IDS = 50
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
ARXIV_PREFIX = re.compile(r"^arxiv:", re.I)

# Text inside string literals, quoted identifiers and comments is blanked before the keyword checks.
LITERALS_AND_COMMENTS = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|\[[^\]]*\]|--[^\n]*|/\*.*?\*/", re.S)
# REPLACE is left out because replace() is an ordinary string function; the authorizer blocks REPLACE INTO.
FORBIDDEN = re.compile(r"\b(PRAGMA|ATTACH|DETACH|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|VACUUM|REINDEX|ANALYZE|UPSERT)\b", re.I)
ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

SCHEMA_NOTES = [
    "papers: one row per corpus paper. year is the first arXiv submission year. citation_count comes from "
    "Semantic Scholar (OpenAlex where Semantic Scholar reported none) and was fetched once at ingest time.",
    "authors / paper_authors: join papers.arxiv_id = paper_authors.arxiv_id and paper_authors.author_id = authors.id. "
    "Author names are matched exactly as written on arXiv.",
    "results: benchmark numbers reported by each paper. scene is NULL for a dataset average. "
    "higher_is_better is 1 for metrics like PSNR and SSIM and 0 for LPIPS, Chamfer distance and times; "
    "rank with ORDER BY value DESC when higher_is_better = 1 and ASC when it is 0. "
    "verified = 0 rows were extracted automatically and are not hand-checked; say so when you report them. "
    "Numbers from different settings (resolution, splits) are not directly comparable.",
]

logger = logging.getLogger("mcp.papers_db")
mcp = FastMCP("papers_db")


def as_json(payload: dict) -> str:
    """Compact JSON: FastMCP would indent a returned dict, which costs the model tokens."""
    return json.dumps(payload, ensure_ascii=False)


def db_path() -> Path:
    return (ROOT / os.environ.get("PAPERS_DB_PATH", "index/papers.sqlite")).resolve()


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)


def check_sql(sql: str) -> str:
    """Return the statement without a trailing semicolon, or raise ToolError explaining the rejection."""
    sql = sql.strip()
    if not sql:
        raise ToolError("sql must not be empty.")
    if len(sql) > MAX_SQL_CHARS:
        raise ToolError(f"sql is too long ({len(sql)} chars); keep it under {MAX_SQL_CHARS}.")
    sql = sql.removesuffix(";").rstrip()
    bare = LITERALS_AND_COMMENTS.sub(" ", sql)
    if ";" in bare:
        raise ToolError("Only a single SQL statement is allowed per call.")
    if not re.match(r"\s*(SELECT|WITH)\b", bare, re.I):
        raise ToolError("Only read-only SELECT or WITH queries are allowed.")
    forbidden = FORBIDDEN.search(bare)
    if forbidden:
        raise ToolError(f"{forbidden.group(1).upper()} is not allowed; only read-only SELECT or WITH queries are.")
    return sql


def _authorizer(action: int, *_args) -> int:
    return sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _cell(value: object) -> object:
    """Make one result value JSON-safe and short: blobs are described, not dumped, and long text is cut."""
    if isinstance(value, bytes):
        return f"<blob, {len(value)} bytes>"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + f"... [{len(value) - MAX_CELL_CHARS} more chars]"
    return value


@mcp.tool()
def describe_schema() -> str:
    """Describe the paper database: every table with its columns, types and row count, plus notes on
    what the columns mean and how to rank benchmark results. Call this before writing SQL for run_sql."""
    with closing(connect()) as conn:
        tables = []
        for (name, ddl) in conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"):
            columns = [{"name": c[1], "type": c[2], "nullable": not c[3], "primary_key": bool(c[5])}
                       for c in conn.execute(f"PRAGMA table_info({name})")]
            rows = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            tables.append({"name": name, "row_count": rows, "columns": columns, "ddl": ddl})
        benchmarks = [r[0] for r in conn.execute("SELECT DISTINCT benchmark FROM results ORDER BY 1")]
        metrics = [r[0] for r in conn.execute("SELECT DISTINCT metric FROM results ORDER BY 1")]
    return as_json({"tables": tables, "notes": SCHEMA_NOTES, "results_benchmarks": benchmarks, "results_metrics": metrics})


@mcp.tool()
async def run_sql(sql: str) -> str:
    """Run one read-only SQL query (SQLite dialect) against the paper database and return the rows.

    Use this for counts, rankings, filters and joins over paper metadata and benchmark results, e.g.
    the most-cited papers of a year, papers by an author, or the best PSNR on a benchmark. Call
    describe_schema first. Exactly one SELECT or WITH statement is allowed; PRAGMA, ATTACH and any
    statement that writes are rejected. At most 200 rows are returned ("truncated": true if more
    matched or the output got too large) and a query is stopped after 2 seconds. Text values
    longer than 1000 characters are cut, and blobs are described rather than returned.

    Returns {"sql", "columns", "rows", "row_count", "truncated"}. Report the exact SQL you ran.
    To look up known papers by arXiv ID, get_paper_metadata is simpler.
    """
    statement = check_sql(sql)
    # In a worker thread, so a slow query doesn't stall other calls on this server.
    return await asyncio.to_thread(_run_sql, statement)


def _run_sql(statement: str) -> str:
    deadline = time.monotonic() + QUERY_TIME_LIMIT_S
    conn = connect()
    try:
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        cursor = conn.execute(statement)
        rows = cursor.fetchmany(MAX_ROWS + 1)
        columns = [d[0] for d in cursor.description or ()]
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc):
            raise ToolError(f"Query stopped after {QUERY_TIME_LIMIT_S:.0f} s; add filters or a LIMIT.") from exc
        if "not authorized" in str(exc):
            raise ToolError("The query tried a non-read operation; only read-only SELECT queries are allowed.") from exc
        raise ToolError(f"SQL error: {exc}. Check table and column names with describe_schema.") from exc
    except sqlite3.Error as exc:
        raise ToolError(f"SQL error: {exc}.") from exc
    finally:
        conn.close()
    kept, size = [], 0
    for row in rows[:MAX_ROWS]:
        row = [_cell(v) for v in row]
        size += len(as_json({"r": row}))
        if size > MAX_RESPONSE_CHARS:
            break
        kept.append(row)
    return as_json({
        "sql": statement,
        "columns": columns,
        "rows": kept,
        "row_count": len(kept),
        "truncated": len(rows) > len(kept),
    })


@mcp.tool()
def get_paper_metadata(arxiv_ids: list[str]) -> str:
    """Look up corpus papers by arXiv ID: title, year, venue, citation_count, primary_category, url and
    authors (in byline order). Accepts up to 50 IDs like "2308.04079" (a version suffix such as v2 is ignored).

    Returns {"papers": [...], "missing": [...]}, where missing lists IDs that are not in the corpus.
    Use this to enrich citations or resolve IDs returned by run_sql; for anything that needs
    filtering, counting or ranking, use run_sql instead.
    """
    if not arxiv_ids:
        raise ToolError("arxiv_ids must contain at least one ID.")
    if len(arxiv_ids) > MAX_METADATA_IDS:
        raise ToolError(f"get_paper_metadata accepts at most {MAX_METADATA_IDS} IDs, got {len(arxiv_ids)}.")
    ids = list(dict.fromkeys(re.sub(r"v\d+$", "", ARXIV_PREFIX.sub("", i.strip())) for i in arxiv_ids))
    malformed = [i for i in ids if not ARXIV_ID.match(i)]
    if malformed:
        raise ToolError(f"Malformed arXiv IDs {malformed}; expected the form 2308.04079.")
    marks = ",".join("?" * len(ids))
    with closing(connect()) as conn:
        conn.row_factory = sqlite3.Row
        papers = {r["arxiv_id"]: {**dict(r), "authors": []}
                  for r in conn.execute(f"SELECT * FROM papers WHERE arxiv_id IN ({marks})", ids)}
        for r in conn.execute(
            f"SELECT pa.arxiv_id, a.name FROM paper_authors pa JOIN authors a ON a.id = pa.author_id "
            f"WHERE pa.arxiv_id IN ({marks}) ORDER BY pa.rowid", ids
        ):
            papers[r["arxiv_id"]]["authors"].append(r["name"])
    return as_json({"papers": [papers[i] for i in ids if i in papers], "missing": [i for i in ids if i not in papers]})


def main() -> None:
    # force=True replaces the rich handler FastMCP installs when the server object is created.
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), stream=sys.stderr, force=True,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
    if not db_path().exists():
        raise SystemExit(f"no paper database at {db_path()}; run ingest/build_db.py")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
