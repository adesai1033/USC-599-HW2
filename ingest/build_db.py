"""Metadata (arXiv + Semantic Scholar) -> index/papers.sqlite.

  python ingest/build_db.py

Reads data/selected.json (arXiv metadata plus Semantic Scholar citation counts and venues,
written by fetch_arxiv.py select) and rebuilds PAPERS_DB_PATH (default index/papers.sqlite)
from scratch. The database is written to a temporary file and moved into place at the end.

Citation fallback: Semantic Scholar sometimes reports 0 or nothing for a well-known paper (it
reports 0 for NeRF). For those papers the count comes from OpenAlex instead. OpenAlex keeps the
arXiv preprint and the published version as separate works, so the fallback searches by title
and takes the highest count among works with exactly the same title.

The results table is created empty; extract_results.py fills it.
"""

import json
import logging
import os
import re
import sqlite3
from pathlib import Path

import httpx
from dotenv import load_dotenv

from fetch_arxiv import USER_AGENT, request_with_retry

ROOT = Path(__file__).resolve().parent.parent
SELECTED_PATH = ROOT / "data" / "selected.json"
OPENALEX_WORKS = "https://api.openalex.org/works"

SCHEMA = """
CREATE TABLE papers (
    arxiv_id         TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    year             INTEGER NOT NULL,
    venue            TEXT,
    citation_count   INTEGER,
    primary_category TEXT NOT NULL,
    url              TEXT NOT NULL
);
CREATE TABLE authors (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE paper_authors (
    arxiv_id  TEXT NOT NULL REFERENCES papers(arxiv_id),
    author_id INTEGER NOT NULL REFERENCES authors(id),
    PRIMARY KEY (arxiv_id, author_id)
);
CREATE TABLE results (
    arxiv_id         TEXT NOT NULL REFERENCES papers(arxiv_id),
    benchmark        TEXT NOT NULL,
    scene            TEXT,              -- NULL for a dataset average
    metric           TEXT NOT NULL,
    value            REAL NOT NULL,
    higher_is_better INTEGER NOT NULL CHECK (higher_is_better IN (0, 1)),
    setting          TEXT,
    verified         INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1))
);
CREATE INDEX idx_papers_year ON papers(year);
CREATE INDEX idx_paper_authors_author ON paper_authors(author_id);
CREATE INDEX idx_results_lookup ON results(benchmark, metric);
"""

logger = logging.getLogger("build_db")


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def openalex_citations(client: httpx.Client, title: str) -> int | None:
    """Highest cited_by_count among OpenAlex works whose title matches exactly, or None."""
    params = {"search": title, "per-page": 10, "select": "display_name,cited_by_count"}
    works = request_with_retry(client, "GET", OPENALEX_WORKS, params=params).json()["results"]
    counts = [w["cited_by_count"] for w in works
              if normalize_title(w["display_name"] or "") == normalize_title(title)]
    return max(counts) if counts else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env")
    db_path = (ROOT / os.environ.get("PAPERS_DB_PATH", "index/papers.sqlite")).resolve()
    papers = json.loads(SELECTED_PATH.read_text())

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        for paper in papers:
            if paper.get("citation_count"):
                continue
            fallback = openalex_citations(client, paper["title"])
            logger.info("%s: Semantic Scholar reports %r citations; OpenAlex fallback gives %r",
                        paper["arxiv_id"], paper.get("citation_count"), fallback)
            if fallback is not None:
                paper["citation_count"] = fallback

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".tmp")
    tmp_path.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp_path)
    conn.executescript(SCHEMA)
    author_ids: dict[str, int] = {}
    for paper in papers:
        conn.execute(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?)",
            (paper["arxiv_id"], paper["title"], int(paper["published"][:4]), paper.get("venue") or None,
             paper.get("citation_count"), paper["primary_category"], paper["url"]),
        )
        for name in dict.fromkeys(paper["authors"]):  # de-duplicate, keep order
            if name not in author_ids:
                author_ids[name] = conn.execute("INSERT INTO authors (name) VALUES (?)", (name,)).lastrowid
            conn.execute("INSERT INTO paper_authors VALUES (?, ?)", (paper["arxiv_id"], author_ids[name]))
    conn.commit()
    conn.close()
    os.replace(tmp_path, db_path)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("papers", "authors", "paper_authors", "results")}
    logger.info("wrote %s: %s", db_path, counts)
    for row in conn.execute("SELECT arxiv_id, citation_count, title FROM papers ORDER BY citation_count DESC LIMIT 5"):
        logger.info("top cited: %s %6d  %s", row[0], row[1], row[2][:60])
    conn.close()


if __name__ == "__main__":
    main()
