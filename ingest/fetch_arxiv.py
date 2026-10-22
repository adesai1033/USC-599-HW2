"""arXiv API -> data/raw/*.pdf + metadata.

Two steps, both free (no paid APIs):

  python ingest/fetch_arxiv.py select     # query arXiv + Semantic Scholar, pick papers
  python ingest/fetch_arxiv.py download   # fetch PDFs for the selected papers

`select` writes data/candidates.json (every candidate seen, for auditing),
data/selected.json (the corpus), and data/selection_report.md (for human review).
`download` reads data/selected.json and writes data/raw/<arxiv_id>.pdf, skipping files
that already exist.
"""

import argparse
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "ingest" / "corpus_spec.yaml"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
SELECTED_PATH = DATA_DIR / "selected.json"
REPORT_PATH = DATA_DIR / "selection_report.md"

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
USER_AGENT = "csci599-paper-assistant/0.1 (course project corpus ingest)"

ARXIV_DELAY_S = 3.0  # arXiv asks for one request every 3 seconds
ARXIV_PAGE_SIZE = 100
RESULTS_PER_QUERY = 200
S2_BATCH_SIZE = 400
MAX_RETRIES = 6

ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
VERSION_SUFFIX = re.compile(r"v\d+$")

logger = logging.getLogger("fetch_arxiv")


def load_spec() -> dict:
    with SPEC_PATH.open() as f:
        return yaml.safe_load(f)


def request_with_retry(client: httpx.Client, method: str, url: str, **kwargs) -> httpx.Response:
    """Send a request, backing off on 429 / 5xx / transport errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            logger.warning("transport error (%s), attempt %d/%d", type(exc).__name__, attempt, MAX_RETRIES)
        else:
            if response.status_code == 200:
                return response
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
            logger.warning("HTTP %d from %s, attempt %d/%d", response.status_code, url, attempt, MAX_RETRIES)
        time.sleep(min(10 * attempt, 60))
    raise RuntimeError(f"giving up on {url} after {MAX_RETRIES} attempts")


def parse_entries(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", ATOM):
        raw_id = entry.find("a:id", ATOM).text.rsplit("/abs/", 1)[-1]
        papers.append(
            {
                "arxiv_id": VERSION_SUFFIX.sub("", raw_id),
                "version": raw_id,
                "title": " ".join(entry.find("a:title", ATOM).text.split()),
                "abstract": " ".join(entry.find("a:summary", ATOM).text.split()),
                "authors": [a.find("a:name", ATOM).text for a in entry.findall("a:author", ATOM)],
                "published": entry.find("a:published", ATOM).text[:10],
                "primary_category": entry.find("x:primary_category", ATOM).get("term"),
                "categories": [c.get("term") for c in entry.findall("a:category", ATOM)],
                "url": f"https://arxiv.org/abs/{VERSION_SUFFIX.sub('', raw_id)}",
            }
        )
    return papers


class ArxivClient:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client
        self.last_request = 0.0

    def _get(self, params: dict) -> list[dict]:
        wait = ARXIV_DELAY_S - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            response = request_with_retry(self.client, "GET", ARXIV_API, params=params)
        finally:
            self.last_request = time.monotonic()
        return parse_entries(response.text)

    def search(self, query: str, categories: list[str], start: date, end: date) -> list[dict]:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        window = f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
        full_query = f"({query}) AND ({cats}) AND {window}"
        results: list[dict] = []
        for offset in range(0, RESULTS_PER_QUERY, ARXIV_PAGE_SIZE):
            page = self._get(
                {
                    "search_query": full_query,
                    "start": offset,
                    "max_results": ARXIV_PAGE_SIZE,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                }
            )
            results.extend(page)
            if len(page) < ARXIV_PAGE_SIZE:
                break
        return results

    def by_ids(self, arxiv_ids: list[str]) -> list[dict]:
        return self._get({"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)})


def fetch_citations(client: httpx.Client, arxiv_ids: list[str]) -> dict[str, dict]:
    """Return {arxiv_id: {citation_count, venue}} from Semantic Scholar; missing papers are omitted."""
    out: dict[str, dict] = {}
    for i in range(0, len(arxiv_ids), S2_BATCH_SIZE):
        batch = arxiv_ids[i : i + S2_BATCH_SIZE]
        response = request_with_retry(
            client,
            "POST",
            S2_BATCH,
            params={"fields": "citationCount,venue"},
            json={"ids": [f"ARXIV:{a}" for a in batch]},
        )
        for arxiv_id, record in zip(batch, response.json()):
            if record:
                out[arxiv_id] = {
                    "citation_count": record.get("citationCount") or 0,
                    "venue": record.get("venue") or None,
                }
        time.sleep(1.0)
    return out


def exclusion_reason(paper: dict, spec: dict) -> str | None:
    """Return why a candidate fails the spec's filters, or None if it passes."""
    arxiv = spec["arxiv"]
    if paper["arxiv_id"] in (spec["exclude"].get("arxiv_ids") or []):
        return "manually excluded"
    if not set(paper["categories"]) & set(arxiv["categories"]):
        return "category"
    published = date.fromisoformat(paper["published"])
    if not arxiv["date_range"]["start"] <= published <= arxiv["date_range"]["end"]:
        return "date"
    for field, key in (("title", "title_terms"), ("abstract", "abstract_terms")):
        for term in spec["exclude"].get(key, []):
            if re.search(rf"\b{re.escape(term)}\b", paper[field], re.IGNORECASE):
                return f"{field} term '{term}'"
    return None


def select_topic(topic: str, cfg: dict, seeds: list[dict], pool: list[dict], recent_since: date) -> list[dict]:
    """Seeds first, then top-cited recent papers up to min_recent, then top-cited overall up to quota."""
    chosen = list(seeds)
    by_citations = sorted(pool, key=lambda p: p["citation_count"], reverse=True)

    def is_recent(paper: dict) -> bool:
        return date.fromisoformat(paper["published"]) >= recent_since

    for paper in by_citations:
        if sum(map(is_recent, chosen)) >= cfg["min_recent"] or len(chosen) >= cfg["quota"]:
            break
        if is_recent(paper) and paper not in chosen:
            chosen.append(paper)
    for paper in by_citations:
        if len(chosen) >= cfg["quota"]:
            break
        if paper not in chosen:
            chosen.append(paper)
    for paper in chosen:
        paper["topic"] = topic
    return chosen


def write_report(selected: list[dict], candidates: dict[str, dict], spec: dict) -> None:
    excluded = [p for p in candidates.values() if p.get("excluded")]
    lines = [
        "# Corpus selection report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from `ingest/corpus_spec.yaml`.",
        f"Selected **{len(selected)}** papers from {len(candidates)} candidates "
        f"({len(excluded)} excluded by filters).",
        "",
    ]
    for topic, cfg in spec["topics"].items():
        papers = [p for p in selected if p["topic"] == topic]
        lines += [
            f"## {topic} ({len(papers)}/{cfg['quota']})",
            "",
            "| arXiv | Year | Cites | Seed | Title |",
            "|---|---|---|---|---|",
        ]
        for p in papers:
            title = p["title"].replace("|", "\\|")
            seed = "★" if p.get("is_seed") else ""
            lines.append(
                f"| [{p['arxiv_id']}]({p['url']}) | {p['published'][:4]} | {p['citation_count']} | {seed} | {title} |"
            )
        lines.append("")
    lines += ["## Excluded by filters (first 60)", "", "| arXiv | Reason | Title |", "|---|---|---|"]
    for p in excluded[:60]:
        lines.append(f"| {p['arxiv_id']} | {p['excluded']} | {p['title'].replace('|', '\\|')} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def cmd_select() -> None:
    spec = load_spec()
    arxiv_cfg = spec["arxiv"]
    start, end = arxiv_cfg["date_range"]["start"], arxiv_cfg["date_range"]["end"]
    recent_since = spec["recent_since"]

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        arxiv = ArxivClient(client)

        seed_topics = {s["arxiv_id"]: s["topic"] for s in spec["seed_papers"]}
        seeds = {p["arxiv_id"]: p for p in arxiv.by_ids(list(seed_topics))}
        missing = set(seed_topics) - set(seeds)
        if missing:
            raise RuntimeError(f"seed IDs not found on arXiv: {sorted(missing)}")
        logger.info("fetched %d seed papers", len(seeds))

        candidates: dict[str, dict] = {}
        topic_hits: dict[str, list[str]] = {topic: [] for topic in spec["topics"]}
        for topic, cfg in spec["topics"].items():
            for query in cfg["queries"]:
                results = arxiv.search(query, arxiv_cfg["categories"], start, end)
                logger.info("%-22s %4d results  %s", topic, len(results), query)
                for paper in results:
                    candidates.setdefault(paper["arxiv_id"], paper)
                    topic_hits[topic].append(paper["arxiv_id"])

        all_ids = sorted(set(candidates) | set(seeds))
        logger.info("looking up citation counts for %d papers on Semantic Scholar", len(all_ids))
        citations = fetch_citations(client, all_ids)

    for papers in (candidates, seeds):
        for arxiv_id, paper in papers.items():
            paper.update(citations.get(arxiv_id, {"citation_count": 0, "venue": None}))
    for paper in candidates.values():
        paper["excluded"] = exclusion_reason(paper, spec)
    for arxiv_id, paper in seeds.items():
        paper["is_seed"] = True
        if reason := exclusion_reason(paper, spec):
            logger.warning("seed %s would fail filter (%s); keeping it because it is a seed", arxiv_id, reason)

    taken: set[str] = set(seeds)
    selected: list[dict] = []
    for topic, cfg in spec["topics"].items():
        topic_seeds = [seeds[a] for a, t in seed_topics.items() if t == topic]
        pool = [
            candidates[a]
            for a in dict.fromkeys(topic_hits[topic])
            if a not in taken and not candidates[a]["excluded"]
        ]
        chosen = select_topic(topic, cfg, topic_seeds, pool, recent_since)
        taken.update(p["arxiv_id"] for p in chosen)
        selected.extend(chosen)
        logger.info("%-22s selected %d/%d (pool %d)", topic, len(chosen), cfg["quota"], len(pool))

    DATA_DIR.mkdir(exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(list(candidates.values()), indent=2))
    SELECTED_PATH.write_text(json.dumps(selected, indent=2))
    write_report(selected, candidates, spec)

    count = spec["paper_count"]
    level = logging.INFO if count["min"] <= len(selected) <= count["max"] else logging.WARNING
    logger.log(level, "selected %d papers (allowed %d-%d); report: %s", len(selected), count["min"], count["max"], REPORT_PATH)


def cmd_download() -> None:
    spec = load_spec()
    max_bytes = spec["pdf"]["max_size_mb"] * 1024 * 1024
    selected = json.loads(SELECTED_PATH.read_text())
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True) as client:
        for i, paper in enumerate(selected, 1):
            target = RAW_DIR / f"{paper['arxiv_id']}.pdf"
            if target.exists():
                continue
            try:
                response = request_with_retry(client, "GET", ARXIV_PDF.format(arxiv_id=paper["version"]))
                if not response.content.startswith(b"%PDF"):
                    raise ValueError("response is not a PDF")
                if len(response.content) > max_bytes:
                    raise ValueError(f"PDF is {len(response.content) // 2**20} MB, over the limit")
                target.write_bytes(response.content)
                logger.info("[%d/%d] %s (%d KB)", i, len(selected), paper["arxiv_id"], len(response.content) // 1024)
            except (RuntimeError, ValueError, httpx.HTTPStatusError) as exc:
                failed.append(paper["arxiv_id"])
                logger.warning("[%d/%d] %s skipped: %s", i, len(selected), paper["arxiv_id"], exc)
            time.sleep(ARXIV_DELAY_S)

    have = sum((RAW_DIR / f"{p['arxiv_id']}.pdf").exists() for p in selected)
    logger.info("%d/%d PDFs present in %s; %d failed: %s", have, len(selected), RAW_DIR, len(failed), failed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["select", "download"])
    args = parser.parse_args()
    {"select": cmd_select, "download": cmd_download}[args.command]()


if __name__ == "__main__":
    main()
