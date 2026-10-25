"""LLM extraction of benchmark results -> results table (verified=0).

  python ingest/extract_results.py --dry-run               # pages and estimated cost; no API calls
  python ingest/extract_results.py                         # every paper in data/raw/
  python ingest/extract_results.py --papers 2308.04079     # just these papers

Run order after a rebuild: build_db.py (creates an empty results table) -> extract_results.py ->
seed_results.py (hand-checked rows, verified=1).

Table bodies are not in data/parsed/ (parse_pdfs.py drops numeric blocks as equation fragments), so
this reads the raw PDFs. A page is sent when its text has a table caption and a known metric name. The
fast-tier model gets the rendered page (which keeps the table layout) plus its text (which keeps the
digits exact), and returns the paper's own results with prompts/extract_results.md.

Benchmark and metric names are normalized here, and higher_is_better comes from
seed_results.KNOWN_METRICS, never from the model. Each row then goes through seed_results.validate_row.
For each paper, its verified=0 rows are replaced and verified=1 rows are never touched, so a re-run is
safe. The model's reply for each page is cached in data/extracted_raw/, so re-running after a change
to the normalization costs nothing (--refresh asks the model again). Every written row is also listed, with its page and table, in data/extracted_results.csv for
hand-checking.
"""

import argparse
import asyncio
import base64
import csv
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

import pymupdf
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm import get_llm  # noqa: E402
from seed_results import KNOWN_METRICS, validate_row  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
SHEET_PATH = ROOT / "data" / "extracted_results.csv"
CACHE_DIR = ROOT / "data" / "extracted_raw"  # the model's reply per page, so a re-run doesn't pay again
PROMPT_PATH = ROOT / "prompts" / "extract_results.md"
SHEET_COLUMNS = ["arxiv_id", "benchmark", "scene", "metric", "value", "higher_is_better", "setting", "page", "table"]

MAX_PAGES_PER_PAPER = 6
DPI = 150
CONCURRENCY = 8
# gemini-3.8-flash paid tier, ai.google.dev/gemini-api/docs/pricing, checked 2026-10-24 (valid to 2026-12-31).
PRICE_PER_M_INPUT = 0.75
PRICE_PER_M_OUTPUT = 3.75
# For --dry-run only: a rendered page plus the prompt, and a typical structured reply.
EST_IMAGE_TOKENS = 1100
EST_PROMPT_TOKENS = 900
EST_OUTPUT_TOKENS = 850  # the first full run averaged 823 per page

TABLE_CAPTION = re.compile(r"\bTab(?:le|\.)\s*\d", re.IGNORECASE)
METRIC_WORD = re.compile(r"PSNR|SSIM|LPIPS|Chamfer|F-score|FPS", re.IGNORECASE)

# Benchmark spellings (lowercased, letters and digits only) -> canonical name.
BENCHMARK_ALIASES = {
    "mipnerf360": "Mip-NeRF 360", "mip360": "Mip-NeRF 360", "360": "Mip-NeRF 360",
    "mipnerf360dataset": "Mip-NeRF 360",
    "tanksandtemples": "Tanks and Temples", "tankstemples": "Tanks and Temples", "tt": "Tanks and Temples",
    "tandt": "Tanks and Temples", "tnt": "Tanks and Temples",
    "deepblending": "Deep Blending", "db": "Deep Blending",
    "llff": "LLFF", "nerfllff": "LLFF", "realforwardfacing": "LLFF", "forwardfacing": "LLFF",
    "dtu": "DTU",
    "blender": "Blender", "nerfsynthetic": "Blender", "syntheticnerf": "Blender", "blendersynthetic": "Blender",
    "nerfsyntheticblender": "Blender", "syntheticblender": "Blender", "synthetic": "Blender",
    "scannet": "ScanNet", "scannetv2": "ScanNet", "scannetpp": "ScanNet++",
    "replica": "Replica",
}
# Metric spellings (lowercased, letters and digits only) -> a KNOWN_METRICS name.
METRIC_ALIASES = {
    "psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS",
    "chamfer": "Chamfer", "chamferdistance": "Chamfer", "cd": "Chamfer", "chamferl1": "Chamfer",
    "fscore": "F-score", "f1": "F-score",
    "accuracy": "Accuracy", "acc": "Accuracy", "completeness": "Completeness", "comp": "Completeness",
    "trainingtime": "Training time", "traintime": "Training time",
    "fps": "FPS", "memory": "Memory", "mem": "Memory",
}

logger = logging.getLogger("extract_results")


class ExtractedResult(BaseModel):
    table: str
    benchmark: str
    scene: str | None = None
    metric: str
    value: float
    setting: str | None = None


class PageResults(BaseModel):
    results: list[ExtractedResult]


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", text.lower()).replace("+", "pp")


def normalize_benchmark(name: str) -> str:
    """The canonical benchmark name, or the name as written when it isn't a known alias."""
    return BENCHMARK_ALIASES.get(_key(name), name.strip())


def normalize_metric(name: str) -> str | None:
    return METRIC_ALIASES.get(_key(name))


def to_row(arxiv_id: str, result: ExtractedResult, known_ids: set[str]) -> dict | None:
    """A validated results row (seed_results.validate_row) or None. higher_is_better is the metric's."""
    metric = normalize_metric(result.metric)
    if metric is None:
        return None
    scene = (result.scene or "").strip()
    if _key(scene) in ("avg", "average", "mean", "all", ""):
        scene = ""
    row, _ = validate_row({
        "arxiv_id": arxiv_id, "benchmark": normalize_benchmark(result.benchmark), "scene": scene,
        "metric": metric, "value": str(round(result.value, 4)),  # unit conversions leave long fractions
        "higher_is_better": str(KNOWN_METRICS[metric]), "setting": result.setting or "",
    }, known_ids)
    return row


def table_pages(pdf_path: Path) -> list[int]:
    """0-based pages with a table caption and a metric name, at most MAX_PAGES_PER_PAPER."""
    with pymupdf.open(pdf_path) as doc:
        pages = [i for i, page in enumerate(doc)
                 if TABLE_CAPTION.search(text := page.get_text()) and METRIC_WORD.search(text)]
    return pages[:MAX_PAGES_PER_PAPER]


def page_message(pdf_path: Path, page_no: int, prompt: str) -> HumanMessage:
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no]
        png = page.get_pixmap(dpi=DPI).tobytes("png")
        text = page.get_text()
    return HumanMessage(content=[
        {"type": "text", "text": f"{prompt}\n\n## Page {page_no + 1}, extracted text\n\n{text}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode()}"}},
    ])


async def extract_page(model, message: HumanMessage, label: str, usage: dict) -> list[ExtractedResult] | None:
    """One structured call, retried once; a page that fails twice is skipped and logged (None)."""
    structured = model.with_structured_output(PageResults, include_raw=True)
    for attempt in (1, 2):
        try:
            reply = await structured.ainvoke([message])
        except Exception as exc:
            logger.warning("%s: attempt %d failed: %s: %s", label, attempt, type(exc).__name__, exc)
            continue
        meta = getattr(reply["raw"], "usage_metadata", None) or {}
        usage["input"] += meta.get("input_tokens", 0)
        usage["output"] += meta.get("output_tokens", 0)
        if reply["parsed"] is not None:
            return reply["parsed"].results
        logger.warning("%s: attempt %d returned unparseable output", label, attempt)
    usage["skipped_pages"] += 1
    return None


def write_paper(conn: sqlite3.Connection, arxiv_id: str, rows: list[dict]) -> int:
    """Replace this paper's verified=0 rows; verified=1 rows (and their keys) are kept as they are."""
    key = ("arxiv_id = :arxiv_id AND benchmark = :benchmark AND scene IS :scene AND metric = :metric "
           "AND setting IS :setting")
    written = 0
    with conn:
        conn.execute("DELETE FROM results WHERE arxiv_id = ? AND verified = 0", (arxiv_id,))
        for row in rows:
            if conn.execute(f"SELECT 1 FROM results WHERE {key}", row).fetchone():
                continue  # a hand-checked row already holds this key
            conn.execute(
                "INSERT INTO results (arxiv_id, benchmark, scene, metric, value, higher_is_better, setting, verified) "
                "VALUES (:arxiv_id, :benchmark, :scene, :metric, :value, :higher_is_better, :setting, 0)", row)
            written += 1
    return written


async def extract_paper(model, arxiv_id: str, title: str, known_ids: set[str], gate: asyncio.Semaphore,
                        usage: dict, refresh: bool = False) -> list[dict]:
    """Every distinct validated row for one paper, each with its page and table for the review sheet."""
    pdf_path = RAW_DIR / f"{arxiv_id}.pdf"
    prompt = PROMPT_PATH.read_text().replace("{title}", title).replace("{arxiv_id}", arxiv_id)

    async def one(page_no: int) -> list[tuple[int, ExtractedResult]]:
        cached = CACHE_DIR / f"{arxiv_id}-p{page_no + 1:02d}.json"
        if cached.exists() and not refresh:
            found = PageResults.model_validate_json(cached.read_text()).results
        else:
            async with gate:
                found = await extract_page(model, page_message(pdf_path, page_no, prompt),
                                           f"{arxiv_id} p.{page_no + 1}", usage)
            if found is None:
                return []
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached.write_text(PageResults(results=found).model_dump_json())
        return [(page_no, r) for r in found]

    seen, rows = set(), []
    for page_results in await asyncio.gather(*(one(p) for p in table_pages(pdf_path))):
        for page_no, result in page_results:
            row = to_row(arxiv_id, result, known_ids)
            if row is None:
                usage["dropped"] += 1
                continue
            natural_key = (row["benchmark"], row["scene"], row["metric"], row["setting"])
            if natural_key in seen:
                continue
            seen.add(natural_key)
            rows.append(row | {"page": page_no + 1, "table": result.table})
    return drop_unlabeled_repeats(rows)


def drop_unlabeled_repeats(rows: list[dict]) -> list[dict]:
    """Drop a row without a setting when another table repeats its number with one.

    Papers repeat a headline number in a later table (often the appendix) that names its setting, e.g.
    31.28 in "Table 1" and again as "G.T. Intrinsic" in "Table 11". Rows of the same table are kept:
    there, an empty setting is the paper's main method and a labeled one is a variant.
    """
    labeled = {(r["benchmark"], r["scene"], r["metric"], r["value"], r["table"]) for r in rows if r["setting"]}
    return [r for r in rows if r["setting"] or not any(
        key[:4] == (r["benchmark"], r["scene"], r["metric"], r["value"]) and key[4] != r["table"] for key in labeled)]


def update_sheet(rows_by_paper: dict[str, list[dict]]) -> None:
    """Rewrite the review sheet: rows of the papers processed now replace their earlier rows."""
    kept = []
    if SHEET_PATH.exists():
        with SHEET_PATH.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["arxiv_id"] not in rows_by_paper]
    with SHEET_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
        for rows in rows_by_paper.values():
            writer.writerows({**r, "scene": r["scene"] or "", "setting": r["setting"] or ""} for r in rows)


async def run(papers: list[tuple[str, str]], db_path: Path, refresh: bool) -> dict:
    model = get_llm("fast", thinking_level="low")
    gate = asyncio.Semaphore(CONCURRENCY)
    usage = {"input": 0, "output": 0, "skipped_pages": 0, "dropped": 0, "written": 0}
    conn = sqlite3.connect(db_path)
    known_ids = {r[0] for r in conn.execute("SELECT arxiv_id FROM papers")}
    rows_by_paper: dict[str, list[dict]] = {}
    try:
        tasks = {asyncio.create_task(extract_paper(model, a, t, known_ids, gate, usage, refresh)): a for a, t in papers}
        for done, task in enumerate(asyncio.as_completed(tasks), 1):
            rows = await task
            arxiv_id = rows[0]["arxiv_id"] if rows else None
            if arxiv_id:
                rows_by_paper[arxiv_id] = rows
                usage["written"] += write_paper(conn, arxiv_id, rows)
            logger.info("%d/%d papers done (%d rows so far)", done, len(papers), usage["written"])
        # Papers that yielded nothing still lose their old unverified rows.
        for arxiv_id, _ in papers:
            if arxiv_id not in rows_by_paper:
                write_paper(conn, arxiv_id, [])
                rows_by_paper[arxiv_id] = []
    finally:
        conn.close()
    update_sheet(rows_by_paper)
    return usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="count pages and estimate cost, without API calls")
    parser.add_argument("--papers", nargs="+", metavar="ARXIV_ID", help="only these papers")
    parser.add_argument("--refresh", action="store_true", help="ignore replies cached in data/extracted_raw/")
    parser.add_argument("--db", type=Path, default=ROOT / os.environ.get("PAPERS_DB_PATH", "index/papers.sqlite"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    conn = sqlite3.connect(args.db.resolve())
    papers = [(a, t) for a, t in conn.execute("SELECT arxiv_id, title FROM papers ORDER BY arxiv_id")
              if (RAW_DIR / f"{a}.pdf").exists() and (not args.papers or a in args.papers)]
    conn.close()

    pages = {a: table_pages(RAW_DIR / f"{a}.pdf") for a, _ in papers}
    papers = [(a, t) for a, t in papers if pages[a]]
    to_ask = {a: [p for p in page_list if args.refresh or not (CACHE_DIR / f"{a}-p{p + 1:02d}.json").exists()]
              for a, page_list in pages.items()}
    n_pages = sum(len(p) for p in to_ask.values())
    text_tokens = 0
    for arxiv_id, page_list in to_ask.items():
        with pymupdf.open(RAW_DIR / f"{arxiv_id}.pdf") as doc:
            text_tokens += sum(len(doc[p].get_text()) // 4 for p in page_list)
    est_in = text_tokens + n_pages * (EST_IMAGE_TOKENS + EST_PROMPT_TOKENS)
    est_out = n_pages * EST_OUTPUT_TOKENS
    logger.info("%d papers with result tables, %d pages (%d not cached); est. %d input + %d output tokens, "
                "about $%.2f", len(papers), sum(len(p) for p in pages.values()), n_pages, est_in, est_out,
                est_in / 1e6 * PRICE_PER_M_INPUT + est_out / 1e6 * PRICE_PER_M_OUTPUT)
    if args.dry_run:
        return

    usage = asyncio.run(run(papers, args.db.resolve(), args.refresh))
    cost = usage["input"] / 1e6 * PRICE_PER_M_INPUT + usage["output"] / 1e6 * PRICE_PER_M_OUTPUT
    logger.info("wrote %d unverified rows (%d model rows dropped by validation, %d pages skipped); "
                "%d input + %d output tokens, about $%.2f. Review sheet: %s", usage["written"], usage["dropped"],
                usage["skipped_pages"], usage["input"], usage["output"], cost, SHEET_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
