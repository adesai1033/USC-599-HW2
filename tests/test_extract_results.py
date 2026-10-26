"""Phase 8: ingest/extract_results.py with a fake model and a generated PDF (no API calls)."""

import asyncio
import sqlite3
import sys
from pathlib import Path

import pymupdf
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ingest"))

import extract_results  # noqa: E402
from build_db import SCHEMA  # noqa: E402
from extract_results import ExtractedResult, PageResults  # noqa: E402

KNOWN = {"2308.04079"}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "papers.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO papers VALUES ('2308.04079', '3DGS', 2023, 'TOG', 11022, 'cs.GR', 'u')")
    conn.commit()
    yield conn
    conn.close()


def result(**fields) -> ExtractedResult:
    return ExtractedResult(**({"table": "Tab. 1", "benchmark": "Mip-NeRF 360", "metric": "PSNR", "value": 27.21} | fields))


def test_names_are_normalized_and_direction_comes_from_the_metric():
    row = extract_results.to_row("2308.04079", result(benchmark="T&T", metric="lpips", value=0.183, scene="Avg."), KNOWN)
    assert row == {"arxiv_id": "2308.04079", "benchmark": "Tanks and Temples", "scene": None, "metric": "LPIPS",
                   "value": 0.183, "higher_is_better": 0, "setting": None}
    assert extract_results.to_row("2308.04079", result(benchmark="MipNeRF-360", metric="Chamfer distance"), KNOWN)["benchmark"] == "Mip-NeRF 360"
    assert extract_results.to_row("2308.04079", result(benchmark="My New Set"), KNOWN)["benchmark"] == "My New Set"
    assert extract_results.to_row("2308.04079", result(value=41.5500001, metric="Training time"), KNOWN)["value"] == 41.55


def test_unknown_metrics_and_papers_are_dropped():
    assert extract_results.to_row("2308.04079", result(metric="FID"), KNOWN) is None
    assert extract_results.to_row("9999.99999", result(), KNOWN) is None


def test_a_number_repeated_with_its_setting_in_another_table_is_kept_once():
    def row(value, setting, table):
        return {"benchmark": "Tanks and Temples", "scene": None, "metric": "PSNR", "value": value, "setting": setting, "table": table}

    rows = [row(31.28, None, "Table 1"), row(31.28, "G.T. Intrinsic", "Table 11"), row(30.9, "Heuristic", "Table 11"),
            row(0.237, None, "Table 2"), row(0.237, "w/GLO", "Table 2")]    # same table: main method and a variant
    kept = extract_results.drop_unlabeled_repeats(rows)
    assert [(r["value"], r["setting"]) for r in kept] == [(31.28, "G.T. Intrinsic"), (30.9, "Heuristic"),
                                                          (0.237, None), (0.237, "w/GLO")]


def test_tanks_and_temples_in_an_nsvf_table_is_the_masked_object_subset():
    def row(benchmark, table):
        return {"benchmark": benchmark, "page": 13, "table": table}

    rows = [row("Synthetic-NSVF", "Table 1"), row("Tanks and Temples", "Table 1"), row("Tanks and Temples", "Table 2")]
    assert [r["benchmark"] for r in extract_results.relabel_nsvf_tanks(rows)] == [
        "Synthetic-NSVF", "Tanks and Temples (NSVF masked objects)", "Tanks and Temples"]
    assert extract_results.normalize_benchmark("Syn.-NSVF") == "Synthetic-NSVF"


def insert(conn, value, verified, metric="PSNR"):
    conn.execute("INSERT INTO results VALUES ('2308.04079', 'Mip-NeRF 360', NULL, ?, ?, ?, NULL, ?)",
                 (metric, value, int(metric != "LPIPS"), verified))


def test_rewrite_replaces_unverified_rows_and_keeps_verified_ones(db):
    insert(db, 27.21, 1)                    # hand-checked
    insert(db, 0.5, 0, metric="LPIPS")      # stale extraction
    rows = [extract_results.to_row("2308.04079", r, KNOWN)
            for r in (result(value=99.0), result(metric="SSIM", value=0.815))]
    assert extract_results.write_paper(db, "2308.04079", rows) == 1
    got = db.execute("SELECT metric, value, verified FROM results ORDER BY metric").fetchall()
    assert got == [("PSNR", 27.21, 1), ("SSIM", 0.815, 0)]
    extract_results.write_paper(db, "2308.04079", rows)  # idempotent
    assert db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 2


class FakeModel:
    """Answers each page with `replies` in order; None stands for a failed call."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    def with_structured_output(self, schema, include_raw=False):
        async def answer(messages):
            self.messages.append(messages[0])
            reply = next(self.replies)
            if reply is None:
                raise TimeoutError("provider timed out")
            return {"raw": AIMessage("", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
                    "parsed": PageResults(results=reply)}

        return RunnableLambda(answer)


def make_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)


def test_only_table_pages_are_sent_and_duplicates_are_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(extract_results, "RAW_DIR", tmp_path)
    monkeypatch.setattr(extract_results, "CACHE_DIR", tmp_path / "cache")
    make_pdf(tmp_path / "2308.04079.pdf", ["Introduction, no tables", "Table 1: PSNR on Mip-NeRF 360", "Tab. 2 LPIPS"])
    model = FakeModel([None, [result(), result(metric="FID")], [result(), result(metric="LPIPS", value=0.214)]])
    usage = {"input": 0, "output": 0, "skipped_pages": 0, "dropped": 0}
    rows = asyncio.run(extract_results.extract_paper(model, "2308.04079", "3DGS", KNOWN, asyncio.Semaphore(1), usage))

    assert [(r["metric"], r["page"]) for r in rows] == [("PSNR", 2), ("LPIPS", 3)]
    assert usage == {"input": 20, "output": 10, "skipped_pages": 0, "dropped": 1}
    # The first call for page 2 failed and was retried; the image and the paper's title went with each call.
    assert len(model.messages) == 3
    content = model.messages[0].content
    assert "3DGS (arXiv:2308.04079)" in content[0]["text"] and content[1]["image_url"]["url"].startswith("data:image/png")

    # A second run reads the cached replies and makes no model calls.
    again = asyncio.run(extract_results.extract_paper(FakeModel([]), "2308.04079", "3DGS", KNOWN, asyncio.Semaphore(1), usage))
    assert again == rows and usage["input"] == 20
