"""Hand-checked benchmark results (ingest/results_seed.csv) -> the results table, verified=1.

  python ingest/seed_results.py                      # ingest/results_seed.csv into PAPERS_DB_PATH
  python ingest/seed_results.py --csv my.csv --db /tmp/papers.sqlite

Each CSV row is copied by hand from a paper's own table:
  arxiv_id,benchmark,scene,metric,value,higher_is_better,setting,source_note
scene and setting may be empty (empty scene = a dataset average). source_note (e.g. "Tab. 1, p.7") is
for the person checking the row; the results table has no column for it.

Every row is validated first: the arXiv ID must be in papers, value must be a number, metric must be a
known metric (spelled as below, any case), higher_is_better must be 0 or 1 and match that metric's
direction. Bad rows are rejected with a reason; good rows are written.

Rows are upserted on (arxiv_id, benchmark, scene, metric, setting), so running the script twice
changes nothing. build_db.py rebuilds the database from scratch, so run this again after it.
"""

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["arxiv_id", "benchmark", "scene", "metric", "value", "higher_is_better", "setting", "source_note"]

# Canonical metric name -> higher_is_better.
KNOWN_METRICS = {
    "PSNR": 1,
    "SSIM": 1,
    "LPIPS": 0,
    "Chamfer": 0,          # Chamfer distance
    "F-score": 1,
    "Accuracy": 0,         # DTU-style accuracy: a distance, lower is better
    "Completeness": 0,     # DTU-style completeness: a distance, lower is better
    "Training time": 0,
    "FPS": 1,
    "Memory": 0,
}
_METRIC_BY_LOWER = {name.lower(): name for name in KNOWN_METRICS}


def validate_row(row: dict, known_ids: set[str]) -> tuple[dict | None, str | None]:
    """Return (clean row, None) or (None, reason)."""
    arxiv_id = (row.get("arxiv_id") or "").strip()
    if arxiv_id not in known_ids:
        return None, f"arXiv ID {arxiv_id!r} is not in papers"
    benchmark = (row.get("benchmark") or "").strip()
    if not benchmark:
        return None, "benchmark is empty"
    metric = _METRIC_BY_LOWER.get((row.get("metric") or "").strip().lower())
    if metric is None:
        return None, f"metric {row.get('metric')!r} is not one of {sorted(KNOWN_METRICS)}"
    try:
        value = float((row.get("value") or "").strip())
    except ValueError:
        return None, f"value {row.get('value')!r} is not a number"
    raw_hib = (row.get("higher_is_better") or "").strip()
    if raw_hib not in ("0", "1"):
        return None, f"higher_is_better {raw_hib!r} must be 0 or 1"
    if int(raw_hib) != KNOWN_METRICS[metric]:
        return None, f"higher_is_better={raw_hib} contradicts {metric} (expected {KNOWN_METRICS[metric]})"
    return {
        "arxiv_id": arxiv_id,
        "benchmark": benchmark,
        "scene": (row.get("scene") or "").strip() or None,
        "metric": metric,
        "value": value,
        "higher_is_better": int(raw_hib),
        "setting": (row.get("setting") or "").strip() or None,
    }, None


def upsert(conn: sqlite3.Connection, row: dict) -> str:
    """Insert or update one result on its natural key; return "inserted", "updated" or "unchanged"."""
    key = "arxiv_id = :arxiv_id AND benchmark = :benchmark AND scene IS :scene AND metric = :metric AND setting IS :setting"
    existing = conn.execute(f"SELECT value, higher_is_better, verified FROM results WHERE {key}", row).fetchall()
    if not existing:
        conn.execute(
            "INSERT INTO results (arxiv_id, benchmark, scene, metric, value, higher_is_better, setting, verified) "
            "VALUES (:arxiv_id, :benchmark, :scene, :metric, :value, :higher_is_better, :setting, 1)", row)
        return "inserted"
    if len(existing) == 1 and tuple(existing[0]) == (row["value"], row["higher_is_better"], 1):
        return "unchanged"
    # Replace every row on this key, including duplicates and unverified extractions, with this one.
    conn.execute(f"DELETE FROM results WHERE {key}", row)
    conn.execute(
        "INSERT INTO results (arxiv_id, benchmark, scene, metric, value, higher_is_better, setting, verified) "
        "VALUES (:arxiv_id, :benchmark, :scene, :metric, :value, :higher_is_better, :setting, 1)", row)
    return "updated"


def seed(csv_path: Path, db_path: Path) -> tuple[dict[str, int], list[str]]:
    """Validate and upsert every row. Returns (counts by outcome, rejection messages)."""
    counts = {"inserted": 0, "updated": 0, "unchanged": 0, "rejected": 0}
    rejections: list[str] = []
    conn = sqlite3.connect(db_path)
    try:
        known_ids = {r[0] for r in conn.execute("SELECT arxiv_id FROM papers")}
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != COLUMNS:
                raise SystemExit(f"{csv_path}: header must be {','.join(COLUMNS)}, got {reader.fieldnames}")
            with conn:  # one transaction: all valid rows, or none if something unexpected fails
                for line, raw in enumerate(reader, start=2):
                    row, reason = validate_row(raw, known_ids)
                    if row is None:
                        counts["rejected"] += 1
                        rejections.append(f"line {line}: {reason}")
                        continue
                    counts[upsert(conn, row)] += 1
    finally:
        conn.close()
    return counts, rejections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "ingest" / "results_seed.csv")
    parser.add_argument("--db", type=Path, default=ROOT / os.environ.get("PAPERS_DB_PATH", "index/papers.sqlite"))
    args = parser.parse_args()
    counts, rejections = seed(args.csv, args.db.resolve())
    for message in rejections:
        print(f"rejected {message}")
    print(", ".join(f"{n} {outcome}" for outcome, n in counts.items()))
    sys.exit(1 if rejections else 0)


if __name__ == "__main__":
    main()
