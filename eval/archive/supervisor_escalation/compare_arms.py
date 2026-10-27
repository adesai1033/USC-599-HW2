"""Compare supervisor-routing arms: routing accuracy, latency, cost, errors, RAGAS.

  .venv/bin/python eval/archive/supervisor_escalation/compare_arms.py

Reads answers/<arm>.jsonl (local only, gitignored) and results/<arm>.scores.jsonl.
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from eval.cost import query_cost  # noqa: E402

ARMS = ["sup-control", "sup-escalate", "sup-senior"]
# Which first dispatch is acceptable for each question type.
OK_FIRST = {"lookup": {"research"}, "compare": {"research", "summary"}, "lit_review": {"summary"},
            "metadata": {"data"}, "not_in_corpus": {"research"}}
OK_TYPE = {"lookup": {"lookup"}, "compare": {"compare"}, "lit_review": {"lit_review", "compare"},
           "metadata": {"metadata"}, "not_in_corpus": {"lookup", "current_events", "general"}}


def load(label):
    return [json.loads(l) for l in (HERE / "answers" / f"{label}.jsonl").open()]


def scores(label):
    p = HERE / "results" / f"{label}.scores.jsonl"
    return {r["id"]: r["scores"] for r in map(json.loads, p.open())} if p.exists() else {}


def qc(r):
    c = query_cost(r)
    return c["models"] + c["embeddings"] + c["compute"]


def med(xs):
    return st.median(xs) if xs else float("nan")


print("| Arm | first dispatch OK | task_type OK | dispatches/q | supervisor calls on Claude | errors | median TTFT s | median total s | mean $/q |")
print("|---|---|---|---|---|---|---|---|---|")
per_q = {}
for arm in ARMS:
    rs = load(arm)
    first_ok = sum(bool(r["dispatches"]) and r["dispatches"][0] in OK_FIRST[r["task_type"]] for r in rs)
    type_ok = sum(bool(r["decisions"]) and r["decisions"][0]["task_type"] in OK_TYPE[r["task_type"]] for r in rs)
    sup = [d for r in rs for d in r["routing"] if d["call_site"] == "supervisor"]
    strong = sum(d["tier"] == "strong" for d in sup)
    errs = [e for r in rs for e in r["errors"]]
    cost = st.mean(qc(r) for r in rs)
    print(f"| {arm} | {first_ok}/{len(rs)} | {type_ok}/{len(rs)} | {st.mean(len(r['dispatches']) for r in rs):.2f} | "
          f"{strong}/{len(sup)} | {len(errs)} {sorted(set(errs))} | {med([r['timings']['ttft_ms'] for r in rs])/1000:.1f} | "
          f"{med([r['timings']['total_ms'] for r in rs])/1000:.1f} | ${cost:.4f} |")
    for r in rs:
        per_q.setdefault(r["id"], {})[arm] = r

print("\nBy question type (median total s / mean $):")
for t in OK_FIRST:
    cells = []
    for arm in ARMS:
        rs = [r for r in load(arm) if r["task_type"] == t]
        cells.append(f"{arm}: {med([r['timings']['total_ms'] for r in rs])/1000:.1f}s ${st.mean(qc(r) for r in rs):.3f}")
    print(f"- {t}: " + "; ".join(cells))

print("\nQuestions where dispatch sequences differ:")
for qid, arms in per_q.items():
    seqs = {a: arms[a]["dispatches"] for a in ARMS}
    if len({tuple(s) for s in seqs.values()}) > 1:
        print(f"- {qid} ({arms[ARMS[0]]['task_type']}): " + "; ".join(f"{a[4:]}={s}" for a, s in seqs.items()))

print("\nEscalation reasons (escalate arm):")
reasons = {}
for r in load("sup-escalate"):
    for d in r["routing"]:
        if d["call_site"] == "supervisor" and d["tier"] == "strong":
            key = d["reason"].split(":")[0]
            key = "".join(c for c in key if not c.isdigit())
            reasons[key] = reasons.get(key, 0) + 1
for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
    print(f"- {v}× {k}")

names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
if all(scores(a) for a in ARMS):
    print("\n| Arm | n | " + " | ".join(names) + " |\n|---|---|" + "---|" * 4)
    for arm in ARMS:
        s = [v for v in scores(arm).values() if v]
        print(f"| {arm} | {len(s)} | " + " | ".join(
            f"{st.mean([x[n] for x in s if x[n] is not None]):.3f}" for n in names) + " |")
    for t in ("lookup", "compare", "lit_review"):
        ids = [q for q, a in per_q.items() if a[ARMS[0]]["task_type"] == t]
        row = []
        for arm in ARMS:
            sc = scores(arm)
            row.append(arm[4:] + " " + "/".join(
                f"{st.mean([sc[q][n] for q in ids if sc.get(q) and sc[q][n] is not None]):.2f}" for n in names))
        print(f"- {t} (f/ar/cp/cr): " + "; ".join(row))
