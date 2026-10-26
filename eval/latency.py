"""TTFT / retrieval / LLM / total, cold vs warm, against the deployed URL.

  .venv/bin/python eval/latency.py https://paper-assistant-....run.app warm --repeats 3
  .venv/bin/python eval/latency.py https://paper-assistant-....run.app cold --repeats 1

Each query is posted to /chat/stream in a fresh session. Stage times come from when each SSE event
arrives at this client, so they include the network and match what a user sees:

  start      request sent -> load_history entered (container start on a cold request, then queueing)
  history    load_history enter -> exit (Firestore read)
  route      supervisor enter -> exit, first decision
  worker     worker enter -> exit (model calls plus tool calls; `retrieval` is the tool part)
  route2     the supervisor's second decision (worker findings -> synthesize)
  first_tok  synthesize enter -> first token
  ttft       request sent -> first token;   total: request sent -> done

retrieval and llm also come from the server's done event (summed tool-call time and summed model-call
time). For a cold run, the service must have scaled to zero first (no request for 15+ minutes); only
the first request of a cold run is cold, so use --repeats 1. Results go to
eval/results/latency-<label>.jsonl and a Markdown table to latency-<label>.md.
"""

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx

RESULTS = Path(__file__).resolve().parent / "results"
WORKERS = ("research", "data", "summary")

# One query per path through the graph, none of them in eval/questions.jsonl.
QUERIES = [
    ("lookup", "How does Plenoxels render without a neural network?"),
    ("compare", "Compare how NeuS and 2D Gaussian Splatting reconstruct surfaces."),
    ("data", "Which methods report the lowest LPIPS on Mip-NeRF 360?"),
    ("current_events", "What Gaussian splatting papers were posted on arXiv this week?"),
    ("no_tools", "What is a signed distance function?"),
]
STAGES = ["start", "history", "route", "worker", "route2", "first_tok", "ttft", "retrieval", "llm", "total"]


def run_query(client: httpx.Client, base: str, question: str) -> dict:
    t0 = time.perf_counter()
    marks: dict[str, float] = {}
    worker_spans: list[tuple[float, float]] = []
    supervisor_spans: list[tuple[float, float]] = []
    entered: dict[str, float] = {}
    done, buffer = None, ""
    with client.stream("POST", f"{base}/chat/stream", json={"session_id": f"lat-{uuid.uuid4().hex[:12]}",
                                                              "message": question},
                       timeout=httpx.Timeout(320, connect=120)) as response:
        for chunk in response.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                now = time.perf_counter() - t0
                name = next((line[6:].strip() for line in frame.splitlines() if line.startswith("event:")), None)
                data = "\n".join(line[5:].strip() for line in frame.splitlines() if line.startswith("data:"))
                payload = json.loads(data) if data else {}
                if name == "trace" and payload["kind"] in ("enter", "exit"):
                    node = payload["node"]
                    if payload["kind"] == "enter":
                        entered[node] = now
                        marks.setdefault(f"{node}_enter", now)
                    elif node in entered:
                        span = (entered.pop(node), now)
                        if node in WORKERS:
                            worker_spans.append(span)
                        elif node == "supervisor":
                            supervisor_spans.append(span)
                        marks.setdefault(f"{node}_exit", now)
                elif name == "token":
                    marks.setdefault("first_token", now)
                elif name == "done":
                    done = payload
                    marks["done"] = now
    ms = lambda seconds: round(seconds * 1000) if seconds is not None else None  # noqa: E731
    span_ms = lambda spans: ms(sum(b - a for a, b in spans)) if spans else None  # noqa: E731
    stages = {
        "start": ms(marks.get("load_history_enter")),
        "history": ms(marks["load_history_exit"] - marks["load_history_enter"]) if "load_history_exit" in marks else None,
        "route": span_ms(supervisor_spans[:1]),
        "worker": span_ms(worker_spans),
        "route2": span_ms(supervisor_spans[1:]),
        "first_tok": ms(marks["first_token"] - marks["synthesize_enter"]) if "first_token" in marks and "synthesize_enter" in marks else None,
        "ttft": ms(marks.get("first_token")),
        "retrieval": done["timings"]["retrieval_ms"] if done else None,
        "llm": done["timings"]["llm_ms"] if done else None,
        "total": ms(marks.get("done")),
    }
    return {"status": response.status_code, "stages": stages, "usage": done["usage"] if done else None,
            "server_timings": done["timings"] if done else None}


def table(label: str, rows: list[dict]) -> str:
    lines = [f"## Latency: {label} ({len(rows)} requests; median ms per stage)", "",
             "| Query | n | " + " | ".join(STAGES) + " |", "|---|---|" + "---|" * len(STAGES)]
    for kind in dict.fromkeys(r["kind"] for r in rows):
        members = [r for r in rows if r["kind"] == kind]
        cells = []
        for stage in STAGES:
            values = [m["stages"][stage] for m in members if m["stages"][stage] is not None]
            cells.append(str(round(statistics.median(values))) if values else "–")
        lines.append(f"| {kind} | {len(members)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("label", help="e.g. warm or cold")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    RESULTS.mkdir(exist_ok=True)
    rows = []
    with httpx.Client() as client, (RESULTS / f"latency-{args.label}.jsonl").open("w") as out:
        for repeat in range(args.repeats):
            for kind, question in QUERIES:
                row = {"label": args.label, "repeat": repeat, "kind": kind, "question": question,
                       **run_query(client, base, question)}
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(kind, row["status"], row["stages"], flush=True)
    (RESULTS / f"latency-{args.label}.md").write_text(table(args.label, rows))
    print(table(args.label, rows))


if __name__ == "__main__":
    main()
