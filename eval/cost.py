"""Token usage per provider -> cost per query.

  .venv/bin/python eval/cost.py baseline [gemini claude gpt ...]

Reads eval/results/<label>.jsonl from `run_ragas.py answer` (each record holds the done event's
per-provider token usage, timings, and the tools called) and prints a Markdown table of cost per query
by question type and by provider, written to eval/results/cost-<labels>.md.

Model tokens are priced from PRICES. Embedding cost is estimated from the search_corpus calls (one
query embedding each, about QUERY_TOKENS tokens). Tavily is counted in credits (a basic search or an
extract is one credit). Cloud Run compute is estimated from each request's wall time at 1 vCPU and 2 GiB;
with concurrency 10 a busy instance shares that time, so this is an upper bound per request.
"""

import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# USD per 1M tokens (input, output), checked 2026-10-26:
#   gemini-3.8-flash  ai.google.dev/gemini-api/docs/pricing (paid tier, rate valid to 2026-12-31)
#   claude-sonnet-5-5 platform.claude.com/docs/en/about-claude/pricing
#   gpt-6.1-sol       developers.openai.com/api/docs/pricing (standard, <=272K context)
PRICES = {"google": (0.75, 3.75), "anthropic": (2.00, 10.00), "openai": (2.00, 10.00)}
EMBEDDING_PRICE = 0.02  # text-embedding-3-small, per 1M tokens
QUERY_TOKENS = 30
# Cloud Run request-based billing, us-west1 (Tier 1), from cloud.google.com/run/pricing. NOT re-checked:
# the page didn't render on 2026-10-26. Compute is a small share of the total either way.
VCPU_SECOND = 0.000024
GIB_SECOND = 0.0000025
MEMORY_GIB = 2


def query_cost(record: dict) -> dict:
    models = sum(t["input"] / 1e6 * PRICES[p][0] + t["output"] / 1e6 * PRICES[p][1]
                 for p, t in record["usage"].items() if p in PRICES)
    searches = sum(1 for t in record["tools"] if t == "search_corpus")
    seconds = record["timings"]["total_ms"] / 1000
    return {
        "models": models,
        "embeddings": searches * QUERY_TOKENS / 1e6 * EMBEDDING_PRICE,
        "compute": seconds * (VCPU_SECOND + MEMORY_GIB * GIB_SECOND),
        "tavily_credits": sum(1 for t in record["tools"] if t.startswith("tavily")),
        "tokens_in": sum(t["input"] for t in record["usage"].values()),
        "tokens_out": sum(t["output"] for t in record["usage"].values()),
    }


def load(label: str) -> list[dict]:
    with (RESULTS / f"{label}.jsonl").open() as f:
        return [r | {"cost": query_cost(r)} for r in map(json.loads, f)]


def row(name: str, records: list[dict]) -> str:
    total = [r["cost"]["models"] + r["cost"]["embeddings"] + r["cost"]["compute"] for r in records]
    mean = lambda key: statistics.mean(r["cost"][key] for r in records)  # noqa: E731
    return (f"| {name} | {len(records)} | {mean('tokens_in'):,.0f} | {mean('tokens_out'):,.0f} | "
            f"${mean('models'):.4f} | ${mean('embeddings'):.6f} | ${mean('compute'):.4f} | "
            f"{mean('tavily_credits'):.1f} | ${statistics.mean(total):.4f} | ${statistics.median(total):.4f} |")


def main() -> None:
    labels = sys.argv[1:] or ["baseline"]
    lines = ["| Run / group | n | input tok | output tok | models | embeddings | Cloud Run | Tavily credits | "
             "mean $/query | median $/query |", "|---|---|---|---|---|---|---|---|---|---|"]
    for label in labels:
        records = load(label)
        lines.append(row(f"**{label}: all**", records))
        for task in dict.fromkeys(r["task_type"] for r in records):
            lines.append(row(f"{label}: {task}", [r for r in records if r["task_type"] == task]))
        by_provider: dict[str, list[float]] = {}
        for r in records:
            for provider, t in r["usage"].items():
                if provider in PRICES:
                    by_provider.setdefault(provider, []).append(
                        t["input"] / 1e6 * PRICES[provider][0] + t["output"] / 1e6 * PRICES[provider][1])
        lines.append(f"| {label}: share by provider | | | | " + ", ".join(
            f"{p} ${sum(v) / len(records):.4f}" for p, v in by_provider.items()) + " | | | | | |")
    text = "\n".join(lines) + "\n"
    (RESULTS / f"cost-{'-'.join(labels)}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
