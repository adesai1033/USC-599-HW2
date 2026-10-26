"""RAGAS metrics over eval/questions.jsonl -> eval/results/, summarized in docs/evaluation.md.

Two steps in two virtualenvs. RAGAS 0.4.3 needs openai<2 and langchain-community<0.4, which the app's
pinned stack can't use, so it lives in .venv-eval (eval/requirements-ragas.txt):

  .venv/bin/python eval/run_ragas.py answer baseline                    # run the graph, save answers
  .venv/bin/python eval/run_ragas.py answer gpt --provider openai --ids q01 q19
  .venv-eval/bin/python eval/run_ragas.py score baseline                # judge them, write the summary

answer runs each question through the real graph and MCP servers, with in-memory history and a fresh
session per question, the same way POST /chat/stream does (streaming.graph_frames). It saves the answer,
the corpus chunks retrieved in that run (in retrieval order, texts fetched with get_chunks), routing,
timings and token usage to eval/results/<label>.jsonl. --provider sends every call to one provider, for
the provider comparison; the default is the normal router.

score adds four metrics (names from RAGAS 0.4.3's metrics.collections): faithfulness, answer_relevancy,
context_precision_with_reference and context_recall. The judge is JUDGE_MODEL with
text-embedding-3-small for answer relevancy. Citation tags are removed from the answer before judging.
Questions without ground truth or without retrieved chunks (metadata and not-in-corpus) are skipped:
their evidence is SQL rows, which these metrics don't cover. Results go to <label>.scores.jsonl and a
Markdown summary to <label>.md.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"

JUDGE_MODEL = "gpt-6.1-sol"  # fixed so runs stay comparable; OpenAI is only the fallback tier in the app
JUDGE_REASONING = "medium"   # "low" flipped verdicts on borderline paraphrases between runs
EMBEDDING_MODEL = "text-embedding-3-small"
PROVIDER_TIERS = {"google": "fast", "anthropic": "strong", "openai": "fallback"}
CITATION_TAG = re.compile(r"\[(?:\d{4}\.\d{4,5}#[^\]\s]+|web:[0-9a-f]{8}|db:[^\]\s]+|unverified)\]")
MAX_CHUNK_IDS = 20  # get_chunks accepts at most this many IDs per call


def load_questions(ids: list[str] | None) -> list[dict]:
    with QUESTIONS.open() as f:
        questions = [json.loads(line) for line in f]
    return [q for q in questions if not ids or q["id"] in ids]


def parse_sse(frames: list[str]) -> list[tuple[str, dict]]:
    events = []
    for frame in frames:
        name = next((line[6:].strip() for line in frame.splitlines() if line.startswith("event:")), None)
        data = "\n".join(line[5:].strip() for line in frame.splitlines() if line.startswith("data:"))
        if name:
            events.append((name, json.loads(data) if data else {}))
    return events


def force_provider(provider: str) -> None:
    """Send every model call to one provider's tier, whatever the router decides."""
    import llm

    tier = PROVIDER_TIERS[provider]
    original = llm.get_llm
    llm.get_llm = lambda _tier, **kwargs: original(tier, **kwargs)


async def chunk_texts(mcp, chunk_ids: list[str]) -> list[str]:
    """Texts of the retrieved chunks, through the corpus server's own get_chunks tool."""
    tool = next(t for t in mcp.tools_for("summary") if t.name == "get_chunks")
    texts: dict[str, str] = {}
    for start in range(0, len(chunk_ids), MAX_CHUNK_IDS):
        reply = await tool.ainvoke({"chunk_ids": chunk_ids[start:start + MAX_CHUNK_IDS]})
        payload = json.loads(reply if isinstance(reply, str) else reply[0]["text"])
        texts |= {c["chunk_id"]: c["text"] for c in payload["chunks"]}
    return [texts[i] for i in chunk_ids if i in texts]


async def answer(label: str, ids: list[str] | None, provider: str | None) -> Path:
    sys.path.insert(0, str(ROOT))
    from langchain_core.messages import HumanMessage

    from graph.build import build_graph
    from mcp_client import MCPTools, install_log_redaction, load_manifests
    from state_store import InMemoryHistory
    from streaming import graph_frames

    install_log_redaction()
    if provider:
        force_provider(provider)
    mcp = MCPTools(load_manifests())
    await mcp.start()
    out = RESULTS / f"{label}.jsonl"
    RESULTS.mkdir(exist_ok=True)
    try:
        graph = build_graph(mcp=mcp, history=InMemoryHistory())
        with out.open("w") as f:
            for q in load_questions(ids):
                mcp.refresh()
                final: dict = {}
                start = time.perf_counter()
                inputs = {"session_id": f"eval-{label}-{q['id']}", "messages": [HumanMessage(q["question"])], "step": 0}
                frames = [frame async for frame in graph_frames(graph, inputs, session_id=inputs["session_id"],
                                                                 start=start, final_state=final)]
                events = parse_sse(frames)
                text = "".join(d["text"] for n, d in events if n == "token")
                retrieved = final.get("retrieved_chunk_ids", [])
                record = {
                    **q, "label": label, "provider": provider or "routed", "answer": text,
                    "retrieved_ids": retrieved, "contexts": await chunk_texts(mcp, retrieved) if retrieved else [],
                    "routing": [d for n, d in events if n == "routing"],
                    "dispatches": [d.worker for d in final.get("dispatches", [])],
                    "tools": [d["detail"].split("(")[0] for n, d in events if n == "trace" and d["kind"] == "tool_call"],
                    "citations": next((d for n, d in events if n == "citations"), {}),
                    "errors": [e.kind for e in final.get("errors", [])],
                    **next(d for n, d in events if n == "done"),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"{q['id']}: {record['timings']['total_ms']} ms, {len(retrieved)} chunks, "
                      f"{len(record['citations'].get('valid', []))} valid citations, dispatches {record['dispatches']}",
                      flush=True)
    finally:
        await mcp.aclose()
    return out


def judge():
    """The judge LLM and embeddings, with a shim for RAGAS 0.4.3's model-name check.

    RAGAS sends max_completion_tokens (and drops temperature) only for model names it parses as
    gpt-5..gpt-19; "gpt-6.1-sol" fails the parse, so the shim renames the parameter itself.
    """
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms.base import llm_factory

    load_dotenv(ROOT / ".env")
    client = AsyncOpenAI(max_retries=4)
    create = client.chat.completions.create
    usage = {"input": 0, "output": 0}

    async def reasoning_create(*args, **kwargs):
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        kwargs["reasoning_effort"] = JUDGE_REASONING
        reply = await create(*args, **kwargs)
        if getattr(reply, "usage", None):  # judge cost
            usage["input"] += reply.usage.prompt_tokens
            usage["output"] += reply.usage.completion_tokens
        return reply

    client.chat.completions.create = reasoning_create
    llm = llm_factory(JUDGE_MODEL, client=client, max_tokens=8192)
    return llm, embedding_factory("openai", model=EMBEDDING_MODEL, client=client), usage


async def score(label: str) -> None:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    llm, embeddings, usage = judge()
    metrics = {
        "faithfulness": (Faithfulness(llm=llm), lambda r: dict(user_input=r["question"], response=r["clean"],
                                                               retrieved_contexts=r["contexts"])),
        "answer_relevancy": (AnswerRelevancy(llm=llm, embeddings=embeddings),
                             lambda r: dict(user_input=r["question"], response=r["clean"])),
        "context_precision": (ContextPrecisionWithReference(llm=llm),
                              lambda r: dict(user_input=r["question"], reference=r["ground_truth"],
                                             retrieved_contexts=r["contexts"])),
        "context_recall": (ContextRecall(llm=llm), lambda r: dict(user_input=r["question"], reference=r["ground_truth"],
                                                                  retrieved_contexts=r["contexts"])),
    }
    with (RESULTS / f"{label}.jsonl").open() as f:
        records = [json.loads(line) for line in f]
    gate = asyncio.Semaphore(4)

    async def one(record: dict) -> dict:
        if not record.get("ground_truth") or not record["contexts"] or record["task_type"] in ("metadata", "not_in_corpus"):
            return record | {"scores": None}
        record = record | {"clean": CITATION_TAG.sub("", record["answer"])}
        scores = {}
        async with gate:
            for name, (metric, args) in metrics.items():
                try:
                    scores[name] = (await metric.ascore(**args(record))).value
                except Exception as exc:  # one failed judgment shouldn't lose the run
                    print(f"{record['id']} {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    scores[name] = None
        print(f"{record['id']}: " + ", ".join(f"{k} {v:.2f}" if v is not None else f"{k} -" for k, v in scores.items()),
              flush=True)
        return record | {"scores": scores}

    scored = await asyncio.gather(*(one(r) for r in records))
    with (RESULTS / f"{label}.scores.jsonl").open("w") as f:
        for record in scored:
            f.write(json.dumps({k: record[k] for k in ("id", "task_type", "question", "scores")}) + "\n")
    (RESULTS / f"{label}.md").write_text(summary(label, scored, usage))
    print(summary(label, scored, usage))


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None and v == v]  # drop None and NaN
    return sum(values) / len(values) if values else None


def summary(label: str, scored: list[dict], judge_usage: dict) -> str:
    names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    rows = [r for r in scored if r["scores"]]
    fmt = lambda v: f"{v:.3f}" if v is not None else "–"  # noqa: E731
    lines = [f"## RAGAS: {label} (judge {JUDGE_MODEL}, {len(rows)} questions)", "",
             "| Group | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    groups = {"all": rows} | {t: [r for r in rows if r["task_type"] == t] for t in ("lookup", "compare", "lit_review")}
    for group, members in groups.items():
        if members:
            lines.append(f"| {group} | {len(members)} | " +
                         " | ".join(fmt(mean([m["scores"][n] for m in members])) for n in names) + " |")
    lines += ["", "| id | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {r['id']} | " + " | ".join(fmt(r["scores"][n]) for n in names) + " |" for r in rows]
    lines += ["", f"Judge tokens: {judge_usage['input']} input, {judge_usage['output']} output.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    run = steps.add_parser("answer", help="run the graph on the questions (app venv)")
    run.add_argument("label")
    run.add_argument("--ids", nargs="+", help="only these question IDs")
    run.add_argument("--provider", choices=sorted(PROVIDER_TIERS), help="send every model call to this provider")
    judged = steps.add_parser("score", help="judge saved answers with RAGAS (eval venv)")
    judged.add_argument("label")
    args = parser.parse_args()
    os.environ.setdefault("STATE_BACKEND", "memory")
    if args.step == "answer":
        print(asyncio.run(answer(args.label, args.ids, args.provider)))
    else:
        asyncio.run(score(args.label))


if __name__ == "__main__":
    main()
