"""Run the orchestration graph from the terminal.

  python scripts/cli.py "How does 3D Gaussian Splatting render a novel view?"   # one shot
  python scripts/cli.py                                                         # REPL
  python scripts/cli.py --mcp-config tests/fixtures/mcp_config.broken_corpus.json "..."  # failure probe

Trace events print as dim lines; the answer streams live. Each turn is independent (no history yet).
The MCP servers start once and are reused across REPL turns.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from graph.build import DEFAULT_CONFIG, build_graph  # noqa: E402
from mcp_client import MCPTools, install_log_redaction, load_manifests  # noqa: E402

TTY = sys.stdout.isatty()
DIM, RED, RESET = ("\033[2m", "\033[1;31m", "\033[0m") if TTY else ("", "", "")


def format_trace(event: dict) -> str:
    step = f"[step {event['step']}]" if event.get("step") is not None else "[step -]"
    who = event["node"] if not event.get("agent") or event["agent"] == event["node"] else f"{event['node']}/{event['agent']}"
    if event["kind"] == "decision":
        text = f"{who} decision → {event['detail']}"
    else:
        text = f"{who} {event['kind']}" + (f": {event['detail']}" if event.get("detail") else "")
    if event.get("ms") is not None:
        text += f" ({event['ms']} ms)"
    return f"{DIM}{step} {text}{RESET}"


async def run_turn(graph, mcp: MCPTools, question: str) -> None:
    start = time.perf_counter()
    mcp.refresh()  # one reconnect attempt per turn for unhealthy servers, in the background
    final: dict = {}
    streamed: list[str] = []
    answering = False
    inputs = {"session_id": uuid.uuid4().hex, "messages": [HumanMessage(question)], "step": 0}
    # subgraphs=True: trace events written inside worker subgraphs reach this stream too.
    async for namespace, mode, payload in graph.astream(
        inputs, config=DEFAULT_CONFIG, stream_mode=["custom", "messages", "values"], subgraphs=True
    ):
        if mode == "custom":
            if answering:
                print()
                answering = False
            print(format_trace(payload), flush=True)
        elif mode == "messages":
            chunk, metadata = payload
            # Only the answer: supervisor calls stream structured-output JSON through the same channel.
            if metadata.get("langgraph_node") == "synthesize" and "synthesize" in metadata.get("tags", []) and chunk.text:
                if not answering:
                    print()
                    answering = True
                print(chunk.text, end="", flush=True)
                streamed.append(chunk.text)
        elif not namespace:  # the root graph's state, not a worker subgraph's
            final = payload
    if answering:
        print()
    # Text the stream didn't carry, e.g. the notice synthesize adds when the model fails.
    answer = final["messages"][-1].text if final.get("messages") else ""
    if answer.strip() != "".join(streamed).strip():
        print(answer.removeprefix("".join(streamed).rstrip()).strip())

    print_sources(final)
    print_errors(final.get("errors", []))
    summary = (
        f"steps={final.get('step', 0)} "
        f"dispatches={[d.worker for d in final.get('dispatches', [])]} "
        f"findings={len(final.get('findings', []))} "
        f"total={round((time.perf_counter() - start) * 1000)} ms"
    )
    print(f"\n{DIM}— {summary}{RESET}")


def print_sources(final: dict) -> None:
    """Retrieved chunk IDs, the answer's validated citations, and any unverified ones."""
    retrieved = final.get("retrieved_chunk_ids", [])
    if retrieved:
        print(f"\n{DIM}Retrieved chunks ({len(retrieved)}): {', '.join(retrieved)}{RESET}")
    report = final.get("answer_citations")
    if report and report.valid:
        print("\nCitations:")
        for i, c in enumerate(report.valid, 1):
            where = ", ".join(x for x in [c.title, f"§{c.section}" if c.section else None,
                                          f"p.{c.page}" if c.page is not None else None] if x)
            print(f"  {i}. [{c.id}] {where or '(no metadata)'} — {c.url or ''}")
    invalid = report.invalid if report else []
    line = f"UNVERIFIED CITATIONS: {', '.join(invalid) if invalid else '(none)'}"
    print(f"\n{RED}{line}{RESET}" if invalid else f"\n{DIM}{line}{RESET}")


def print_errors(errors: list) -> None:
    if not errors:
        return
    by_kind: dict[str, list[str]] = {}
    for e in errors:
        by_kind.setdefault(e.kind, []).append(e.agent)
    print(f"\n{DIM}Errors: " + "; ".join(f"{kind} ×{len(agents)} ({', '.join(dict.fromkeys(agents))})"
                                      for kind, agents in by_kind.items()) + RESET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="ask one question; omit for a REPL")
    parser.add_argument("--mcp-config", metavar="PATH",
                        help="MCP server manifests to use instead of MCP_SERVERS_CONFIG (for failure probes)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    install_log_redaction()
    # Gemini drops "additionalProperties" from MCP tool schemas and warns on every call; it's harmless.
    logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
    os.environ["LOG_LEVEL"] = "WARNING"  # quiet the MCP servers' own startup logs
    mcp = MCPTools(load_manifests(args.mcp_config))
    await mcp.start()
    for name, status in mcp.status.items():
        print(f"{DIM}[mcp] {name}: {status.state}" + (f" ({status.error})" if status.error else "") + RESET)
    try:
        graph = build_graph(mcp=mcp)
        if args.question:
            await run_turn(graph, mcp, " ".join(args.question))
            return
        print("3D vision paper assistant — empty line or Ctrl-D to quit.")
        while True:
            try:
                question = (await asyncio.to_thread(input, "\n> ")).strip()
            except EOFError:
                break
            if not question:
                break
            try:
                await run_turn(graph, mcp, question)
            except Exception as exc:  # keep the REPL alive; the graph itself degrades gracefully
                print(f"error: {type(exc).__name__}: {exc}")
    finally:
        await mcp.aclose()


if __name__ == "__main__":
    asyncio.run(main())
