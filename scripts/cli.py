"""Run the orchestration graph from the terminal.

  python scripts/cli.py "How does 3D Gaussian Splatting render a novel view?"   # one shot
  python scripts/cli.py                                                         # REPL

Trace events print as dim lines; the answer streams live. Each turn is independent (no history yet).
"""

import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from graph.build import DEFAULT_CONFIG, build_graph  # noqa: E402

TTY = sys.stdout.isatty()
DIM, RESET = ("\033[2m", "\033[0m") if TTY else ("", "")


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


async def run_turn(graph, question: str) -> None:
    start = time.perf_counter()
    final: dict = {}
    answering = False
    inputs = {"session_id": uuid.uuid4().hex, "messages": [HumanMessage(question)], "step": 0}
    async for mode, payload in graph.astream(inputs, config=DEFAULT_CONFIG, stream_mode=["custom", "messages", "values"]):
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
        else:
            final = payload
    if answering:
        print()

    errors = final.get("errors", [])
    summary = (
        f"steps={final.get('step', 0)} "
        f"dispatches={[f'{d.worker}' for d in final.get('dispatches', [])]} "
        f"findings={len(final.get('findings', []))} "
        f"errors={[e.kind for e in errors]} "
        f"total={round((time.perf_counter() - start) * 1000)} ms"
    )
    print(f"\n{DIM}— {summary}{RESET}")


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    graph = build_graph()
    if len(sys.argv) > 1:
        await run_turn(graph, " ".join(sys.argv[1:]))
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
            await run_turn(graph, question)
        except Exception as exc:  # keep the REPL alive; the graph itself degrades gracefully
            print(f"error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
