"""Final answer node: streams the answer from the findings, tagged "synthesize" for token consumers."""

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.registry import PROMPTS_DIR
from graph.state import AgentError, Finding, OrchestratorState
from graph.supervisor import latest_question
from graph.trace import emit_trace
from llm import get_llm

logger = logging.getLogger("graph.synthesize")

UNAVAILABLE_ANSWER = "Sorry, the language model is unavailable right now, so I can't write an answer. Please try again."


def _format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "(none)"
    blocks = []
    for i, finding in enumerate(findings, 1):
        cited = ", ".join(c.id for c in finding.citations) or "none"
        blocks.append(f"### Finding {i} (from {finding.agent}; citations: {cited})\n{finding.content}")
    return "\n\n".join(blocks)


def _format_problems(errors: list[AgentError]) -> str:
    # Only the user-relevant summary each error carries; never raw tracebacks.
    return "\n".join(f"- {e.message}" for e in errors) or "(none)"


def make_synthesize(llm: BaseChatModel | None) -> Callable[[OrchestratorState], Awaitable[dict]]:
    model = llm if llm is not None else get_llm("fast")  # Phase 5 swaps this line for router.route()
    system_prompt = (PROMPTS_DIR / "synthesize.md").read_text()

    async def synthesize(state: OrchestratorState) -> dict:
        step = state.get("step")
        start = time.perf_counter()
        emit_trace("synthesize", "enter", step=step)
        context = "\n\n".join([
            f"Question: {latest_question(state)}",
            f"Task type: {state.get('task_type') or 'general'}",
            f"Today's date: {date.today().isoformat()}",
            f"Findings:\n{_format_findings(state.get('findings', []))}",
            f"Problems:\n{_format_problems(state.get('errors', []))}",
        ])
        update: dict = {}
        parts: list[str] = []
        try:
            async for chunk in model.astream(
                [SystemMessage(system_prompt), HumanMessage(context)], config={"tags": ["synthesize"]}
            ):
                parts.append(chunk.text)
            answer = "".join(parts)
        except Exception as exc:
            logger.warning("synthesize failed: %s: %s", type(exc).__name__, str(exc)[:200])
            emit_trace("synthesize", "error", step=step, detail=f"model call failed: {type(exc).__name__}")
            answer = "".join(parts) or UNAVAILABLE_ANSWER
            update["errors"] = [AgentError(agent="synthesize", kind="synthesize_failed",
                                           message="The language model failed while writing the answer.")]
        emit_trace("synthesize", "exit", step=step, ms=round((time.perf_counter() - start) * 1000))
        return update | {"messages": [AIMessage(answer)]}

    return synthesize
