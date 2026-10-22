"""Final answer node: streams the answer from the findings, tagged "synthesize" for token consumers.

After streaming, the answer's citations are validated against the union of the findings' citations
and stored in answer_citations. The tokens have already gone out, so invalid IDs are reported, not
rewritten. If the model call fails, the final message is a fixed notice (appended to any tokens that
already streamed), so the graph always ends with an AIMessage.
"""

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
from rag.citations import validate

logger = logging.getLogger("graph.synthesize")

ANSWER_FAILED_NOTICE = "Sorry, the answer couldn't be generated because the language model failed. Please try again."


def _format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "(none)"
    blocks = []
    for i, finding in enumerate(findings, 1):
        cited = ", ".join(c.id for c in finding.citations) or "none"
        grounded = "yes" if finding.grounded else "no (no verified citations)"
        blocks.append(f"### Finding {i} (from {finding.agent}; grounded: {grounded}; citations: {cited})\n{finding.content}")
    return "\n\n".join(blocks)


def _format_problems(errors: list[AgentError]) -> str:
    # Only the user-relevant summary each error carries; never raw tracebacks.
    return "\n".join(f"- {e.message}" for e in errors) or "(none)"


def make_synthesize(llm: BaseChatModel | None) -> Callable[[OrchestratorState], Awaitable[dict]]:
    model = llm if llm is not None else get_llm("fast", thinking_level="low")  # Phase 5 swaps this for router.route()
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
        first_token_ms: int | None = None
        reasoning_tokens: int | None = None
        try:
            async for chunk in model.astream(
                [SystemMessage(system_prompt), HumanMessage(context)], config={"tags": ["synthesize"]}
            ):
                if chunk.text and first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - start) * 1000)
                parts.append(chunk.text)
                usage = chunk.usage_metadata or {}
                reasoning_tokens = usage.get("output_token_details", {}).get("reasoning", reasoning_tokens)
            answer = "".join(parts)
        except Exception as exc:
            logger.exception("synthesize model call failed")
            emit_trace("synthesize", "error", step=step, detail=f"synthesize_error: {type(exc).__name__}")
            # Keep whatever already streamed; the notice says the answer is incomplete.
            streamed = "".join(parts).rstrip()
            answer = f"{streamed}\n\n{ANSWER_FAILED_NOTICE}" if streamed else ANSWER_FAILED_NOTICE
            update["errors"] = [AgentError(agent="synthesize", kind="synthesize_error",
                                           message="The language model failed while writing the answer.")]

        allowed = {c.id: c for f in state.get("findings", []) for c in f.citations}
        report = validate(answer, allowed)
        if report.invalid:
            emit_trace("synthesize", "error", step=step, detail=f"unverified citations in the answer: {', '.join(report.invalid)}")
        update["answer_citations"] = report
        # The split between waiting (first token) and writing shows where synthesize's time goes.
        stats = [f"first token {first_token_ms / 1000:.1f} s" if first_token_ms is not None else "no tokens",
                 f"{len(answer.split())} words"]
        if reasoning_tokens is not None:
            stats.append(f"{reasoning_tokens} reasoning tokens")
        emit_trace("synthesize", "exit", step=step, detail=", ".join(stats), ms=round((time.perf_counter() - start) * 1000))
        return update | {"messages": [AIMessage(answer)]}

    return synthesize
