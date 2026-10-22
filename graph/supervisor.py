"""Supervisor node: structured output -> SupervisorDecision, then guards that can force `synthesize`.

Guards run after the model returns, in this order; the first that fires sets next="synthesize",
records an AgentError and emits an error trace event:
  a. the structured output failed or did not validate  -> supervisor_parse_error
  b. step >= MAX_SUPERVISOR_STEPS                       -> step_limit
  c. the chosen worker is not available                 -> worker_unavailable
  d. same worker and instruction as the last dispatch   -> repeated_dispatch
"""

import logging
import re
import time
from collections.abc import Awaitable, Callable, Collection

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.registry import AGENTS, PROMPTS_DIR
from config import max_supervisor_steps
from graph.state import AgentError, Dispatch, OrchestratorState, SupervisorDecision
from graph.trace import emit_trace
from llm import get_llm

logger = logging.getLogger("graph.supervisor")

FINDING_PREVIEW_CHARS = 600


def latest_question(state: OrchestratorState) -> str:
    """The text of the most recent user message."""
    for message in reversed(state.get("messages", [])):
        if message.type == "human":
            return message.text
    return ""


def _normalize(instruction: str) -> str:
    return re.sub(r"\s+", " ", instruction).strip().rstrip(".!?").casefold()


def _roster(worker_names: Collection[str]) -> str:
    lines = []
    for name, spec in AGENTS.items():
        status = "" if name in worker_names else " (unavailable right now)"
        lines.append(f"- `{name}`{status}: {spec.description}")
    lines += [f"- `{name}`: (no description)" for name in worker_names if name not in AGENTS]
    lines.append("- `synthesize`: write the final answer from the findings gathered so far.")
    return "\n".join(lines)


def _context(state: OrchestratorState, step: int, max_steps: int) -> str:
    """The question plus everything gathered this turn, as the supervisor's user message."""
    parts = [f"Question: {latest_question(state)}", f"Decision {step} of at most {max_steps}."]
    dispatches = state.get("dispatches", [])
    findings = state.get("findings", [])
    errors = state.get("errors", [])
    parts.append("Dispatches so far:\n" + ("\n".join(f"- {d.worker}: {d.instruction}" for d in dispatches) or "(none)"))
    parts.append("Findings so far:\n" + ("\n".join(
        f"- from {f.agent}: {f.content[:FINDING_PREVIEW_CHARS]}{'...' if len(f.content) > FINDING_PREVIEW_CHARS else ''}"
        for f in findings
    ) or "(none)"))
    parts.append("Errors so far:\n" + ("\n".join(f"- {e.agent} ({e.kind}): {e.message}" for e in errors) or "(none)"))
    return "\n\n".join(parts)


def _unavailable_message(worker: str) -> str:
    spec = AGENTS.get(worker)
    capability = f" ({spec.description})" if spec else ""
    return f"The {worker} worker is not available, so its capability{capability} could not be used."


def make_supervisor(
    llm: BaseChatModel | None, worker_names: Collection[str]
) -> Callable[[OrchestratorState], Awaitable[dict]]:
    """Build the supervisor node. `worker_names` are the workers wired into the graph."""
    model = llm if llm is not None else get_llm("fast")  # Phase 5 swaps this line for router.route()
    decider = model.with_structured_output(SupervisorDecision).with_config(tags=["supervisor"])
    system_prompt = (PROMPTS_DIR / "supervisor.md").read_text().replace("{workers}", _roster(worker_names))

    async def supervisor(state: OrchestratorState) -> dict:
        step = state.get("step", 0) + 1
        max_steps = max_supervisor_steps()
        start = time.perf_counter()
        emit_trace("supervisor", "enter", step=step)

        decision: SupervisorDecision | None = None
        failure = ""
        try:
            raw = await decider.ainvoke([SystemMessage(system_prompt), HumanMessage(_context(state, step, max_steps))])
            decision = raw if isinstance(raw, SupervisorDecision) else SupervisorDecision.model_validate(raw)
        except Exception as exc:  # provider error, unparseable JSON or a schema violation
            failure = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("supervisor structured output failed: %s", failure)

        if decision is not None:
            emit_trace("supervisor", "decision", step=step,
                       detail=f"{decision.next} ({decision.task_type}): {decision.reason}")

        error: AgentError | None = None
        if decision is None:
            error = AgentError(agent="supervisor", kind="supervisor_parse_error",
                               message=f"The supervisor's decision could not be read ({failure}).")
        elif decision.next != "synthesize":
            dispatches = state.get("dispatches", [])
            if step >= max_steps:
                error = AgentError(agent="supervisor", kind="step_limit",
                                   message=f"Reached the limit of {max_steps} supervisor steps; answering with what was gathered.")
            elif decision.next not in worker_names:
                error = AgentError(agent=decision.next, kind="worker_unavailable", message=_unavailable_message(decision.next))
            elif dispatches and dispatches[-1].worker == decision.next and (
                _normalize(dispatches[-1].instruction) == _normalize(decision.instruction)
            ):
                error = AgentError(agent=decision.next, kind="repeated_dispatch",
                                   message=f"The same request was just sent to {decision.next}; not repeating it.")

        update: dict = {"step": step, "task_type": decision.task_type if decision else "general"}
        if error is not None:
            emit_trace("supervisor", "error", step=step, agent=error.agent, detail=f"{error.kind}: {error.message}")
            update |= {"next": "synthesize", "instruction": "", "errors": [error]}
        elif decision.next == "synthesize":
            update |= {"next": "synthesize", "instruction": ""}
        else:
            update |= {"next": decision.next, "instruction": decision.instruction,
                       "dispatches": [Dispatch(worker=decision.next, instruction=decision.instruction)]}
        emit_trace("supervisor", "exit", step=step, detail=f"next={update['next']}", ms=round((time.perf_counter() - start) * 1000))
        return update

    return supervisor
