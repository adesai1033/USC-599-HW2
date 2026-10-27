"""Supervisor node: structured output -> SupervisorDecision, then guards that can force `synthesize`.

Guards run after the model returns, in this order; the first that fires sets next="synthesize",
records an AgentError and emits an error trace event:
  a. the model call raised                                    -> supervisor_error
     the structured output failed or did not validate         -> supervisor_parse_error
  b. step >= MAX_SUPERVISOR_STEPS                             -> step_limit
  c. the chosen worker is not available                       -> worker_unavailable
  d. the chosen worker already failed this turn, and either it
     has used its MAX_WORKER_RETRIES or the instruction repeats
     one it already got                                       -> retry_limit
  e. same worker and instruction as the last dispatch         -> repeated_dispatch
Before calling the model at all: if every worker has failed past its retries, go to synthesize.
"""

import logging
import re
import time
from collections.abc import Awaitable, Callable, Collection

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from agents.registry import AGENTS, PROMPTS_DIR
from config import max_supervisor_steps, max_worker_retries, supervisor_timeout_s
from graph.history import format_earlier_turns
from graph.state import AgentError, Dispatch, Finding, OrchestratorState, RoutingDecision, SupervisorDecision
from graph.trace import emit_routing, emit_trace
from llm import FALLBACK, Models, Tier, resolve_models
from router import RouteFeatures, invoke_with_fallback, route

logger = logging.getLogger("graph.supervisor")

FINDING_PREVIEW_CHARS = 600
FAILURE_KINDS = {"worker_timeout", "transport", "provider_error", "worker_exception"}  # from graph/guard.py
ERROR_SUMMARY = {
    "worker_timeout": "ran too long and was stopped",
    "transport": "lost its connection to a tool server",
    "provider_error": "its language model provider returned an error",
    "worker_exception": "failed unexpectedly",
    "history_unavailable": "earlier turns of the conversation couldn't be loaded",
    "worker_unavailable": "is not available",
    "invalid_citation": "cited sources it never retrieved; marked [unverified]",
    "attribution_mismatch": "cited a source about a different paper",
    "retry_limit": "will not be retried again this turn",
    "repeated_dispatch": "the same request was already sent",
    "step_limit": "step limit reached",
    "supervisor_parse_error": "a decision could not be read",
    "supervisor_error": "the decision model call failed",
}


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
    parts = []
    if earlier := format_earlier_turns(state):
        parts.append(f"Earlier in this conversation (oldest first):\n\n{earlier}")
    parts += [f"Question: {latest_question(state)}", f"Decision {step} of at most {max_steps}."]
    dispatches = state.get("dispatches", [])
    findings = state.get("findings", [])
    errors = state.get("errors", [])
    parts.append("Dispatches so far:\n" + ("\n".join(f"- {d.worker}: {d.instruction}" for d in dispatches) or "(none)"))
    parts.append("Findings so far:\n" + ("\n".join(_finding_line(f) for f in findings) or "(none)"))
    parts.append("Errors so far:\n" + (_error_summary(state) or "(none)"))
    return "\n\n".join(parts)


def _finding_line(finding: Finding) -> str:
    """A finding for the supervisor: a preview of its text, plus every paper it cited, never truncated,
    so a later instruction can hand those identifiers to the next worker."""
    preview = finding.content[:FINDING_PREVIEW_CHARS] + ("..." if len(finding.content) > FINDING_PREVIEW_CHARS else "")
    papers = list(dict.fromkeys(c.arxiv_id for c in finding.citations if c.arxiv_id))
    line = f"- from {finding.agent} (grounded: {'yes' if finding.grounded else 'no'}): {preview}"
    if papers:
        line += f"\n  Papers cited: {', '.join(papers)}"
    if web := [f"{c.id} {c.url}" if c.url else c.id for c in finding.citations if c.source == "web"]:
        line += f"\n  Web sources cited: {', '.join(web)}"  # URLs, so a later worker can read them
    return line


def _failures(state: OrchestratorState) -> dict[str, int]:
    """How many times each worker failed (guard errors) this turn."""
    counts: dict[str, int] = {}
    for error in state.get("errors", []):
        if error.kind in FAILURE_KINDS:
            counts[error.agent] = counts.get(error.agent, 0) + 1
    return counts


def _error_summary(state: OrchestratorState) -> str:
    """One compact line per error, e.g. "- research failed: transport (lost its connection to a tool server)"."""
    failures, retries = _failures(state), max_worker_retries()
    lines = []
    for e in state.get("errors", []):
        verb = "failed" if e.kind in FAILURE_KINDS else "-"
        line = f"- {e.agent} {verb}: {e.kind} ({ERROR_SUMMARY.get(e.kind, 'see errors')})"
        if e.kind in FAILURE_KINDS:
            line += f"; retries left: {max(0, retries + 1 - failures[e.agent])}"
        lines.append(line)
    return "\n".join(dict.fromkeys(lines))


def _unavailable_message(worker: str) -> str:
    spec = AGENTS.get(worker)
    capability = f" ({spec.description})" if spec else ""
    return f"The {worker} worker is not available, so its capability{capability} could not be used."


def make_supervisor(
    llm: BaseChatModel | Models | None, worker_names: Collection[str]
) -> Callable[[OrchestratorState], Awaitable[dict]]:
    """Build the supervisor node. `worker_names` are the workers wired into the graph.

    `llm` is one model for every tier (tests) or a Models resolver; None means the configured providers.
    """
    # Routing is a short classification: low thinking measured 2.5 s median vs 3.2 s, same decisions.
    models = resolve_models(llm, thinking_level="low")
    deciders: dict[Tier, Runnable] = {}

    def decider(tier: Tier) -> Runnable:
        if tier not in deciders:
            deciders[tier] = models(tier).with_structured_output(SupervisorDecision).with_config(tags=["supervisor"])
        return deciders[tier]

    system_prompt = (PROMPTS_DIR / "supervisor.md").read_text().replace("{workers}", _roster(worker_names))

    async def supervisor(state: OrchestratorState) -> dict:
        step = state.get("step", 0) + 1
        max_steps = max_supervisor_steps()
        start = time.perf_counter()
        emit_trace("supervisor", "enter", step=step)

        failures, retries = _failures(state), max_worker_retries()
        if worker_names and all(failures.get(w, 0) > retries for w in worker_names):
            # Nothing left to dispatch: skip the model call.
            task_type, reason = state.get("task_type") or "general", "every worker failed past its retries"
            emit_trace("supervisor", "decision", step=step, detail=f"synthesize: {reason}",
                       decision={"next": "synthesize", "instruction": "", "task_type": task_type, "reason": reason})
            emit_trace("supervisor", "exit", step=step, detail="next=synthesize", ms=round((time.perf_counter() - start) * 1000))
            return {"step": step, "next": "synthesize", "instruction": "", "task_type": task_type}

        routing = [route("supervisor", RouteFeatures(task_type=state.get("task_type") or "general"))]
        emit_routing(routing[0], step=step)

        def switched(fallback: RoutingDecision) -> None:
            routing.append(fallback)
            emit_routing(fallback, step=step)

        messages = [SystemMessage(system_prompt), HumanMessage(_context(state, step, max_steps))]
        decision: SupervisorDecision | None = None
        failure_kind, failure = "", ""
        try:
            raw, _ = await invoke_with_fallback(routing[0], lambda: decider(routing[0].tier).ainvoke(messages),
                                                lambda: decider(FALLBACK[routing[0].tier]).ainvoke(messages), switched,
                                                # Routing takes 2-5 s; a stalled provider call (one took 53 s) moves on.
                                                timeout=supervisor_timeout_s())
            decision = raw if isinstance(raw, SupervisorDecision) else SupervisorDecision.model_validate(raw)
        except ValueError as exc:  # OutputParserException and pydantic's ValidationError: unreadable output
            failure_kind, failure = "supervisor_parse_error", type(exc).__name__
            logger.warning("supervisor structured output could not be read: %s", type(exc).__name__)
        except Exception as exc:  # the model call itself failed: provider error, timeout, network
            failure_kind, failure = "supervisor_error", type(exc).__name__
            logger.exception("supervisor model call failed")

        if decision is not None:
            emit_trace("supervisor", "decision", step=step,
                       detail=f"{decision.next} ({decision.task_type}): {decision.reason}",
                       decision=decision.model_dump())

        error: AgentError | None = None
        if decision is None:
            message = ("The supervisor's decision could not be read" if failure_kind == "supervisor_parse_error"
                       else "The supervisor's model call failed")
            error = AgentError(agent="supervisor", kind=failure_kind, message=f"{message} ({failure}).")
        elif decision.next != "synthesize":
            dispatches = state.get("dispatches", [])
            if step >= max_steps:
                error = AgentError(agent="supervisor", kind="step_limit",
                                   message=f"Reached the limit of {max_steps} supervisor steps; answering with what was gathered.")
            elif decision.next not in worker_names:
                error = AgentError(agent=decision.next, kind="worker_unavailable", message=_unavailable_message(decision.next))
            elif failures.get(decision.next) and (
                failures[decision.next] > retries
                or any(d.worker == decision.next and _normalize(d.instruction) == _normalize(decision.instruction)
                       for d in dispatches)
            ):
                error = AgentError(agent=decision.next, kind="retry_limit",
                                   message=f"{decision.next} already failed this turn and was not retried "
                                           "(retry limit reached, or the same request was already tried).")
            elif dispatches and dispatches[-1].worker == decision.next and (
                _normalize(dispatches[-1].instruction) == _normalize(decision.instruction)
            ):
                error = AgentError(agent=decision.next, kind="repeated_dispatch",
                                   message=f"The same request was just sent to {decision.next}; not repeating it.")

        update: dict = {"step": step, "task_type": decision.task_type if decision else "general", "routing": routing}
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
