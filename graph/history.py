"""load_history / save_history nodes: the session's earlier turns in, this turn out.

load_history puts the last HISTORY_TURNS turns in front of the current question in `messages` (as
human/AI pairs); save_history stores the turn after synthesize. A store failure never fails the turn:
loading falls back to no history, and a failed save is logged and traced.
"""

import logging
import time
from collections.abc import Awaitable, Callable

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from config import history_turns
from graph.state import AgentError, CitationReport, OrchestratorState
from graph.trace import emit_trace
from rag.citations import parse_citation_ids
from state_store import History

logger = logging.getLogger("graph.history")

HISTORY_UNAVAILABLE = "Earlier turns of this conversation couldn't be loaded, so this answer doesn't use them."
# How much of each earlier answer the supervisor and synthesize see; cited IDs are always listed in full.
EARLIER_ANSWER_CHARS = 1500


def _current_question(state: OrchestratorState) -> AnyMessage | None:
    return next((m for m in reversed(state.get("messages", [])) if m.type == "human"), None)


def earlier_turns(state: OrchestratorState) -> list[tuple[str, str]]:
    """(question, answer) pairs for the turns before the current question, oldest first."""
    messages = state.get("messages", [])
    last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=0)
    pairs, question = [], None
    for message in messages[:last_human]:
        if message.type == "human":
            question = message.text
        elif message.type == "ai" and question is not None:
            pairs.append((question, message.text))
            question = None
    return pairs


def format_earlier_turns(state: OrchestratorState) -> str:
    """The earlier turns as a numbered block, each answer cut to EARLIER_ANSWER_CHARS plus its cited IDs."""
    blocks = []
    for i, (question, answer) in enumerate(earlier_turns(state), 1):
        cited = parse_citation_ids(answer)
        text = answer[:EARLIER_ANSWER_CHARS] + (" [...]" if len(answer) > EARLIER_ANSWER_CHARS else "")
        block = f"Turn {i}. User: {question}\nAssistant: {text}"
        if cited:
            block += f"\nIDs cited in that answer, in order: {', '.join(cited)}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _history_messages(turns, question: AnyMessage) -> list[AnyMessage]:
    messages: list[AnyMessage] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
    for turn in turns:
        messages += [HumanMessage(turn.question), AIMessage(turn.answer)]
    return [*messages, question]


def make_load_history(history: History) -> Callable[[OrchestratorState], Awaitable[dict]]:
    async def load_history(state: OrchestratorState) -> dict:
        start = time.perf_counter()
        emit_trace("load_history", "enter")
        question = _current_question(state)
        try:
            turns = await history.load(state["session_id"], history_turns())
        except Exception as exc:
            logger.warning("loading history failed for session_id=%s: %s: %s", state["session_id"],
                           type(exc).__name__, exc)
            emit_trace("load_history", "error", detail=f"history unavailable: {type(exc).__name__}")
            return {"errors": [AgentError(agent="history", kind="history_unavailable", message=HISTORY_UNAVAILABLE)]}
        emit_trace("load_history", "exit", detail=f"{len(turns)} earlier turn{'s' * (len(turns) != 1)}",
                   ms=round((time.perf_counter() - start) * 1000))
        if not turns or question is None:
            return {}
        return {"messages": _history_messages(turns, question)}

    return load_history


def make_save_history(history: History) -> Callable[[OrchestratorState], Awaitable[dict]]:
    async def save_history(state: OrchestratorState) -> dict:
        start = time.perf_counter()
        emit_trace("save_history", "enter")
        if any(e.kind == "synthesize_error" for e in state.get("errors", [])):
            emit_trace("save_history", "exit", detail="not saved: the answer failed")
            return {}
        answer = state["messages"][-1].text if state.get("messages") else ""
        report = state.get("answer_citations") or CitationReport()
        question = _current_question(state)
        try:
            turn = await history.append(state["session_id"], question.text if question else "", answer, report.valid)
        except Exception as exc:
            logger.warning("saving history failed for session_id=%s: %s: %s", state["session_id"],
                           type(exc).__name__, exc)
            emit_trace("save_history", "error", detail=f"turn not saved: {type(exc).__name__}")
            return {}
        emit_trace("save_history", "exit", detail=f"saved turn {turn.n}", ms=round((time.perf_counter() - start) * 1000))
        return {}

    return save_history
