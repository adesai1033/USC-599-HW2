"""Worker subgraph factory: worker_agent <-> worker_tools loop, ending in one Finding.

  START -> worker_agent --(tool calls?)--> worker_tools -> worker_agent ... -> END

worker_agent calls the model with the agent's tools bound. worker_tools runs them through
ToolNode(handle_tool_errors=True) and emits tool_call / tool_result trace events. After
WORKER_MAX_TOOL_ROUNDS rounds, the model is called once more without tools to summarize.

The parent-graph node maps the subgraph's result to a Finding and to retrieved_chunk_ids (every chunk
any tool returned). Citations are validated against the sources this run's tools returned: an ID
that no tool returned is replaced with [unverified] and reported as an invalid_citation error, and
the finding is grounded only if at least one valid citation remains.
"""

import json
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import date
from typing import Annotated, TypedDict
from urllib.parse import urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agents.registry import AgentSpec
from config import worker_max_tool_rounds
from graph.state import AgentError, Citation, Finding, OrchestratorState
from graph.supervisor import latest_question
from graph.trace import emit_trace
from rag.citations import attribution_mismatches, mark_unverified, validate

WorkerNode = Callable[[OrchestratorState], Awaitable[dict]]

SUMMARIZE_NOW = (
    "You have used all of your tool calls. Write your findings now from the tool results above, "
    "citing only IDs that appear in them. Do not call any more tools."
)
CORPUS_TOOLS = {"search_corpus", "get_chunks"}
FIX_ATTRIBUTION = (
    "Some claims name a paper but cite a source that never mentions it:\n{problems}\n\n"
    "Rewrite your findings in full. For each of these claims, cite a tool result above that is about the paper "
    "it names, or remove the claim. Keep everything else as it was. Do not call any tools."
)


class WorkerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    instruction: str
    rounds: int
    step: int | None  # the supervisor step that dispatched this worker, for trace events


def tool_text(message: ToolMessage) -> str:
    """A tool result's text, whether the content is a string or a list of content blocks."""
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content if isinstance(b, dict) and b.get("type") == "text")


def _payload(message: ToolMessage) -> dict:
    if message.status == "error":
        return {}
    try:
        payload = json.loads(tool_text(message))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _chunks(message: ToolMessage) -> list[dict]:
    """Chunk records in a search_corpus or get_chunks result."""
    if message.name not in CORPUS_TOOLS:
        return []
    payload = _payload(message)
    records = payload.get("results") if message.name == "search_corpus" else payload.get("chunks")
    return [r for r in records or [] if isinstance(r, dict) and isinstance(r.get("chunk_id"), str)]


def _web_results(message: ToolMessage) -> list[dict]:
    """Normalized Tavily results (each with a source_id)."""
    results = _payload(message).get("results") if message.name and message.name.startswith("tavily_") else None
    return [r for r in results or [] if isinstance(r, dict) and isinstance(r.get("source_id"), str)]


def summarize_args(args: dict) -> str:
    text = json.dumps(args, ensure_ascii=False)
    return text if len(text) <= 120 else text[:117] + "..."


def summarize_result(message: ToolMessage) -> str:
    """One line for the trace, e.g. "6 chunks: 2308.04079#method-03, ..."."""
    if message.status == "error":
        return "error: " + tool_text(message)[:150]
    if chunks := _chunks(message):
        ids = [c["chunk_id"] for c in chunks]
        return f"{len(ids)} chunks: " + ", ".join(ids[:3]) + (", ..." if len(ids) > 3 else "")
    if web := _web_results(message):
        listed = [f"{r['source_id']} ({urlsplit(r['url']).netloc})" for r in web[:3]]
        return f"{len(web)} results: " + ", ".join(listed) + (", ..." if len(web) > 3 else "")
    payload = _payload(message)
    if "papers" in payload:
        missing = payload.get("missing") or []
        return f"{len(payload['papers'])} papers" + (f", missing {missing}" if missing else "")
    if "results" in payload or "chunks" in payload:
        return "0 results" + (f" ({payload['note']})" if payload.get("note") else "")
    return f"{len(tool_text(message))} chars"


def allowed_citations(messages: list[AnyMessage]) -> dict[str, Citation]:
    """Every source the tools returned in this run, as the citation it would become, keyed by ID."""
    allowed: dict[str, Citation] = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        for c in _chunks(message):
            arxiv_id = c.get("arxiv_id") or c["chunk_id"].split("#", 1)[0]
            allowed[c["chunk_id"]] = Citation(
                id=c["chunk_id"], source="corpus", arxiv_id=arxiv_id, title=c.get("title"), section=c.get("section"),
                page=c.get("page"), url=c.get("url") or f"https://arxiv.org/abs/{arxiv_id}",
            )
        for r in _web_results(message):
            allowed[r["source_id"]] = Citation(id=r["source_id"], source="web", title=r.get("title") or None, url=r.get("url"))
    return allowed


def source_texts(messages: list[AnyMessage]) -> dict[str, str]:
    """For each source returned in this run, the text that identifies it, for attribution checks."""
    texts: dict[str, str] = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            for c in _chunks(message):
                texts[c["chunk_id"]] = f"{c.get('arxiv_id', '')} {c.get('title', '')}"
            for r in _web_results(message):
                texts[r["source_id"]] = f"{r['url']} {r.get('title', '')} {r.get('content', '')}"
    return texts


def _describe(mismatches: list[tuple[str, list[str], list[str]]]) -> str:
    return "\n".join(f"- \"{claim[:200]}\" names {', '.join(ids)} but cites {', '.join(cited)}"
                     for claim, cited, ids in mismatches)


def retrieved_chunk_ids(messages: list[AnyMessage]) -> list[str]:
    """Every chunk ID returned by search_corpus or get_chunks in this run, cited or not, in order."""
    ids: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            ids += [c["chunk_id"] for c in _chunks(message) if c["chunk_id"] not in ids]
    return ids


def make_worker(
    spec: AgentSpec,
    tools: list[BaseTool] | Callable[[], list[BaseTool]],
    llm: BaseChatModel,
    *,
    unavailable_notes: Iterable[str] | Callable[[], Iterable[str]] = (),
) -> WorkerNode:
    """Build the parent-graph node for one worker.

    `tools` and `unavailable_notes` may be callables. They are then read at the start of every run, so
    a server that went down (or came back) since the graph was built is reflected in the next run.
    """
    current_tools = tools if callable(tools) else (lambda: tools)
    current_notes = unavailable_notes if callable(unavailable_notes) else (lambda: unavailable_notes)
    base_prompt = spec.prompt_file.read_text()

    def build_subgraph(run_tools: list[BaseTool], system_prompt: str):
        bound = llm.bind_tools(run_tools) if run_tools else llm
        tool_node = ToolNode(run_tools, handle_tool_errors=True)

        async def worker_agent(state: WorkerState) -> dict:
            messages = [SystemMessage(system_prompt), *state["messages"]]
            max_rounds = worker_max_tool_rounds()
            if state["rounds"] >= max_rounds:
                emit_trace(spec.name, "decision", step=state["step"], agent=spec.name,
                           detail=f"tool round cap ({max_rounds}) reached; summarizing without tools")
                return {"messages": [await llm.ainvoke([*messages, HumanMessage(SUMMARIZE_NOW)])]}
            return {"messages": [await bound.ainvoke(messages)]}

        async def worker_tools(state: WorkerState, config: RunnableConfig) -> dict:
            calls = state["messages"][-1].tool_calls
            for call in calls:
                emit_trace(spec.name, "tool_call", step=state["step"], agent=spec.name,
                           detail=f"{call['name']}({summarize_args(call['args'])})")
            start = time.perf_counter()
            update = await tool_node.ainvoke(state, config)
            ms = round((time.perf_counter() - start) * 1000)
            for message in update["messages"]:
                emit_trace(spec.name, "tool_result", step=state["step"], agent=spec.name,
                           detail=f"{message.name}: {summarize_result(message)}", ms=ms)
            return {"messages": update["messages"], "rounds": state["rounds"] + 1}

        graph = StateGraph(WorkerState)
        graph.add_node("worker_agent", worker_agent)
        graph.add_node("worker_tools", worker_tools)
        graph.add_edge(START, "worker_agent")
        graph.add_conditional_edges("worker_agent", tools_condition, {"tools": "worker_tools", END: END})
        graph.add_edge("worker_tools", "worker_agent")  # the loop-back: the model sees every tool result
        return graph.compile(name=spec.name)

    async def worker(state: OrchestratorState) -> dict:
        system_prompt = base_prompt
        if notes := list(current_notes()):
            system_prompt += "\n\n## Unavailable right now\n\n" + "\n".join(f"- {n}" for n in notes)
        task = (
            f"Instruction from the supervisor: {state.get('instruction', '')}\n\n"
            f"Original user question: {latest_question(state)}\n\n"
            f"Task type: {state.get('task_type') or 'general'}\n\n"
            f"Today's date: {date.today().isoformat()}"  # so "this month" means something
        )
        result = await build_subgraph(list(current_tools()), system_prompt).ainvoke(
            {"messages": [HumanMessage(task)], "instruction": state.get("instruction", ""), "rounds": 0,
             "step": state.get("step")},
            {"recursion_limit": 2 * worker_max_tool_rounds() + 4},
        )
        messages = result["messages"]
        final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), AIMessage(""))
        content = final.text
        errors: list[AgentError] = []

        # A claim that names one paper but cites another's source is wrong even when the ID was
        # retrieved, so the retrieval check below can't catch it. Give the model one chance to fix it.
        sources = source_texts(messages)
        if mismatches := attribution_mismatches(content, sources):
            emit_trace(spec.name, "error", step=state.get("step"), agent=spec.name,
                       detail=f"attribution check: {len(mismatches)} claim(s) cite a source that doesn't mention "
                              "the paper they name; asking for a fix")
            fix = HumanMessage(FIX_ATTRIBUTION.format(problems=_describe(mismatches)))
            revised = (await llm.ainvoke([SystemMessage(system_prompt), *messages, fix])).text
            content = revised or content
            if mismatches := attribution_mismatches(content, sources):
                emit_trace(spec.name, "error", step=state.get("step"), agent=spec.name,
                           detail=f"attribution check: {len(mismatches)} claim(s) still mismatched")
                errors.append(AgentError(
                    agent=spec.name, kind="attribution_mismatch",
                    message="Some findings cite a source that doesn't mention the paper they name; don't use "
                            "those citations for those claims:\n" + _describe(mismatches),
                ))

        # Only IDs that this run's tools returned may be cited. Others become [unverified].
        report = validate(content, allowed_citations(messages))
        if report.invalid:
            content = mark_unverified(content, set(report.invalid))
            emit_trace(spec.name, "error", step=state.get("step"), agent=spec.name,
                       detail=f"invalid citations replaced with [unverified]: {', '.join(report.invalid)}")
            errors.append(AgentError(
                agent=spec.name, kind="invalid_citation",
                message=f"The {spec.name} worker cited sources it never retrieved; they are marked [unverified]: "
                        + ", ".join(report.invalid),
            ))

        update: dict = {
            "findings": [Finding(agent=spec.name, content=content, citations=report.valid, grounded=bool(report.valid))],
            "retrieved_chunk_ids": retrieved_chunk_ids(messages),
        }
        return update | ({"errors": errors} if errors else {})

    return worker
