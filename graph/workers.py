"""Worker subgraph factory: worker_agent <-> worker_tools loop, ending in one Finding.

  START -> worker_agent --(tool calls?)--> worker_tools -> worker_agent ... -> END

worker_agent calls the model with the agent's tools bound. worker_tools runs them through
ToolNode(handle_tool_errors=True) and emits tool_call / tool_result trace events, then appends the
remaining tool-round budget to the history. After WORKER_MAX_TOOL_ROUNDS rounds, the model is called
once more without tools to summarize. The history is append-only throughout.

Every model call is routed (router.route() over the task type, SQL errors so far and papers in the
instruction). A decision is emitted when the run starts and whenever the tier changes, e.g. data
escalating after a failed run_sql. A provider error retries on the fallback tier, and a tier that
failed stays on the fallback for the rest of the run.

The parent-graph node maps the subgraph's result to a Finding and to retrieved_chunk_ids (every chunk
any tool returned). Citations are validated against the sources this run's tools returned: the model
gets one chance to replace an ID that no tool returned, then any left is replaced with [unverified]
and reported as an invalid_citation error, and
the finding is grounded if at least one valid citation remains or a run_sql query returned rows.
Citable sources are chunks (search_corpus, get_chunks), web results (Tavily) and database rows
(db:<arxiv_id> for every arXiv ID in a run_sql or get_paper_metadata result).
"""

import json
import operator
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import date
from typing import Annotated, TypedDict
from urllib.parse import urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agents.registry import AgentSpec
from config import worker_max_tool_rounds
from graph.state import AgentError, Citation, Finding, OrchestratorState, RoutingDecision
from graph.supervisor import latest_question
from graph.trace import emit_routing, emit_trace
from llm import Models, Tier, resolve_models
from rag.citations import attribution_mismatches, mark_unverified, validate
from router import RouteFeatures, invoke_with_fallback, papers_in_scope, route, sql_errored

WorkerNode = Callable[[OrchestratorState], Awaitable[dict]]

SUMMARIZE_NOW = (
    "You have used all of your tool calls. Write your findings now from the tool results above, "
    "citing only IDs that appear in them. Do not call any more tools."
)
CORPUS_TOOLS = {"search_corpus", "get_chunks"}
ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}")
FIX_ATTRIBUTION = (
    "Some claims name a paper but cite a source that never mentions it:\n{problems}\n\n"
    "Rewrite your findings in full. For each of these claims, cite a tool result above that is about the paper "
    "it names, or remove the claim. Keep everything else as it was. Do not call any tools."
)
FIX_CITATIONS = (
    "These citation IDs were not returned by any tool in this run: {ids}.\n\n"
    "Rewrite your findings in full. Replace each with an ID that appears in the tool results above and supports "
    "the claim, or remove the claim. Keep everything else as it was. Do not call any tools."
)


class WorkerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    instruction: str
    task_type: str
    rounds: int
    step: int | None  # the supervisor step that dispatched this worker, for trace events
    routing: Annotated[list[RoutingDecision], operator.add]


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


def _sql_result(message: ToolMessage) -> dict | None:
    """A successful run_sql payload ({sql, columns, rows, row_count, truncated}), or None."""
    payload = _payload(message) if message.name == "run_sql" else {}
    return payload if isinstance(payload.get("rows"), list) and isinstance(payload.get("sql"), str) else None


def _db_papers(message: ToolMessage) -> dict[str, str | None]:
    """arXiv IDs (with a title when the result has one) in a run_sql or get_paper_metadata result."""
    papers: dict[str, str | None] = {}
    if message.name == "get_paper_metadata":
        for paper in _payload(message).get("papers") or []:
            if isinstance(paper, dict) and isinstance(paper.get("arxiv_id"), str):
                papers[paper["arxiv_id"]] = paper.get("title")
    elif (result := _sql_result(message)) is not None:
        columns = [str(c).lower() for c in result.get("columns") or []]
        title_at = columns.index("title") if "title" in columns else None
        for row in result["rows"]:
            title = row[title_at] if title_at is not None and title_at < len(row) else None
            for cell in row:
                if isinstance(cell, str) and ARXIV_ID.fullmatch(cell):
                    papers[cell] = papers.get(cell) or (title if isinstance(title, str) else None)
    return papers


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
    if (result := _sql_result(message)) is not None:
        return f"{result.get('row_count', len(result['rows']))} rows" + (" (truncated)" if result.get("truncated") else "")
    payload = _payload(message)
    if isinstance(payload.get("tables"), list):
        return f"schema: {len(payload['tables'])} tables"
    if "papers" in payload:
        missing = payload.get("missing") or []
        return f"{len(payload['papers'])} papers" + (f", missing {missing}" if missing else "")
    if "results" in payload or "chunks" in payload:
        return "0 results" + (f" ({payload['note']})" if payload.get("note") else "")
    return f"{len(tool_text(message))} chars"


def round_budget(used: int, max_rounds: int) -> str:
    """The tool-round budget: appended to the task and after every tool round, never to the system prompt.

    The history stays append-only, so the prompt prefix a reply was made from never changes. Claude
    rejects a replayed thinking block whose prefix (system prompt included) differs from the original.
    """
    left = max_rounds - used
    if left <= 0:
        return f"## Tool budget\n\nYou have used all {max_rounds} rounds of tool calls. Write your findings now."
    last = " This is your last round: make every call you still need in this one response." if left == 1 else ""
    return (f"## Tool budget\n\nRounds of tool calls used: {used} of {max_rounds} ({left} left). Several calls in one "
            f"response count as one round. Once you have evidence for every part of the instruction, stop calling "
            f"tools and write your findings.{last}")


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
        for arxiv_id, title in _db_papers(message).items():
            existing = allowed.get(f"db:{arxiv_id}")
            allowed[f"db:{arxiv_id}"] = Citation(
                id=f"db:{arxiv_id}", source="db", arxiv_id=arxiv_id, title=title or (existing.title if existing else None),
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
    return allowed


def executed_queries(messages: list[AnyMessage]) -> tuple[list[str], bool]:
    """The SQL run_sql executed successfully in this run, and whether any of it returned rows."""
    queries, any_rows = [], False
    for message in messages:
        if isinstance(message, ToolMessage) and (result := _sql_result(message)) is not None:
            queries.append(result["sql"])
            any_rows = any_rows or bool(result["rows"])
    return queries, any_rows


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


def failed_tiers(routing: list[RoutingDecision]) -> set[str]:
    """Tiers that failed with a provider error this run (each fallback decision follows the one that failed)."""
    return {routing[i - 1].tier for i, d in enumerate(routing) if d.fallback and i > 0}


def make_worker(
    spec: AgentSpec,
    tools: list[BaseTool] | Callable[[], list[BaseTool]],
    llm: BaseChatModel | Models,
    *,
    unavailable_notes: Iterable[str] | Callable[[], Iterable[str]] = (),
) -> WorkerNode:
    """Build the parent-graph node for one worker.

    `llm` is one model for every tier (tests) or a Models resolver that the router picks from per call.
    `tools` and `unavailable_notes` may be callables. They are then read at the start of every run, so
    a server that went down (or came back) since the graph was built is reflected in the next run.
    """
    current_tools = tools if callable(tools) else (lambda: tools)
    models = resolve_models(llm)

    def max_tool_rounds() -> int:
        return spec.max_tool_rounds or worker_max_tool_rounds()

    current_notes = unavailable_notes if callable(unavailable_notes) else (lambda: unavailable_notes)
    base_prompt = spec.prompt_file.read_text()

    async def routed_call(decision: RoutingDecision, step: int | None, previous: list[RoutingDecision],
                          call: Callable[[Tier], Awaitable[AIMessage]]) -> tuple[AIMessage, list[RoutingDecision]]:
        """Run one model call on the decision's tier, with fallback. Returns the reply and the new decisions."""
        if decision.tier in failed_tiers(previous):
            return await call("fallback"), []  # this tier already failed this run; don't wait for it again
        new: list[RoutingDecision] = []
        last = next((d for d in reversed(previous) if not d.fallback), None)
        if last is None or last.tier != decision.tier:
            new.append(decision)
            emit_routing(decision, step=step)

        def switched(fallback: RoutingDecision) -> None:
            new.append(fallback)
            emit_routing(fallback, step=step)

        reply, _ = await invoke_with_fallback(decision, lambda: call(decision.tier), lambda: call("fallback"), switched)
        return reply, new

    def build_subgraph(run_tools: list[BaseTool], system_prompt: str):
        bound_models: dict[Tier, Runnable] = {}

        def bound(tier: Tier) -> Runnable:
            if tier not in bound_models:
                bound_models[tier] = models(tier).bind_tools(run_tools) if run_tools else models(tier)
            return bound_models[tier]

        tool_node = ToolNode(run_tools, handle_tool_errors=True)

        async def worker_agent(state: WorkerState) -> dict:
            max_rounds = max_tool_rounds()
            messages = [SystemMessage(system_prompt), *state["messages"]]
            decision = route(spec.name, RouteFeatures(task_type=state["task_type"], sql_errored=sql_errored(state["messages"]),
                                                      papers_in_scope=papers_in_scope(state["instruction"])))
            if state["rounds"] >= max_rounds:
                emit_trace(spec.name, "decision", step=state["step"], agent=spec.name,
                           detail=f"tool round cap ({max_rounds}) reached; summarizing without tools")
                reply, new = await routed_call(decision, state["step"], state["routing"], lambda tier: models(tier).ainvoke(
                    [*messages, HumanMessage(SUMMARIZE_NOW)]))
            else:
                reply, new = await routed_call(decision, state["step"], state["routing"],
                                               lambda tier: bound(tier).ainvoke(messages))
            return {"messages": [reply], "routing": new}

        async def worker_tools(state: WorkerState, config: RunnableConfig) -> dict:
            calls = state["messages"][-1].tool_calls
            for call in calls:
                emit_trace(spec.name, "tool_call", step=state["step"], agent=spec.name,
                           detail=f"{call['name']}({summarize_args(call['args'])})")
            start = time.perf_counter()
            update = await tool_node.ainvoke(state, config)
            ms = round((time.perf_counter() - start) * 1000)
            # The tools in one round run together, so the round's time goes on its first result only.
            for i, message in enumerate(update["messages"]):
                emit_trace(spec.name, "tool_result", step=state["step"], agent=spec.name,
                           detail=f"{message.name}: {summarize_result(message)}", ms=ms if i == 0 else None)
            # The model doesn't stop on its own; telling it the budget lets it plan its last searches.
            budget = HumanMessage(round_budget(state["rounds"] + 1, max_tool_rounds()))
            return {"messages": [*update["messages"], budget], "rounds": state["rounds"] + 1}

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
            f"Today's date: {date.today().isoformat()}\n\n"  # so "this month" means something
            f"{round_budget(0, max_tool_rounds())}"
        )
        result = await build_subgraph(list(current_tools()), system_prompt).ainvoke(
            {"messages": [HumanMessage(task)], "instruction": state.get("instruction", ""),
             "task_type": state.get("task_type") or "general", "rounds": 0, "step": state.get("step"), "routing": []},
            {"recursion_limit": 2 * max_tool_rounds() + 4},
        )
        messages = result["messages"]
        routing: list[RoutingDecision] = list(result["routing"])
        final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), AIMessage(""))
        content = final.text
        errors: list[AgentError] = []

        history = [SystemMessage(system_prompt), *messages]

        async def ask_to_fix(request: str) -> str:
            """One rewrite of the findings on the run's last routed tier (or its fallback, if that tier failed)."""
            nonlocal routing
            decision = next(d for d in reversed(routing) if not d.fallback)
            history.append(HumanMessage(request))
            reply, new = await routed_call(decision, state.get("step"), routing, lambda t: models(t).ainvoke(history))
            history.append(reply)
            routing = routing + new
            return reply.text

        # A claim that names one paper but cites another's source is wrong even when the ID was
        # retrieved, so the retrieval check below can't catch it. Give the model one chance to fix it.
        sources = source_texts(messages)
        if mismatches := attribution_mismatches(content, sources):
            emit_trace(spec.name, "error", step=state.get("step"), agent=spec.name,
                       detail=f"attribution check: {len(mismatches)} claim(s) cite a source that doesn't mention "
                              "the paper they name; asking for a fix")
            content = await ask_to_fix(FIX_ATTRIBUTION.format(problems=_describe(mismatches))) or content
            if mismatches := attribution_mismatches(content, sources):
                emit_trace(spec.name, "error", step=state.get("step"), agent=spec.name,
                           detail=f"attribution check: {len(mismatches)} claim(s) still mismatched")
                errors.append(AgentError(
                    agent=spec.name, kind="attribution_mismatch",
                    message="Some findings cite a source that doesn't mention the paper they name; don't use "
                            "those citations for those claims:\n" + _describe(mismatches),
                ))

        # Only IDs that this run's tools returned may be cited. The model gets one chance to replace
        # others (usually a neighboring chunk ID it guessed); what remains becomes [unverified].
        allowed = allowed_citations(messages)
        report = validate(content, allowed)
        if report.invalid:
            emit_trace(spec.name, "error", step=state.get("step"), agent=spec.name,
                       detail=f"citations no tool returned: {', '.join(report.invalid)}; asking for a fix")
            content = await ask_to_fix(FIX_CITATIONS.format(ids=", ".join(report.invalid))) or content
            report = validate(content, allowed)
        if report.invalid:
            content = mark_unverified(content, set(report.invalid))
            emit_trace(spec.name, "error", step=state.get("step"), agent=spec.name,
                       detail=f"invalid citations replaced with [unverified]: {', '.join(report.invalid)}")
            errors.append(AgentError(
                agent=spec.name, kind="invalid_citation",
                message=f"The {spec.name} worker cited sources it never retrieved; they are marked [unverified]: "
                        + ", ".join(report.invalid),
            ))

        # Grounded: a verified citation survived, or a query returned rows (an aggregate cites no single paper).
        queries, any_rows = executed_queries(messages)
        update: dict = {
            "findings": [Finding(agent=spec.name, content=content, citations=report.valid,
                                 grounded=bool(report.valid) or any_rows, queries=queries)],
            "retrieved_chunk_ids": retrieved_chunk_ids(messages),
            "routing": routing,
        }
        return update | ({"errors": errors} if errors else {})

    return worker
