"""OrchestratorState plus Finding, Citation, RoutingDecision, AgentError, Dispatch and SupervisorDecision."""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

TaskType = Literal["lookup", "compare", "lit_review", "metadata", "current_events", "general"]


class Citation(BaseModel):
    id: str  # a chunk ID like "2308.04079#method-03", or "web:N"
    source: Literal["corpus", "web"]
    arxiv_id: str | None = None
    title: str | None = None
    section: str | None = None
    page: int | None = None
    url: str | None = None


class Finding(BaseModel):
    agent: str
    content: str
    citations: list[Citation] = []


class RoutingDecision(BaseModel):
    call_site: str
    agent: str
    provider: str
    model: str
    tier: str
    dimension: str
    reason: str
    fallback: bool = False


class AgentError(BaseModel):
    agent: str
    kind: str  # e.g. supervisor_parse_error, step_limit, worker_unavailable, repeated_dispatch
    message: str


class Dispatch(BaseModel):
    worker: str
    instruction: str


class SupervisorDecision(BaseModel):
    next: Literal["research", "data", "summary", "synthesize"] = Field(
        description="The worker to run next, or 'synthesize' to write the final answer now."
    )
    instruction: str = Field(description="What the chosen worker should do; empty when next is 'synthesize'.")
    task_type: TaskType = Field(description="The kind of question being answered.")
    reason: str = Field(description="One line explaining the choice.")


class OrchestratorState(TypedDict):
    session_id: str
    messages: Annotated[list[AnyMessage], add_messages]  # history + this turn
    next: str
    instruction: str
    task_type: str
    step: int
    findings: Annotated[list[Finding], operator.add]     # agent, content, citations
    retrieved_chunk_ids: Annotated[list[str], operator.add]
    routing: Annotated[list[RoutingDecision], operator.add]
    errors: Annotated[list[AgentError], operator.add]
    dispatches: Annotated[list[Dispatch], operator.add]  # worker + instruction, in order
