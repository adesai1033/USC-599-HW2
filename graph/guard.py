"""Worker failure wrapper: a worker that raises or runs too long becomes an AgentError, never a crashed graph.

guarded(name, node_fn) runs the worker under WORKER_TIMEOUT_S and catches Exception (never
CancelledError or other BaseExceptions). The failure is logged with its traceback, classified, and
returned as a state update holding one AgentError, no finding, and the routing decisions the worker
emitted before it failed, so the graph continues to the supervisor. Kinds:
  worker_timeout    the worker ran past WORKER_TIMEOUT_S
  transport         an MCP connection or session error
  provider_error    an LLM API error (auth, rate limit, 5xx, bad request)
  worker_exception  anything else
The message is fixed per kind: no tracebacks, URLs or provider response bodies.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import anyio
from langchain_core.exceptions import ModelError
from mcp.shared.exceptions import McpError

from config import worker_timeout_s
from graph.state import AgentError, OrchestratorState
from graph.trace import emit_trace, recording

logger = logging.getLogger("graph.guard")

WorkerNode = Callable[[OrchestratorState], Awaitable[dict]]

TRANSPORT_ERRORS = (McpError, ConnectionError, EOFError, anyio.ClosedResourceError, anyio.BrokenResourceError,
                    anyio.EndOfStream)
# Provider SDK errors that reach us unwrapped; LangChain's own wrappers subclass ModelError.
PROVIDER_MODULES = ("google.genai", "google.api_core", "openai", "anthropic", "langchain_google_genai",
                    "langchain_openai", "langchain_anthropic")

MESSAGES = {
    "worker_timeout": "The {name} worker took longer than {timeout:.0f} s and was stopped.",
    "transport": "The {name} worker lost its connection to a tool server.",
    "provider_error": "The {name} worker's language model provider returned an error.",
    "worker_exception": "The {name} worker failed unexpectedly.",
}


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception, its causes and contexts, and the members of any exception group."""
    seen, stack = [], [exc]
    while stack:
        current = stack.pop()
        if current is None or any(current is s for s in seen):
            continue
        seen.append(current)
        stack += [current.__cause__, current.__context__]
        if isinstance(current, BaseExceptionGroup):
            stack += list(current.exceptions)
    return seen


def is_provider_error(exc: BaseException) -> bool:
    """An LLM API error (auth, rate limit, timeout, 5xx, bad request) anywhere in the exception's chain."""
    return any(isinstance(e, ModelError) or type(e).__module__.startswith(PROVIDER_MODULES) for e in _chain(exc))


def classify(exc: BaseException) -> str:
    """One of worker_timeout, transport, provider_error or worker_exception."""
    chain = _chain(exc)
    if isinstance(exc, TimeoutError):
        return "worker_timeout"
    if is_provider_error(exc):
        return "provider_error"
    if any(isinstance(e, TRANSPORT_ERRORS) or type(e).__module__.startswith("mcp") for e in chain):
        return "transport"
    return "worker_exception"


def guarded(name: str, node_fn: WorkerNode) -> WorkerNode:
    """Wrap a worker node so that a failure becomes an AgentError and the graph continues."""

    async def node(state: OrchestratorState) -> dict:
        timeout = worker_timeout_s()
        start = time.perf_counter()
        try:
            with recording() as routing:
                return await asyncio.wait_for(node_fn(state), timeout)
        except Exception as exc:  # CancelledError is a BaseException and propagates
            kind = classify(exc)
            logger.exception("worker %r failed (%s)", name, kind)
            ms = round((time.perf_counter() - start) * 1000)
            emit_trace(name, "error", step=state.get("step"), agent=name, detail=kind, ms=ms)
            error = AgentError(agent=name, kind=kind, message=MESSAGES[kind].format(name=name, timeout=timeout))
            return {"errors": [error]} | ({"routing": routing} if routing else {})

    return node
