"""MultiServerMCPClient setup, persistent sessions, discovery, per-agent filtering, failure handling.

Each server in MCP_SERVERS_CONFIG (default mcp_config.json) gets one long-lived session, opened in
its own task so that a server that fails to start, crashes or disconnects never affects the
others. Tools are discovered once per session and exposed only if the manifest's "tools" list
names them; agents then receive only the tools in their allowlist in agents/registry.py.

Every tool call goes through an interceptor that sends it on the server's current session. If
the server is down, the call returns the manifest's "description" as a tool error instead of
raising, and a server that was up and then disconnected is restarted on a later call.

Tavily results are reshaped before the model sees them: each result gets a deterministic
source_id ("web:" + 8 hex chars of the URL's sha1), so [web:...] citations stay stable across
searches. An unparseable or off-schema Tavily payload becomes a clean tool error.

The Tavily URL carries the API key as a query parameter. It is built here from TAVILY_API_KEY.
install_log_redaction() removes it, and every other provider key, from all log output; failure
messages are redacted too.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

from agents.registry import AGENTS

ROOT = Path(__file__).resolve().parent
START_TIMEOUT_S = 30.0
CALL_TIMEOUT_S = 60.0
RESTART_COOLDOWN_S = 30.0
MAX_RESULT_CHARS = 100_000  # longer tool output is cut before it reaches a model
SECRET_ENV_VARS = ("TAVILY_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
TAVILY_TOOLS = {"tavily_search", "tavily_extract"}
TRACKING_PARAM = re.compile(r"^(utm_|fbclid$|gclid$)")
# One paper, one ID: arxiv.org/abs/2610.04351, /html/2610.04351v1, /pdf/2610.04351v2.pdf, arXiv's own mirrors,
# and third-party pages that are keyed by the arXiv ID (alphaxiv.org/overview/2610.04351, huggingface.co/papers/...).
ARXIV_HOSTS = {"arxiv.org", "export.arxiv.org", "ar5iv.labs.arxiv.org", "ar5iv.org"}
ARXIV_MIRROR_HOSTS = {"alphaxiv.org", "huggingface.co", "semanticscholar.org", "api.semanticscholar.org",
                      "emergentmind.com", "papers.cool", "scholar.archive.org", "paperswithcode.com"}
ARXIV_PAGE = re.compile(
    r"^/(?:abs|html|pdf)/(?P<id>\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?(?:\.pdf)?/?$"
)
ARXIV_MIRROR_PAGE = re.compile(r"^/(?:abs|pdf|overview|papers|arxiv|paper/arxiv)/(?P<id>\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?$")
REQUEST_TIMEOUT_CODE = 408  # McpError code the MCP SDK uses for a request that timed out

ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)\}")
SECRET_PARAM = re.compile(r"(tavilyApiKey=)[^&\s'\"<>]+", re.I)

logger = logging.getLogger("mcp_client")


def redact(text: str) -> str:
    """Remove API keys from text: the Tavily key as a query parameter, and any provider key on its own."""
    text = SECRET_PARAM.sub(r"\1<redacted>", text)
    for name in SECRET_ENV_VARS:
        key = os.environ.get(name, "")
        if len(key) >= 8:
            text = text.replace(key, "<redacted>")
    return text


class RedactingFilter(logging.Filter):
    """Redacts secrets from a record's message and traceback before any handler formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        clean = redact(message)
        if clean != message:
            record.msg, record.args = clean, None
        if record.exc_info and not record.exc_text:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        return True


def install_log_redaction() -> None:
    """Attach RedactingFilter to every handler, including uvicorn's own. Call after logging is configured."""
    loggers = [logging.getLogger(), *(l for l in logging.root.manager.loggerDict.values() if isinstance(l, logging.Logger))]
    for handler in {h for lg in loggers for h in lg.handlers}:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())


@dataclass
class ServerStatus:
    name: str
    description: str
    tools: frozenset[str]
    state: Literal["starting", "up", "down"] = "starting"
    error: str | None = None
    was_up: bool = False
    last_attempt: float = 0.0


def load_manifests(path: str | Path | None = None) -> dict[str, dict]:
    path = Path(path or os.environ.get("MCP_SERVERS_CONFIG") or ROOT / "mcp_config.json")
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text())


def _expand(value: str) -> str:
    def lookup(match: re.Match) -> str:
        if not os.environ.get(match.group(1)):
            raise ValueError(f"environment variable {match.group(1)} is not set")
        return os.environ[match.group(1)]

    return ENV_REF.sub(lookup, value)


def build_connection(manifest: dict) -> dict:
    """Turn one manifest into a langchain-mcp-adapters connection dict."""
    session_kwargs = {"read_timeout_seconds": timedelta(seconds=CALL_TIMEOUT_S)}
    if manifest["transport"] == "stdio":
        # "python" means this interpreter, so servers run in the same venv as the app.
        command = sys.executable if manifest["command"] == "python" else manifest["command"]
        env = {k: os.environ[k] for k in manifest.get("env_passthrough", []) if k in os.environ}
        return {"transport": "stdio", "command": command, "args": manifest.get("args", []),
                "env": env, "cwd": str(ROOT), "session_kwargs": session_kwargs}
    if manifest["transport"] == "streamable_http":
        return {"transport": "streamable_http", "url": _expand(manifest["url"]), "session_kwargs": session_kwargs}
    raise ValueError(f"unsupported transport {manifest['transport']!r}")


def normalize_url(url: str) -> str:
    """Canonical form of a URL for source IDs.

    http and https are the same, "www." is dropped, as are the fragment, tracking parameters and a
    trailing slash. Every arXiv page of a paper (abs, html, pdf, any version), and a third-party page
    keyed by its arXiv ID, maps to the paper's arXiv abs page.
    """
    parts = urlsplit(url.strip())
    scheme = "https" if parts.scheme.lower() in ("http", "https") else parts.scheme.lower()
    host = parts.netloc.lower().removeprefix("www.")
    if host in ARXIV_HOSTS and (page := ARXIV_PAGE.match(parts.path)):
        return f"https://arxiv.org/abs/{page['id']}"
    if host in ARXIV_MIRROR_HOSTS and (page := ARXIV_MIRROR_PAGE.match(parts.path)):
        return f"https://arxiv.org/abs/{page['id']}"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not TRACKING_PARAM.match(k)))
    return urlunsplit((scheme, host, parts.path.rstrip("/"), query, ""))


def web_source_id(url: str) -> str:
    """Deterministic citation ID for a web source: the same URL always gets the same ID."""
    return "web:" + hashlib.sha1(normalize_url(url).encode()).hexdigest()[:8]


class TavilyError(Exception):
    """Tavily reported an error (e.g. a 400 for a bad query) inside an otherwise successful MCP result."""


def normalize_tavily(tool_name: str, text: str) -> str:
    """Reshape a Tavily payload to {results: [{source_id, url, title, content}], ...}.

    Raises TavilyError if Tavily reported an error, and ValueError if the payload is malformed.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("response is not JSON") from exc
    if isinstance(payload, dict) and payload.get("error"):
        detail = payload.get("detail")
        reason = detail.get("error") if isinstance(detail, dict) else detail
        raise TavilyError(str(reason or payload["error"])[:300])
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("response has no results list")
    normalized: dict[str, dict] = {}
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("url"), str) or not result["url"].strip():
            raise ValueError("a result has no url")
        source_id = web_source_id(result["url"])
        if source_id in normalized:
            continue  # another page of the same source, e.g. a paper's arXiv html page after its abs page
        content = result.get("raw_content") if tool_name == "tavily_extract" else result.get("content")
        normalized[source_id] = {
            "source_id": source_id,
            "url": result["url"],
            "title": result.get("title") or "",
            "content": content or result.get("content") or "",
        }
    out: dict = {"results": list(normalized.values())}
    if payload.get("query"):
        out["query"] = payload["query"]
    if payload.get("answer"):
        out["answer"] = payload["answer"]
    if payload.get("failed_results"):
        out["failed_results"] = payload["failed_results"]
    return json.dumps(out, ensure_ascii=False)


def _normalize_tavily_result(tool_name: str, result: CallToolResult) -> CallToolResult:
    if result.isError:
        return result
    text = "".join(b.text for b in result.content if isinstance(b, TextContent))
    try:
        return CallToolResult(content=[TextContent(type="text", text=normalize_tavily(tool_name, text))])
    except TavilyError as exc:
        # Tavily's own message tells the model how to fix the call, e.g. "provide search terms".
        return _error_result(f"{tool_name} failed: {redact(str(exc))}")
    except ValueError as exc:
        logger.warning("%s returned a malformed response: %s", tool_name, exc)
        return _error_result(f"{tool_name} returned a malformed response ({exc}); web results are unavailable for this call.")


def _error_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)


def _cap(result: CallToolResult) -> CallToolResult:
    """Cut text content beyond MAX_RESULT_CHARS so one oversized result can't flood a model's context."""
    for block in result.content:
        if isinstance(block, TextContent) and len(block.text) > MAX_RESULT_CHARS:
            cut = len(block.text) - MAX_RESULT_CHARS
            block.text = block.text[:MAX_RESULT_CHARS] + f"\n... [truncated {cut} chars]"
    return result


class MCPTools:
    """Persistent sessions to every configured MCP server, and the tools each agent may use."""

    def __init__(self, manifests: dict[str, dict] | None = None) -> None:
        self.manifests = manifests if manifests is not None else load_manifests()
        self.status = {
            name: ServerStatus(name, m["description"], frozenset(m.get("tools", [])))
            for name, m in self.manifests.items()
        }
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, BaseTool] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._stop = asyncio.Event()
        self._restart_locks = {name: asyncio.Lock() for name in self.manifests}
        self._background: set[asyncio.Task] = set()

    async def start(self) -> None:
        """Connect to every server concurrently. Failures are recorded in status, never raised."""
        await asyncio.gather(*(self._connect(name) for name in self.manifests))
        up = [n for n, s in self.status.items() if s.state == "up"]
        down = {n: s.error for n, s in self.status.items() if s.state == "down"}
        logger.info("MCP servers up: %s; down: %s; tools: %s", up, down or "none", sorted(self._tools))

    async def _connect(self, name: str) -> None:
        status = self.status[name]
        status.state, status.error, status.last_attempt = "starting", None, time.monotonic()
        try:
            connection = build_connection(self.manifests[name])
        except (KeyError, ValueError) as exc:
            self._mark_down(name, f"bad manifest: {exc}")
            return
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        self._tasks[name] = asyncio.create_task(self._serve(name, connection, ready), name=f"mcp:{name}")
        try:
            tools = await asyncio.wait_for(asyncio.shield(ready), START_TIMEOUT_S)
        except TimeoutError:
            self._tasks[name].cancel()
            self._mark_down(name, f"did not start within {START_TIMEOUT_S:.0f} s")
            return
        except Exception as exc:  # the server failed to start; _serve has marked it down
            logger.warning("MCP server %r failed to start: %s", name, redact(f"{type(exc).__name__}: {exc}"))
            return
        exposed = {t.name: t for t in tools if t.name in status.tools}
        missing = status.tools - exposed.keys()
        if missing:
            logger.warning("MCP server %r does not provide declared tools %s", name, sorted(missing))
        clash = exposed.keys() & {n for n, t in self._tools.items() if t.metadata.get("server") != name}
        if clash:
            self._mark_down(name, f"tool names {sorted(clash)} already provided by another server")
            return
        for tool in exposed.values():
            tool.metadata = {**(tool.metadata or {}), "server": name}
        self._tools.update(exposed)
        status.state, status.was_up = "up", True

    async def _serve(self, name: str, connection: dict, ready: asyncio.Future) -> None:
        """Own one server's session for its whole life, so it is opened and closed in the same task."""
        client = MultiServerMCPClient({name: connection})
        session: ClientSession | None = None
        try:
            async with client.session(name) as session:
                tools = await load_mcp_tools(
                    session, server_name=name, tool_interceptors=[self._call], handle_tool_errors=True
                )
                self._sessions[name] = session
                ready.set_result(tools)
                await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._mark_down(name, f"{type(exc).__name__}: {exc}")
            if not ready.done():
                ready.set_exception(exc)
        finally:
            # A restart may already have stored a newer session under this name; leave that one alone.
            if session is not None and self._sessions.get(name) is session:
                self._sessions.pop(name, None)

    def _mark_down(self, name: str, reason: str) -> None:
        status = self.status[name]
        status.state, status.error = "down", redact(reason)
        self._sessions.pop(name, None)
        logger.warning("MCP server %r is down: %s", name, status.error)

    def refresh(self) -> None:
        """Retry, in the background, every server that is down and past its cooldown.

        Call once per request: a server that failed at startup (e.g. Tavily during a network blip
        on a cold start) then comes back without waiting for the instance to be replaced.
        """
        for name, status in self.status.items():
            if status.state == "down" and time.monotonic() - status.last_attempt >= RESTART_COOLDOWN_S:
                task = asyncio.create_task(self._restart(name))
                self._background.add(task)
                task.add_done_callback(self._background.discard)

    async def _restart(self, name: str) -> None:
        async with self._restart_locks[name]:
            status = self.status[name]
            if status.state == "up" or time.monotonic() - status.last_attempt < RESTART_COOLDOWN_S:
                return
            logger.info("restarting MCP server %r", name)
            old = self._tasks.pop(name, None)
            if old:
                old.cancel()  # closes the dead session in the task that opened it
            await self._connect(name)

    async def _call(self, request: MCPToolCallRequest, _handler) -> CallToolResult:
        """Interceptor for every tool call: route to the server's current session, never raise."""
        name, status = request.server_name, self.status[request.server_name]
        if status.was_up and status.state != "up":
            await self._restart(name)  # also waits out a restart another call already started
        session = self._sessions.get(name)
        if session is None:
            return _error_result(status.description)
        try:
            result = await session.call_tool(request.name, request.args)
            if request.name in TAVILY_TOOLS:
                result = _normalize_tavily_result(request.name, result)
            return _cap(result)
        except McpError as exc:
            if exc.error.code == REQUEST_TIMEOUT_CODE:
                return _error_result(f"{request.name} timed out after {CALL_TIMEOUT_S:.0f} s; try a narrower request.")
            self._mark_down(name, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # transport failure: the server crashed or the connection dropped
            self._mark_down(name, f"{type(exc).__name__}: {exc}")
        return _error_result(status.description)

    def tools_for(self, agent: str) -> list[BaseTool]:
        """The agent's allowlisted tools that were discovered, sorted by name. Missing ones are logged."""
        allowed = AGENTS[agent].tools
        missing = allowed - self._tools.keys()
        if missing:
            logger.warning("agent %r is missing allowlisted tools %s", agent, sorted(missing))
        return [self._tools[n] for n in sorted(allowed & self._tools.keys())]

    def unavailable_notes(self, agent: str) -> list[str]:
        """Descriptions of servers that should provide one of the agent's tools but are down or didn't
        provide it, for the agent's system prompt."""
        allowed = AGENTS[agent].tools
        return [
            s.description for s in self.status.values()
            if s.tools & allowed and (s.state != "up" or (s.tools & allowed) - self._tools.keys())
        ]

    async def aclose(self) -> None:
        for task in self._background:
            task.cancel()
        self._stop.set()
        tasks = list(self._tasks.values())
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=10)
            for task in pending:
                task.cancel()
        self._tasks.clear()
        self._sessions.clear()
