"""Settings from env: model IDs, RAG_TOP_K, step limits, state backend."""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# Local development only: the image excludes .env, and real env vars always win.
load_dotenv()

REQUIRED = ("GOOGLE_API_KEY", "GEMINI_MODEL")
# Match .env.example; verified against the provider docs (see README).
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class Settings:
    port: int
    google_api_key: str
    gemini_model: str
    log_level: str
    # The strong tier is optional at startup: a missing key fails Claude's calls, which then fall back
    # to Gemini, instead of stopping the app.
    anthropic_api_key: str = ""
    anthropic_model: str = ""

    def __repr__(self) -> str:
        return (
            f"Settings(port={self.port}, gemini_model={self.gemini_model!r}, anthropic_model={self.anthropic_model!r}, "
            f"log_level={self.log_level!r}, api keys=<redacted>)"
        )


def _env_number(name: str, default: float, minimum: float, kind: type = int) -> float:
    """A numeric setting read at call time, so tests and probes can change it per run."""
    value = kind(os.environ.get(name, str(default)))
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}, got {value}.")
    return value


def max_supervisor_steps() -> int:
    """MAX_SUPERVISOR_STEPS (default 6): supervisor decisions per turn before it must synthesize."""
    return _env_number("MAX_SUPERVISOR_STEPS", 6, 1)


def worker_max_tool_rounds() -> int:
    """WORKER_MAX_TOOL_ROUNDS (default 4): tool-calling rounds a worker gets before it must summarize."""
    return _env_number("WORKER_MAX_TOOL_ROUNDS", 4, 1)


def max_worker_retries() -> int:
    """MAX_WORKER_RETRIES (default 1): times a worker that failed this turn may be dispatched again."""
    return _env_number("MAX_WORKER_RETRIES", 1, 0)


def worker_timeout_s() -> float:
    """WORKER_TIMEOUT_S (default 60): wall-clock limit for one worker run."""
    return _env_number("WORKER_TIMEOUT_S", 60, 0.001, float)


def tool_timeout_s() -> float:
    """TOOL_TIMEOUT_S (default 30): limit for one MCP tool call."""
    return _env_number("TOOL_TIMEOUT_S", 30, 0.001, float)


def supervisor_timeout_s() -> float:
    """SUPERVISOR_TIMEOUT_S (default 20): wait for the supervisor's model before retrying on the other tier."""
    return _env_number("SUPERVISOR_TIMEOUT_S", 20, 0.001, float)


def strong_context_tokens() -> int:
    """STRONG_CONTEXT_TOKENS (default 8000): estimated synthesize context above which it uses the strong tier."""
    return _env_number("STRONG_CONTEXT_TOKENS", 8000, 1)


def supervisor_escalation() -> bool:
    """SUPERVISOR_ESCALATION (default off): let the supervisor move to the strong tier for hard decisions
    (re-planning after a failed or ungrounded worker, a follow-up turn, combining multi-paper findings)."""
    return os.environ.get("SUPERVISOR_ESCALATION", "").strip().lower() in ("1", "true", "yes", "on")


def mcp_reconnect_backoff_s() -> float:
    """MCP_RECONNECT_BACKOFF_S (default 30): wait after a failed reconnect before trying a server again."""
    return _env_number("MCP_RECONNECT_BACKOFF_S", 30, 0, float)


def mcp_start_timeout_s() -> float:
    """MCP_START_TIMEOUT_S (default 90): wait for an MCP server to start and list its tools. A Cloud Run
    cold start on 1 vCPU takes the stdio servers about 25 s, and once more than 30 s."""
    return _env_number("MCP_START_TIMEOUT_S", 90, 0.001, float)


def history_turns() -> int:
    """HISTORY_TURNS (default 6): earlier turns of the session loaded into each new turn."""
    return _env_number("HISTORY_TURNS", 6, 0)


def state_backend() -> str:
    """STATE_BACKEND: "firestore" on Cloud Run, "memory" (the default) locally and in tests."""
    backend = os.environ.get("STATE_BACKEND", "memory").strip().lower()
    if backend not in ("memory", "firestore"):
        raise RuntimeError(f"STATE_BACKEND must be 'memory' or 'firestore', got {backend!r}.")
    return backend


def firestore_project_id() -> str | None:
    """FIRESTORE_PROJECT_ID; None lets the client take the project from ADC."""
    return os.environ.get("FIRESTORE_PROJECT_ID", "").strip() or None


def firestore_timeout_s() -> float:
    """FIRESTORE_TIMEOUT_S (default 5): limit for one history read or write."""
    return _env_number("FIRESTORE_TIMEOUT_S", 5, 0.001, float)


@lru_cache
def get_settings() -> Settings:
    """Read settings from the environment; raise if a required variable is missing."""
    missing = [name for name in REQUIRED if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in .env for local runs or via --set-env-vars on Cloud Run."
        )
    return Settings(
        port=int(os.environ.get("PORT", "8080")),
        google_api_key=os.environ["GOOGLE_API_KEY"].strip(),
        gemini_model=os.environ["GEMINI_MODEL"].strip(),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "").strip() or DEFAULT_ANTHROPIC_MODEL,
    )
