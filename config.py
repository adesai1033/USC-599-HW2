"""Settings from env: model IDs, RAG_TOP_K, step limits, state backend."""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# Local development only: the image excludes .env, and real env vars always win.
load_dotenv()

REQUIRED = ("GOOGLE_API_KEY", "GEMINI_MODEL")


@dataclass(frozen=True)
class Settings:
    port: int
    google_api_key: str
    gemini_model: str
    log_level: str

    def __repr__(self) -> str:
        return (
            f"Settings(port={self.port}, gemini_model={self.gemini_model!r}, "
            f"log_level={self.log_level!r}, google_api_key=<redacted>)"
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


def mcp_reconnect_backoff_s() -> float:
    """MCP_RECONNECT_BACKOFF_S (default 30): wait after a failed reconnect before trying a server again."""
    return _env_number("MCP_RECONNECT_BACKOFF_S", 30, 0, float)


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
    )
