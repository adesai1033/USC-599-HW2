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


def max_supervisor_steps() -> int:
    """MAX_SUPERVISOR_STEPS (default 6): supervisor decisions per turn before it must synthesize."""
    steps = int(os.environ.get("MAX_SUPERVISOR_STEPS", "6"))
    if steps < 1:
        raise RuntimeError(f"MAX_SUPERVISOR_STEPS must be at least 1, got {steps}.")
    return steps


def worker_max_tool_rounds() -> int:
    """WORKER_MAX_TOOL_ROUNDS (default 4): tool-calling rounds a worker gets before it must summarize."""
    rounds = int(os.environ.get("WORKER_MAX_TOOL_ROUNDS", "4"))
    if rounds < 1:
        raise RuntimeError(f"WORKER_MAX_TOOL_ROUNDS must be at least 1, got {rounds}.")
    return rounds


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
