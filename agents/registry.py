"""Per-agent tool allowlist, prompt file and default tier."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass(frozen=True)
class AgentSpec:
    tools: frozenset[str]
    prompt_file: Path
    default_tier: Literal["fast", "strong"]


AGENTS: dict[str, AgentSpec] = {
    "research": AgentSpec(
        tools=frozenset({"search_corpus", "get_paper_metadata", "tavily_search"}),
        prompt_file=PROMPTS_DIR / "research.md",
        default_tier="fast",
    ),
    "data": AgentSpec(
        tools=frozenset({"describe_schema", "run_sql", "get_paper_metadata"}),
        prompt_file=PROMPTS_DIR / "data.md",
        default_tier="fast",
    ),
    "summary": AgentSpec(
        tools=frozenset({"search_corpus", "get_chunks", "get_paper_metadata", "tavily_extract"}),
        prompt_file=PROMPTS_DIR / "summary.md",
        default_tier="strong",
    ),
}
