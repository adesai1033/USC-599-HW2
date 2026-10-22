"""Per-agent tool allowlist, prompt file and default tier.

AGENTS holds every specialized agent's spec. mcp_client.py filters MCP tools with these allowlists.
WORKERS holds the agents that are wired into the graph as worker nodes; an agent is added to it in
the phase that implements it (research in Phase 4b, data in 4d).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str  # one line, shown to the supervisor
    prompt_file: Path
    tools: frozenset[str]
    default_tier: Literal["fast", "strong"]
    max_tool_rounds: int | None = None  # overrides WORKER_MAX_TOOL_ROUNDS for this agent


AGENTS: dict[str, AgentSpec] = {
    "research": AgentSpec(
        name="research",
        description="Answers technical questions by reading paper text in the 3D vision corpus and the live web, "
        "with citations: how methods work, what a paper proposes, recent work. Not for counting or ranking "
        "papers by metadata or benchmark results.",
        prompt_file=PROMPTS_DIR / "research.md",
        tools=frozenset({"search_corpus", "get_paper_metadata", "tavily_search"}),
        default_tier="fast",
    ),
    "data": AgentSpec(
        name="data",
        description="Answers questions about paper metadata (years, venues, authors, citation counts) and reported "
        "benchmark results (PSNR, SSIM, LPIPS, Chamfer, speed) using SQL over the paper database. Does not read "
        "paper text.",
        prompt_file=PROMPTS_DIR / "data.md",
        tools=frozenset({"describe_schema", "run_sql", "get_paper_metadata"}),
        default_tier="fast",
        max_tool_rounds=5,  # schema -> explore values -> query -> one retry -> metadata
    ),
    "summary": AgentSpec(
        name="summary",
        description="Writes multi-paper summaries and literature reviews from the corpus, grouped by paper, "
        "with citations kept and disagreements between papers flagged.",
        prompt_file=PROMPTS_DIR / "summary.md",
        tools=frozenset({"search_corpus", "get_chunks", "get_paper_metadata", "tavily_extract"}),
        default_tier="strong",
    ),
}

WORKERS: dict[str, AgentSpec] = {name: AGENTS[name] for name in ("research", "data")}


def available_workers() -> list[str]:
    """Names of the agents the graph can dispatch to."""
    return list(WORKERS)
