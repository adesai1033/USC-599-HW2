"""Test-only stdio MCP server. Stands in for corpus and tavily, plus tools to probe failure handling."""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake")


@mcp.tool()
def search_corpus(query: str) -> str:
    """Fake corpus search."""
    return f"search:{query}"


@mcp.tool()
def get_chunks(chunk_ids: list[str]) -> str:
    """Fake chunk lookup."""
    return ",".join(chunk_ids)


@mcp.tool()
def tavily_search(query: str) -> str:
    """Fake web search."""
    return f"web:{query}"


@mcp.tool()
def tavily_extract(urls: list[str]) -> str:
    """Fake web extract."""
    return ",".join(urls)


@mcp.tool()
def echo(text: str) -> str:
    """Return text unchanged."""
    return text


@mcp.tool()
def crash() -> str:
    """Kill the server process mid-call, as a crashed or disconnected server would."""
    os._exit(1)


if __name__ == "__main__":
    mcp.run(transport="stdio")
