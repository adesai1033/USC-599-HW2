"""MCP server (stdio) over the Chroma corpus: search_corpus, get_chunks.

  python mcp_servers/corpus.py

Needs OPENAI_API_KEY to embed queries. Reads OPENAI_EMBEDDING_MODEL and CHROMA_PERSIST_DIR.
The index is loaded once at startup, before the server accepts requests.
"""

import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.runnables import Runnable  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402

from rag.retriever import CHUNK_FIELDS, SearchRequest, build_retriever, open_store  # noqa: E402

K_MIN, K_MAX = 1, 12
MAX_QUERY_CHARS = 2000
MAX_FILTER_IDS = 50
MAX_CHUNK_IDS = 20
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
CHUNK_ID = re.compile(r"^\d{4}\.\d{4,5}#[a-z0-9-]+-\d{2,}$")

logger = logging.getLogger("mcp.corpus")
mcp = FastMCP("corpus")


@dataclass(frozen=True)
class Corpus:
    store: Chroma
    retriever: Runnable[SearchRequest, list[dict]]
    arxiv_ids: frozenset[str]
    sections: Counter


@lru_cache
def corpus() -> Corpus:
    embeddings = OpenAIEmbeddings(
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), timeout=20, max_retries=2
    )
    store = open_store(embeddings)
    metadatas = store.get(include=["metadatas"])["metadatas"]
    logger.info("corpus index loaded: %d chunks", len(metadatas))
    return Corpus(
        store=store,
        retriever=build_retriever(store),
        arxiv_ids=frozenset(m["arxiv_id"] for m in metadatas),
        sections=Counter(m["section"] for m in metadatas),
    )


def as_json(payload: dict) -> str:
    """Compact JSON: FastMCP would indent a returned dict, which costs the model tokens."""
    return json.dumps(payload, ensure_ascii=False)


def normalize_arxiv_id(raw: str) -> str:
    """'arXiv:2308.04079v2' -> '2308.04079'."""
    return re.sub(r"v\d+$", "", raw.strip().removeprefix("arXiv:").removeprefix("arxiv:"))


def validate_arxiv_ids(raw: list[str] | None) -> tuple[str, ...]:
    if not raw:
        return ()
    if len(raw) > MAX_FILTER_IDS:
        raise ToolError(f"arxiv_ids accepts at most {MAX_FILTER_IDS} IDs, got {len(raw)}.")
    ids = tuple(dict.fromkeys(normalize_arxiv_id(i) for i in raw))
    malformed = [i for i in ids if not ARXIV_ID.match(i)]
    if malformed:
        raise ToolError(f"Malformed arXiv IDs {malformed}; expected the form 2308.04079.")
    unknown = [i for i in ids if i not in corpus().arxiv_ids]
    if unknown:
        raise ToolError(f"Not in the corpus: {unknown}. Search without arxiv_ids, or use web search for these papers.")
    return ids


def validate_sections(raw: list[str] | None) -> tuple[str, ...]:
    if not raw:
        return ()
    sections = tuple(dict.fromkeys(s.strip().lower() for s in raw))
    unknown = [s for s in sections if s not in corpus().sections]
    if unknown:
        common = [s for s, _ in corpus().sections.most_common(12)]
        raise ToolError(f"Unknown sections {unknown}. Common section names: {common}.")
    return sections


@mcp.tool()
async def search_corpus(
    query: str, k: int = 6, arxiv_ids: list[str] | None = None, sections: list[str] | None = None
) -> str:
    """Semantic search over the corpus of ~220 arXiv papers on 3D computer vision (NeRF, 3D Gaussian
    Splatting, neural surfaces, multi-view stereo, feed-forward reconstruction, dynamic scenes, neural SLAM).

    Use this to find passages that answer a question. Returns up to k chunks, best first, each with
    chunk_id, arxiv_id, title, section, page, url, score (higher is more relevant) and text. Cite
    chunks by their chunk_id. An empty result means the corpus has nothing relevant.

    Args:
        query: A natural-language question or description of what to find.
        k: Number of chunks to return, 1 to 12. Default 6.
        arxiv_ids: Optional. Restrict the search to these papers, e.g. ["2308.04079"].
        sections: Optional. Restrict to these sections. "abstract", "introduction", "related-work",
            "conclusion" and "captions" (figure and table captions) exist for nearly every paper; other
            section names are each paper's own headings (e.g. "method" in one paper,
            "differentiable-3d-gaussian-splatting" in another), so filter on those only if you have
            seen them in an earlier result.

    To read specific chunks you already have IDs for, use get_chunks instead.
    """
    query = query.strip()
    if not query:
        raise ToolError("query must not be empty.")
    if len(query) > MAX_QUERY_CHARS:
        raise ToolError(f"query is too long ({len(query)} chars); keep it under {MAX_QUERY_CHARS}.")
    if isinstance(k, bool) or not isinstance(k, int) or not K_MIN <= k <= K_MAX:
        raise ToolError(f"k must be an integer from {K_MIN} to {K_MAX}, got {k!r}.")
    request = SearchRequest(query=query, k=k, arxiv_ids=validate_arxiv_ids(arxiv_ids), sections=validate_sections(sections))
    results = await corpus().retriever.ainvoke(request)
    response = {"query": query, "k": k, "results": results}
    if not results and request.sections:
        response["note"] = "No chunks match these filters. Section names vary by paper; retry without sections."
    elif not results:
        response["note"] = "No matching chunks in the corpus."
    return as_json(response)


@mcp.tool()
async def get_chunks(chunk_ids: list[str]) -> str:
    """Fetch the full text and metadata of specific corpus chunks by chunk_id.

    Use this when you already have chunk IDs (from an earlier search_corpus call or a citation) and
    need their exact text. Returns {"chunks": [...], "missing": [...]}; each chunk has chunk_id,
    arxiv_id, title, section, page, url and text. IDs look like "2308.04079#method-03".
    At most 20 IDs per call. To find new passages, use search_corpus instead.
    """
    if not chunk_ids:
        raise ToolError("chunk_ids must contain at least one ID.")
    if len(chunk_ids) > MAX_CHUNK_IDS:
        raise ToolError(f"get_chunks accepts at most {MAX_CHUNK_IDS} IDs, got {len(chunk_ids)}.")
    ids = list(dict.fromkeys(i.strip() for i in chunk_ids))
    malformed = [i for i in ids if not CHUNK_ID.match(i)]
    if malformed:
        raise ToolError(f"Malformed chunk IDs {malformed}; expected the form 2308.04079#method-03.")
    found = corpus().store.get(ids=ids, include=["metadatas", "documents"])
    by_id = {
        i: {**{f: m[f] for f in CHUNK_FIELDS}, "text": doc}
        for i, m, doc in zip(found["ids"], found["metadatas"], found["documents"])
    }
    return as_json({"chunks": [by_id[i] for i in ids if i in by_id], "missing": [i for i in ids if i not in by_id]})


def main() -> None:
    # force=True replaces the rich handler FastMCP installs when the server object is created.
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), stream=sys.stderr, force=True,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "openai", "chromadb", "mcp.server.lowlevel.server"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    corpus()  # load the index before accepting requests
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
