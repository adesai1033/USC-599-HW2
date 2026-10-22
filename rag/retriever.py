"""Chroma + embeddings retriever as an LCEL runnable.

The index built by ingest/embed.py is opened from a private temporary copy. Chroma writes to its
directory when it opens a collection, and the index baked into the image is read-only, so the
shipped files are never opened directly.
"""

import atexit
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = "papers"  # must match ingest/embed.py
CHUNK_FIELDS = ("chunk_id", "arxiv_id", "title", "section", "page", "url")


@dataclass(frozen=True)
class SearchRequest:
    query: str
    k: int
    arxiv_ids: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()


def open_store(embeddings: Embeddings, persist_dir: str | Path | None = None) -> Chroma:
    """Open the chunk collection from a temporary copy of `persist_dir` (default CHROMA_PERSIST_DIR)."""
    source = Path(persist_dir or ROOT / os.environ.get("CHROMA_PERSIST_DIR", "index/chroma")).resolve()
    if not (source / "chroma.sqlite3").exists():
        raise FileNotFoundError(f"no Chroma index at {source}; run ingest/embed.py")
    workdir = tempfile.mkdtemp(prefix="chroma-")
    atexit.register(shutil.rmtree, workdir, ignore_errors=True)
    copy = Path(workdir) / "chroma"
    shutil.copytree(source, copy)
    for path in [copy, *copy.rglob("*")]:
        path.chmod(path.stat().st_mode | 0o200)  # the copy must be writable even if the source isn't
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(copy),
        create_collection_if_not_exists=False,
    )


def _where(request: SearchRequest) -> dict | None:
    clauses = []
    if request.arxiv_ids:
        clauses.append({"arxiv_id": {"$in": list(request.arxiv_ids)}})
    if request.sections:
        clauses.append({"section": {"$in": list(request.sections)}})
    if len(clauses) > 1:
        return {"$and": clauses}
    return clauses[0] if clauses else None


def build_retriever(store: Chroma) -> Runnable[SearchRequest, list[dict]]:
    """Runnable: SearchRequest -> chunks [{chunk_id, arxiv_id, title, section, page, url, score, text}], best first."""

    async def search(request: SearchRequest) -> list[dict]:
        hits = await store.asimilarity_search_with_score(request.query, k=request.k, filter=_where(request))
        return [
            # The collection uses cosine distance, so 1 - distance is cosine similarity (higher is better).
            {**{f: doc.metadata[f] for f in CHUNK_FIELDS}, "score": round(1 - distance, 4), "text": doc.page_content}
            for doc, distance in hits
        ]

    return RunnableLambda(search, name="corpus_retriever")
