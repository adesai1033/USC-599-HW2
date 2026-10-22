"""Chroma + embeddings retriever as an LCEL runnable.

The index built by ingest/embed.py is opened from a private temporary copy. Chroma writes to its
directory when it opens a collection, and the index baked into the image is read-only, so the
shipped files are never opened directly. Each copy is tagged with its process ID, and copies left
behind by a process that was killed are removed (on Cloud Run /tmp is memory, so they would add up).
"""

import atexit
import math
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


def _remove_orphaned_copies() -> None:
    for path in Path(tempfile.gettempdir()).glob("chroma-*-*"):
        pid = path.name.split("-")[1]
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            shutil.rmtree(path, ignore_errors=True)
        except PermissionError:
            pass  # the process exists but belongs to another user


def open_store(embeddings: Embeddings, persist_dir: str | Path | None = None) -> Chroma:
    """Open the chunk collection from a temporary copy of `persist_dir` (default CHROMA_PERSIST_DIR)."""
    source = Path(persist_dir or ROOT / os.environ.get("CHROMA_PERSIST_DIR", "index/chroma")).resolve()
    if not (source / "chroma.sqlite3").exists():
        raise FileNotFoundError(f"no Chroma index at {source}; run ingest/embed.py")
    _remove_orphaned_copies()
    workdir = tempfile.mkdtemp(prefix=f"chroma-{os.getpid()}-")
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


def build_retriever(
    store: Chroma, citation_counts: dict[str, int] | None = None, citation_prior: float = 0.0
) -> Runnable[SearchRequest, list[dict]]:
    """Runnable: SearchRequest -> chunks [{chunk_id, arxiv_id, title, section, page, url, score, text}], best first.

    With a citation_prior > 0, a wider set of candidates is re-ranked by
    cosine similarity + citation_prior * log10(1 + citations of the chunk's paper).
    Follow-up papers restate a famous method in their own words, often closer to a question than the
    original paper's text, and this lets the paper that introduced it rank first. "score" stays the
    plain cosine similarity.
    """
    counts = citation_counts or {}

    def ranking(arxiv_id: str, similarity: float) -> float:
        return similarity + citation_prior * math.log10(1 + counts.get(arxiv_id, 0))

    async def search(request: SearchRequest) -> list[dict]:
        fetch = max(4 * request.k, 40) if citation_prior > 0 else request.k
        hits = await store.asimilarity_search_with_score(request.query, k=fetch, filter=_where(request))
        # The collection uses cosine distance, so 1 - distance is cosine similarity (higher is better).
        scored = [(doc, 1 - distance) for doc, distance in hits]
        scored.sort(key=lambda hit: ranking(hit[0].metadata["arxiv_id"], hit[1]), reverse=True)
        return [
            {**{f: doc.metadata[f] for f in CHUNK_FIELDS}, "score": round(similarity, 4), "text": doc.page_content}
            for doc, similarity in scored[:request.k]
        ]

    return RunnableLambda(search, name="corpus_retriever")
