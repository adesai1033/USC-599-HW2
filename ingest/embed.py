"""Embed chunks into the Chroma index.

  python ingest/embed.py --dry-run   # count tokens and estimate cost; no API calls
  python ingest/embed.py

Reads data/chunks.jsonl and writes the persistent Chroma collection COLLECTION in
CHROMA_PERSIST_DIR (default index/chroma/), using OPENAI_EMBEDDING_MODEL and cosine distance.

The run is incremental. Each chunk's metadata stores a hash of its text, so a re-run embeds
only new or changed chunks, and deletes chunks that are no longer in chunks.jsonl. Batches are
written as they finish, so an interrupted run keeps its progress.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import chromadb
import tiktoken
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

COLLECTION = "papers"
BATCH_SIZE = 256
PRICE_PER_M_TOKENS = 0.02  # text-embedding-3-small, developers.openai.com/api/docs/pricing, checked 2026-10-21
METADATA_FIELDS = ("chunk_id", "arxiv_id", "title", "year", "section", "page", "url")

logger = logging.getLogger("embed")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report what would be embedded, without API calls")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    load_dotenv(ROOT / ".env")
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    persist_dir = (ROOT / os.environ.get("CHROMA_PERSIST_DIR", "index/chroma")).resolve()

    chunks = load_chunks()
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": model},
        embedding_function=None,
    )
    if collection.metadata.get("embedding_model") != model:
        sys.exit(f"{persist_dir} was built with {collection.metadata.get('embedding_model')}, not {model}; "
                 "move it aside and re-run to rebuild.")

    existing = collection.get(include=["metadatas"])
    existing_hash = {i: m.get("text_sha") for i, m in zip(existing["ids"], existing["metadatas"])}
    wanted = {c["chunk_id"] for c in chunks}
    todo = [c for c in chunks if existing_hash.get(c["chunk_id"]) != text_hash(c["text"])]
    stale = [i for i in existing_hash if i not in wanted]

    encoding = tiktoken.encoding_for_model(model)
    tokens = sum(len(encoding.encode(c["text"])) for c in todo)
    logger.info("%d chunks: %d to embed (%d tokens, est. $%.4f), %d up to date, %d stale to delete",
                len(chunks), len(todo), tokens, tokens / 1e6 * PRICE_PER_M_TOKENS,
                len(chunks) - len(todo), len(stale))
    if args.dry_run:
        return

    if stale:
        collection.delete(ids=stale)
    embedder = OpenAIEmbeddings(model=model, max_retries=5)
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        vectors = embedder.embed_documents([c["text"] for c in batch])
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[{**{k: c[k] for k in METADATA_FIELDS}, "text_sha": text_hash(c["text"])} for c in batch],
        )
        logger.info("embedded %d/%d", start + len(batch), len(todo))
    logger.info("collection %r in %s now holds %d chunks", COLLECTION, persist_dir, collection.count())


if __name__ == "__main__":
    main()
