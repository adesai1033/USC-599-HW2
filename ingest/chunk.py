"""Section-aware + recursive chunking.

  python ingest/chunk.py

Reads data/parsed/*.json and data/selected.json and writes data/chunks.jsonl, one chunk per line:
  {chunk_id, arxiv_id, title, year, section, page, url, text}

- Each paper's arXiv abstract becomes one chunk with section="abstract".
- Body sections are split recursively: paragraphs into sentences, and sentences into words when
  a sentence is too long. The pieces are packed into chunks of up to CHUNK_TOKENS, with about
  OVERLAP_TOKENS of trailing sentences repeated at the start of the next chunk in the same
  section. Chunks never cross a section boundary.
- Figure and table captions are packed into smaller chunks with section="captions".
- Chunk IDs are {arxiv_id}#{section_slug}-{nn}. The numbering continues across sections that
  share a slug, so IDs are unique within a paper and stable when the same parse is re-chunked.

Token counts are estimated at about 4 characters per token. That's close enough to size chunks
well under the embedding model's input limit.
"""

import json
import logging
import math
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"
SELECTED_PATH = DATA_DIR / "selected.json"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

CHUNK_TOKENS = 800
OVERLAP_TOKENS = 100
CAPTION_CHUNK_TOKENS = 300
MIN_TAIL_TOKENS = 150  # a section's last chunk smaller than this is merged into the previous one
MAX_MERGED_TOKENS = 1000  # ...unless the merged chunk would exceed this
CHARS_PER_TOKEN = 4

ABBREVIATIONS = {
    "al", "fig", "figs", "eq", "eqs", "e.g", "i.e", "vs", "sec", "secs", "tab", "cf", "resp",
    "approx", "no", "ref", "refs", "etc", "dr", "prof", "inc", "ch",
}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(“\"])")

logger = logging.getLogger("chunk")


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def is_continuation(previous: str, following: str) -> bool:
    """True when `following` continues the sentence `previous` was cut off in."""
    return not previous.rstrip().endswith((".", "!", "?", ":")) and following[:1].islower()


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, re-joining splits that follow a known abbreviation ("et al.", "Fig.")."""
    sentences: list[str] = []
    for piece in SENTENCE_END.split(text):
        # lstrip so a bracketed reference like "(Tab. 3)" still matches the abbreviation.
        last_word = sentences[-1].rsplit(" ", 1)[-1].lstrip("([").rstrip(".").lower() if sentences else ""
        if sentences and (last_word in ABBREVIATIONS or len(last_word) == 1):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def split_words(text: str, max_tokens: int) -> list[str]:
    words, pieces, current = text.split(), [], []
    for word in words:
        if current and est_tokens(" ".join([*current, word])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def units_for(paragraphs: list[dict], max_tokens: int) -> list[dict]:
    """Break paragraphs into sentence units (words if a sentence is too long): {text, page, para_start}.

    Sentence-sized units let chunk boundaries and overlaps fall between sentences, while
    para_start keeps the paragraph breaks.
    """
    units = []
    for paragraph in paragraphs:
        pieces = []
        for sentence in split_sentences(paragraph["text"]):
            pieces.extend([sentence] if est_tokens(sentence) <= max_tokens else split_words(sentence, max_tokens))
        for i, piece in enumerate(pieces):
            units.append({"text": piece, "page": paragraph["page"], "para_start": i == 0})
    return units


def join_units(units: list[dict]) -> str:
    out = ""
    for unit in units:
        sep = "" if not out else ("\n\n" if unit["para_start"] else " ")
        out += sep + unit["text"]
    return out


def pack(units: list[dict], max_tokens: int, overlap_tokens: int) -> list[dict]:
    """Greedy packing with a sentence-level overlap. Returns [{text, page}]."""
    units = [{**u, "idx": i} for i, u in enumerate(units)]
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for unit in units:
        if current and est_tokens(join_units([*current, unit])) > max_tokens:
            chunks.append(current)
            overlap: list[dict] = []
            for prev in reversed(current):
                if est_tokens(join_units([prev, *overlap])) > overlap_tokens:
                    break
                overlap.insert(0, {**prev, "para_start": False})
            # Never let the overlap plus the next unit exceed the limit.
            while overlap and est_tokens(join_units([*overlap, unit])) > max_tokens:
                overlap.pop(0)
            current = overlap
        current.append(unit)
    if current:
        chunks.append(current)

    if len(chunks) > 1 and est_tokens(join_units(chunks[-1])) < MIN_TAIL_TOKENS:
        seen = {u["idx"] for u in chunks[-2]}
        tail = [u for u in chunks[-1] if u["idx"] not in seen]
        if est_tokens(join_units([*chunks[-2], *tail])) <= MAX_MERGED_TOKENS * max_tokens / CHUNK_TOKENS:
            chunks.pop()
            chunks[-1].extend(tail)
    return [{"text": join_units(c), "page": c[0]["page"]} for c in chunks]


def chunk_paper(parsed: dict, meta: dict) -> list[dict]:
    arxiv_id = parsed["arxiv_id"]
    base = {
        "arxiv_id": arxiv_id,
        "title": meta["title"],
        "year": int(meta["published"][:4]),
        "url": meta["url"],
    }
    counters: Counter = Counter()

    def make(section: str, text: str, page: int) -> dict:
        n = counters[section]
        counters[section] += 1
        return {"chunk_id": f"{arxiv_id}#{section}-{n:02d}", **base, "section": section, "page": page, "text": text}

    chunks = [make("abstract", meta["abstract"], 1)]
    for section in parsed["sections"]:
        paragraphs = []
        for p in section["paragraphs"]:
            # A subheading is folded into the next paragraph so it stays with its content.
            if p.get("subheading"):
                paragraphs.append({"text": p["text"] + ".", "page": p["page"], "subheading": True})
            elif paragraphs and paragraphs[-1].get("subheading"):
                head = paragraphs.pop()
                paragraphs.append({"text": f"{head['text']} {p['text']}", "page": head["page"]})
            elif paragraphs and is_continuation(paragraphs[-1]["text"], p["text"]):
                # Text split at a column or page break: rejoin it into one paragraph.
                paragraphs[-1] = {**paragraphs[-1], "text": f"{paragraphs[-1]['text']} {p['text']}"}
            else:
                paragraphs.append(p)
        units = units_for([p for p in paragraphs if not p.get("subheading")], CHUNK_TOKENS)
        for piece in pack(units, CHUNK_TOKENS, OVERLAP_TOKENS):
            chunks.append(make(section["section"], piece["text"], piece["page"]))

    caption_units = units_for(parsed["captions"], CAPTION_CHUNK_TOKENS)
    for piece in pack(caption_units, CAPTION_CHUNK_TOKENS, 0):
        chunks.append(make("captions", piece["text"], piece["page"]))
    return chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    selected = {p["arxiv_id"]: p for p in json.loads(SELECTED_PATH.read_text())}
    all_chunks: list[dict] = []
    missing = []
    for arxiv_id, meta in selected.items():
        path = PARSED_DIR / f"{arxiv_id}.json"
        if not path.exists():
            missing.append(arxiv_id)
            continue
        all_chunks.extend(chunk_paper(json.loads(path.read_text()), meta))

    ids = [c["chunk_id"] for c in all_chunks]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise RuntimeError(f"duplicate chunk IDs: {duplicates[:10]}")

    with CHUNKS_PATH.open("w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = sorted(est_tokens(c["text"]) for c in all_chunks)
    body = sorted(est_tokens(c["text"]) for c in all_chunks if c["section"] not in ("abstract", "captions"))
    q = statistics.quantiles(body, n=20)
    logger.info("%d papers, %d chunks -> %s (%d papers missing a parse: %s)",
                len(selected) - len(missing), len(all_chunks), CHUNKS_PATH, len(missing), missing)
    logger.info("body chunk tokens (est.): p5=%d median=%d p95=%d max=%d", q[0], statistics.median(body), q[-1], body[-1])
    logger.info("by section kind: %s", dict(Counter(
        c["section"] if c["section"] in ("abstract", "captions") else "body" for c in all_chunks
    )))
    logger.info("total est. tokens to embed: %d", sum(sizes))


if __name__ == "__main__":
    main()
