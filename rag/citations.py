"""Parses cited IDs out of model text. Validation against this turn's retrieval comes in Phase 4c.

Corpus chunks are cited as [2308.04079#method-03] and web sources as [web:1a2b3c4d]. A bracket may
hold several IDs separated by commas or semicolons, and citations may be adjacent ([a][b]).
"""

import re

# The corpus has only new-style arXiv IDs (YYMM.NNNNN), so old-style IDs (cs/0601001) are not matched.
CHUNK_ID = re.compile(r"\d{4}\.\d{4,5}#[a-z0-9]+(?:-[a-z0-9]+)*-\d{2,}")
WEB_ID = re.compile(r"web:[0-9a-f]{8}")
BRACKET = re.compile(r"\[([^\[\]\n]{1,400})\]")


def parse_citation_ids(text: str) -> list[str]:
    """Every cited corpus or web ID, in order of first appearance, without duplicates."""
    ids: list[str] = []
    for group in BRACKET.finditer(text):
        for token in re.split(r"[,;]", group.group(1)):
            token = token.strip()
            if (CHUNK_ID.fullmatch(token) or WEB_ID.fullmatch(token)) and token not in ids:
                ids.append(token)
    return ids


def is_web_id(citation_id: str) -> bool:
    return citation_id.startswith("web:")
