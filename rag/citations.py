"""Parses cited IDs out of model text, and finds claims whose citation is about a different paper.

Validation against this turn's retrieval comes in Phase 4c.

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


# A bare arXiv ID in prose, e.g. "LoCoSplat (arXiv:2610.04351)"; not the arXiv part of a chunk ID.
ARXIV_MENTION = re.compile(r"(?<![\d.#])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d#])")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z*])")


def _strip_citations(text: str) -> str:
    """Remove bracketed citations so the arXiv IDs inside chunk IDs don't count as mentions."""
    def drop(group: re.Match) -> str:
        tokens = [t.strip() for t in re.split(r"[,;]", group.group(1))]
        return " " if all(CHUNK_ID.fullmatch(t) or WEB_ID.fullmatch(t) for t in tokens) else group.group(0)

    return BRACKET.sub(drop, text)


def attribution_mismatches(text: str, source_texts: dict[str, str]) -> list[tuple[str, list[str], list[str]]]:
    """Claims that name a paper by arXiv ID but cite only sources that never mention that paper.

    `source_texts` maps a citation ID to the text that identifies its source (for a chunk, its paper's
    arXiv ID and title; for a web result, its URL, title and content). Citations not in it are skipped:
    whether they were retrieved at all is the validator's job. Returns (claim, cited IDs, arXiv IDs).
    """
    mismatches = []
    for line in text.splitlines():
        for claim in SENTENCE_BREAK.split(line):
            cited = [c for c in parse_citation_ids(claim) if c in source_texts]
            mentioned = sorted(set(ARXIV_MENTION.findall(_strip_citations(claim))))
            if cited and mentioned and not any(m in source_texts[c] for c in cited for m in mentioned):
                mismatches.append((claim.strip(), cited, mentioned))
    return mismatches
