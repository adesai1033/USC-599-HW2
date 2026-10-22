"""PyMuPDF -> section-split text.

  python ingest/parse_pdfs.py            # parse every PDF in data/raw/ not yet parsed
  python ingest/parse_pdfs.py --force    # re-parse everything
  python ingest/parse_pdfs.py 2308.04079 # parse specific papers

Writes data/parsed/<arxiv_id>.json:
  {arxiv_id, title, sections: [{section, heading, paragraphs: [{text, page}]}],
   captions: [{text, page}], stats: {...}}

The PDF's own title block and abstract are dropped. chunk.py indexes the clean abstract from
the arXiv metadata instead. The references section is dropped, and any appendix after it is
kept.
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
SELECTED_PATH = DATA_DIR / "selected.json"

BOLD_FLAG = 16
BOLD_FONT = re.compile(r"Bold|CMBX|Semibold|SemiBold|Medi|Black|Heavy|\.B$", re.IGNORECASE)
FULL_WIDTH_RATIO = 0.55  # a block wider than this fraction of the page spans both columns
REPEAT_MIN_PAGES = 3  # short text repeated on this many pages is a running header/footer

KNOWN_HEADINGS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related-work",
    "related works": "related-work",
    "background": "background",
    "preliminaries": "preliminaries",
    "preliminary": "preliminaries",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "our approach": "method",
    "proposed method": "method",
    "experiments": "experiments",
    "experiment": "experiments",
    "experimental results": "experiments",
    "evaluation": "experiments",
    "results": "results",
    "discussion": "discussion",
    "limitations": "limitations",
    "limitations and future work": "limitations",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "conclusion and future work": "conclusion",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "references": "references",
    "bibliography": "references",
    "appendix": "appendix",
    "supplementary material": "appendix",
    "supplementary": "appendix",
}
DROP_SECTIONS = {"abstract", "references", "acknowledgments"}

NUMBERED_HEADING = re.compile(r"^(?P<num>(\d{1,2}|[A-H])(\.\d{1,2}){0,2})\.?\s+(?P<title>[A-Z][^\n]{1,80})$")
CAPTION = re.compile(r"^(Figure|Fig\.|Table)\s*[0-9A-Z]+[.:]?\s", re.IGNORECASE)
ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}(v\d+)?\s*\[")
PAGE_NUMBER = re.compile(r"^\d{1,3}$")
WORD = re.compile(r"[A-Za-z]{3,}")
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
SECTION_NUMBER = re.compile(r"^(\d{1,2}|[A-H])(\.\d{1,2}){0,2}\.?$")

logger = logging.getLogger("parse_pdfs")


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "section"


def clean(text: str) -> str:
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    return re.sub(r"\s*\n\s*", " ", text).strip()


def block_lines(block: dict) -> list[dict]:
    """Return the block's lines as {text, size, bold}, merging a bare section number with the next line."""
    lines = []
    for line in block["lines"]:
        spans = [s for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        lines.append(
            {
                "text": "".join(s["text"] for s in line["spans"]).strip(),
                "size": max(s["size"] for s in spans),
                "bold": all(s["flags"] & BOLD_FLAG or BOLD_FONT.search(s["font"]) for s in spans),
            }
        )
    merged: list[dict] = []
    for line in lines:
        if merged and SECTION_NUMBER.match(merged[-1]["text"]) and line["text"][:1].isupper():
            merged[-1] = {**line, "text": f"{merged[-1]['text']} {line['text']}"}
        else:
            merged.append(line)
    return merged


def is_equation_fragment(text: str) -> bool:
    """Short blocks dominated by symbols, digits or single letters (displayed math, garbled LaTeX)."""
    words = WORD.findall(text)
    letters = sum(c.isalpha() for c in text)
    if len(text) < 200 and len(words) < 4:
        return True
    return letters / max(len(text), 1) < 0.55


def reading_order(blocks: list[dict], page_width: float) -> list[dict]:
    """Order blocks for 1- or 2-column layouts.

    Full-width blocks split the page into horizontal bands. Within each band, the left
    column is read top to bottom, then the right column.
    """
    mid = page_width / 2
    blocks = sorted(blocks, key=lambda b: b["bbox"][1])
    ordered: list[dict] = []
    band: list[dict] = []

    def flush() -> None:
        left = [b for b in band if b["bbox"][0] < mid]
        right = [b for b in band if b["bbox"][0] >= mid]
        ordered.extend(sorted(left, key=lambda b: b["bbox"][1]))
        ordered.extend(sorted(right, key=lambda b: b["bbox"][1]))
        band.clear()

    for block in blocks:
        x0, _, x1, _ = block["bbox"]
        if (x1 - x0) > FULL_WIDTH_RATIO * page_width:
            flush()
            ordered.append(block)
        else:
            band.append(block)
    flush()
    return ordered


def classify_heading(line: dict, body_size: float) -> tuple[str, str, int] | None:
    """Return (slug, heading title, level) if this line looks like a section heading."""
    text = line["text"].strip()
    if len(text) > 90 or text.endswith((",", ";")) or (text.endswith(".") and not SECTION_NUMBER.match(text)):
        return None
    match = NUMBERED_HEADING.match(text)
    title = (match["title"] if match else text).strip().rstrip(":.")
    if not WORD.search(title):
        return None
    emphasized = line["bold"] or line["size"] >= body_size + 0.8 or (title.isupper() and len(title) > 3)
    if not emphasized:
        return None
    known = KNOWN_HEADINGS.get(title.lower())
    if match:
        level = 1 if match["num"][0].isalpha() and "." not in match["num"] else match["num"].count(".") + 1
        return known or slugify(title), title, level
    if known:
        return known, title, 1
    return None


def furniture_texts(pages: list) -> set[str]:
    """Short block texts (digits ignored) that repeat on several pages: running headers and footers."""
    seen: Counter = Counter()
    for _, _, _, blocks in pages:
        keys = {re.sub(r"\d+", "#", clean(b["_raw"])) for b in blocks if len(b["_raw"]) < 150}
        seen.update(keys)
    return {key for key, n in seen.items() if n >= REPEAT_MIN_PAGES}


def split_unheaded_introduction(sections: list[dict], abstract: str, stats: Counter) -> None:
    """Some papers start the introduction right after the abstract with no heading.

    Text in the PDF's abstract section beyond the length of the arXiv abstract is moved into an
    `introduction` section instead of being dropped with the abstract.
    """
    if any(s["section"] == "introduction" for s in sections):
        return
    for i, section in enumerate(sections):
        if section["section"] != "abstract":
            continue
        seen, cut = 0, None
        for j, paragraph in enumerate(section["paragraphs"]):
            seen += len(paragraph["text"])
            if seen >= 0.8 * len(abstract):
                cut = j + 1
                break
        if cut is not None and cut < len(section["paragraphs"]):
            intro = {"section": "introduction", "heading": None, "paragraphs": section["paragraphs"][cut:]}
            section["paragraphs"] = section["paragraphs"][:cut]
            sections.insert(i + 1, intro)
            stats["unheaded_introduction"] = 1
        return


def parse_pdf(path: Path, title: str, abstract: str) -> dict:
    doc = pymupdf.open(path)
    pages = []
    sizes: Counter = Counter()
    for page_no, page in enumerate(doc, 1):
        data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        blocks = [b for b in data["blocks"] if b.get("type") == 0 and b.get("lines")]
        for b in blocks:
            b["_raw"] = "\n".join("".join(s["text"] for s in line["spans"]) for line in b["lines"])
            for line in b["lines"]:
                for s in line["spans"]:
                    sizes[round(s["size"], 1)] += len(s["text"])
        pages.append((page_no, page.rect.width, page.rect.height, blocks))
    doc.close()
    body_size = sizes.most_common(1)[0][0] if sizes else 10.0
    repeated = furniture_texts(pages)

    sections: list[dict] = []
    captions: list[dict] = []
    stats: Counter = Counter()
    current = {"section": "front-matter", "heading": None, "paragraphs": []}

    def add_paragraph(lines: list[str], page_no: int) -> None:
        text = clean("\n".join(lines))
        if not text:
            return
        if is_equation_fragment(text):
            stats["equation_fragments"] += 1
        else:
            current["paragraphs"].append({"text": text, "page": page_no})

    for page_no, width, height, blocks in pages:
        for block in reading_order(blocks, width):
            raw = clean(block["_raw"])
            if not raw:
                continue
            _, y0, _, y1 = block["bbox"]
            if (
                ARXIV_STAMP.search(raw)
                or PAGE_NUMBER.match(raw)
                or re.sub(r"\d+", "#", raw) in repeated
                or ((y1 < height * 0.06 or y0 > height * 0.94) and len(raw) < 120)
            ):
                stats["furniture"] += 1
                continue
            if CAPTION.match(raw):
                captions.append({"text": raw, "page": page_no})
                continue

            buffer: list[str] = []
            for line in block_lines(block):
                heading = classify_heading(line, body_size)
                if not heading:
                    buffer.append(line["text"])
                    continue
                add_paragraph(buffer, page_no)
                buffer = []
                slug, heading_title, level = heading
                if level == 1 or current["section"] == "front-matter":
                    sections.append(current)
                    current = {"section": slug, "heading": heading_title, "paragraphs": []}
                    stats["headings"] += 1
                else:
                    current["paragraphs"].append({"text": heading_title, "page": page_no, "subheading": True})
                    stats["subheadings"] += 1
            add_paragraph(buffer, page_no)
    sections.append(current)
    split_unheaded_introduction(sections, abstract, stats)

    kept = []
    for section in sections:
        if section["section"] in DROP_SECTIONS or section["section"] == "front-matter":
            stats[f"dropped_{section['section']}"] += 1
            continue
        if any(not p.get("subheading") for p in section["paragraphs"]):
            kept.append(section)

    if not kept:
        # No headings detected: keep everything except references as one section.
        body = [p for s in sections if s["section"] != "references" for p in s["paragraphs"]]
        kept = [{"section": "body", "heading": None, "paragraphs": body}]
        stats["no_headings_fallback"] = 1

    return {
        "arxiv_id": path.stem,
        "title": title,
        "sections": kept,
        "captions": captions,
        "stats": {
            **stats,
            "pages": len(pages),
            "sections_kept": len(kept),
            "body_chars": sum(len(p["text"]) for s in kept for p in s["paragraphs"]),
            "captions": len(captions),
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arxiv_ids", nargs="*", help="parse only these papers")
    parser.add_argument("--force", action="store_true", help="re-parse papers already in data/parsed/")
    args = parser.parse_args()

    papers = {p["arxiv_id"]: p for p in json.loads(SELECTED_PATH.read_text())}
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = [RAW_DIR / f"{a}.pdf" for a in args.arxiv_ids] if args.arxiv_ids else sorted(RAW_DIR.glob("*.pdf"))

    failed, fallbacks = [], []
    for path in paths:
        out = PARSED_DIR / f"{path.stem}.json"
        if out.exists() and not args.force and not args.arxiv_ids:
            continue
        try:
            paper = papers.get(path.stem, {})
            parsed = parse_pdf(path, paper.get("title", ""), paper.get("abstract", ""))
        except Exception:
            logger.exception("failed to parse %s", path.name)
            failed.append(path.stem)
            continue
        out.write_text(json.dumps(parsed, indent=1, ensure_ascii=False))
        s = parsed["stats"]
        if s.get("no_headings_fallback"):
            fallbacks.append(path.stem)
        logger.info(
            "%s: %d pages, %d sections, %d captions, %d chars, %d equation fragments dropped",
            path.stem, s["pages"], s["sections_kept"], s["captions"], s["body_chars"], s.get("equation_fragments", 0),
        )
    logger.info("done; %d failed %s; %d with no headings detected %s", len(failed), failed, len(fallbacks), fallbacks)


if __name__ == "__main__":
    main()
