# Process Log

## AI tools used

## Development narrative

## Verification narrative

## What I learned

---

## Change log

Dated record of each code change, newest first.

### 2026-10-21 — Phase 3: MCP servers and client

**Dependencies (approved):** `mcp==1.30.0` and `langchain-mcp-adapters==0.3.2`. The latest
`mcp` is 2.3.0, but AGENTS.md specifies the 1.x SDK and the adapters require `mcp<2`.

**`rag/retriever.py`:** an LCEL runnable over the Chroma collection with `arxiv_ids` and
`sections` filters. It returns chunk metadata plus a cosine-similarity score.
- **Finding:** Chroma 1.5.9 can't open a read-only directory ("attempt to write a readonly
  database"), and the index in the image is read-only for `appuser`.
- **Workaround:** the retriever copies the index to a temp directory at startup and opens the
  copy. The shipped files are never written.

**`mcp_servers/corpus.py`** (FastMCP, stdio): `search_corpus` and `get_chunks`.
- **Validation:** `k` must be 1–12; arXiv IDs are normalized and must be in the corpus;
  section names must exist; query and ID-list sizes are capped. Problems come back as clear
  tool errors.
- **Startup:** the index loads once, before the server accepts requests.

**`mcp_servers/papers_db.py`** (FastMCP, stdio): `describe_schema`, `run_sql` and
`get_paper_metadata`.
- **`run_sql` guard layers:**
  - exactly one SELECT/WITH statement, checked with string literals and comments blanked out;
  - a forbidden-keyword check (PRAGMA, ATTACH, DDL, DML);
  - a `mode=ro` connection;
  - a SQLite authorizer that allows only reads;
  - a 200-row cap;
  - a 2-second limit enforced by a progress handler.
- **Schema notes:** `describe_schema` explains how to rank by `higher_is_better` and when to
  flag unverified rows.

**`mcp_client.py`:** one persistent session per server, each owned by its own task.
- **Isolation:** a server that won't start, crashes or disconnects doesn't affect the others.
- **Calls:** every tool call goes through an interceptor that uses the server's current
  session. If the server is down, the call returns the manifest's description as a tool error.
  A crashed server restarts on a later call, after a cooldown.
- **Secrets:** the Tavily URL is built from `TAVILY_API_KEY`. A logging filter redacts the key
  from every log message and traceback, and failure messages are redacted too.
- **Filtering:** per-agent tool lists come from the new `agents/registry.py`.
- **`mcp_config.json`:** each manifest lists its tools, which env vars to pass to the server,
  and an "unavailable" description for the agents' prompts.

**Fixes found while testing:**
- FastMCP pretty-prints returned dicts, which costs tokens, so tools return compact JSON
  strings.
- FastMCP replaces the logging setup at import, so each server reconfigures with
  `force=True`.
- An empty result with a section filter now says that section names vary by paper. The 3DGS
  paper has no "method" section.

**Tests:**
- `tests/test_phase3.py` (34 tests, offline; 43 including Phase 1) covers:
  - the SQL guard (12 rejected statements) and `higher_is_better` ranking;
  - the row cap and timeout;
  - corpus `k` bounds and validation, against a fake-embedding index;
  - exact per-agent tool lists;
  - a server that won't start, a missing env var, and a tool error;
  - a crash mid-call, followed by a restart;
  - the Tavily key never appearing in log output.
- `tests/test_phase3_live.py` (`RUN_LIVE=1`, Phase 3's "done when"): every tool listed and
  called through the client, and a broken corpus manifest leaving the other servers working.
  Both pass.
- **In the container** (`paper-assistant:phase3`, uid 10001, `index/` not writable): all
  three servers come up in about 3 s. The corpus index loads through the temp-copy
  workaround (4,867 chunks). All five corpus and DB tools succeed, `DROP TABLE` is
  rejected, and the raw Tavily key appears in none of the output.

### 2026-10-21 — Chunk spot-check, chunker fix, `eval/questions.jsonl`

**Spot-check:** read 20 random chunks (seed 599).
- **Looked good:** reading order was correct in all 20, with no two-column interleaving, and
  the section labels were right.
- **Minor, left as is:**
  - Removing display equations leaves sentences like "minimizing the following loss:"
    dangling.
  - About 6% of body chunks still have short runs of inline math.
  - Dehyphenation joins real compounds ("lowresources").
  - Sub-figure captions repeat.
  - One TSGS caption swallowed a body paragraph. The longest caption in the corpus is 1,333
    characters, so this is rare.

**Fixed in `chunk.py`:**
- Bracketed references such as "(Tab. 3)" or "(Fig. 2)" were split as sentence ends, because
  the abbreviation check saw "(tab". That affected 652 places.
- Caption chunks put a paragraph break between every sentence.

After re-chunking: 4,867 chunks and 0 split references. The index still needs a re-embed of
950 changed chunks, about $0.005, waiting on approval.

**`eval/questions.jsonl`:** 29 questions (18 lookup, 5 compare, 2 lit review, 3 metadata,
1 not in corpus) covering 26 papers.
- Every reference answer was written from sentences pulled out of the chunks, not from memory.
- Three claims I'd written from memory were checked against the text. One wasn't supported
  (NeuS "occlusion-aware") and was replaced.
- Metadata answers come from `papers.sqlite`.
- No benchmark-ranking questions yet, because the `results` table is empty until
  `extract_results.py` runs.

### 2026-10-21 — `ingest/build_db.py`

**Built:** `build_db.py` rebuilds `index/papers.sqlite` from `data/selected.json` using the
AGENTS.md schema: `papers`, `authors`, `paper_authors` and an empty `results` table.
- **Constraints:** `results` has CHECK constraints on `higher_is_better` and `verified`.
- **Indexes:** on year, author and (benchmark, metric).
- **Safe rebuild:** the DB is written to a temp file and swapped in at the end.

**Citation fallback:** Semantic Scholar reports 0 citations for NeRF, so zero or missing counts
fall back to OpenAlex.
- OpenAlex splits a paper's arXiv preprint and published version into separate records
  (NeRF: 524 on the arXiv DOI, 5,324 on the ECCV version).
- The fallback searches by title and takes the highest count among exact title matches. NeRF
  was the only paper that needed it.

**Result:** 220 papers, 922 authors, 1,297 author links, no missing citation counts, and 1 paper
with no venue. Top cited: 3DGS 11,022, Instant-NGP 6,490, NeRF 5,324.

### 2026-10-21 — `ingest/embed.py`

**Built:** `embed.py` embeds `data/chunks.jsonl` with `text-embedding-3-small` into the
persistent Chroma collection `papers` in `index/chroma/`, using cosine distance.
- **Metadata:** every chunk keeps `chunk_id, arxiv_id, title, year, section, page, url`.
- **Incremental:** each chunk stores a hash of its text, so a re-run embeds only new or
  changed chunks and deletes ones that were removed. Batches of 256 are written as they
  finish.
- **Dry run:** `--dry-run` gives an exact tiktoken count and a cost estimate without any API
  calls.
- **Model check:** the script refuses to mix embedding models in one index.

**Dependencies (approved):** `langchain-openai==1.7.2`, `langchain-chroma==1.1.0` and
`chromadb==1.5.9`, added to `requirements.txt` because the runtime retriever needs them.

**Result:**
- 4,870 chunks, 2,132,107 tokens, about $0.043 at $0.02 per 1M tokens (price checked
  2026-10-21). The index is 121 MB.
- A second run embedded nothing.

**Retrieval check:** 10 test queries through `langchain-chroma`, top 6 each. All 10 were
relevant, and the target paper ranked first for Instant-NGP, Mip-NeRF, SplaTAM, BARF and
D-NeRF. For the 3DGS rendering query, the top 6 came from related Gaussian Splatting papers
rather than the original. Something to watch in RAGAS.

### 2026-10-21 — `ingest/chunk.py`

**Built:** `chunk.py` reads `data/parsed/` and writes `data/chunks.jsonl` with the fields
`chunk_id, arxiv_id, title, year, section, page, url, text`.
- **Abstracts:** each paper's arXiv abstract becomes one chunk with `section="abstract"`.
- **Body:** each section is split into sentences (and into words if a sentence is too long),
  then packed greedily into chunks of up to 800 tokens with about 100 tokens of overlap.
  Chunks never cross section boundaries, and a final chunk under 150 tokens is merged into
  the previous one (merged size capped at 1,000).
- **Splitting details:** sentence splitting ignores abbreviations such as "et al." and
  "Fig.". Subheadings stay attached to their first paragraph, and paragraphs cut at a
  column or page break are rejoined.
- **Captions:** packed into chunks of up to 300 tokens with `section="captions"`.
- **Chunk IDs:** `{arxiv_id}#{section_slug}-{nn}`, numbered per slug within a paper. The run
  fails if any ID is duplicated.
- **Token counts:** estimated at 4 characters per token, which avoids adding a tokenizer
  dependency.

**Bugs found and fixed while checking the output**
- When a short final chunk was merged into the previous one, the overlap sentences were
  duplicated. Units are now de-duplicated by index.
- Only 35% of consecutive chunks actually overlapped, because whole paragraphs were the
  packing unit. Packing by sentence raised that to 90%; the rest sit next to a single
  sentence longer than 100 tokens.
- There were mid-sentence paragraph breaks at column and page splits. 0 remain.

**Result:** 220 papers produce 4,870 chunks: 220 abstract, 3,714 body and 936 caption. Body
chunks are about 126 tokens at p5, 732 at the median, 798 at p95 and 877 at most. About
2.44M tokens in total to embed.

### 2026-10-21 — Full download and parse; parser fixes from corpus-wide checks

**Done:** downloaded all 220 PDFs (3.7 GB, gitignored) and parsed all 220. The first pass
skipped 14 PDFs over the old 40 MB cap, and the second pass at the 80 MB cap fetched them.
`caffeinate -is` kept the Mac awake for the download, retry and parse.

**Fixes from checking the whole corpus**
- **IEEE-style headings:** Roman-numeral headings ("I. INTRODUCTION") weren't recognised, so
  4 SLAM papers had 1–2 sections (SemGauss-SLAM kept 3.7k characters) and 4 more had no
  headings at all. The numbered-heading pattern now accepts I–X.
- **False headings:** 7 of 2,045 headings were table cells or axis labels ("FPS", "Avg.",
  "Iterations (×103)", one SVD formula) merged with a number on the line above. Number
  merging now only happens in 2-line, heading-only blocks. Titles must be at least 4
  characters and at least 80% letters.

**Final parse stats:** 220/220 parsed with 0 failures and 0 single-section fallbacks. Every
paper has an introduction section. The median paper has 9 sections and 34k characters
(minimum 12k); the corpus totals 7.9M characters. 3,860 captions, with every paper having
some. No odd headings remain.

**Spot check** (20 random paragraphs, seed 599): reading order across columns is correct,
inline math is readable, and there's no garbled LaTeX. Paragraphs split at column or page
breaks start mid-sentence; chunk.py joins paragraphs within a section, so this resolves
there. Subfigure captions starting "(a) …" stay in the body text, which is a minor issue.

### 2026-10-21 — `ingest/parse_pdfs.py`; PDF download started

**Built:** `parse_pdfs.py` (PyMuPDF, `pymupdf==2.0`, approved). For each PDF it writes
`data/parsed/<id>.json`.
- **Reading order:** for two-column pages, full-width blocks split the page into bands, and
  each band is read left column then right column.
- **Headings:** detected line by line, so a heading inside a paragraph block is still found.
  Numbered, known-name and ALL-CAPS headings count when they're bold (including LaTeX `CMBX`
  fonts) or larger than the body text. A bare "1" line is merged with the heading on the next
  line. Subsections fold into their top-level section.
- **Captions:** kept as a separate list (`Figure N` / `Fig.` / `Table N`).
- **Removed:**
  - equation and table fragments (short blocks, or mostly non-letters)
  - running headers and footers (short text that repeats on 3 or more pages, plus margin
    text, arXiv stamps and page numbers)
  - the front matter, the PDF's abstract, references and acknowledgments
- **Fallbacks:** an introduction with no heading is recovered by comparing against the arXiv
  abstract. If no headings are found at all, everything goes into a single `body` section.

**Debugging:** the first version lost most of the text in Instant-NGP and NeRF and dropped
the Mip-NeRF 360 introduction. The causes were headings merged into paragraph blocks, ALL-CAPS
headings split across two lines, LaTeX bold fonts not setting the bold flag, and an
introduction with no heading. After the fixes, all four test papers have sensible sections:
NeRF 11, Mip-NeRF 360 13, Instant-NGP 12, 3DGS 11. I sampled the dropped fragments and they're
flattened results tables and math, not prose.

**Known limitation:** rejoining words hyphenated across a line break also merges real
compounds ("coordinate-based" becomes "coordinatebased").

**Download:** the size cap in `corpus_spec.yaml` went from 40 to 80 MB after 7 papers were
skipped for being 40–60 MB, including the Plenoxels and Nerfstudio seeds. A second download
pass will fetch them.

### 2026-10-21 — Hand-curated corpus exclusions

**Done:** after reviewing the borderline papers, I added four IDs to `exclude.arxiv_ids`
and re-ran `select`. IGEV stereo matching (2303.06615) was kept, as were the LiDAR and
generative-scene papers.

| Removed | Replaced by (same topic) |
|---|---|
| WIRE (2301.05187) | S-NeRF: NeRF for Street Views (2303.00749) |
| The Temporal Opportunist (2104.14540) | MVSTER (2204.07346) |
| MVImgNet (2303.06042) | IterMVS (2112.05126) |
| Best 3D Scene Representation for Robotics (2512.03422) | GS-LIVO (2501.08672) |

**Result:** still 220 papers, with every topic at quota.

### 2026-10-21 — `ingest/fetch_arxiv.py`, select step run

**Built:** `fetch_arxiv.py` with two steps, `select` and `download`. Neither calls a paid API.
- **`select`:** runs 17 arXiv queries, keeps papers that match the categories and date range,
  and applies the exclusions. It then looks up citation counts in batches on Semantic Scholar
  and fills each topic's quota (seeds, then recent papers, then the most cited).
- **`select` outputs:** `data/candidates.json`, `data/selected.json` and
  `data/selection_report.md`.
- **`download`:** fetches each selected PDF to `data/raw/`, skips files that already exist,
  checks the bytes really are a PDF, and enforces the size cap.
- **Rate limits:** waits 3 s between arXiv requests and retries 429 and 5xx responses with
  backoff. The User-Agent identifies the project only.
- **Dependency:** added `pyyaml==6.0.3` to `requirements-dev.txt` (approved).

**Spec changes:** after the first run I tightened the title exclusions (humans, faces,
portraits, medical, CT and tomography, instance segmentation, grasping, adversarial and
poisoning papers, "recent advances" surveys). I also added `exclude.arxiv_ids` for hand
curation.

**Results:** 1,874 candidates; 146 excluded by filters; 220 selected, with every topic at
quota. By year: 2020 14, 2021 41, 2022 21, 2023 43, 2024 60, 2025 40, 2026 1.

**Found:** Semantic Scholar reports `citationCount: 0` for NeRF (2003.08934). This is an
error in their record, not in our code. NeRF is still included because it's a seed, but
`build_db.py` needs a citation fallback or "most-cited" queries will be wrong.

**Still open:** hand review of the borderline papers in the report, then the `download` step.

### 2026-10-21 — Phase 2 started: `ingest/corpus_spec.yaml`

**Done:** wrote the corpus spec covering scope, the `cs.CV`/`cs.GR` categories, a date range
of 2020-01-01 to 2026-10-21, and a target of 220 papers (min 150, max 300).
- **44 seed papers across 8 topics:** NeRF, fast radiance fields, Gaussian Splatting,
  surfaces, MVS, feed-forward, dynamic, and SLAM.
- **Fill-out queries:** each topic has arXiv queries and a quota. The quotas sum to 220.
  Candidates are ranked by citation count, with a minimum number of papers from 2025 or
  later per topic.
- **Exclusions:** title and abstract terms that filter out surveys, avatars, text-to-3D,
  detection, segmentation and pose papers.

**Verified:** checked every seed ID against the arXiv API. Two IDs I had from memory were
wrong (Neuralangelo and PatchmatchNet) and were corrected to 2306.03092 and 2012.01411. The
YAML parses, all 44 seeds are unique, and the quotas sum to the target.

**Decision pending:** MVSNet (1804.02505) is from 2018, before the 2020 start date, so it
was left out.

### 2026-10-21 — Phase 0 smoke test passed

**Done:** ran a one-off curl script (not committed) that made one call to each provider,
one Tavily search, and one Firestore write, read and delete. Key values were never printed.

**Results** (all PASS)
- Gemini `gemini-3.8-flash`, Claude `claude-sonnet-5-5`, and OpenAI `gpt-6.1-sol`
  (via the Responses API) each replied "ok".
- The `text-embedding-3-small` embedding came back with 1536 dimensions.
- Tavily search returned the 3DGS paper (arXiv 2308.04079).
- The Firestore `(default)` database is Native mode in `us-west1`. The write, read and
  delete all succeeded using my gcloud login.
- The runtime service account `paper-assistant-run` has only `roles/datastore.user`.

**Still open:** Firestore access as the service account itself gets tested from Cloud Run
in Phase 7. Tavily through the Remote MCP endpoint (key in the URL) gets tested in Phase 3.

### 2026-10-21 — Phase 1 deployed to Cloud Run (revision `paper-assistant-00001-2hw`)

**Done:** ran `./deploy.sh` (Cloud Build to Artifact Registry, then Cloud Run in `us-west1`
as `paper-assistant-run`). No code changes.

**Verified on the deployed URL** (`https://paper-assistant-428458746699.us-west1.run.app`)
- `GET /health` returns `{"status":"ok"}`.
- `POST /chat/stream` sends many `token` frames followed by exactly one `done`.
  TTFT was 7.3 s and the total 14.5 s on a warm instance; most of the TTFT is Gemini
  thinking time.

**Still open**
- Confirm the stream isn't being buffered on the way through, using timestamped curl output.
- Test a request from a cold start.
- The UI shows Markdown and LaTeX as raw text.
- Gemini cites papers from its own memory. The citation validator in a later phase will
  block this.

### 2026-10-21 16:53 PDT — Phase 1 walking skeleton (`dd9eaff`)

**Goal:** before any multi-agent code, prove that FastAPI, SSE token streaming, the static
frontend, Docker and Cloud Run all work end to end.

**Built**
- `config.py`: reads settings from env (`PORT`, `GOOGLE_API_KEY`, `GEMINI_MODEL`,
  `LOG_LEVEL`), loads `.env` locally, fails fast on missing keys, and hides the key in `repr`.
- `llm.py`: `get_llm("fast")` returns a streaming `ChatGoogleGenerativeAI` using
  `GEMINI_MODEL`, with a 60 s timeout and 2 retries.
- `streaming.py`: `sse_event()` encodes frames, and `with_heartbeat()` sends `: ping` every 15 s.
- `main.py`: `GET /health`, `POST /chat/stream` (events `token` / `error` / `done` with
  `ttft_ms` and `total_ms`), and `static/` served at `/`. The stream always ends with `done`
  and returns HTTP 200 even on errors.
- `static/`: vanilla chat UI that reads the stream with fetch and a ReadableStream, buffers
  partial SSE frames, shows TTFT and total, and supports light and dark mode.
- `Dockerfile` (multi-stage, `/opt/venv`, non-root `appuser`), `.dockerignore`,
  `.gcloudignore`, `.gitignore` (changed to ignore `index/`), and `deploy.sh` (Cloud Build to
  Artifact Registry, then Cloud Run with the runtime service account).
- `requirements*.txt` pinned with `==`, `.env.example` with verified model IDs, and a
  `README.md` stub.
- `tests/`: 9 pytest tests with the LLM mocked.

**Verified**
- `pytest tests/`: all 9 tests pass.
- Local curl against live Gemini: tokens arrive incrementally (TTFT about 5.2 s, total about 6.7 s).
- A bad API key gives `error` then `done` with HTTP 200, and the key never appears in the logs.
- The browser UI streams and shows TTFT and total.
- The Docker image builds and runs as uid 10001 with `-e PORT=8080`. `.env` and `tests/`
  are not in the image.

**Not done yet:** `deploy.sh` has been written but not run. Nothing has been pushed.
