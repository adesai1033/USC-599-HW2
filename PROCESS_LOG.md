# Process Log

## AI tools used

## Development narrative

## Verification narrative

## What I learned

---

## Change log

Dated record of each code change, newest first.

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
