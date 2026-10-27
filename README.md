# CSCI 599 Assignment 2 — 3D Vision Paper Research Assistant

A research assistant for 3D computer vision papers: NeRF, 3D Gaussian Splatting, neural surfaces,
multi-view stereo, feed-forward reconstruction and neural SLAM. You ask in natural language, for
example:
- "How does 3DGS render a novel view?"
- "Which methods report the best PSNR on Mip-NeRF 360?"
- "What 3D reconstruction papers came out this month?"
- "Summarize the second one"

**How it answers:**
- **Workers:** a LangGraph **supervisor** routes the work to three workers:
  - `research`: corpus RAG plus web search;
  - `data`: SQL over a paper and benchmark database;
  - `summary`: multi-paper reviews.
- **Tools:** the workers reach their tools through three MCP servers.
- **Answer:** it streams token by token over SSE, with citations to specific paper chunks, a visible
  routing decision per model call, and an inspectable trace.
- **Providers:** generation is split between Gemini and Claude, and each is the other's fallback.
- **History:** conversations persist in Firestore, so they survive Cloud Run instance replacement.

- **Deployed:** https://paper-assistant-7j3km77uuq-uw.a.run.app (Cloud Run, `us-west1`).
- **Architecture:** [`docs/architecture.md`](docs/architecture.md), with the diagrams, state
  schema, agents, routing, RAG, MCP and durable state.
- **Orchestration flow:** [`docs/orchestration_flow.md`](docs/orchestration_flow.md), one question
  traced through the supervisor, workers, MCP servers and failure paths.
- **Evaluation:** [`docs/evaluation.md`](docs/evaluation.md), with RAGAS, latency, cost and the
  provider comparison.

## Local setup

Requires Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt     # runtime + ingest + test dependencies
cp .env.example .env                    # then fill in real keys; .env is gitignored
```

**Keys** (`.env`):

| Key | Required? | Used for |
|---|---|---|
| `GOOGLE_API_KEY` | required | Gemini |
| `ANTHROPIC_API_KEY` | optional | Claude. Without it, Claude calls fail and fall back to Gemini. |
| `OPENAI_API_KEY` | required for corpus search | query embeddings |
| `TAVILY_API_KEY` | required for web search | Tavily |

**Built index:** the app also needs `index/chroma/` and `index/papers.sqlite`. They're gitignored, so
build them with the [ingestion](#ingestion) steps below. Ingestion runs offline and never at
request time.

## Run locally

```bash
uvicorn main:app --port 8080            # STATE_BACKEND defaults to memory (not durable)
```

- **Browser:** open http://localhost:8080. Startup waits for the MCP servers, a few seconds locally,
  before the port opens.
- **Terminal:** ask a question with a live trace and streamed answer, or open a REPL:

  ```bash
  python scripts/cli.py "How does 3D Gaussian Splatting render a novel view?"
  python scripts/cli.py                     # REPL
  ```

- **curl:**

  ```bash
  curl -N -X POST localhost:8080/chat/stream -H 'Content-Type: application/json' \
    -d '{"session_id":"t1","message":"What is NeRF?"}'
  ```

**Tests:**

```bash
pytest tests/                                   # offline; models and MCP results are faked in tests only
RUN_LIVE=1 pytest tests/test_phase3_live.py     # every tool against the real servers (about 2 Tavily credits)
```

**Failure probes:**

```bash
python scripts/cli.py --mcp-config tests/fixtures/mcp_config.broken_corpus.json "How does NeRF work?"
ANTHROPIC_API_KEY=invalid python scripts/cli.py "Compare NeRF and 3DGS"   # falls back to Gemini
```

## Ingestion

Each step is offline and incremental. The ones that call a paid API have a `--dry-run` that
estimates the cost first.

```bash
python ingest/fetch_arxiv.py select       # arXiv + Semantic Scholar -> data/selected.json (220 papers; review data/selection_report.md)
python ingest/fetch_arxiv.py download     # PDFs -> data/raw/
python ingest/parse_pdfs.py               # PyMuPDF -> data/parsed/ (section-split text, captions)
python ingest/chunk.py                    # -> data/chunks.jsonl (4,741 chunks)
python ingest/embed.py --dry-run && python ingest/embed.py   # -> index/chroma/ (about $0.04)
python ingest/build_db.py                 # -> index/papers.sqlite (papers, authors, empty results)
python ingest/extract_results.py --dry-run && python ingest/extract_results.py   # benchmark rows, verified=0 (about $4)
python ingest/seed_results.py             # 99 hand-checked rows from ingest/results_seed.csv, verified=1
```

- **Scope:** the corpus is defined once in `ingest/corpus_spec.yaml`: categories, date range, 44
  seed papers, per-topic queries and quotas, and exclusions.
- **Results:** `extract_results.py` caches each page's model reply in `data/extracted_raw/`, so a
  re-run after a code change is free.
- **Order:** after a database rebuild, run `build_db.py → extract_results.py → seed_results.py`.

## Evaluation

RAGAS 0.4.3 pins `openai<2`, which conflicts with the app, so it has its own virtualenv:

```bash
python3.13 -m venv .venv-eval && .venv-eval/bin/pip install -r eval/requirements-ragas.txt
.venv/bin/python eval/run_ragas.py answer baseline                  # run the graph on eval/questions.jsonl
.venv-eval/bin/python eval/run_ragas.py score baseline              # judge (gpt-6.1-sol), about $4 per pass
.venv/bin/python eval/run_ragas.py answer gemini --provider google  # force one provider
.venv/bin/python eval/latency.py https://<service-url> warm --repeats 3
.venv/bin/python eval/cost.py baseline gemini
```

Results are written to `eval/results/` and summarized in [`docs/evaluation.md`](docs/evaluation.md).

## Docker

```bash
docker build -t paper-assistant .
docker run --rm -p 8080:8080 --env-file .env paper-assistant
```

The image is multi-stage on `python:3.13-slim`, runs as the non-root `appuser`, and puts its venv in
`/opt/venv`. It bakes in the pre-built `index/`, read-only. `.dockerignore` and `.gcloudignore`
exclude `data/`, `.env`, both venvs, `tests/`, `eval/` and `ingest/`, none of which run in the
container. `.gcloudignore` must not exclude `index/`, because
without it `gcloud builds submit` would fall back to `.gitignore` and ship an image with no corpus.

## Deploy (Cloud Run)

```bash
set -a; source .env; set +a
./deploy.sh
```

**What `deploy.sh` does:**
- **Project:** `GCP_PROJECT_ID` and `GCP_REGION` from `.env` select the project and region; unset,
  they default to ours (`usc-599-hw2`, `us-west1`).
- **Build:** builds the image with Cloud Build and pushes it to Artifact Registry
  (`${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/images/paper-assistant`).
- **Deploy:** deploys the Cloud Run service `paper-assistant` with `--memory 2Gi --cpu 1
  --concurrency 10 --timeout 300 --min-instances 0 --max-instances 3`.
- **Identity:** the service runs as `paper-assistant-run@${GCP_PROJECT_ID}.iam.gserviceaccount.com`.
- **Settings:** it passes every key and model ID with `--set-env-vars` from shell variables, never
  literals, plus `STATE_BACKEND=firestore`.

Traffic follows the latest revision.

### Fresh GCP project setup

To deploy into a new project, set `GCP_PROJECT_ID` (and `GCP_REGION` if not `us-west1`) in `.env`;
`deploy.sh` needs no edits. Then, once per project:

```bash
set -a; source .env; set +a
PROJECT_ID="$GCP_PROJECT_ID"; REGION="$GCP_REGION"
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com firestore.googleapis.com

# Artifact Registry repo for the image
gcloud artifacts repositories create images --repository-format=docker --location="$REGION"

# Firestore (Native mode) for conversation history
gcloud firestore databases create --location="$REGION" --type=firestore-native

# Least-privilege runtime identity: Firestore access only (no key file; the client uses ADC)
gcloud iam service-accounts create paper-assistant-run --display-name="paper-assistant runtime"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:paper-assistant-run@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role=roles/datastore.user
```

- **Why a service account:** without `--service-account`, Cloud Run would run as the Compute Engine
  default account, which has project-wide Editor.
- **Permissions:** the account running `deploy.sh` needs permission to deploy Cloud Run services and
  to act as `paper-assistant-run`; a project Owner has both.
- **Build errors:** on some new projects the Cloud Build service account also needs
  `roles/artifactregistry.writer` to push the image. Grant it if the build fails with a permission
  error.
- **Checks after deploying:**
  - A cold-start request.
  - A multi-turn conversation that survives a new revision, which proves history is in Firestore.

## Providers and model IDs

Model IDs verified against the provider docs on 2026-10-21:

| Env var | Model ID | Source |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.8-flash` | https://ai.google.dev/gemini-api/docs/models |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | https://platform.claude.com/docs/en/about-claude/models/overview |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | https://developers.openai.com/api/docs/models |

**Tiers** (`llm.py`):
- **fast** = Gemini (`GEMINI_MODEL`).
- **strong** = Claude (`ANTHROPIC_MODEL`). Its thinking is limited to the gaps between tool calls
  with `thinking={"type": "between_tools"}`; Sonnet 5.5 rejects `disabled`.

**Fallback:** each tier is the other's fallback. A provider error, or a stalled supervisor call, on
Gemini retries once on Claude, and vice versa. A missing `ANTHROPIC_API_KEY` doesn't stop startup:
Claude's calls fail and fall back to Gemini.

**OpenAI** is used only for embeddings (`OPENAI_EMBEDDING_MODEL`, needed for corpus search) and as
the RAGAS judge in `eval/`. It was a third chat tier until Phase 9, when two things showed it wasn't
worth keeping:
- The forced-OpenAI run hit worker timeouts on every multi-paper question.
- When its credits ran out, embeddings broke too, so a fallback on the same provider failed together
  with corpus search.

**Routing:** `router.py` picks the tier per call; see the routing table in
[`docs/architecture.md`](docs/architecture.md).

## MCP servers

Three servers, declared in `mcp_config.json` (path in `MCP_SERVERS_CONFIG`) and connected by
`mcp_client.py` through `MultiServerMCPClient`, one persistent session per server:

| Server | Transport | Tools | Source |
|---|---|---|---|
| `corpus` | stdio | `search_corpus`, `get_chunks` | ours, `mcp_servers/corpus.py` |
| `papers_db` | stdio | `describe_schema`, `run_sql`, `get_paper_metadata` | ours, `mcp_servers/papers_db.py` |
| `tavily` | streamable HTTP | `tavily_search`, `tavily_extract` | Tavily Remote MCP (see Attribution) |

Each of our servers runs standalone: `python mcp_servers/corpus.py` (needs `OPENAI_API_KEY` to embed
queries) or `python mcp_servers/papers_db.py`.

## SSE contract

`POST /chat/stream` with body `{"session_id": str, "message": str}` returns `text/event-stream`, with
`Cache-Control: no-cache` and `X-Accel-Buffering: no`.

| Event | Payload |
|---|---|
| `routing` | a `RoutingDecision {call_site, agent, provider, model, tier, dimension, reason, fallback}` |
| `trace` | `{step, node, kind, agent, detail, ms}`; a supervisor `decision` also carries `{next, task_type, reason, instruction}` |
| `token` | `{text}`, from `synthesize` only |
| `citations` | `{valid: [{id, arxiv_id, title, section, page, url, source}], invalid: [id]}` |
| `error` | `{message}`: user-safe, no stack traces or secrets |
| `done` | `{session_id, timings: {ttft_ms, retrieval_ms, llm_ms, total_ms}, usage: {provider: {input, output}}}` |

The stream always ends with `done`, including after an `error`. A `: ping` heartbeat comment is sent
every 15 s.

**Other endpoints:**
- `GET /health`.
- `GET /sessions/{session_id}/turns`, which restores a conversation in the UI.

## Cost and access notes

Prices checked 2026-10-26; full tables are in [`docs/evaluation.md`](docs/evaluation.md#cost-per-query).

- **Per query (normal routing):**
  - Mean $0.079, median $0.040.
  - By type: lookup about $0.04, compare about $0.15, literature review about $0.32.
  - Claude is about 70% of the spend.
- **Model prices:** Gemini 3.8 Flash $0.75 / $3.75 per 1M input / output tokens. Claude Sonnet 5.5
  $2 / $10. `text-embedding-3-small` $0.02 per 1M.
- **One-time ingestion:** embeddings about $0.04, results extraction about $4.
- **Evaluation:** about $4 per 25-question RAGAS pass for the judge, plus about $2 to answer the
  questions.
- **Tavily:** one credit per search or extract.
- **Cloud Run:** `min-instances=0`, so idle time costs nothing. A cold start pays for container
  start, MCP server spawn and the Chroma load, about 50 s before the first request is handled.
- **Access:** all three API keys are pay-as-you-go:
  - An exhausted OpenAI balance breaks corpus search, because query embeddings use it.
  - An invalid or exhausted Anthropic key degrades to Gemini.

## Attribution

- **Tavily Remote MCP server** (`https://mcp.tavily.com/mcp/`), the official hosted MCP server
  from Tavily, provides live web search (`tavily_search`) and page extraction (`tavily_extract`).
  We use it unmodified; the API key is passed in the `tavilyApiKey` query parameter.
