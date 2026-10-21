# CSCI 599 Assignment 2 — 3D Vision Paper Research Assistant

A research assistant for 3D computer vision papers (NeRF, 3D Gaussian Splatting, multi-view
stereo, feed-forward reconstruction, neural SLAM). Answers stream token-by-token over SSE to
a vanilla HTML/JS frontend served by FastAPI, deployed on Google Cloud Run.

**Status:** Phase 1 (walking skeleton). The app has a single Gemini Flash call streamed over
SSE. Agents, RAG, MCP, routing and durable state come in later phases.

## Local setup

Requires Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # then fill in real keys; .env is gitignored
uvicorn main:app --reload --port 8080
```

Open http://localhost:8080. Required variables: `GOOGLE_API_KEY` and `GEMINI_MODEL`. The app
refuses to start if either is missing. `PORT` (default 8080) and `LOG_LEVEL` (default `INFO`)
are optional.

Stream a response from the command line:

```bash
curl -N -X POST localhost:8080/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"t","message":"What is NeRF?"}'
```

Run the tests (the LLM is mocked):

```bash
pytest tests/
```

## Docker

```bash
docker build -t paper-assistant .
docker run --rm -p 8080:8080 -e PORT=8080 \
  -e GOOGLE_API_KEY -e GEMINI_MODEL paper-assistant
```

## Deploy (Cloud Run)

```bash
set -a; source .env; set +a
./deploy.sh
```

`deploy.sh` builds the image with Cloud Build, pushes it to
`us-west1-docker.pkg.dev/usc-599-hw2/images/paper-assistant`, and deploys the Cloud Run
service `paper-assistant` in `us-west1`. The service runs as
`paper-assistant-run@usc-599-hw2.iam.gserviceaccount.com`. Keys are passed from shell
variables and are never written into the repo.

## SSE contract

`POST /chat/stream` with body `{"session_id": str, "message": str}`. Phase 1 events:
`token {text}`, `error {message}`, `done {session_id, timings: {ttft_ms, total_ms}}`. The
stream always ends with `done`. A `: ping` heartbeat comment is sent every 15 s.

## Providers and model IDs

Model IDs verified against the provider docs on 2026-10-21:

| Env var | Model ID | Source |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.8-flash` | https://ai.google.dev/gemini-api/docs/models |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | https://platform.claude.com/docs/en/about-claude/models/overview |
| `OPENAI_MODEL` | `gpt-6.1-sol` | https://developers.openai.com/api/docs/models |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | https://developers.openai.com/api/docs/models |

## Ingestion

## Architecture

## Evaluation

## Cost and access notes

## Fresh GCP project setup

## Attribution
