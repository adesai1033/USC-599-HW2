# Build plan

The phased plan this project follows. Each phase has a "done when" check. Progress is
recorded in the change log in `PROCESS_LOG.md`.

## Week 1: Foundations

### Phase 0 — Setup (day 1)
- Create the repo with AGENTS.md, .env.example, .gitignore, and an explicit .gcloudignore.
- Get API keys for Gemini, Anthropic, OpenAI, and Tavily. Check the current model IDs on each provider's docs page and pin them.
- Set up GCP: create the project, enable Cloud Run, Cloud Build, Artifact Registry, and Firestore, create a Native-mode Firestore database, and a service account with roles/datastore.user.
- **Done when:** a short script gets a reply from all three providers and one Tavily search result.

### Phase 1 — Deploy a skeleton (days 1–2)
- Build FastAPI with /health, POST /chat/stream streaming tokens from one real LLM call, and a bare static/index.html that reads the stream with fetch().
- Write the Dockerfile (venv in /opt/venv, non-root user, PORT from env) and deploy.sh, then deploy.
- **Done when:** the live Cloud Run URL streams tokens into the page.
- Why first: SSE buffering, Docker permissions, and env-var problems are much easier to fix with 50 lines of code than with 3,000.

### Phase 2 — Corpus and ingestion (days 2–5, overlapping phase 1)
- Write corpus_spec.yaml. Start from about 30 landmark papers you can name: NeRF, Mip-NeRF 360, Instant-NGP, Plenoxels, TensoRF, 3DGS, Mip-Splatting, 2DGS, MVSNet, DUSt3R, MASt3R, and so on. Fill out to 150–300 with arXiv keyword searches, then skim the list and remove off-topic papers.
- Run fetch_arxiv → parse_pdfs → chunk → embed → Chroma, then build_db for metadata. Hold off on results extraction for now.
- Spot-check about 20 random chunks for parsing problems: two-column text in the wrong order, garbled equations, missing captions.
- Write eval/questions.jsonl (25–30 questions with ground truth) now, while you're reading the papers.
- **Done when:** for 10 test queries, the top 6 retrieved chunks are clearly relevant when you read them yourself.

## Week 2: The multi-agent system

### Phase 3 — MCP servers (days 5–6)
- Write corpus.py and papers_db.py, each with input validation and runnable on its own.
- Write mcp_client.py: persistent stdio sessions, Tavily via the URL parameter with the key redacted from logs, and per-agent tool filtering.
- Test the SQL guard and the three MCP failure modes.
- **Done when:** a script lists and calls every tool, and a broken manifest doesn't stop the others from working.

### Phase 4 — The graph, one worker at a time (days 6–9)
- Write state.py, then supervisor → research → synthesize working end to end from a command-line script.
- Add data, then summary. Add the failure wrapper, the step limit, and the citation validator.
- **Done when:** the canonical queries route to the right workers in the CLI, and a worker that raises an exception doesn't crash the graph.
- This is the 25% criterion, so give it the most time.

### Phase 5 — Provider routing (days 9–10)
- Write llm.py (tiers), router.py (rules plus fallback), and emit a RoutingDecision for every LLM call. Add unit tests.
- **Done when:** the canonical queries produce a mix of fast-tier and strong-tier decisions, and a bad Anthropic key falls back to OpenAI.

### Phase 6 — Streaming and frontend (days 10–12)
- Map LangGraph stream events to the SSE contract: routing, trace, token, citations, error, done.
- Build out the frontend: a routing badge per turn, a collapsible trace panel, and citation chips linking to arXiv.
- **Done when:** the local UI shows all four required parts of the frontend criterion.

## Week 3: Production and evidence

### Phase 7 — Durable state (day 12)
- Write state_store.py (Firestore and memory backends) and the load_history/save_history nodes.
- **Done when:** "summarize the second one" works, and a session survives deploying a new revision.

### Phase 8 — Real deploy and hardening (days 13–14)
- Deploy the full system. Test cold start, run the failure probes, and try 15–20 questions you didn't rehearse.
- Now run extract_results.py and hand-verify 50–100 rows so the Data agent has trustworthy numbers.

### Phase 9 — Evaluation (days 14–16)
- Run RAGAS, then fix whatever scores worst, usually chunking, k, or the prompt, and run it again. Budget one full iteration.
- Measure latency (cold and warm, TTFT broken down by stage), cost per query, and compare providers.

### Phase 10 — Documentation (days 16–17)
- Write architecture.md, evaluation.md, the README, and PROCESS_LOG.md. Keep rough notes from day 1 so the process log isn't written from memory.
- Keep a couple of days spare as buffer.
