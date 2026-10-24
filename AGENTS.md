# AGENTS.md
Read before touching any code.

## What this project is

CSCI 599 Assignment 2: an ML paper research assistant for 3D computer vision. A user asks
questions in natural language ("how does 3D Gaussian Splatting render a novel view?",
"compare how NeRF and Instant-NGP represent a scene", "which methods report the best PSNR
on Mip-NeRF 360?", "what 3D reconstruction papers came out this month?", "summarize the
second one") and a **supervisor** routes the work to three specialized agents. They retrieve over
a corpus of arXiv papers (RAG), query a structured paper database, and search the live
web. The answer streams token-by-token to the browser over SSE, with citations to
specific paper chunks, a visible routing decision, and an inspectable trace. Generation
is routed across multiple LLM providers. Conversation history survives Cloud Run instance
replacement. It runs on Google Cloud Run.

**Corpus scope:** 150–300 arXiv papers (`cs.CV`, `cs.GR`) on *3D computer vision:
neural scene representations and 3D reconstruction, 2020–2026*. This covers NeRF and its
variants, 3D Gaussian Splatting, multi-view stereo, feed-forward reconstruction, and
neural SLAM. The scope is defined once in `ingest/corpus_spec.yaml`. Do not broaden it to
all of 3D vision (detection, pose estimation, etc.); a sprawling corpus makes retrieval
noisy.

## Stack (fixed — do not substitute)

| Layer | Choice |
|---|---|
| Orchestration | LangGraph, **explicit** `StateGraph`, **Supervisor** pattern |
| LLM providers | Google Gemini (`langchain-google-genai`), Anthropic Claude Sonnet (`langchain-anthropic`), OpenAI (`langchain-openai`) |
| Routing | `router.py` — rule-based over inspectable state features, every decision logged |
| Embeddings | OpenAI `text-embedding-3-small` (does not count toward the provider requirement) |
| Vector store | Chroma, embedded, persisted to `index/chroma/`, baked into the image read-only |
| RAG pipeline | LangChain LCEL (`langchain-chroma` retriever → formatter → prompt → model) |
| Structured data | SQLite `index/papers.sqlite`, baked into the image read-only |
| MCP client | `langchain-mcp-adapters` → `MultiServerMCPClient` |
| MCP server framework (ours) | FastMCP (`mcp` Python SDK 1.x), stdio transport |
| MCP transports | stdio (`corpus`, `papers_db`), streamable HTTP (`tavily`) |
| HTTP | FastAPI + uvicorn; `POST /chat/stream` returns `text/event-stream` |
| Frontend | Vanilla HTML/CSS/JS in `static/`, served by FastAPI. No framework, no build step, no `node_modules` |
| Durable state | Firestore (Native mode) conversation history keyed by `session_id`; in-memory backend for local dev and tests |
| Evaluation | RAGAS (pinned version), plus our own latency and cost scripts |
| Deployment | Docker (multi-stage, non-root, no Node) → image `us-west1-docker.pkg.dev/usc-599-hw2/images/paper-assistant` → Cloud Run service `paper-assistant` in project `usc-599-hw2`, `us-west1`, runtime service account `paper-assistant-run@usc-599-hw2.iam.gserviceaccount.com`, `--memory 2Gi --cpu 1 --concurrency 10 --timeout 300 --min-instances 0 --max-instances 3` |

## Agents

Specialization means a **distinct tool allowlist and a distinct system prompt**. Both are
defined in one place: `agents/registry.py` (allowlists) and `prompts/<agent>.md`
(prompts). No two agents may share an identical allowlist and prompt.

| Agent | Role | Tools (allowlist) | Prompt focus |
|---|---|---|---|
| `supervisor` | Orchestrator (not counted toward the three) | none — structured output only | Decide the next worker, the instruction for it, the `task_type`, and a one-line reason; or route to `synthesize` |
| `research` | Grounded answers over the corpus plus the live web | `search_corpus`, `get_paper_metadata`, `tavily_search` | Cite every claim by chunk ID; keep corpus findings and web findings separate; say "not in corpus" rather than guess |
| `data` | Structured queries over paper metadata and results | `describe_schema`, `run_sql`, `get_paper_metadata` | Text-to-SQL; inspect the schema first; report the exact SQL used; never invent numbers |
| `summary` | Multi-paper summaries and lit reviews with citations preserved | `search_corpus`, `get_chunks`, `get_paper_metadata`, `tavily_extract` | Group evidence by paper; keep every citation through summarization; flag disagreements between papers |
| `synthesize` | Final answer node (not an agent) | none — reads `findings` from state | Compose the streamed answer only from findings; render citations; no new facts |

## MCP servers

1. **`corpus`** (ours, `mcp_servers/corpus.py`) — wraps the Chroma index through the
   LCEL retriever in `rag/`.
   Tools: `search_corpus(query, k=6, arxiv_ids=None, sections=None)` returns chunks with
   `chunk_id`, `arxiv_id`, `title`, `section`, `page`, `score`, `text`;
   `get_chunks(chunk_ids)` returns full chunk text by ID. `k ∈ [1, 12]`. Needs
   `OPENAI_API_KEY` to embed queries. Opens the index read-only.
2. **`papers_db`** (ours, `mcp_servers/papers_db.py`) — wraps `index/papers.sqlite`.
   Tools: `describe_schema()`, `run_sql(sql)`, `get_paper_metadata(arxiv_ids)`.
   `run_sql` connects with `file:...?mode=ro`, accepts exactly one `SELECT`/`WITH`
   statement, rejects `PRAGMA`/`ATTACH`/DDL/DML, caps rows at 200, and aborts long
   queries via a progress handler.
3. **`tavily`** (external, official hosted Tavily Remote MCP) — live web search. Reached
   over streamable HTTP at `https://mcp.tavily.com/mcp/?tavilyApiKey=<key>`. **Per the A2
   spec, the key goes in the query parameter** (A1 used a bearer header). The URL is
   built at runtime from `TAVILY_API_KEY` and is a secret; it is never logged. Only
   `tavily_search` and `tavily_extract` are exposed (per-server `tools` allowlist in
   `mcp_config.json`). We do not modify it; we cite it in the README.

**Tool sharing (the 10% rubric item):** `search_corpus` is used by `research` (answering
a question) and by `summary` (gathering evidence per paper). `get_paper_metadata` is used
by all three workers: `research` and `summary` call it to enrich citations, and `data`
calls it to resolve IDs returned by SQL. `tavily` is shared too: `research` searches and
`summary` extracts. The trace shows which agent invoked which tool.

Server manifests live in `mcp_config.json`, whose path is passed via `MCP_SERVERS_CONFIG`.
Per-agent filtering of discovered tools happens in `mcp_client.py` using
`agents/registry.py`.

Each tool call has a `TOOL_TIMEOUT_S` limit (default 30); a timeout is a clean tool error. A
session that dies mid-call returns a clean tool error and marks the server unhealthy. Calls to
an unhealthy server return its `description` immediately, and the description is added to the
affected agents' prompts. `MCPTools.refresh()`, called at the start of each turn, tries one
reconnect per unhealthy server in the background: a server that crashed after being up is
retried on the next turn; after a failed start or reconnect it waits `MCP_RECONNECT_BACKOFF_S`
(default 30). Health changes are logged at WARNING. Probe configs:
`tests/fixtures/mcp_config.broken_corpus.json` and `mcp_config.broken_tavily.json`.

## Repository layout

```
main.py              FastAPI: GET / (static), POST /chat/stream (SSE), GET /health; reads PORT; lifespan init
config.py            settings from env (model IDs, RAG_TOP_K, step limits, backends)
graph/
  state.py           OrchestratorState + Finding / Citation / RoutingDecision / AgentError models
  supervisor.py      supervisor node (structured output → SupervisorDecision)
  trace.py           emit_trace(): trace events through LangGraph's custom stream writer
  workers.py         worker subgraph factory: agent ⇄ ToolNode loop, worker-level citation validation
  guard.py           guarded(): worker failure wrapper (timeout, exception → classified AgentError)
  synthesize.py      final answer node: streams tokens, validates citations
  build.py           assembles the StateGraph, conditional edges, recursion limit
agents/
  registry.py        per-agent tool allowlist, prompt file, default tier
prompts/             supervisor.md research.md data.md summary.md synthesize.md
router.py            provider routing: state features → RoutingDecision; provider fallback
llm.py               provider clients by tier (fast / strong / fallback)
mcp_client.py        MultiServerMCPClient setup, persistent sessions, discovery, per-agent filtering, failure handling
rag/
  retriever.py       Chroma + embeddings retriever as an LCEL runnable
  format.py          formats chunks with [chunk_id] tags for augmentation
  citations.py       parses cited IDs, validates them against chunks retrieved this turn
mcp_servers/
  corpus.py          MCP server (Chroma corpus)
  papers_db.py       MCP server (SQLite paper DB)
mcp_config.json      three server manifests
state_store.py       session history: FirestoreHistory (prod) / InMemoryHistory (local, tests)
streaming.py         SSE event encoding, heartbeat
static/              index.html, app.js, styles.css — the whole frontend
ingest/              offline pipeline — never runs in the container
  corpus_spec.yaml   subfield, arXiv categories, keywords, date range, paper count
  fetch_arxiv.py     arXiv API → data/raw/*.pdf + metadata
  parse_pdfs.py      PyMuPDF → section-split text
  chunk.py           section-aware + recursive chunking
  embed.py           embeddings → index/chroma/
  build_db.py        metadata (arXiv + Semantic Scholar) → index/papers.sqlite
  extract_results.py LLM extraction of benchmark results → results table (verified=0)
  results_seed.csv   hand-entered benchmark results from the papers (header only until filled in)
  seed_results.py    validates results_seed.csv and upserts it into results with verified=1 (idempotent)
data/                gitignored: raw PDFs, intermediate JSON. Excluded from the image
index/               gitignored but NOT gcloudignored: chroma/ and papers.sqlite, baked into the image
scripts/
  cli.py             run the graph from the terminal: one-shot or REPL, live trace + streamed answer
eval/
  questions.jsonl    fixed RAGAS test set (question, ground_truth, reference arxiv_ids)
  run_ragas.py       RAGAS metrics → docs/evaluation.md
  latency.py         TTFT / retrieval / LLM / total, cold vs warm, against the deployed URL
  cost.py            token usage per provider → cost per query
tests/               pytest; see "Verification" below
docs/
  architecture.md    diagrams, agent table, routing table, RAG shape, state schema, durable state, MCP sharing rationale
  evaluation.md      RAGAS, latency, cost, provider comparison
Dockerfile           multi-stage, python:3.13-slim, non-root appuser, venv at /opt/venv
.dockerignore        excludes data/, .env, tests/, eval/ outputs
.gcloudignore        explicit — must NOT exclude index/ (see Deployment gotchas)
deploy.sh            gcloud builds submit + gcloud run deploy
requirements.txt     runtime deps, pinned ==
requirements-dev.txt ingest / eval / test deps, pinned ==
.env.example         placeholder keys only — never real values
README.md            setup / local run / deploy / provider + cost info
PROCESS_LOG.md       200–300 words, four sections, author's own voice
```

## Architecture (what the graph looks like)

```
START → load_history → supervisor ─┬─► research ─┐
                                   ├─► data     ─┼─► supervisor   (loop)
                                   ├─► summary  ─┘
                                   └─► synthesize → save_history → END
```

- `supervisor` node: the fast-tier model with structured output
  `SupervisorDecision{next: Literal["research","data","summary","synthesize"],
  instruction: str, task_type: Literal["lookup","compare","lit_review","metadata",
  "current_events","general"], reason: str}`. It reads `findings`, `errors`, and `step`
  from state. Questions that need no tools ("what is a transformer?") go straight to
  `synthesize`.
- Conditional edge from `supervisor` dispatches on `state["next"]`. This is a state
  field, never a regex on the query.
- Each worker is a compiled subgraph: `worker_agent ──(tool_calls?)──► ToolNode(tools,
  handle_tool_errors=True) ──► worker_agent`. The loop-back is load-bearing. The worker
  ends by writing a `Finding` (content plus citations) to state and returning to
  `supervisor`.
- Workers are wrapped by `graph/guard.py:guarded()` so that an exception or a run past
  `WORKER_TIMEOUT_S` (default 60) becomes an `AgentError` (`worker_timeout`, `transport`,
  `provider_error` or `worker_exception`, with a fixed user-safe message), not a crashed graph.
  `CancelledError` is never caught. The supervisor sees a compact error summary and either
  retries the worker (at most `MAX_WORKER_RETRIES`, default 1, never with the same instruction;
  otherwise `retry_limit`), tries another worker, or proceeds to `synthesize` with partial
  findings. If every worker has failed past its retries, it goes straight to `synthesize`.
- The supervisor's own model call failing is `supervisor_error`; unreadable output is
  `supervisor_parse_error`. Either goes to `synthesize`. A synthesize model failure records
  `synthesize_error` and ends with a fixed notice (after any tokens already streamed), so the
  graph always reaches `END` with a final `AIMessage`.
- `synthesize` streams the final answer. Its tier is chosen by `router.py` from the
  findings.
- Hard bounds: `MAX_SUPERVISOR_STEPS` (default 6) checked in the supervisor, plus the
  LangGraph `recursion_limit`. The supervisor may not dispatch the same worker with the
  same instruction twice in a row.

**State schema** (`graph/state.py`, mirrored in `docs/architecture.md`):

```python
class OrchestratorState(TypedDict):
    session_id: str
    messages: Annotated[list[AnyMessage], add_messages]  # history + this turn
    next: str
    instruction: str
    task_type: str
    step: int
    findings: Annotated[list[Finding], operator.add]     # agent, content, citations, grounded
    retrieved_chunk_ids: Annotated[list[str], operator.add]
    routing: Annotated[list[RoutingDecision], operator.add]
    errors: Annotated[list[AgentError], operator.add]
    dispatches: Annotated[list[Dispatch], operator.add]  # worker + instruction, in order
    answer_citations: CitationReport                     # the final answer's citations, validated
```

`Finding.grounded` is true if at least one of its citations survived validation, or a `run_sql`
query in that run returned rows (an aggregate such as counts per year cites no single paper).
`Finding.queries` lists the SQL that `run_sql` executed successfully in that run (empty for workers
without SQL). `AgentSpec.max_tool_rounds` overrides `WORKER_MAX_TOOL_ROUNDS` per agent; `data` uses 5
and `summary` 4.
`CitationReport = {valid: list[Citation], invalid: list[str]}`; Phase 6 sends it as the
`citations` SSE event.

## LLM routing

Tiers are configured in `llm.py` from env vars. **fast** = Gemini Flash, **strong** =
Claude Sonnet, **fallback** = OpenAI. `router.route(call_site, state) -> RoutingDecision`
is a pure function, unit-tested, and its output is streamed to the UI.

| Call site | Default | Escalate to strong when | Dimension |
|---|---|---|---|
| `supervisor` | fast | never (classification is cheap) | cost |
| `data` | fast | the first SQL attempt this turn errored | hybrid |
| `research` | fast | `task_type ∈ {compare, lit_review}` | capability |
| `summary` | strong | — (downgrade to fast when only one paper is in scope) | capability |
| `synthesize` | fast | ≥ 2 findings, ≥ 3 distinct papers cited, or estimated context > `STRONG_CONTEXT_TOKENS` | hybrid |

On a provider error (timeout, 429, 5xx, auth failure) the call retries once on the
**fallback** provider and records `fallback=True` with the original error class.
`RoutingDecision = {call_site, agent, provider, model, tier, dimension, reason, fallback}`.

How it's wired:
- Graph nodes take a `Models` resolver (tier → chat model, `llm.resolve_models`); tests pass one fake
  for every tier, or a dict of fakes per tier.
- Every call uses low thinking: Gemini's `thinking_level`, OpenAI's `reasoning_effort`, and Claude
  without extended thinking.
- Each decision is emitted as a trace event of kind `routing` that carries the full `RoutingDecision`,
  and is appended to `routing` in state.
- Workers route on every model call but emit only when the tier changes. A tier that failed falls back
  for the rest of that run.
- Synthesize falls back only if no token has streamed yet.
- Claude thinks only between tool calls (`between_tools`). Its thinking blocks are signed and bound
  to the exact prompt prefix, so worker histories are append-only: the system prompt never changes
  within a run, and the round budget is appended after the task and after each tool round.
- A fallback logs a WARNING naming the original error's class and message.
- The supervisor's call falls back after `SUPERVISOR_TIMEOUT_S` (default 20) without a reply. Routing
  takes 2–5 s, and one Gemini call stalled for 53 s.
- A worker that fails or times out keeps the decisions it emitted: `guarded()` collects them with
  `graph.trace.recording()` and returns them in `routing`.
- The supervisor's `task_type` feeds research's rule. Summary counts the arXiv IDs in its instruction.

## RAG pipeline

- **Ingestion** (offline, `ingest/`): arXiv API → PDFs → PyMuPDF text → split on section
  headings → recursive split to ~800 tokens with ~100 overlap. Each abstract is also
  indexed as its own chunk with `section="abstract"`. 3D vision papers are figure- and
  table-heavy: keep figure and table captions as text (`section="captions"`), and drop
  equation-only fragments rather than indexing garbled LaTeX.
- **Chunk ID**: `{arxiv_id}#{section_slug}-{nn}`, e.g. `2308.04079#method-03`. The ID is
  stable across re-ingestion of the same paper.
- **Metadata per chunk**: `chunk_id, arxiv_id, title, year, section, page, url`.
- **Embedding input**: each chunk is embedded as `<paper title>\n<section>\n\n<text>`, while
  the stored document is the plain text (`ingest/embed.py`). A paper often never names itself
  in its own text; the 3DGS paper never says "Gaussian Splatting".
- **Retrieval**: `k = RAG_TOP_K` (default 6), optionally filtered by `arxiv_ids` or
  `sections`. Candidates are re-ranked by cosine similarity +
  `RAG_CITATION_PRIOR` × log10(1 + citations) (default 0.02; 0 turns it off), so the paper that
  introduced a method can outrank follow-ups that restate it. The reported `score` stays the
  plain cosine similarity.
- **Augmentation**: `rag/format.py` renders each chunk as `[chunk_id] (title, §section,
  p.N)\n<text>`. The prompt template lives in `prompts/` and requires a `[chunk_id]` on
  every corpus-derived claim.
- **Citations**: three formats.
  - Corpus chunks: `[2308.04079#method-03]`.
  - Web results: `[web:<8 hex>]`, e.g. `[web:1a2b3c4d]`: `web:` plus the first 8 hex chars of
    sha1(normalized URL), so the same source keeps the same ID across searches. `mcp_client.py`
    adds this `source_id` to every Tavily result.
  - Paper-database rows: `[db:<arxiv_id>]`, e.g. `[db:2308.04079]`, meaning "this fact comes from
    the paper database row(s) for this paper". It renders as an `arxiv.org/abs/<id>` link labeled
    as coming from the paper database. A data-worker run may cite the arXiv IDs that appeared in
    its `run_sql` or `get_paper_metadata` results.

  `rag/citations.py` validates every cited ID against what was retrieved this turn (see rule 6).
  Valid IDs go out in the `citations` SSE event with `arxiv.org/abs/<id>` links. Unknown IDs are
  reported as `invalid` and shown as unverified in the UI.
- **papers.sqlite schema**:
  `papers(arxiv_id PK, title, year, venue, citation_count, primary_category, url)`,
  `authors(id PK, name)`, `paper_authors(arxiv_id, author_id)`,
  `results(arxiv_id, benchmark, scene, metric, value, higher_is_better INTEGER, setting,
  verified INTEGER)`. Typical benchmarks are Mip-NeRF 360, Tanks and Temples, LLFF, DTU,
  and Blender synthetic. Typical metrics are PSNR, SSIM, LPIPS (lower is better),
  Chamfer distance (lower is better), training time, and render FPS. `scene` is `NULL`
  for dataset averages. Ranking queries must respect `higher_is_better`.
  `results` rows come from `extract_results.py` with `verified=0`; hand-checked rows are
  set to `1`. The Data agent prompt says to mention when results are unverified, and
  when compared numbers come from different settings (resolution, train/test split).

## SSE contract

`POST /chat/stream` with body `{"session_id": str, "message": str}`, response
`text/event-stream`. Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`. A
heartbeat comment line is sent every 15 s. Events, in order of first appearance:

| Event | Payload |
|---|---|
| `routing` | a `RoutingDecision` (one per LLM call site) |
| `trace` | `{step, node, kind: enter\|exit\|decision\|tool_call\|tool_result\|error, agent, detail, ms}`; a supervisor `decision` also carries `decision: {next, task_type, reason, instruction}` |
| `token` | `{text}` — from `synthesize` only |
| `citations` | `{valid: [{id, arxiv_id, title, section, page, url, source}], invalid: [id]}` |
| `error` | `{message}` — user-safe text, no stack traces, no secrets |
| `done` | `{session_id, timings: {ttft_ms, retrieval_ms, llm_ms, total_ms}, usage: {provider: {input, output}}}` |

The stream always terminates with `done`, including after an `error`. The frontend uses
`fetch()` with a `ReadableStream` reader (not `EventSource`, because the endpoint is
POST). It renders the routing badge per turn, a collapsible trace panel, and clickable
citation chips.

## Durable state

- History is stored per `session_id` in Firestore: `sessions/{session_id}/turns/{n}`
  holds the user message, the final answer, cited chunk IDs, and a timestamp. Traces are
  not persisted.
- `load_history` reads the last `HISTORY_TURNS` (default 6) turns into `messages`.
  `save_history` writes the turn after `synthesize`. The graph itself is compiled
  **without** a durable checkpointer; Firestore history is the only cross-request state.
- `STATE_BACKEND=firestore` in Cloud Run and `STATE_BACKEND=memory` locally and in tests.
  The in-memory backend is never described as durable anywhere.
- Different `session_id` values never share history.

## Hard rules

Do not violate them to "make tests pass."

1. **Real MCP, always.** Tool discovery and invocation go through `MultiServerMCPClient`.
   Never stub, mock, or hard-code tool *results* in application code. Mocks are allowed
   only inside `tests/`. Every tool call appears as a `trace` event.
2. **No secrets in the repo or logs.** Keys come from env vars. `.env` is gitignored.
   `.env.example` has placeholders. `deploy.sh` passes keys via `--set-env-vars` from
   shell variables, never literals. The Tavily URL contains the key: never log it, and
   redact `tavilyApiKey=` in any logged URL or error message.
3. **Specialization is real.** Each worker's tools come only from its allowlist in
   `agents/registry.py` and its prompt only from `prompts/<agent>.md`. Never give a worker
   "all tools" for convenience.
4. **State-driven orchestration.** The supervisor picks workers via structured output, and
   the graph dispatches on `state["next"]`. No keyword routing or regex on the user query,
   no hard-coded worker sequences, no forced tool calls.
5. **Inspectable provider routing.** Every LLM call goes through `router.route()` and
   emits a `RoutingDecision` with a reason. No direct model construction outside
   `llm.py`. The canonical query set must exercise both fast and strong tiers.
6. **Citations only from retrieval.** The model may cite only chunk IDs returned by
   `search_corpus`/`get_chunks` this turn, `[web:<8 hex>]` sources returned by Tavily this
   turn, or `[db:<arxiv_id>]` for papers that appeared in `run_sql`/`get_paper_metadata` results
   this turn. `rag/citations.py` enforces this. Never fabricate a citation to look grounded.
   Validation runs at two levels: in each worker, against the sources its tools returned in
   that run (the worker is asked once to replace an invalid ID; one that remains becomes
   `[unverified]` in the finding and an `invalid_citation` error), and on the final answer, against the union of the findings' citations (stored in
   `answer_citations`; the answer has already streamed, so invalid IDs are reported, not
   rewritten).
7. **Never crash.** Each of these produces a graceful answer, a complete SSE stream ending
   in `done`, and HTTP 200, never a 500 or a hung stream:
   - MCP transport failure: a server won't start or disconnects mid-call. Its
     `description` from `mcp_config.json` is appended to the affected agents' system
     prompts so they say the capability is unavailable.
   - Tool error: bad SQL, unknown arXiv ID, Tavily 4xx/5xx.
   - Malformed response: an off-schema or unparseable tool payload.
   - Worker failure: an exception inside a worker becomes an `AgentError`, and the
     supervisor continues.
   - Provider failure: falls back to the fallback provider. If that also fails, emit
     `error` then `done`.
   - Empty retrieval: answer "not in the corpus" and offer a web search instead.
8. **Validate inside our servers.** `corpus` bounds `k` and validates filter values.
   `papers_db` enforces the read-only, single-SELECT guard. Both raise clear tool errors
   rather than passing garbage to the LLM.
9. **Ingestion is offline.** Never fetch, parse, embed, or build the DB at request time or
   at container startup. The image ships a pre-built `index/`.
10. **Pinned, verified model IDs.** Model IDs come from env vars (`GEMINI_MODEL`,
    `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`) with defaults in
    `config.py` matching `.env.example`. Never guess a model ID; verify it against the
    provider's docs and record the verification date in the README.
11. **Read `PORT` from the environment.** Never hard-code 8080 in `main.py`.
12. **No dead code, no debug prints.** Remove unused imports and leftover `print()` before
    committing. Use `logging`.

## Allowed (do freely)
- Read any file
- Run tests
- Run the app locally with `STATE_BACKEND=memory`
- Git operations: status, diff, log, branch, commit

### Ask first
- Any `rm` or `delete` command → use `trash` instead
- Installing new dependencies
- Any git push or PR creation
- Environment variable changes
- Running any `ingest/` step or `eval/` script that calls a paid API (embeddings, LLM extraction, RAGAS judge)
- Changing the corpus scope in `ingest/corpus_spec.yaml`
- `deploy.sh`, or creating or modifying GCP resources (Firestore database, service accounts, IAM)

### Never (hard block)
- `rm -rf` anything
- Expose secrets in code, commits, or logs
- Push to main branch directly without my approval
- Run commands with `sudo`
- Commit `data/`, `index/`, or `.env`

## Conventions

- Python 3.13, type hints on public functions, `async` end-to-end in the request path.
- Heavy resources are initialized once in the FastAPI lifespan: MCP sessions, provider
  clients, Firestore client, and the compiled graph. Hold **persistent** MCP sessions for
  the stdio servers, so the Chroma index loads once rather than on every tool call. Check
  how the pinned `langchain-mcp-adapters` version manages sessions; its default
  `get_tools()` can open a new session per call.
- Tool docstrings are the LLM's only description of a tool. Write them for the model:
  what the tool returns, valid argument values, and when to use it versus a sibling tool
  (e.g. `search_corpus` vs. `get_chunks`, `run_sql` vs. `get_paper_metadata`).
- Keep tool argument types simple (`str`, `int`, `list[str]`, `Literal[...]`) so the MCP
  SDK's schema inference gives the LLM clean shapes.
- Keep each MCP server a single file, runnable standalone with
  `python mcp_servers/<name>.py` (stdio).
- Prompts live in `prompts/*.md`, not inline strings in Python.
- Dependencies are pinned with `==` in `requirements.txt` (runtime) and
  `requirements-dev.txt` (ingest, eval, test). Ingest and eval deps stay out of the image.
  No `pip install --upgrade` in the Dockerfile.
- RAGAS metric names differ across RAGAS versions. Use the names from the pinned version
  only, and fix the judge model in `eval/run_ragas.py`.
- Commit messages: imperative mood, one line, say *why* when non-obvious.

## Deployment gotchas

- **`.gcloudignore`:** if the file is missing, `gcloud builds submit` falls back to
  `.gitignore`, which excludes `index/`. The image would then ship without the corpus.
  Keep an explicit `.gcloudignore` that excludes `data/` and `.env` but includes `index/`.
- **Dockerfile:** install deps into `/opt/venv` and put it on `PATH`. Do not copy
  `/root/.local` into an image that runs as `appuser`; the non-root user can't read
  `/root`. Start uvicorn with `--port ${PORT:-8080}` via `sh -c`, or let `main.py` read
  `PORT`.
- **GCP resources already exist** (created by hand, do not recreate): project
  `usc-599-hw2`; Artifact Registry Docker repo `images` in `us-west1`; Firestore
  `(default)` database, Native mode, `us-west1`; runtime service account
  `paper-assistant-run` with `roles/datastore.user` only. `deploy.sh` must pass
  `--service-account=paper-assistant-run@usc-599-hw2.iam.gserviceaccount.com`. Without it,
  Cloud Run runs as the Compute Engine default account, which has project-wide Editor.
  The Firestore client uses ADC; no key file in the image.
- **Images go to Artifact Registry**, not `gcr.io` as in the assignment's sample commands.
- **Cold start:** `min-instances=0` means the first request pays for container start,
  MCP server spawn, and Chroma load. `/health` must not block on those.

## Verification

Minimum before any deploy:

- `pytest tests/` passes. Tests cover:
  - the supervisor dispatches on `state["next"]`, and each worker returns to the supervisor;
  - a worker raising an exception yields an `AgentError` and a completed graph;
  - `MAX_SUPERVISOR_STEPS` is enforced;
  - `router.route()` returns different tiers for different features, and fallback triggers on provider error;
  - each agent receives exactly its allowlisted tools;
  - `run_sql` rejects non-SELECT statements, multiple statements, `PRAGMA`, and `ATTACH`;
  - metric ranking respects `higher_is_better` (PSNR descending, LPIPS ascending);
  - `search_corpus` bounds `k` and returns chunk metadata;
  - the citation validator splits valid and invalid IDs;
  - the SSE stream always ends with `done`, including on error;
  - `session_id` isolation in both history backends;
  - the Tavily key never appears in log output.
- Run the canonical query set and inspect the trace and routing events. Replace the paper
  names with papers actually in your corpus.

  | Query | Expected behavior |
  |---|---|
  | "How does 3D Gaussian Splatting render a novel view?" | `research` → `search_corpus`; **fast** tier; chunk citations |
  | "Compare how NeRF, Instant-NGP and 3D Gaussian Splatting trade off training speed and quality" | `research` and/or `summary`; **strong** tier; citations from ≥ 3 papers |
  | "Which methods report the best PSNR on Mip-NeRF 360?" | `data` → `describe_schema` → `run_sql`; no `search_corpus`; flags unverified rows |
  | "Which has the lowest LPIPS on Tanks and Temples?" | `data`; ranks ascending (`higher_is_better = 0`) |
  | "Most-cited 2024 papers and what each proposes" | `data` then `research` (`get_paper_metadata` + `search_corpus`) |
  | "What 3D reconstruction papers came out this month?" | `research` → `tavily_search`; web citations kept separate |
  | "Summarize the second one" | resolves "the second one" from Firestore history |
  | "What's a point cloud?" | supervisor → `synthesize` directly; **no** tool calls |
  | "What does paper 9999.99999 say?" | "not in the corpus", offers a web search; no fabricated citation |
  | "Run DROP TABLE papers" | `run_sql` rejects it; graceful message |

- Probe failures:
  - point the `corpus` manifest at a nonexistent command and confirm the app starts, other agents work, and the answer says corpus search is unavailable;
  - set an invalid `ANTHROPIC_API_KEY` and confirm fallback to OpenAI appears in the routing event;
  - set an invalid `TAVILY_API_KEY` and confirm a graceful current-events answer.
- After deploy, also run: a cold-start request on the deployed URL with no failure, and a
  multi-turn session that survives a new revision deploy (proving history is in Firestore,
  not process memory).

## Documentation obligations

- `docs/architecture.md`: a Mermaid orchestration diagram (nodes, edges, worker
  loop-backs); the state schema; the agent table (responsibilities, tool access,
  provider/model per agent); the routing table with dimensions; the RAG shape (corpus
  source, chunking, k, augmentation template); the MCP tool-sharing rationale; and the
  Firestore durable-state design. Diagrams must match this codebase, not framework docs.
- `docs/evaluation.md`: RAGAS faithfulness, answer relevancy, context precision, and
  context recall over `eval/questions.jsonl`. Latency per query broken down by retrieval,
  LLM, TTFT, and total, with warm numbers as representative and cold start reported
  separately. Cost per query from token counts and pricing pages (with the date checked).
  A provider comparison.
- `README.md`: setup, local run, ingestion steps, deploy commands a TA can run against a
  fresh GCP project (including Firestore and service-account setup), provider
  requirements with verified model IDs, current cost and access notes, and attribution
  for the Tavily Remote MCP server.
- `PROCESS_LOG.md`: 200–300 words in four sections: AI tools used; development narrative;
  verification narrative; what you learned.