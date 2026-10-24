# Process Log

## AI tools used

## Development narrative

## Verification narrative

## What I learned

---

## Change log

Dated record of each code change, newest first.

### 2026-10-23 — Phase 6: the graph streams over SSE, and the frontend shows routing, trace and citations

**Backend**
- `streaming.graph_frames()` runs the graph and maps its output to the SSE contract:
  - each routing trace becomes a `routing` event (the full `RoutingDecision`);
  - every other trace event becomes `trace`;
  - synthesize chunks become `token`;
  - the validated report becomes `citations`;
  - a `synthesize_error` or a crashed run becomes `error`;
  - `done` is always sent last.
- Answer text that never streamed (the failure notice) goes out as a final token.
- `done` carries the timings and the token usage:
  - `ttft_ms`, `retrieval_ms` (the sum of tool round times), `llm_ms` and `total_ms`;
  - `usage` per provider, from a callback handler keyed on LangChain's `ls_provider`.
- `main.py` now runs the real graph. The MCP servers and the graph start in a background task, so
  `/health` answers during a cold start, and `/chat/stream` waits for that task. If startup failed,
  the stream gives `error` then `done`. The Phase 1 single-model path is gone.
- **Supervisor decision:** the `decision` trace event now also carries `{next, task_type, reason,
  instruction}` for the UI. AGENTS.md's trace `kind` list now includes `decision`.
- **Worker tool time:** a tool round's time is recorded on its first result only, so
  `retrieval_ms` doesn't count a parallel round twice.

**Frontend** (vanilla JS): each answer is a card with:
- **Routing badge:** the path that actually ran (`data metadata → summary lit_review →
  synthesize`), with failed workers struck through. It has a provider pill per LLM call, colored
  by tier and marked ⚠ on a fallback, and a "Why this route" list of every supervisor and router
  reason.
- **Answer:** an escape-first Markdown renderer (paragraphs, headings, lists, tables, code).
- **Citation chips:** `[id]` tags become chips, and after `citations` they link out.
  - Corpus chips link to arXiv, with title, section and page on hover.
  - Web chips link to their page, labeled by hostname.
  - `[db:]` chips are labeled "paper database".
  - Unverified IDs are struck through.
- **Sources list:** grouped by paper.
- **Trace panel:** collapsible, with step, node, kind, detail and ms. Rows append live.
- **Footer:** TTFT, retrieval, LLM, total, and tokens per provider.

**Verified**
- `pytest tests/` gives 163 passed, 2 skipped. New tests are in `tests/test_sse_graph.py` (event
  order, routing payloads, structured decision, citation split, usage and retrieval time, crash →
  error + done, no supervisor JSON in tokens). `tests/test_phase1.py` now serves fake graphs
  through a `make_runtime` seam.
- **Live with curl:** routing/trace, then 51 tokens, then citations and done.
- **Live in Chrome, all four frontend parts shown:**
  - The comparison went to summary on the strong tier, with 12 sources across 3 papers.
  - Most-cited 2024 went data → summary → synthesize on the strong tier, with `[db:]` chips.
  - The point cloud question went straight to synthesize with no tools.
  - "This month" used web chips.
  - Mip-NeRF 360 PSNR went data → research. The `results` table is still unseeded, so the
    numbers came from the corpus.
- **Broken-corpus probe:** the app starts. Research falls back to Tavily, the answer says corpus
  search was unavailable, and the stream ends in `done`.

**Still open**
- TTFT is 25–50 s on multi-agent questions. Phase 9 will measure it.
- History (Phase 7): each turn is still independent.
- Seeding `results`.

### 2026-10-23 — Phase 5 follow-ups: Claude's thinking blocks, DROP TABLE, routing records, stalls

**1. The remaining Claude 400 was the same thinking-signature error.** The probe caught its message:
"messages.1.content.10: Invalid `signature` in `thinking` block".
- `between_tools` does not turn thinking off. It limits thinking to the gaps between tool calls, and
  that failed call had 76 reasoning tokens. Those blocks are signed and bound to the exact prompt
  prefix, and the round budget still rewrote the system prompt every call.
- **Fix:** worker histories are now append-only.
  - The system prompt is fixed for the run.
  - The round budget goes at the end of the task message, and as a message after each tool round.
  - The two fix requests (attribution, then citations) also append to one history.
  - `between_tools` stays, for latency.
- **Verified:**
  - A test checks that each call's messages extend the previous call's, and that the system prompt
    is identical across calls.
  - Live, Claude with its default (adaptive) thinking produced signed thinking blocks on middle calls
    (`[0, 1, 1, 0]` and `[0, 1, 0]` per call). Later calls replayed them with no 400. Before the fix,
    this setup failed on the second call.
  - 8 graph runs on Claude (comparison ×4, hand-off ×2, anti-aliasing ×2) had no fallbacks and no
    errors. The 3DGS lookup on Gemini works with the budget message after tool results.

**2. DROP TABLE now reaches `run_sql`.**
- `prompts/supervisor.md`: a request to run a SQL statement or change the database goes to the
  database worker, with the statement quoted.
- `prompts/data.md`: pass a given statement to `run_sql` unchanged and report its error.
- **Live:** "Run DROP TABLE papers" and "Delete all papers from 2020" go to data, `run_sql`, "Only
  read-only SELECT or WITH queries are allowed." The point-cloud question still goes straight to
  synthesize.
- Bonus: after the SQL error, `data` escalated to strong mid-run. This was the first live check of
  that rule, and of a Gemini history passed to Claude.

**3. Failed workers keep their routing decisions.**
- `graph.trace.recording()` collects every decision emitted inside it.
- `guarded()` returns them in `routing` on failure or timeout. Tests cover a double provider failure
  and a timeout.

**4. One-offs:**
- **Stalled supervisor call:** `invoke_with_fallback(timeout=...)`. The supervisor now falls back
  to OpenAI after `SUPERVISOR_TIMEOUT_S` (default 20, new in `config.py` and `.env.example`) with
  "no reply within 20 s from google". Routing normally takes 2–5 s; one Gemini call took 53 s.
- **Unretrieved citations:** a worker that cites an ID no tool returned is asked once to replace it
  (`FIX_CITATIONS`). Whatever remains is still marked `[unverified]` as before.

**Verified:** `pytest tests/` gives 154 passed, 2 skipped.

### 2026-10-23 — Phase 5b: routing wired into the graph

**What changed:**
- **Models resolver** (`llm.resolve_models`): every node takes a model per tier. Tests still pass
  one fake for all tiers, or a fake per tier.
- **Supervisor:** routed with fallback.
- **Workers:**
  - Routed on every model call, but a decision is emitted only when the tier changes (`data`
    escalates after a failed `run_sql`).
  - A tier that failed stays on the fallback for the rest of the run.
  - The round-cap summary call and the attribution fix use the run's tier.
- **Synthesize:** routed from findings and context size. It falls back only before the first token.
- **Streaming and state:** each decision streams as a trace event of kind `routing` that carries the
  full `RoutingDecision`, and is stored in `routing` in state.
- **CLI:** prints each decision and a routing summary.
- **Tests:** 9 new in `tests/test_routing_graph.py`; 149 passed, 2 skipped.

**Bug found live: Claude's thinking blocks.** Every summary run on Claude failed with a 400 and fell
back to OpenAI.
- Sonnet 5.5 thinks by default, and its signed thinking blocks are bound to the exact prompt
  prefix.
- Our round budget rewrites the system prompt every call, so the second call was rejected ("The
  `system` prompt differs from the one this block was created with").
- **Fix:** the strong tier runs with thinking off, `thinking={"type": "between_tools"}`. Sonnet 5.5
  rejects `{"type": "disabled"}`.
- Since then Claude's replies carry only text and tool-use blocks. Smoke test: Claude and OpenAI
  both accept a Gemini tool-call history, so a mid-run tier switch needs no message rewriting.

**Canonical queries** (routing per call site):

| Query | Workers | Non-fast routing | Notes |
|---|---|---|---|
| How does 3DGS render | research | none | 30 s |
| Compare NeRF / Instant-NGP / 3DGS | summary | summary strong, synthesize strong (4 papers) | |
| Best PSNR on Mip-NeRF 360 | data → research | synthesize strong (2 findings) | |
| Lowest LPIPS on T&T | data → research | synthesize strong (2 findings) | |
| Most-cited 2024 | data → summary | summary strong, synthesize strong | |
| This month | research | none | |
| What's a point cloud | none | none | supervisor → synthesize, no tools, 5 s |
| 9999.99999 | research | none | |
| DROP TABLE | none | none | answered without tools |

- The mix of fast and strong holds. The supervisor is always fast.
- `data` never escalated: none of its SQL errored in these runs. The unit test covers it.

**Probes:**
- **Invalid `ANTHROPIC_API_KEY`:** summary and synthesize both fell back to OpenAI
  (`AnthropicAuthenticationError`), and the answer completed with no unverified citations.
- **Both providers failing** (invalid Anthropic key, nonexistent `OPENAI_MODEL`; not an invalid
  OpenAI key, which would also break corpus embeddings):
  - workers fail as `provider_error`, and the supervisor retries and then stops at the step limit;
  - synthesize falls back, fails, and ends with the fixed notice, exit 0.
  - Worker routing decisions from runs that raised are streamed but not kept in state.

**Summary on Sonnet** (comparison ×8, hand-off ×2, `WORKER_TIMEOUT_S=300` for the probe):
- Comparison summary 24–35 s (Gemini: 23–24 s); totals 37–64 s (Gemini: 38–40 s).
- Synthesize on Sonnet takes 9–11 s.
- Every finding had Across papers.
- **Margin:** the worst run without a fallback leaves 25 s under the 60 s timeout. One run that fell
  back mid-summary took 47 s, which leaves 13 s.

**Still open:**
- One of the 14 Claude summary runs since the fix fell back after an `AnthropicInvalidRequestError`
  of unknown cause. It didn't recur in 9 more runs. The fallback warning now logs the provider's
  message, so the next one will show the cause.
- One Gemini supervisor call took 52.6 s for 221 tokens: provider latency (turn total 103 s).
- One comparison run cited an unretrieved chunk (`2201.05989#captions-04`); the worker validator
  marked it `[unverified]`.
- One hand-off: the supervisor passed 5 of the 10 IDs, which confirms the earlier guess.

### 2026-10-22 — Phase 5a: provider tiers, the router and fallback (not wired in yet)

Phase 5 is split: 5a adds the modules with unit tests and leaves the graph unchanged; 5b wires them
in and runs the live checks.
- **`llm.py`:** `get_llm` now has three tiers.
  - fast: Gemini, unchanged.
  - strong: `ChatAnthropic`, no extended thinking, 60 s timeout, 1 SDK retry.
  - fallback: `ChatOpenAI`, `reasoning_effort` from the caller's thinking level, 60 s, 1 retry.
  - Also `PROVIDERS` and `model_id(tier)`.
- **`config.py`:** Anthropic and OpenAI keys and models. They're not required at startup: a missing
  key becomes a provider error, which falls back. Model IDs default to the `.env.example` values. New
  `STRONG_CONTEXT_TOKENS` (default 8000), added to `.env.example`.
- **`router.py`:**
  - `route(call_site, RouteFeatures)` is a pure function implementing the AGENTS.md table, with a
    reason that names the deciding values.
  - Feature helpers: `sql_errored`, `papers_in_scope` (arXiv IDs in the supervisor's instruction, not
    the user query), `cited_papers`, `estimate_tokens`.
  - `invoke_with_fallback` retries once on OpenAI after a provider error and returns a
    `fallback=True` decision naming the error class. Other errors, cancellation, and a failing
    fallback propagate.
- **`graph/guard.py`:** the provider-error check moved into a shared `is_provider_error`, used by
  both the guard and the router.
- **Tests:** `tests/conftest.py` now overrides the Anthropic and OpenAI keys and models too, so tests
  never reach a real provider; it keeps the real keys under `RUN_LIVE=1`.
- **Verification:**
  - Re-checked `gpt-6.1-sol` on OpenAI's models page: reasoning effort low through max.
  - One tiny live call to each provider worked. Sonnet took 14.5 s for "ok", so latency needs
    watching in 5b.
- **Verified:** `pytest tests/` gives 140 passed, 2 skipped (18 new in `tests/test_router.py`).

### 2026-10-22 — Phase 5 prep: the Anthropic dependency

- Installed `langchain-anthropic==1.7.6` (pulls in `anthropic` 1.17.0) for the strong tier and pinned
  it in `requirements.txt`. `langchain-openai==1.7.2` (fallback tier) was already pinned.
- `langchain-core` stayed at 1.6.13 and `pip check` reports no conflicts.
- **Verified:** `pytest tests/` gives 122 passed, 2 skipped.

### 2026-10-22 — Investigation: the summary timeout and the 5-paper hand-off (no code change)

I timed every model and tool call (LangChain callback handler, `WORKER_TIMEOUT_S=300` for the probe
only) over 6 anti-aliasing runs and 5 "most-cited 2024 papers" hand-offs. No run timed out.

**Timeout: mostly the model writing slowly, not our side.**
- Summary worker: 18–26 s (anti-aliasing), 24–32 s (hand-off). Model calls account for all but ~1 s.
- Tool rounds take 0.4–2.2 s, even with 11 `search_corpus` calls in parallel.
- The final write-up call dominates: 1,500–2,900 output tokens at a steady ~125 tokens/s, so 11–23 s.
  The searching calls before it take 2–6 s each.
- The provider spikes now and then: one supervisor call took 9.2 s for 83 tokens (usual ~2.5 s).
- Likely cause of the 163 s turn: it happened before the round budget, when every summary run used
  all 4 rounds plus a forced write-up (~5 model calls on longer inputs, roughly 45–50 s). One spike
  on top would pass 60 s. Unproven: that run's trace was lost.
- Remaining margin: ~28 s in the worst run. Phase 5's move of `summary` to Claude Sonnet may eat
  into it. Options, not implemented: a length budget per paper in `prompts/summary.md`, or a
  timeout per worker.

**2024 hand-off: the supervisor's choice, not a dropped ID.**
- In all 5 runs the supervisor passed 10 IDs and `summary` searched and cited all 10.
- The earlier 5-paper run had 1 round with 5 tool calls. `summary` makes one search per ID (plus
  usually one metadata call) in one round, so that matches about 4–5 IDs handed over: the supervisor
  likely passed fewer. That's allowed, since the question gives no count. My measurement script
  hadn't recorded the instruction; the CLI trace does.

### 2026-10-22 — Summary worker stops on its own: round budget in the prompt

**Correction:** the 4e entry says "Gemini doesn't reliably batch tool calls". The traces show that
was wrong.
- In both comparison runs, round 2 ran the three per-paper searches in parallel.
- The slow run (59 s) differed afterwards. It kept making follow-up calls: an off-topic rasterizer
  search, an unneeded `get_paper_metadata`, a hunt for 3DGS's "Table 1" numbers, and a `get_chunks`
  re-read. It then hit the 6-round cap.

**Step 1 (a round plan in the prompt, cap 4):**
- `prompts/summary.md` got a round plan: find papers (skip if IDs are given), all per-paper
  searches in one round, at most one follow-up.
- `get_paper_metadata` is only for years, venues or authors.
- New caveat that tables, like equations, aren't reliably indexed, so don't hunt for table
  numbers.
- `summary` `max_tool_rounds` went 6 → 4.
- **Result:** the cap fired in 6 of 6 topic runs. The model doesn't stop on its own; the cap did
  the stopping. Totals were 39–47 s.

**Step 2 (`graph/workers.py`):**
- Every model call's system prompt now ends with the tool budget: "Rounds of tool calls used: N of
  M (K left)…".
- On the last round it adds "This is your last round: make every call you still need in this one
  response."
- This applies to all workers.

**Result** (7 runs: comparison ×4, anti-aliasing ×2, 2024 hand-off ×1):
- The cap fired in **0 of 7**; runs took 1–3 rounds.
- **Time:** comparison 37.5–39.8 s (before: 41 s and 59 s), anti-aliasing 38 s, hand-off 44 s.
- **Coverage:** every comparison cited the 3 named papers, anti-aliasing cited 3 and 5 papers,
  every finding had Across papers, and there were no unverified citations or errors.
- **Regression check:** the 3DGS question → research (still hit its 4-round cap once); most cited →
  data.

**Still open:**
- In the first, partly lost measurement batch, one anti-aliasing run hit `worker_timeout` (the
  summary worker ran past 60 s). The guard caught it, the supervisor retried summary and then
  research, and the turn took 163 s. It didn't recur in the 7-run re-run, so it looks intermittent,
  probably slow model calls.
- The 2024 hand-off searched 5 papers in the latest run (10 in earlier ones). I didn't capture
  whether the supervisor passed 5 IDs or summary dropped some.

**Verified:** `pytest tests/` gives 122 passed, 2 skipped. The summary rounds test now checks the
4-round cap and the budget lines.

### 2026-10-22 — Phase 4e: the `summary` worker

There was no separate brief, so the scope came from AGENTS.md's summary row: multi-paper summaries
and literature reviews with `search_corpus`, `get_chunks`, `get_paper_metadata` and
`tavily_extract`, evidence grouped by paper, every citation kept, disagreements flagged.

**Built:**
- **`prompts/summary.md`:**
  - with given IDs, cover each one; with a topic, search first, then pick 3–6 papers;
  - make the per-paper `search_corpus` calls together, in one round;
  - `get_chunks` for full text, `tavily_extract` only for URLs given in the instruction;
  - per-paper blocks, then an **Across papers** section on relations and conflicts.
- **Registry:**
  - `summary` is registered with `max_tool_rounds=6`.
  - Its description covers summaries, reviews and comparisons of 3 or more methods.
  - `research`'s description now excludes multi-paper reviews; one- and two-method comparisons stay
    with research.
- **Supervisor:**
  - the routing rule names the literature-review worker by description, not by name;
  - a rule to put a web source's URL in the instruction for a later worker;
  - the finding view lists each web citation's URL, so `tavily_extract` can use it.
- **Tier:** every worker runs on the fast tier until Phase 5, because `get_llm` has no strong tier
  yet and `summary`'s default is strong.

**Verified:**
- `pytest tests/` gives 122 passed, 2 skipped (5 new tests in `tests/test_summary_worker.py`,
  including a research → summary hand-off of a web URL).
- **Real runs:**
  - "Summarize how anti-aliasing has been addressed in radiance fields" → summary; 14 citations
    from 5 papers (Mip-NeRF, Mip-NeRF 360, Zip-NeRF, Mip-Splatting, Analytic-Splatting). The finding
    has an Across papers section with conflicts (volume rendering vs real-time; Mip-Splatting
    over-smoothing, which Analytic-Splatting addresses), and synthesize kept it.
  - "Compare how NeRF, Instant-NGP and 3DGS trade off training speed and quality" → summary;
    citations from all 3 papers.
  - "Most-cited 2024 papers and what each proposes" now goes data → summary. The summary worker got
    all 10 IDs, searched them in one round, and every paper has a chunk citation as well as its
    `[db:...]`.
  - Routing regression: 3DGS lookup and NeRF vs Instant-NGP → research; most cited → data;
    this month → research; point cloud → none; 9999.99999 → research. No unverified citations in
    any run.

**Still weak:**
- **Batching:** Gemini doesn't reliably batch tool calls. Before the "one round" prompt line both
  runs hit the 6-round cap; after it, one of two comparison runs still did (59 s, against 41 s for
  the other).
- **Model tier:** summary runs on the fast tier until Phase 5 routes it to the strong tier.

### 2026-10-22 — Phase 4d testing finished: benchmark rankings on a seeded copy

**Setup:**
- The real `ingest/results_seed.csv` and `index/papers.sqlite` were left untouched (still 0
  results rows), as agreed.
- I copied the database to a scratch folder and seeded the copy with `seed_results.py` from 42 rows
  read from the papers' own PDF tables:
  - 3DGS (2308.04079) Tab. 1: Plenoxels, INGP-Base/Big, Mip-NeRF 360 and 3DGS 7K/30K on
    Mip-NeRF 360 and Tanks and Temples;
  - 2DGS (2403.17888) Tab. 8–9: 2DGS averages plus two bonsai scene rows.
- The two 2DGS average rows were set to `verified=0` to exercise the unverified flag.
- The app was pointed at the copy with `PAPERS_DB_PATH`.

**Results:**
- **"Best PSNR on Mip-NeRF 360":** data only. The ranking was descending and exactly right: 27.69
  > 27.21 > 27.03 > 25.60 > 25.59 > 25.30 > 23.08.
- **"Lowest LPIPS on Tanks and Temples":** data only. The ranking was ascending and exactly right:
  0.183 < 0.257 < 0.280 < 0.305 < 0.330 < 0.379.
- **Other checks:** the SQL matched names case-insensitively and was printed, citations were `db`
  only, and there were no unverified citations.
- **Per-scene rows:** with the bonsai rows added (31.98 and 31.52), the worker kept dataset
  averages and scenes separate, so the 31.98 scene score didn't top the average ranking. The SQL
  itself didn't filter on `scene IS NULL`.

**Fixed (prompts only):** the data finding reported `verified` per row, but synthesize sometimes
dropped "verified = 0", and nothing said the numbers came from different sources (copied from the
original paper vs reproduced by the 3DGS authors).
- `data.md` now requires a "Comparability:" line.
- `synthesize.md` now requires keeping unverified marks and comparability caveats.
- Re-run: both answers flag 2DGS as unverified and explain where the numbers come from.

**Still open:**
- The worker skips the `SELECT DISTINCT benchmark` step and uses `LIKE` patterns directly. That works
  with today's consistent spellings.
- The real results table is empty until you seed it. The scratch CSV (42 rows, each with a source
  note) can be a starting point, but each row needs a human check before it goes in with
  `verified=1`.

**Verified:** `pytest tests/` gives 117 passed, 2 skipped.

### 2026-10-22 — Benchmark questions run before the results table is seeded (no code change)

I ran "Which methods report the best PSNR on Mip-NeRF 360?" and "Which has the lowest LPIPS on
Tanks and Temples?" with `results` still empty: `ingest/results_seed.csv` has only its header,
and no rows have been entered yet.

**Worked:**
- Both went to `data` first. It ran `describe_schema`, then `SELECT COUNT(*) FROM results`, found
  0 rows, and reported it without inventing numbers.
- The supervisor then sent both questions to `research`, which looked in paper text and on the web.
- No crash, empty UNVERIFIED lines, and the SQL was printed.

**Went wrong:**
- **The PSNR answer stated "27.58 dB average" for 3DGS**, citing
  `2308.04079#per-scene-error-metrics-00`. That chunk's 27.58 / 0.790 / 0.240 are the 3DGS authors'
  reruns of the Mip-NeRF 360 *method*, not 3DGS's own score.
  - The citation was retrieved and the number is in the text, so neither validation check could
    catch the wrong attribution.
  - This is why benchmark numbers should come from hand-checked rows (`seed_results.py`) rather
    than from reading paper text.
- **A wasted query:** after finding `results` empty, `data` ran a pointless title search
  (`title LIKE '%Mip-NeRF 360%'`).
- **Unchecked web numbers:** the LPIPS answer used figures from web sources (NerfBaselines: 2DGS
  0.123 on Truck) that I haven't checked.

**Still untested:** ranking direction (PSNR descending, LPIPS ascending), dataset averages versus
scenes, and flagging verified status and settings. These need seeded rows. Next step: fill in about
20 rows from the papers' own tables, run `python ingest/seed_results.py`, and re-run both questions.

### 2026-10-22 — Phase 4d: the `data` worker

**Real schema** (checked with `describe_schema` and `run_sql` before writing the prompt):
- `papers`: 220 rows (2020:14, 2021:40, 2022:21, 2023:43, 2024:61, 2025:40, 2026:1).
- `authors`: 922 rows; `paper_authors`: 1,297 rows.
- `results`: **0 rows**, so there are no benchmark or metric spellings yet.
- Venues are full names ("Computer Vision and Pattern Recognition", 80 papers), not acronyms.
- `results` has no unique key.

**Built:**
- **Citations:** a third type, `[db:<arxiv_id>]`, in `rag/citations.py` (parser, validator,
  `[unverified]` marking). It's allowed for arXiv IDs that appear in a run's `run_sql` or
  `get_paper_metadata` results, and gets its title from the row when there is one.
- **`Finding`:** new `queries` field (the SQL that ran successfully). `grounded` is also true when a
  query returned rows.
- **`AgentSpec.max_tool_rounds`:** `data` uses 5.
- **`data` registered** in `WORKERS`. `research`'s description now says it reads text and isn't for
  counting or ranking.
- **`prompts/data.md`:** written against the real schema, including the empty-`results` case and
  `LIKE` for venues.
- **Hand-offs:**
  - Supervisor rule: pass every ID the next worker needs, not a sample. The routing rule is phrased
    by description, without hard-coded worker names.
  - Research rule: filter by the given `arxiv_ids`, abstracts first.
  - The supervisor's view of each finding lists every paper it cited.
- **Results seed:** `ingest/results_seed.csv` (header only) and `ingest/seed_results.py`.
  - Checks: the ID is in `papers`, the value is numeric, the metric is known (any case),
    `higher_is_better` is 0 or 1 and matches the metric's direction.
  - Upserts on (arxiv_id, benchmark, scene, metric, setting) with `verified=1`, and is idempotent.
  - Not run on the real index.
- **CLI:** prints each data finding's SQL and labels db citations "(paper database)".

**Found while running the done-when queries:**
- "Most-cited 2024 papers and what each proposes": the supervisor passed only 4 of 10 IDs to
  research. Synthesize then described papers 5–10 with only `[db:...]` citations, though the
  database has titles, not methods.
- Validation can't catch this, because the IDs were retrieved. I fixed it in the prompts:
  - synthesize: a `[db:...]` citation supports only database facts;
  - supervisor: pass every ID.
- The re-run passed all 10 IDs, and every description has a chunk citation.

**Verified:**
- `pytest tests/` gives 117 passed, 2 skipped (11 new tests in `tests/test_data_worker.py`).
- Real runs:
  - most cited → data only;
  - papers per year → data, with counts matching the database;
  - most-cited 2024 + proposals → data → research → synthesize, with db and chunk citations;
  - `DROP TABLE` → declined, row counts and file hash unchanged;
  - "this month" → research.
- No routing mistakes in these runs, and no unverified citations.

### 2026-10-22 — Phase 4c: robustness (the graph can't crash or cite unretrieved sources)

**Built:**
- **`rag/citations.py`:** `validate()` returns a `CitationReport` (`{valid, invalid}`, in order,
  without duplicates), and `mark_unverified()` replaces invalid IDs.
- **Worker-level validation** (`graph/workers.py`): the allowed sources are the ones this run's
  tools returned. An invalid ID becomes `[unverified]`, plus an `invalid_citation` error and a
  trace event. `Finding.grounded` is true only if at least one valid citation remains.
- **Answer-level validation** (`graph/synthesize.py`): the answer's citations are checked against
  the union of the findings' citations and stored in the new state field `answer_citations`.
  Invalid IDs are reported, not rewritten, because the answer has already streamed.
- **`graph/guard.py`:** `guarded()` wraps every worker node with `WORKER_TIMEOUT_S` and
  classifies failures as `worker_timeout`, `transport`, `provider_error` or `worker_exception`.
  - Classification follows the exception's cause chain, using `langchain_core.exceptions.ModelError`
    for providers.
  - Messages are fixed per kind; `CancelledError` propagates.
- **Supervisor:**
  - It sees a compact error summary with retries left.
  - New `retry_limit` guard: at most `MAX_WORKER_RETRIES` retries, never with the same
    instruction.
  - When every worker has failed past its retries, it skips the model and goes to synthesize.
  - A model-call failure is `supervisor_error`; unreadable output (any `ValueError`) stays
    `supervisor_parse_error`.
- **Synthesize:** a model failure records `synthesize_error` and ends with a fixed notice, added
  after any tokens that already streamed.
- **`mcp_client.py`:**
  - Every call gets `read_timeout_seconds=TOOL_TIMEOUT_S`.
  - A dead session gives a clean tool error and marks the server unhealthy. Calls to an
    unhealthy server fail immediately with its description, which also goes into the affected
    worker's prompt; tools and notes are now read at the start of every worker run.
  - `refresh()` at the start of each turn reconnects in the background. A crashed server is
    retried on the next turn; a failed start or reconnect waits `MCP_RECONNECT_BACKOFF_S`.
  - Health changes are logged at WARNING, and the "missing allowlisted tools" warning only when
    that set changes.
  - The old reconnect-on-call behavior is gone.
- **CLI:** `--mcp-config`, MCP health lines at startup, errors grouped by kind, `answer_citations`
  as the numbered list, an UNVERIFIED CITATIONS line, and any failure notice the stream didn't
  carry.
- **Probe fixtures:** `tests/fixtures/mcp_config.broken_corpus.json` and `broken_tavily.json`.

**Verified:**
- `pytest tests/` gives 106 passed, 2 skipped. There are 20 new tests in
  `tests/test_robustness.py`, and the Phase 3/4b tests changed for the new behavior: reconnect
  happens on `refresh()`, not on a call, and unretrieved IDs become `[unverified]`.
- Live probes:
  - **Broken corpus:** the corpus server fails discovery, research has no `search_corpus`, and
    the answer says the corpus is unavailable and uses web sources, with no chunk citations.
  - **Broken Tavily:** the session works but every search returns
    `tavily_search failed: Unauthorized...`; the answer says web search failed.
  - **Corpus server killed mid-REPL** (`pgrep -f mcp_servers/corpus.py`): that turn's
    `search_corpus` returned the unavailable error within 3 ms and the worker used the web. The
    next turn logged "trying to reconnect" and "reconnected", and corpus search worked again.
  - **9999.99999:** "not in the corpus", the worker searched the web itself, and no invented
    citations.
  - **The 4b questions** all show an empty UNVERIFIED CITATIONS line.

**Seen in the runs:**
- **Worker cited an unretrieved source:** in "this month", after the tool-round cap, the worker
  cited `web:ebdd62a3`, which no tool returned. It became `[unverified]`, and synthesize left that
  claim out.
- **No invalid citations in final answers:** none appeared in any final answer.
- **Wrong corpus date range:** the broken-Tavily answer claimed the corpus "only covers literature
  through mid-2024". That's false (it covers 2020 to March 2026) and is the model's invention.
- **Extra dispatch during a reconnect:** on the turn where the corpus server is reconnecting, the
  first research run still sees it as unavailable, and the supervisor dispatches research again.
  That costs about 4 s once.

### 2026-10-22 — Faster comparison answers (synthesize length and thinking)

**Instrumented first:** synthesize's `exit` trace now reports time to first token, word count and
reasoning tokens. Gemini's stream reports 0 reasoning tokens even when it clearly thinks, so that
count isn't reliable.

**What the measurements showed** ("Compare how NeRF and Instant-NGP represent a scene", 3–4 runs
each):

| Setup | Synthesize | First token | Words | Research | Total |
|---|---|---|---|---|---|
| baseline | 11.6–14.9 s | 1.3–3.7 s | 533–608 | 16.6–24.0 s | 34–44 s |
| + length targets | 10.4–20.8 s | 2.9–12.9 s | 378–428 | 19.4–23.6 s | 36–47 s |
| + low thinking | 8.5–13.0 s (median ~9 s) | 1.3–1.8 s | 333–482 | 21.3–28.6 s | 35.5–53 s |

- **My plan's hypothesis was wrong.** At baseline, synthesize was mostly writing, not thinking.
- **Length targets** cut the words by about 30%, but the model then thought longer before its
  first token (up to 12.9 s).
- **Low thinking** on top fixed that.

**Changes:**
- `prompts/synthesize.md` has per-task-type length targets: compare = one table plus at most
  three short paragraphs, about 300–450 words.
- Synthesize uses `thinking_level="low"`.
- `prompts/research.md`: for comparisons, make one search filtered to all the papers' IDs, and
  never re-search a detail a result already covers.

**Quality** across the 7 runs with the final settings (4 compare, 3DGS, "this month", point
cloud):
- every comparison cited both papers;
- every cited chunk ID was in that turn's retrieved list;
- no citations were grouped;
- no long paragraph lacked a citation;
- the 3DGS answer cites only `2308.04079`, and the point-cloud answer made no tool calls.

**Still open:**
- Research is now most of a research turn's time (20–29 s, varying with the number of rounds).
- One run hit a Gemini `504 DEADLINE_EXCEEDED` in the worker, and the whole turn crashed. This is
  the known gap that the Phase 4c worker failure wrapper and the Phase 5 fallback cover.

**Verified:** `pytest tests/` gives 86 passed, 2 skipped.

### 2026-10-22 — Fixed issues 1–3, arXiv mirrors and the `time_range` gap

**arXiv mirrors:** pages on alphaxiv, Hugging Face Papers, Semantic Scholar, Emergent Mind,
papers.cool, scholar.archive.org and Papers with Code that are keyed by an arXiv ID now map to
`https://arxiv.org/abs/<id>`, so they share the paper's web ID.

**Issue 1 (ranking):**
- **Title embeddings alone did not fix it.** Chunks are now embedded as
  `<title>\n<section>\n\n<text>` (a paid re-embed of 2.23M tokens, about $0.045). That moved
  `2308.04079` from outside the top 60 to rank 28, but eval top-6 hits stayed 22/26, because dozens
  of follow-ups also have "Gaussian Splatting" in their titles.
- **What fixed it:** re-ranking candidates by cosine similarity + 0.02 × log10(1 + citations), from
  `papers.sqlite` and configurable as `RAG_CITATION_PRIOR`.
- **Eval-set results:**

  | Index | Citation prior | Top-6 hits | Rank-1 hits | MRR | 3DGS paper's rank |
  |---|---|---|---|---|---|
  | old | 0 | 22/27 | 19 | 0.754 | outside top 80 |
  | new | 0.02 | 25/27 | 21 | 0.822 | 5 |
  | new | 0.05 | 25/27 | 25 | 0.928 | 1 |

  0.05 pushed an anti-aliasing literature-review question from rank 8 to 16, so I chose 0.02.
- **Trade-off:** popularity bias. A rarely cited new paper ranks lower in unfiltered searches.
  Searches filtered by `arxiv_ids` aren't affected.

**Issue 2 (wrong attribution):** `rag/citations.attribution_mismatches()` flags a claim that names
a paper by arXiv ID while citing only sources that never mention it. The worker then gets one
repair call. Anything still mismatched becomes an `attribution_mismatch` `AgentError`, which
synthesize sees. `research.md` now requires the arXiv ID next to each web paper's name, so the
check can apply.
- **Limit:** it can't catch a misattributed claim that doesn't name a paper's arXiv ID.

**Issue 3 (latency):**
- **Thinking levels:** `get_llm` takes `thinking_level`, and Gemini 3.8 Flash rejects "minimal".
  - The supervisor uses "low": median 3.2 → 2.5 s, worst case 5.1 → 2.6 s, same 5/5 routing.
  - Workers use "low": the research worker went from 27 s to 13 s and found the same papers, with
    fewer search rounds and citations.
  - Synthesize keeps the default thinking level.
- **`research.md` changes:**
  - a "When to stop" rule;
  - search with a method's distinctive mechanisms;
  - no `get_paper_metadata` call just for titles;
  - one filtered search for comparisons;
  - always set `time_range` for recent work, with an example;
  - never send a query made only of `site:` operators;
  - don't search for formulas, because equations aren't indexed.
- **Root cause of the 3DGS loop:** the worker spent its last rounds hunting for the alpha-blending
  equation, which ingest drops.

**Results:** four-question CLI runs, against the Phase 4b runs from 2026-10-21.

| Question | Before | After | Tool rounds | Notes |
|---|---|---|---|---|
| 3DGS | 49 s | 31–34 s (two runs) | 4 (cap) → 2 | all citations from `2308.04079` |
| "This month" | 50–62 s | 27 s | 2 | used `time_range` |
| NeRF vs Instant-NGP | 50 s | 48 s | 3 | synthesize took 21 s for a long answer |
| Point cloud | 11 s | 8 s | none | |

Every cited corpus ID was in the retrieved list, and the attribution check didn't fire.

**Verified:** `pytest tests/` gives 86 passed, 2 live tests skipped (4 new tests).

**Process note:** while backing up the index I typed a stray `rm -rf /dev/null`, which AGENTS.md
forbids. It failed with "Operation not permitted" and changed nothing.

### 2026-10-22 — Fixed known issues 4 (citation formatting) and 5 (duplicate web sources)

**Issue 4, citation formatting:**
- `prompts/research.md` and `prompts/synthesize.md` now both require one ID per pair of brackets,
  written side by side as `[a][b]`.
- `synthesize.md` no longer conflicts: instead of "copy verbatim" plus "one per bracket", it says
  to split any grouped IDs it finds in the findings.
- Citation examples in both prompts are now plain text, not code spans, and both prompts say
  never to put citations in code formatting.

**Issue 5, duplicate web sources:**
- `normalize_url` now treats http and https as the same, drops `www.`, and maps every arXiv page
  of a paper (abs, html, pdf, any `vN`, export, ar5iv) to `https://arxiv.org/abs/<id>`.
- `normalize_tavily` keeps only the first result per `source_id`.
- IDs of plain arXiv abs URLs are unchanged, e.g. LoCoSplat is still `web:1fd729ad`.

**Found while verifying:** Tavily reports some failures inside a successful MCP result, e.g.
`{"error": "Search failed", "status": 400, "detail": {"error": "Query cannot consist only of
site: operators..."}}`. The normalizer was calling these "malformed response", so the model
never learned why its query failed. It now returns `tavily_search failed: <Tavily's message>`
as a tool error.

**Verified:**
- `pytest tests/`: 82 passed, 2 live tests skipped (2 new tests).
- I re-ran the 3DGS and "this month" CLI questions. Across 41 citations, none were grouped or in
  backticks, and no paper was cited under two web IDs.

**Still open:** third-party arXiv mirrors (e.g. alphaxiv.org) keep their own IDs. Issues 1–3 are
unchanged. In the re-run, the worker didn't use Tavily's `time_range`, so the "this month"
results were again thin; the answer said so honestly.

### 2026-10-22 — Known issues after Phase 4b (diagnosed, not yet fixed)

The four Phase 4b CLI runs worked end to end, but five things don't work well. Each was traced to
a cause before any fix.

1. **Ranking: a broad 3DGS question doesn't retrieve the 3DGS paper.**
   - For "How does 3D Gaussian Splatting render a novel view?", `2308.04079` is not in the top 60
     results. The worker's keyword-heavy rewrite ranks it 7th.
   - **Cause:** the 3DGS paper never uses the phrase "Gaussian Splatting" or "3DGS" in its own
     text (0 mentions in 27 chunks). The name appears only in the title, and titles aren't
     embedded with the chunks.
   - Meanwhile 90 follow-up papers re-describe 3DGS rendering in their introductions, related
     work and preliminaries (284 chunks, about 0.85 mentions per chunk), and those win the
     similarity search.
   - **Possible fixes:** embed each chunk with its paper title and section; add keyword (BM25)
     search alongside the embeddings; limit results per paper. Embedding with titles needs a
     paid re-embed.
2. **Wrong citation: a claim was attributed to the wrong web source.**
   - In the first "this month" run, DroneSplat was cited to `[web:772d7f0a]`, an MDPI paper
     comparing NeRF and 3DGS on UAV imagery.
   - **Cause:** the summarizing call after the tool-round cap had about 35 web results in
     context, tagged only with opaque IDs, and two of them were on similar drone/UAV topics.
     Nothing checks that a claim matches its source.
   - **Note:** the planned 4c validator only checks that a cited ID was retrieved this turn, so
     it would not catch this, because `web:772d7f0a` was retrieved.
   - **Possible fixes:** show each source's title beside its ID in the worker's evidence, or
     check that each claim matches its source.
3. **Latency: about 50 s per research turn.**
   - **Worker:** it hits the 4-round tool cap on every question, then makes a fifth,
     summarizing call. `prompts/research.md` encourages more calls ("reformulate and retry
     once", "search for each separately", "call get_paper_metadata") and never says when to
     stop. The re-sent context grows with every round.
   - **Supervisor:** each decision takes 3.7–3.9 s for about 600 input and 250 output tokens,
     and about half the output tokens are reasoning. `llm.py` sets no thinking options, so
     Gemini thinks by default.
   - **Tools:** fast. `search_corpus` took 0.3–0.9 s and Tavily 1.5–4 s.
4. **Citation formatting: synthesize groups IDs as `[a, b]` and once wrapped them in backticks.**
   - **Cause:** conflicting prompt instructions. `synthesize.md` says both "copy citations
     verbatim" and "one ID per pair of brackets", and `research.md` never forbids grouping, so
     grouped citations in the findings are copied through.
   - Both prompts also show citation examples inside Markdown code spans, which the model
     imitates.
5. **Duplicate web sources: the same paper gets two web IDs.**
   - **Cause:** `normalize_url` only lowercases the host, drops tracking parameters and trims
     trailing slashes. arXiv's abs, html and pdf pages, `vN` versions and `www.arxiv.org` all
     hash differently, so LoCoSplat was cited as both `abs/2610.04351` and
     `html/2610.04351v1`.

**Plan:** fix 4 and 5 now (small, free). Fix 3 next, by giving the worker a stopping rule and
turning down supervisor thinking. Fix 1 with the re-embed, after approval. Fix 2 as part of
Phase 4c.

**Status (2026-10-22):** 4 and 5 are fixed in `c094375`. Still open:
- third-party arXiv mirrors (e.g. alphaxiv.org) get their own web IDs;
- issues 1, 2 and 3;
- the worker doesn't reliably use Tavily's `time_range`, so a "this month" question found few
  October 2026 papers in the re-run (the answer said so honestly).

Next, in order: mirrors; a worker stopping rule and less supervisor thinking (3), with
`time_range` guidance; a check that a claim's paper matches its cited source (2); and embedding
chunks with their paper title (1, a paid re-embed).

**Status (later on 2026-10-22):**
- Mirrors and issues 1–3 are fixed in `f9ff7e9`. Comparison latency is fixed in `2d0e2f6`.
- `time_range` is now used for "this month" questions.
- **Still open:**
  - Research time (20–29 s per research turn) is now the main cost.
  - The attribution check can't catch a misattributed claim that doesn't name an arXiv ID.
  - The citation prior favors highly cited papers in unfiltered searches.
  - An unhandled Gemini 504 in a worker crashes the turn (Phase 4c failure wrapper, Phase 5
    fallback).

**Status after Phase 4c (`abdea34`):**
- **Fixed:** a worker failure (including a provider 504) is now an `AgentError` and the turn
  completes. Phase 5's fallback provider will add a retry on another provider.
- **Still open:**
  - research time;
  - claims with no citation aren't validated (the broken-Tavily answer invented a corpus date
    range);
  - the citation prior's popularity bias;
  - one extra research dispatch on the turn a server reconnects.

### 2026-10-22 — Phase 4b: worker factory and the `research` worker

**Built:**
- **`graph/workers.py`:** `make_worker(spec, tools, llm)`, a compiled subgraph
  `worker_agent ⇄ worker_tools`. `worker_tools` wraps `ToolNode(handle_tool_errors=True)` and
  emits `tool_call`/`tool_result` traces. After `WORKER_MAX_TOOL_ROUNDS` (default 4) the model
  makes one summarizing call without tools.
- **Worker output:** one `Finding`, whose citations get chunk and web metadata from this run's
  tool results, plus `retrieved_chunk_ids`.
- **`rag/citations.py`:** parses `[chunk_id]` and `[web:<8 hex>]`, including adjacent citations
  and comma-separated groups.
- **`mcp_client.py`:**
  - Tavily results get `source_id = web:` plus 8 hex chars of sha1(normalized URL).
  - A malformed payload becomes a clean tool error.
  - An agent missing an allowlisted tool is logged, and that server's description goes into the
    worker's prompt.
- **Registry and wiring:** `research` is registered, and `build_graph(mcp=...)` builds workers
  from the registry. The CLI opens MCP once and prints the retrieved IDs and a citation list.

**Findings while testing:**
- **Subgraph traces:** trace events written inside a subgraph reach the parent's stream only
  with `astream(..., subgraphs=True)`, so the CLI and the test helper use it.
- **Dates:** the worker didn't know today's date. It searched for "October 2024" papers when
  asked about "this month", and once cited an MDPI page for a different paper. The worker and
  synthesize inputs now include today's date. `prompts/research.md` says to use it and Tavily's
  `time_range`, and notes that arXiv IDs encode YYMM.
- **Retrieval:** a broad search for the 3DGS question didn't return the 3DGS paper in the top 6.
  The worker recovered by filtering by `arxiv_ids`.

**Verified:**
- `pytest tests/`: 80 passed, 2 live tests skipped (14 new worker tests).
- Real CLI runs:
  - 3DGS: the answer cites `2308.04079` chunks.
  - "This month": October 2026 arXiv papers with `[web:xxxxxxxx]` IDs.
  - NeRF vs Instant-NGP: four filtered corpus searches, citing both papers.
  - "What's a point cloud?": no tool calls.

### 2026-10-22 — Phase 4a: orchestration skeleton (no real workers)

**Dependency (approved):** `langgraph==1.2.16`, the latest release; it needs `langchain-core>=1.4.7,<2`.
I read its installed source:
- **`langgraph.config.get_stream_writer()`** reads the writer from the runnable config's runtime.
  It is a no-op in a run that doesn't stream `custom`, and raises `RuntimeError` outside any
  runnable context.
- **Stream modes** are `values`, `updates`, `checkpoints`, `tasks`, `debug`, `messages` and
  `custom`.
- **`with_structured_output`** in `langchain-google-genai` 4.4.2 defaults to `json_schema`.

**Built:**
- **`graph/state.py`:** the Pydantic models, plus `OrchestratorState` with a new `dispatches`
  list, mirrored in AGENTS.md.
- **`graph/trace.py`:** `emit_trace()`.
- **`graph/supervisor.py`:** decides through structured output, then applies the guards in order
  (`supervisor_parse_error`, `step_limit`, `worker_unavailable`, `repeated_dispatch`).
- **`graph/synthesize.py`:** streams the answer with `tags=["synthesize"]`.
- **`graph/build.py`:** `build_graph()` with injectable workers and models, and `DEFAULT_CONFIG`
  with `recursion_limit = 2 × MAX_SUPERVISOR_STEPS + 4`.
- **`scripts/cli.py`**, **`prompts/supervisor.md`** and **`prompts/synthesize.md`**.
- **`config.max_supervisor_steps()`**.
- **`agents/registry.py`:** `AgentSpec` gains `name` and `description`, and there is a new, empty
  `WORKERS` plus `available_workers()`. `AGENTS` keeps all three allowlists, because
  `mcp_client.py` filters tools with them.

**Design choices:**
- **Supervisor roster:** the supervisor sees every agent in `AGENTS`, with the ones that aren't
  wired in marked unavailable. It routes by what the question needs, so the
  `worker_unavailable` guard records the missing capability and the answer says so instead of
  guessing.
- **`emit_trace` signature:** it takes a `step` keyword, because the trace payload includes the
  supervisor step.
- **CLI stream modes:** the CLI also streams `values`, to get the final state for its summary
  line.

**Verified:**
- `pytest tests/`: 66 passed, 2 live tests skipped. That includes 11 new skeleton tests.
- Real Gemini runs:
  - "What's a point cloud?" goes supervisor → synthesize and streams an answer.
  - The 3DGS question goes to research, hits `worker_unavailable`, and the answer says the
    capability isn't available, with no citations.
  - The REPL ran three questions in a row; the PSNR question was routed to `data`.

### 2026-10-22 — Stress test of Phases 1–3, and fixes

**How:** throwaway scripts (not committed) ran the real app under uvicorn with a fake model. They
sent 60 concurrent streams, a client disconnect mid-stream, a slow first token, a provider error and
malformed bodies. They also sent adversarial SQL to `run_sql`, odd inputs to the corpus tools over
the real index (with fake query embeddings), and concurrency, crashes, hangs and oversized output
through the MCP client.

**Bugs found and fixed:**
- **`run_sql`:**
  - Blob values crashed the tool ("bytes is not JSON serializable"), and `1e999` produced
    invalid JSON (`Infinity`).
  - There was no size bound. `SELECT printf('%.*c', 3000000, 'x') FROM papers` built a 600 MB
    response and reached 1.2 GB RSS, which would take down a 2 GiB instance.
  - **Fix:** a SQLite value-length limit of 100 KB, text cells cut at 1,000 chars, blobs
    described, and the response capped at 100k chars (marked `truncated`).
- **`run_sql` blocking:** the tool was synchronous, so FastMCP ran it on the server's event loop
  and one slow query stalled every other DB call for up to 2 s. It now runs in a worker thread.
- **MCP client, restart race:** after a crash, 4 of 5 concurrent callers got "unavailable"
  even though the restart succeeded. Callers now wait for a restart that is already in progress,
  and restart locks are per server.
- **MCP client, no retries for servers that never started:** a server that failed at startup
  was never retried (e.g. Tavily during a cold-start network blip). The new `refresh()`, called
  once per request from Phase 4, retries it in the background after the cooldown.
- **MCP client, output size:** tool output was unbounded (20 MB passed straight through). It is
  now cut at 100k chars.
- **Redaction:** it covered only the Tavily key and was not installed in the app. An exception
  message containing the Google key was logged verbatim. It now covers every provider key and
  every handler, including uvicorn's, and `main.py` installs it at startup.
- **Request validation:**
  - A whitespace-only message passed and called the model.
  - `session_id` accepted any characters, including `/`, and it becomes a Firestore document
    ID in Phase 7.
  - **Fix:** the message is stripped first, and `session_id` must match `[A-Za-z0-9_-]{1,128}`.
- **Retriever temp copies:** the 116 MB temp copy of the index leaked whenever a corpus server
  was killed rather than exiting normally, and on Cloud Run `/tmp` is memory. Copies are now
  tagged with their process ID, and orphaned ones are removed at startup.
- **Frontend:**
  - `crypto.randomUUID` is missing outside HTTPS and localhost, which broke the whole script.
  - The session ID changed on every reload.
  - Over-long input showed "HTTP 422".
  - **Fix:** a fallback ID kept in `sessionStorage`, `maxlength="8000"` on the input, and a
    readable message for 422 responses.
- **arXiv IDs:** an uppercase `ARXIV:` prefix was rejected as malformed. Normalization is now
  case-insensitive in both servers.

**Held up under load:** 60 concurrent streams all ended in `done`. A disconnect stopped generation
after 3 of 40 tokens. Heartbeats arrived before a slow first token, and an error was followed by
`done`. Path traversal returned 404. The SQL guard held (DELETE inside a CTE, `pragma_table_info`,
`load_extension`). Timeouts and startup hangs were handled. All 4,867 chunk IDs match the
`get_chunks` validator.

**Ingest fixes (re-embed approved):** a review of `ingest/` against the shipped data found these:
- **Bold table headers taken as section headings:** headers like "Method" started fake sections,
  giving 111 duplicate-slug sections and junk table-header chunks. Once a paper uses numbered
  headings, an unnumbered known heading is now accepted only for abstract, references,
  acknowledgments or appendix, or right after references. A slug is never reopened.
- **Body sentences filed as captions:** sentences like "Figure 3 shows..." went into captions.
  The caption number must now be followed by a delimiter or a capitalized word. Roman-numeral
  and letter numbers are still accepted.
- **Acknowledgments indexed:** this covered singular spellings, run-in "Acknowledgements. We
  thank..." (even mid-paragraph) and NeurIPS's "Acknowledgments and Disclosure of Funding".
  More than 1,200 chars after an acknowledgments heading is kept as an unheaded appendix rather
  than dropped.
- **Affiliation lines taken as headings:** e.g. "1 The University of Tokyo 2 RIKEN AIP". In
  front matter, only known headings are accepted now.
- **Trailing hyphen in slugs:** truncation could leave one, producing `--` in chunk IDs.
- **`chunk.py`:** a single over-long "word" is cut to chunk size, and two subheadings in a row
  both keep their text.
- **`embed.py`:** a metadata-only change now updates metadata without re-embedding.

Before writing, the new parser was compared with the old one on all 220 papers:
- body text +0.3%;
- duplicate-slug sections 111 to 15 (the rest are real numbered repeats);
- every paper's first section is now `introduction`;
- the one paper that lost more than 10% lost only table-cell text.

The re-run gave 4,741 chunks (was 4,867). 673 were embedded (294k tokens, about $0.006) and 296
stale ones deleted. Chroma and `chunks.jsonl` hold identical ID sets. On the eval questions, a
reference paper appears in the top 6 for 22 of 25 retrieval questions. q01 still ranks
2403.17888 above the 3DGS paper, which matches the old index.

**Known, not fixed:** a tool call still in flight when the client shuts down waits for the 60 s
read timeout; this only matters at shutdown.

**Verified:** `pytest tests/` gives 55 passed, 2 live tests skipped (12 new regression tests). The
stress scripts were re-run and all issues above are resolved. The rebuilt image passes the
in-container MCP check, and no index copy is left in `/tmp` after exit.

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
