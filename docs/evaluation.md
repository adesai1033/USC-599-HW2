# Evaluation

All numbers were measured on 2026-10-26. Raw per-question results are in `eval/results/`:
- `*.scores.jsonl` and `*.md` hold the RAGAS scores.
- `latency-*.jsonl` and `latency-*.md` hold the latency runs.
- `cost-*.md` holds the cost tables.

The answer files themselves (`<label>.jsonl`, with full chunk texts) are gitignored.

## Method

- **Test set:** `eval/questions.jsonl` has 29 questions, each with a hand-written ground truth and
  its reference arXiv IDs: 18 lookups, 5 compares, 2 literature reviews, 3 metadata questions and 1
  not-in-corpus question.
- **What RAGAS scores:** the 25 corpus questions. The metadata questions are answered from SQL rows
  and the not-in-corpus question has no corpus context, so these metrics don't apply to them. They
  are still run, timed and costed.
- **Answering** (`eval/run_ragas.py answer`): each question runs through the real graph and real MCP
  servers, the same way `POST /chat/stream` does, in a fresh session.
  - The **contexts** are the corpus chunks retrieved in that run, fetched with the `get_chunks` tool.
  - Citation tags are stripped from the answer before judging.
- **Scoring** (`eval/run_ragas.py score`): RAGAS 0.4.3, `ragas.metrics.collections`.
  - **Metrics:** `Faithfulness`, `AnswerRelevancy`, `ContextPrecisionWithReference` and
    `ContextRecall`.
  - **Judge:** `gpt-6.1-sol` at medium reasoning; low reasoning flipped verdicts between runs.
    Embeddings for answer relevancy use `text-embedding-3-small`.
  - **Separate venv:** RAGAS 0.4.3 needs `openai<2` and `langchain-community<0.4`, which conflict with
    the app's pins, so it runs in its own `.venv-eval` (`eval/requirements-ragas.txt`).
  - **Judge cost:** about 1.4M input and 0.14M output tokens per 25-question pass, about $4.

## RAGAS results

| Run | n | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|
| Baseline | 25 | 0.930 | **0.823** | 0.854 | 0.881 |
| After the prompt fix (`iter1`) | 25 | 0.918 | 0.833 | 0.840 | 0.919 |

By question type, for the run after the fix:

| Group | n | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|
| lookup | 18 | 0.917 | 0.838 | 0.876 | 0.930 |
| compare | 5 | 0.915 | 0.876 | 0.886 | 0.911 |
| lit_review | 2 | 0.933 | 0.683 | 0.401 | 0.833 |

### The iteration (one, as budgeted)

- **Diagnosis:** answer relevancy was the lowest metric (0.823). Reading the low scorers showed the
  answers were on topic but opened with a heading and carried detail nobody asked for: ablation
  numbers, implementation specifics, side results. Answer relevancy works by generating the question
  back from the answer, so it penalizes exactly that. Retrieval wasn't the problem: context recall
  was 0.881 and precision 0.854.
- **Fix:** `prompts/synthesize.md` gained a "Shape" section.
  - Open with one or two sentences that answer the question in its own terms, both parts of a
    two-part question.
  - Never start with a heading.
  - Drop details the question didn't ask for.
  - For a literature review, open with the common thread.
- **Result:** relevancy rose 0.823 → 0.833. Context recall also moved, by 0.04, although a prompt
  change can't affect retrieval, so the gain is **within run-to-run noise**. The prompt stays: it
  didn't hurt, and the answers read better. Faithfulness and precision moved by about 0.01, also
  noise-level.

### Where it is weakest

- **Literature reviews:**
  - q24: relevancy 0.61, precision 0.45.
  - q25: relevancy 0.75, precision 0.35.
  The summary worker pulls evidence for many papers, and much of it ranks as irrelevant to the
  reference answer. Multi-paper retrieval, scoping which papers to fetch, is the next thing to fix,
  beyond this phase's one iteration.
- **q09** (BARF's coarse-to-fine encoding; faithfulness 0.65): the answer is on point and cites the
  BARF method section, but it walks through the gradient derivation step by step. My reading, not
  checked against the judge's per-claim verdicts, is that some of those steps restate the chunks
  loosely enough that the judge couldn't match them.
- **q10** (how SplaTAM tracks the camera; relevancy 0.68): the answer opens with a direct answer, as
  the new prompt asks. It then still runs to headed sections of implementation detail (iteration
  counts, loss weights), the habit the prompt fix targeted. The prompt reduced it but didn't remove
  it.

### Caveats

- n = 25 and RAGAS judgments are themselves noisy; differences under about 0.03 aren't meaningful.
- The judge is an LLM. Faithfulness counts claims supported by the retrieved chunks, not whether they
  are true.

## Latency

Measured with `eval/latency.py` against the deployed service. Five queries, one per path through the
graph and none of them in the test set, each run in a fresh session. Stage times come from when each
SSE event reaches the client, so they include the network:

| Stage | Meaning |
|---|---|
| start | request sent → `load_history` entered (container start on a cold request) |
| history | Firestore read |
| route | the supervisor's first decision |
| worker | worker runs (model calls plus tool calls) |
| route2 | the supervisor's decision to synthesize |
| first tok | `synthesize` entered → first token |
| TTFT | request sent → first token |
| retrieval / LLM | server-side sums of tool-call and model-call time, from the `done` event |
| total | request sent → `done` |

**Warm (representative):** 3 repeats each, median ms.

| Query | start | history | route | worker | route2 | first tok | **TTFT** | retrieval | LLM | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| lookup | 54 | 21 | 2,409 | 10,035 | 1,940 | 1,553 | **16,327** | 291 | 21,046 | **21,508** |
| compare | 44 | 19 | 3,272 | 28,895 | 2,264 | 813 | **35,786** | 973 | 42,703 | **43,812** |
| data | 51 | 20 | 2,516 | 21,721 | 2,104 | 761 | **26,887** | 58 | 32,828 | **33,091** |
| current events | 44 | 16 | 2,699 | 19,033 | 2,522 | 1,309 | **26,005** | 8,192 | 22,445 | **27,800** |
| no tools | 49 | 17 | 2,367 | – | – | 1,647 | **4,094** | 0 | 7,199 | **7,374** |

**Cold:** the first request after the service had scaled to zero.

| Query | start | TTFT | total |
|---|---|---|---|
| lookup (the cold request) | 48,166 | 66,328 | 72,147 |

- **Where the time goes:**
  - The worker stage dominates, and inside it the model calls: tool calls are under 1 s, except Tavily
    at about 8 s.
  - Each supervisor decision costs about 2.4 s, and there are two per tool-using turn.
  - Synthesize's first token arrives about 1 s after it starts. Its time is spent writing.
  - Compares are slowest because research runs on Claude and reads more chunks.
- **Cold start:** 48 s of the cold request passes before the graph starts: container start, imports,
  spawning the MCP servers and loading the Chroma index on 1 vCPU. After that the request runs at
  warm speed.
- **Revision note:** these runs were on revision `00004-nxk`. Since then (`00009-tkh`) startup waits
  for the MCP servers before the port opens, which fixed cold instances that lost corpus search. The
  cold path is the same work in a different order, so the cold numbers weren't re-measured.

## Cost per query

`eval/cost.py` prices the per-provider token usage that each run's `done` event reports. Prices were
checked on 2026-10-26:

| Model | Input / output, per 1M tokens | Source |
|---|---|---|
| `gemini-3.8-flash` | $0.75 / $3.75 | ai.google.dev/gemini-api/docs/pricing (paid tier) |
| `claude-sonnet-5-5` | $2.00 / $10.00 | platform.claude.com/docs/en/about-claude/pricing |
| `gpt-6.1-sol` (judge only) | $2.00 / $10.00 | developers.openai.com/api/docs/pricing |
| `text-embedding-3-small` | $0.02 | developers.openai.com/api/docs/pricing |

- **Embeddings:** about 30 tokens per `search_corpus` call, which is negligible.
- **Tavily:** one credit per search or extract.
- **Cloud Run:** estimated from each request's wall time at 1 vCPU and 2 GiB. This is an upper bound,
  since concurrent requests share an instance. The Cloud Run rate was not re-checked: the pricing
  page didn't render on 2026-10-26.

**Baseline (normal routing), 29 questions:**

| Group | n | Input tokens | Output tokens | Models | Cloud Run | Mean $/query | Median $/query |
|---|---|---|---|---|---|---|---|
| **all** | 29 | 41,989 | 3,154 | $0.0780 | $0.0010 | **$0.0790** | **$0.0397** |
| lookup | 18 | 33,611 | 2,732 | $0.0433 | $0.0009 | $0.0442 | $0.0362 |
| compare | 5 | 56,731 | 4,856 | $0.1517 | $0.0016 | $0.1533 | $0.1357 |
| lit_review | 2 | 127,938 | 7,133 | $0.3220 | $0.0015 | $0.3235 | $0.3235 |
| metadata | 3 | 19,567 | 1,097 | $0.0228 | $0.0005 | $0.0233 | $0.0250 |
| not in corpus | 1 | 14,455 | 451 | $0.0125 | $0.0006 | $0.0131 | $0.0131 |

- **Split by provider:** Gemini $0.0225 and Claude $0.0555 per query. Claude handles the minority of
  calls (compares, multi-paper summaries, large syntheses) but 70% of the spend.
- **Phase spend:** a full evaluation pass (answering plus judging 25 questions) costs about $6.
- **Ingestion, one-time:**
  - Embeddings: $0.043.
  - Results extraction: $3.92 on Gemini Flash. A first run cost another $3.95 and was lost to a bug.

## Provider comparison

`run_ragas.py answer --provider P` sends every model call to one provider, whatever the router
decides. Nine questions, chosen to cover each type: lookups q01, q04, q07, q10 and q15; compares q20
and q22; lit review q25; metadata q26.

| Provider (forced) | Mean $/query | Median total time | Errors |
|---|---|---|---|
| Gemini `gemini-3.8-flash` | $0.044 | 35 s | none |
| Claude `claude-sonnet-5-5` | $0.173 | 30 s | none |
| OpenAI `gpt-6.1-sol` | $0.100 | 45 s | 4 worker timeouts on all 3 multi-paper questions; q20 lost both workers |
| *Routed (normal)* | *$0.079* | — | — |

RAGAS on the 7 corpus questions all three answered (q20 excluded, since the OpenAI run returned no
context for it):

| Provider | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| Gemini | 0.941 | 0.795 | 0.803 | 0.833 |
| Claude | 0.964 | 0.820 | 0.784 | 0.917 |
| OpenAI | 0.992 | 0.840 | 0.908 | 0.888 |

- **Gemini vs Claude:** Claude is about 4× Gemini's price for slightly better faithfulness,
  relevancy and recall. Routing only the multi-paper work to Claude gets most of that for $0.079 per
  query. Gemini is good enough for lookups, metadata and routing.
- **OpenAI:** it scores highest here, with two caveats.
  - The judge is also `gpt-6.1-sol`, so part of the lead may be the judge preferring its own model's
    wording.
  - The scores exclude q20, where OpenAI failed outright.
  It was also the slowest: every compare and lit-review question hit the 60 s worker timeout.
- **Decision:** with that, and the Phase 9 outage where exhausted OpenAI credits broke embeddings and
  the fallback at the same moment, OpenAI was dropped as a chat provider. Gemini and Claude now back
  each other up, and OpenAI is kept only for embeddings and as the judge.
