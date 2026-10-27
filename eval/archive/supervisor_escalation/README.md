# Archived experiment: supervisor escalation

Measured on 2026-10-26 and archived the same day: no quality gain over the Gemini supervisor, at
higher cost and latency. The code is not in the live app.

## Contents

| Path | What it is |
|---|---|
| `supervisor_escalation.patch` | The code: `SUPERVISOR_ESCALATION` in `config.py`, the supervisor rule in `router.py`, the features passed by `graph/supervisor.py`, 8 router tests, and `eval/run_ragas.py answer --routing escalate\|senior` (which also records supervisor decisions) |
| `compare_arms.py` | Routing accuracy, latency, cost, errors and RAGAS per arm |
| `results/` | RAGAS summaries (`*.md`) and per-question scores (`*.scores.jsonl`) |
| `answers/` | Full answer records with chunk texts, local only (gitignored, like `eval/results/*.jsonl`) |

The git tag `experiment/supervisor-escalation` marks the commit where the code was live.

## Restoring it

```bash
git apply eval/archive/supervisor_escalation/supervisor_escalation.patch
.venv/bin/python -m pytest tests/test_router.py
.venv/bin/python eval/run_ragas.py answer sup-escalate --routing escalate
```

The patch applies to the code as of commit `287fac3`; if those files have changed since, apply it
with `git apply -3` or check out the tag.

## Results

**Question:** would a stronger supervisor route better, since every turn passes through it?
`SUPERVISOR_ESCALATION` (off by default) moves the supervisor to Claude for hard decisions: re-planning
after a failed or ungrounded worker, the first decision of a follow-up turn, and combining
multi-paper findings. A third arm tested a "senior planner, fast executor" split (supervisor always
Claude, research always Gemini; eval-only, `run_ragas.py answer --routing senior`).

All 29 questions ran under each arm on 2026-10-26, from the same code and prompts, with RAGAS on the
25 corpus questions. Labels `sup-control`, `sup-escalate` and `sup-senior`.

| Arm | First dispatch acceptable | `task_type` correct | Supervisor calls on Claude | Errors | Median TTFT | Median total | Mean $/query |
|---|---|---|---|---|---|---|---|
| Control (current) | 29/29 | 27/29 | 0/59 | 0 | 22.1 s | 28.3 s | $0.074 |
| Escalation | 29/29 | 27/29 | 10/59 | 0 | 24.3 s | 29.7 s | $0.082 |
| Senior planner | 29/29 | 28/29 | 59/59 | 0 | 24.6 s | 30.7 s | $0.086 |

| Arm | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| Control | 0.914 | 0.839 | 0.882 | 0.895 |
| Escalation | 0.920 | 0.846 | 0.875 | 0.907 |
| Senior planner | 0.921 | 0.849 | 0.841 | 0.898 |

- **Routing quality was already at ceiling.** Every arm sent all 29 questions to an acceptable first
  worker, and no arm had a single worker failure to re-plan around. The Gemini supervisor's only
  mistakes were labeling two lookups (q03, q12) as `compare`, which sends research to Claude
  unnecessarily; the senior arm fixed one of the two.
- **Escalation fired 10 times, all on the "enough findings?" decision of a compare or lit-review
  question.** It changed the dispatch sequence once: on q19 Claude asked research for more evidence
  where Gemini asked data for benchmark numbers. q19's faithfulness rose 0.59 → 0.90, but most of
  that is a measurement artifact: RAGAS judges against corpus chunks only, so numbers from SQL rows
  count as unsupported.
- **The quality differences are noise.** Every RAGAS mean is within 0.012 of control except the
  senior arm's context precision (−0.041), and the earlier iteration showed run-to-run swings of
  about 0.04 with no change to retrieval.
- **The costs are real:** escalation adds about 11% per query and 2 s to the median TTFT; the senior
  split adds about 16% and 2.5 s, with every supervisor call on the slower model.
- **Decision:** keep the Gemini supervisor. `SUPERVISOR_ESCALATION` stays in the code, off by
  default. The multi-turn follow-up trigger is untested here, since every eval question is a single
  turn.
