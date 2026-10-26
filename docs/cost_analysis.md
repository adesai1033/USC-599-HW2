# Cost Analysis

The measured cost per query, the pricing sources and the provider comparison are in
[`evaluation.md`](evaluation.md#cost-per-query). This page is the short version. Prices were checked
on 2026-10-26.

## Who pays for what

| Role | Provider | Model | Input / output, per 1M tokens |
|---|---|---|---|
| Supervisor, data, research (default), synthesize (default) | Google | `gemini-3.8-flash` | $0.75 / $3.75 |
| Summary, research on compares and lit reviews, large syntheses | Anthropic | `claude-sonnet-5-5` | $2.00 / $10.00 |
| Fallback | each tier falls back to the other | | |
| Embeddings (query and ingestion) | OpenAI | `text-embedding-3-small` | $0.02 |
| RAGAS judge (eval only) | OpenAI | `gpt-6.1-sol` | $2.00 / $10.00 |

OpenAI is not a chat provider. The routing rules are in [`architecture.md`](architecture.md).

## Measured (2026-10-26, 29-question baseline, normal routing)

- **Mean $0.079 per query** (median $0.040): Gemini $0.0225, Claude $0.0555, Cloud Run about $0.001.
- **By type:** lookup $0.044, compare $0.153, lit review $0.324, metadata $0.023.
- **Forced to one provider** (9 questions): Gemini $0.044, Claude $0.173 per query.
- **Tavily:** one credit per search or extract.
- **Firestore:** up to `HISTORY_TURNS` (6) document reads and one write per turn, negligible at
  coursework scale.

## One-time and evaluation costs

- **Ingestion:** embeddings $0.043; benchmark-result extraction $3.92 on Gemini Flash.
- **Evaluation:** about $6 per full pass (answering plus judging 25 questions); the judge alone is
  about $4.

Re-check each provider's pricing page before quoting these figures later.
