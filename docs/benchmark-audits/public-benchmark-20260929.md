# Public three-pipeline benchmark — 2026-09-29

## Run configuration

- Dataset: `hackathon-resources/questions/eval_public.jsonl` (100 questions)
- Dataset SHA-256: `abddb7d18a6d8ed908f514a7e560fe4950ebe479ebb2cdb75a7456887c10c6e5`
- Pipelines: RAG, GraphRAG, Agentic GraphRAG
- Provider/model: Cloudflare Workers AI, `@cf/meta/llama-3.1-8b-instruct-fast`
- Output: `results/public_clean_20260929.jsonl` (one row per public question, includes answers, traces, token counts, latency, and retrieval metrics)
- EM and token F1 are computed by the repository evaluator. The historical Agentic prompt examples copied from `pub-001` and `pub-002` were removed before this run; regression coverage checks for public question text in the prompt.

## Overall results

| Pipeline | Exact match | Token F1 | Mean latency (ms) | Mean LLM tokens | Mean document MRR | Mean document recall@5 | Mean document precision@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAG | 43/100 (43%) | 0.453 | 1,213.1 | 2,422.8 | 0.5643 | 0.6039 | 0.4477 |
| GraphRAG | 35/100 (35%) | 0.395 | 1,657.3 | 2,083.8 | 0.5019 | 0.3954 | 0.3880 |
| Agentic GraphRAG | 60/100 (60%) | 0.619 | 3,446.2 | 5,113.9 | 0.7372 | 0.5470 | 0.6835 |

## Agentic exact match by question type

| Type | Correct / total | Exact match |
|---|---:|---:|
| Aggregation | 19/21 | 90.5% |
| Temporal | 2/22 | 9.1% |
| Superlative | 10/10 | 100% |
| Multi-hop | 18/28 | 64.3% |
| Lookup | 11/19 | 57.9% |

## Interpretation and limits

- This is the latest measured public answer benchmark in this repository. It is one run, not a confidence interval; accuracy is materially below the 98% goal.
- Agentic has the best EM and mean document MRR in this run, while using more tokens and taking longer than either simpler pipeline. The results do not imply agentic reasoning is cost-effective for every question type.
- Temporal questions are the clearest failure concentration. Multi-hop and lookup also need answer-level error analysis. Retrieval metrics alone cannot establish that answer-bearing evidence was found or that the generated answer is grounded.
- Token F1 is lexical overlap, not semantic completeness or factuality. Mean document recall/precision/MRR use the evaluator's gold-document IDs and must not be represented as citation correctness.
- The benchmark used the same configured provider/model for all arms. It records one date's provider behavior and is not a provider-independent measurement.
- The 98% figure remains a goal. Do not tune against or publish hidden-set accuracy; the hidden dataset contains no answer key for this project.

Reproduce with:

```bash
uv run python -m src.evaluate \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --pipeline all \
  --provider cloudflare \
  --output results/public_clean_20260929.jsonl
```
