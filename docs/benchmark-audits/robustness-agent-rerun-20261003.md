# Robustness sets — agent re-run on the final agent code (2026-10-03)

## Why

The four robustness sets were last run on agent commits `b0a5a25` (paraphrase), `d267836` (two-step) and `59cbd67` (unanswerable, off-template). The agent changed after those runs, ending at `f75fb5c`, so the agent was run again on every set. RAG and GraphRAG rows are unchanged; their commits are listed in each document's provenance.

## Configuration

- Model: Cohere `command-a-03-2025`, temperature 0, LLM response cache off. Embeddings: Cohere `embed-v4.0`, 1024 dimensions.
- Paraphrase on `1ad2759`; two-step, unanswerable and off-template on `ba2db35`. Neither commit changes agent code after `f75fb5c`. `ba2db35` adds a single retry for Cohere's occasional `422 unknown` reply and per-key usage counters; neither changes an answer.
- The first two-step attempt on `1ad2759` stopped at `comp-007` on such a 422. The same question answered correctly when run alone, which is what the retry in `ba2db35` handles.
- Raw rows and manifests (not committed): `results/{set}_agentic_{commit}.jsonl`.

## Agent results

| Set | n | Before | After | Tokens / question, before → after | p50 latency, before → after |
|---|---:|---:|---:|---:|---:|
| Paraphrase | 12 | 12/12 | 12/12 | 1,197 → 983 | 4.4 s → 2.3 s |
| Two-step | 24 | 24/24 | 24/24 | 1,693 → 1,737 | 6.6 s → 3.7 s |
| Unanswerable | 12 | 12/12 declined | 12/12 declined | 7,277 → 5,100 | 14.0 s → 12.1 s |
| Off-template | 12 | 10/12 | 10/12 | 5,103 → 4,833 | 8.8 s → 8.5 s |

Accuracy is unchanged on every set. Latency includes client-side request pacing and varied with provider load during each run.
