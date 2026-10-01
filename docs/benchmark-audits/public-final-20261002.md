# Public benchmark — final configuration (2026-10-02)

## Configuration

- Dataset: `hackathon-resources/questions/eval_public.jsonl` (100 questions), SHA-256 `abddb7d1…c6e5`.
- Corpus: `hackathon-resources/corpus/corpus.jsonl`, SHA-256 `27aef30b…191a`.
- Graph: TigerGraph Savanna 4.2.5, `OlympicsGraph`, reconciled with the parser on 2026-10-01 (zero attribute differences).
- LLM for every call in every pipeline: Cohere `command-a-03-2025`, temperature 0. Embeddings: Cohere `embed-v4.0`, 1024 dimensions.
- LLM response cache disabled: every answer is a live model call. Query embeddings are cached (deterministic vectors).
- Raw rows and manifests (not committed): `results/public_final.jsonl` (all three pipelines, commit `4923d3a`) and `results/public_final_agentic.jsonl` (agent only, commit `cc74fd5`).

The agent was rerun after one fix (`cc74fd5`: reject tool arguments a tool cannot use). `git diff 4923d3a cc74fd5` touches only `src/pipelines/agentic.py` and adds a helper to `toolkit.py` that only the agent calls, so the RAG and GraphRAG rows are the output of the same code. The agent's matched-run result on `4923d3a` is reported below for transparency.

## Results

| Pipeline | Exact match | Strict EM | Token F1 | LLM tokens / question | Latency / question |
|---|---:|---:|---:|---:|---:|
| RAG | **67/100** | 66% | 0.675 | 2,581 | 4.80 s |
| GraphRAG | **98/100** | 97% | 0.983 | 1,094 | 6.21 s |
| Agentic GraphRAG | **98/100** | 97% | 0.983 | 990 | 3.88 s (agent-only run) |
| Agentic GraphRAG, matched run on `4923d3a` | 97/100 | 96% | 0.973 | 1,019 | 5.24 s |

| Question type | n | RAG | GraphRAG | Agentic GraphRAG |
|---|---:|---:|---:|---:|
| Aggregation (count) | 21 | 2/21 · 2,641 tok | 21/21 · 1,450 tok | 21/21 · 812 tok |
| Lookup | 19 | 19/19 · 2,489 tok | 19/19 · 970 tok | 19/19 · 1,037 tok |
| Multi-hop (venue + date) | 28 | 26/28 · 2,509 tok | 27/28 · 962 tok | 27/28 · 861 tok |
| Superlative (ranking) | 10 | 3/10 · 2,769 tok | 9/10 · 1,031 tok | 9/10 · 911 tok |
| Temporal (previous Games) | 22 | 17/22 · 2,608 tok | 22/22 · 1,055 tok | 22/22 · 1,319 tok |

Exact match compares answer content (case, accents, punctuation, and separators ignored); strict EM is the earlier punctuation-sensitive score. Token counts are LLM input + output only; graph queries and retrieval count 0. Latency is wall-clock per question and includes client-side request pacing, which penalises pipelines in proportion to their LLM calls; the agent-only run paces fewer calls per question than the matched three-pipeline run.

### Evidence and cost

| Pipeline | Gold among candidates | Grounded in cited articles | Abstained | Citations / answer | Latency p50 / p95 | Cost / 100 questions |
|---|---:|---:|---:|---:|---:|---:|
| RAG | 67% | 94% (51/54) | 7 | 5.0 | 4.4 s / 6.4 s | $0.65 |
| GraphRAG | 100% | 100% (60/60) | 0 | 2.1 | 6.1 s / 7.5 s | $0.35 |
| Agentic GraphRAG (`cc74fd5`) | 100% | 100% (60/60) | 0 | 2.1 | 3.4 s / 7.3 s | $0.32 |

Grounding covers non-numeric answers (counts are computed by the graph, not stated in one article) and asks whether every claimed value appears in an article that pipeline cited, title included, spacing ignored. "Gold among candidates" credits a tie report that contains the gold answer. Cost uses Cohere Command A list prices ($2.50 / $10.00 per million input / output tokens) and is an estimate, not a bill.

## Agent behaviour (commit `cc74fd5`)

- LLM calls per question: mean 1.16; 86/100 answered with one planning call. Steps (LLM + tool calls): mean 2.30, max 6.
- Strategy changes: 10/100, each recorded with the failed or ambiguous step that triggered it.
- Tools: event_at_venue_date 28, event_attribute 24, count_events 21, previous_edition 19, find_events 12, rank_events 10. No passage search or generated GSQL was needed on this set.
- Stopping: 98 runs stopped on one verified graph value (most also matched a source-article line); 2 finished through the orchestrator with grounded candidates.

## Answers that differ from gold

| Question | Pipelines | What the evidence shows |
|---|---|---|
| pub-044 | GraphRAG, Agentic | Men's épée and women's foil both have 41 competitors; both are returned. Gold names one. |
| pub-099 | GraphRAG, Agentic | Two events share venue "Laura Biathlon & Ski Complex" and date "22 February 2014"; both winners are returned. Gold names one. |
| 33 RAG misses | RAG | 26 of 31 count/ranking questions need 10–40 events and five passages cover about a third; 5 temporal questions retrieve the named year instead of the previous Games; 2 multi-hop picks of near-identical events. |

Both graph-pipeline misses return the gold answer among the reported candidates. Choosing one would be a guess; the pipelines report the tie instead.

## Findings behind the final configuration

- Generated GSQL (earlier design, same model) scored 76%; typed graph tools score 98%.
- `get_superlative_event`'s ACCUM lists lose ORDER BY order; rankings read stored counts for the linked pool.
- 26 events had a missing or wrong Games year from the infobox; titles are now authoritative and the live graph was corrected.
- Entity-linking fixes found in this run's traces: canonical titles passed as phrases (pub-006), exact venue wording outranking a containing name (pub-028), gender counted toward the label (pub-055), and rejecting arguments a tool cannot use (pub-035).

## Reproduce

```bash
uv run python -m src.evaluate --provider cohere --pipeline all \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --output results/public_final.jsonl --resume
uv run python scripts/summarize_results.py results/public_final.jsonl
```
