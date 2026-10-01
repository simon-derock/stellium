# Public benchmark: mandatory hybrid and generated GSQL — 2026-09-30

## Run configuration

- Dataset: `hackathon-resources/questions/eval_public.jsonl`, 100 questions.
- Chat: Cohere `command-a-03-2025`.
- Embeddings: Cohere `embed-v4.0`, 1024 dimensions, queried against the live TigerGraph HNSW index.
- Graph: TigerGraph Savanna 4.2.5, `OlympicsGraph`.
- Retrieval: every phase performs dense HNSW + BM25Plus + RRF + local int8 MiniLM cross-encoder reranking. GraphRAG also asks the LLM to generate guarded read-only GSQL. Agentic performs mandatory initial hybrid retrieval, then selects GSQL and/or more retrieval steps.
- Raw results: `/tmp/stellium_public_final_20260930.jsonl` (local artifact; do not commit result JSONL).

Run the same public benchmark with credentials configured in the environment:

```bash
uv run python -m src.evaluate \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --pipeline all --provider cohere \
  --output /tmp/stellium_public_final_20260930.jsonl
```

## Results

| Pipeline | Exact match | Token F1 | Doc MRR | Recall@5 | Precision@5 | Mean LLM tokens | Mean context tokens | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAG | 60/100 (60%) | 0.6278 | 0.8670 | 0.7362 | 0.4293 | 2,580.2 | 1,927.3 | 5,396 ms |
| GraphRAG | 60/100 (60%) | 0.6369 | 0.8670 | 0.7362 | 0.4293 | 3,659.6 | 2,032.3 | 10,379 ms |
| Agentic GraphRAG | 76/100 (76%) | 0.7970 | 0.8690 | 0.7462 | 0.4313 | 8,291.8 | 8,073.1 | 11,483 ms |

Exact match uses normalized string equality against the provided gold answer. Token F1 measures lexical overlap and is not a semantic judge. MRR/recall/precision measure retrieved gold documents, not answer correctness or entailment. Context token counts are estimates; LLM token counts use provider usage metadata.

### Exact match by question type

| Type | Questions | RAG | GraphRAG | Agentic |
|---|---:|---:|---:|---:|
| Aggregation | 21 | 1/21 (4.8%) | 1/21 (4.8%) | 12/21 (57.1%) |
| Temporal | 22 | 18/22 (81.8%) | 18/22 (81.8%) | 21/22 (95.5%) |
| Superlative | 10 | 0/10 | 0/10 | 0/10 |
| Multi-hop | 28 | 22/28 (78.6%) | 22/28 (78.6%) | 24/28 (85.7%) |
| Lookup | 19 | 19/19 | 19/19 | 19/19 |

## Trace and failure findings

- Agentic recorded 172 LLM calls and 348 trace steps across 100 questions. It invoked `hybrid_search` 105 times (100 required initial calls plus 5 additional calls) and `gsql_query` 65 times.
- Generated GSQL is still unreliable: run logs include rejections for unsupported vertex/edge types, attributes, functions, and query structure, plus TigerGraph runtime failures. The tool guards reject unsafe or unsupported queries; rejected calls lower answer coverage and add latency/tokens.
- Superlative exact match is 0/10 in all phases. Several Agentic outputs identify the right event suffix but omit the full canonical event title required by the gold answer; other questions show genuinely wrong or missing results. This is both an answer-format issue and a retrieval/query correctness issue, not merely a scoring artifact.
- Aggregation is the largest remaining Agentic weakness at 12/21. Fixed GraphRAG matches RAG at 60% overall and takes about twice the mean latency of RAG in this run.
- Agentic gains 16 percentage points over RAG/GraphRAG, at about 3.21× their mean LLM token use and about 2.13× RAG's mean latency.

## Follow-up diagnosis after the measured run

- Superlative instructions were changed to require the complete canonical event title. GraphRAG and Agentic prompts gained concrete schema-valid examples for `HELD_AT` venue/date traversal and `PRECEDES` temporal traversal, and now require double-quoted GSQL string literals. Live read-only checks accepted the examples.
- A focused 10-question superlative run after those changes measured RAG 3/10 EM, GraphRAG 4/10, and Agentic 2/10. Agentic made nine GSQL calls and all nine completed without tool errors, but still often returned a competitor count instead of the event title. This shows that syntax correction alone did not solve answer selection and grounding. The focused sample is diagnostic and is not comparable to or a replacement for the 100-question baseline above.
- A corpus audit found 119 documents with both an infobox competitor total and an explicit opening-paragraph total; four disagree. One is a clear zero-appending corruption: Q2570052 has `competitors: 41000000` while its prose says 41. The parser now corrects this case only when the prose confirms the exact value and the infobox value is that value followed by at least three zeroes. The other three ordinary disagreements are preserved as infobox values.
- The corrected parse was applied to the matching TigerGraph Event vertex as a single-attribute upsert and verified with a live query: Q2570052 now has competitor_count 41. This correction is not represented in the 100-question results above.
- The shared GSQL executor now converts only complete, unambiguous single-quoted literals to double-quoted strings before allowlist validation. A live TigerGraph smoke query using `'fencing'` was normalized and returned Event rows. The full quality gate after this change passed: 214 tests, 3 skipped, Ruff, formatting, and mypy.
- Gemini smoke test returned HTTP 503 with a provider high-demand message. Mistral returned HTTP 429 rate-limited. Neither produced benchmark results; Cohere remains the only successfully scored provider in this run.
- After the live graph correction, a new focused 10-question run using Cohere stopped before writing any result rows: the provider returned HTTP 429 after bounded retries. The corrected graph value is verified, but there is no post-correction answer-quality score yet.
- A 10-question Agentic-only diagnostic using Cloudflare on the corrected graph scored 4/10 EM and 0.467 Token F1 (mean 20,049 LLM tokens and 10.36 s per question). It made 27 generated-GSQL calls, 13 of which returned tool errors; ten successful calls returned empty result sets. Thirteen queries compared `e.sport` directly against lowercase literals, while corpus values use title case. This is a different provider and a different run configuration, not comparable to the Cohere baseline.
- GSQL instructions were then made consistent about accumulators and case-insensitive string comparisons, and the executor began exposing structured canonical Event rows to the agent. A 3-question Cloudflare probe after those changes scored 1/3 EM: one answer was the right title suffix but not the canonical full title; another answer was a number despite the successful graph result containing the correct event name. The two probes are too small and use different prompts from the 10-question run; neither is a public benchmark score. The latest full 100-question result is still the Cohere baseline above.
- The Cohere client's credential handling and request pacing were reworked; traces record no credential values.
- Afterwards, the same 10-question superlative subset was rerun across all three pipelines with the current prompts, structured Event observations, corrected Q2570052 graph value, and Cohere `command-a-03-2025` / `embed-v4.0`:

  | Pipeline | Exact match | Token F1 | Mean LLM tokens | Mean latency |
  |---|---:|---:|---:|---:|
  | RAG | 3/10 (30%) | 0.603 | 2,815.9 | 9,005 ms |
  | GraphRAG | 8/10 (80%) | 0.918 | 4,522.6 | 13,424 ms |
  | Agentic GraphRAG | 8/10 (80%) | 0.857 | 10,537.4 | 16,251 ms |

  The retrieval metrics were identical across phases in this subset (MRR 0.775, recall@5 0.354, precision@5 0.780). Agentic made 19 successful Cohere calls and nine generated-GSQL calls; eight GSQL observations exposed structured Event candidates and one tool call errored. Its exact match tied GraphRAG while using 2.33× the mean LLM tokens and taking 1.21× the mean latency. This small, changed-code diagnostic is not a full public score or a controlled component ablation.

## Interpretation and limits

The run confirms that the three pipelines execute with mandatory hybrid retrieval and that Agentic currently leads on exact match. It does not show that the new GSQL generation or local reranker caused the accuracy change: there is no controlled ablation isolating those components. The 76% Agentic EM is also below the previous 90% run, so this implementation is not yet a metrics win. A full 100-question rerun against the latest prompts and corrected graph data, aggregation diagnosis, answer-level hybrid ablation, and the hidden 50-question submission remain outstanding.

## Provider smoke follow-up — 2026-10-01

- A current-code 100-question Cohere rerun wrote 16 complete rows, then stopped after five retries returned HTTP 429. The partial rows score RAG 5/16 EM, GraphRAG 4/16, and Agentic 14/16, but this is a biased prefix of the dataset and is not a valid benchmark result. It must not replace the 100-question baseline.
- A three-question Agentic smoke using Gemini completed only `pub-001` (1/1 EM, 32.2 s, 15,637 reported LLM tokens) before later calls ended with HTTP 503 after five retries. One question is not a provider accuracy comparison.
- The same three-question Agentic smoke using Mistral ended on HTTP 429 before writing any rows. No Mistral answer or score is available.
- These runs confirm provider availability/rate-limit failures, not that one provider is more or less intelligent. A fair model comparison still needs the same completed question set for each provider under a stable quota window. All raw JSONL artifacts remain under `/tmp` and are intentionally excluded from the repository.
- The Agentic and GraphRAG GSQL prompts were reviewed against the actual guardrail validator. Their eight executable GSQL examples pass validation; the GraphRAG `...` form is a format template, not an executable example. Both prompts now state the validator's allowed function/syntax subset explicitly.
- A one-question Agentic-only smoke with Mistral (`mistral-medium-latest`) returned HTTP 429 and wrote no result row. No accuracy score is available from this attempt.
- A one-question Agentic run on Cloudflare (`pub-001`, aggregation) completed successfully: EM/F1 1.0, two LLM calls, 9,507 reported LLM tokens, and 9.20 s end-to-end. The model called mandatory `hybrid_search`, then generated a bounded `SumAccum` GSQL query; the query passed validation and TigerGraph returned `@@match_count = 5` in 182 ms. It then finished with answer `5`. This validates the end-to-end path for this one aggregation case only, not general GSQL reliability or public-set accuracy. Raw output is in `/tmp/stellium_agentic_cloudflare_20261001.jsonl`.
