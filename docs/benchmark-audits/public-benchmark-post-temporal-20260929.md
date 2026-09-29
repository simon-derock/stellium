# Public three-pipeline benchmark after temporal fixes — 2026-09-29

## Run configuration and integrity

- Dataset: `hackathon-resources/questions/eval_public.jsonl` (100 questions)
- Dataset SHA-256: `abddb7d18a6d8ed908f514a7e560fe4950ebe479ebb2cdb75a7456887c10c6e5`
- Output SHA-256: `9d9542e017aa9409147d3915e10d49fba9da659e52560f33a65e000437409b63`
- All pipelines used Cloudflare Workers AI `@cf/meta/llama-3.1-8b-instruct-fast`
- The run followed the deployed TigerGraph response parsing and temporal query corrections (`PRECEDES` direction and case-insensitive gender filter).
- Previous prompt exemplars copied from `pub-001` and `pub-002` were removed; the prompt regression test checks for public question text.
- Raw 100-row output: `results/public_post_temporal_20260929.jsonl`

Reproduce with:

```bash
uv run python -m src.evaluate \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --pipeline all \
  --provider cloudflare \
  --output results/public_post_temporal_20260929.jsonl
```

## Overall metrics

| Pipeline | Exact match | Token F1 | Mean latency | Mean LLM tokens | Mean document MRR |
|---|---:|---:|---:|---:|---:|
| RAG | 42/100 (42%) | 0.438 | 1,120.9 ms | 2,423 | 0.588 |
| GraphRAG | 32/100 (32%) | 0.367 | 1,567.0 ms | 2,079 | invalid† |
| Agentic GraphRAG | 84/100 (84%) | 0.859 | 2,219.1 ms | 3,830 | invalid† |

† A later audit found that the GraphRAG and Agentic pipelines converted retrieved document IDs through unordered sets before evaluation. The stored MRR, recall@5, and precision@5 for those pipelines do not reliably represent evidence order and must not be used as ranking results. RAG preserves its HNSW order and its MRR remains valid. A first-seen-order fix is implemented; new ranking metrics require a fresh benchmark. This issue does not affect answer, token, or latency metrics.

Agentic category results:

| Type | Exact match | Token F1 | Mean tokens |
|---|---:|---:|---:|
| Aggregation | 21/21 (100%) | 1.000 | 4,262 |
| Temporal | 21/22 (95.5%) | 0.955 | 3,126 |
| Superlative | 7/10 (70%) | 0.784 | 3,477 |
| Multi-hop | 19/28 (67.9%) | 0.718 | 4,186 |
| Lookup | 16/19 (84.2%) | 0.842 | 3,830 |

## What this says about retrieval and the Jina model

The Jina model and TigerGraph vector index were unchanged. The Agentic score moved from 60/100 to 84/100 after temporal GSQL corrections, while targeted temporal EM moved from 2/22 to 20/22 after fixing directed-edge traversal and case-sensitive gender matching. This is evidence that those graph-query defects were a major source of the temporal failures; it does not prove dense retrieval is optimal or that embeddings never contribute to errors.

Dense retrieval's 95% gold-document hit@30 and BM25's 88% gold-document hit@5 / 96% hit@30 are document-coverage metrics. They do not prove answer-bearing chunk coverage, complete evidence, or correct final answers.

## Pipeline scope and observed tool selection

- RAG uses TigerVector HNSW top-5 only. It does not call GSQL, BM25, RRF, or a reranker.
- The measured fixed GraphRAG run used an LLM entity-extraction step, fixed GSQL lookups/multi-hop when its extracted fields permitted, dense top-3 retrieval, then one synthesis call. The code was subsequently changed to use hybrid passage retrieval by default; this historical benchmark does not measure that change.
- Agentic exposes GSQL aggregation/temporal/superlative/multi-hop/lookup, dense vector search, and hybrid search. The orchestrator makes tool selection dynamically within a four-iteration bound.
- In this 100-question Agentic run, `hybrid_search` was selected for **0/100** questions; `vector_search` was selected for 4/100. Tool-call counts were GSQL aggregate 30, temporal 31, superlative 7, multi-hop 40, and lookup 40 (a question can call multiple tools).
- Consequently, this run's 84% Agentic result does **not** establish a contribution from BM25, RRF, or the cross-encoder. Their separate retrieval audit measures BM25 candidate document coverage only. An answer-level hybrid ablation remains missing.
- The trace records only `ReActOrchestrator` as an agent. Retrieval/reasoning capabilities are tools, not separate specialized agent instances. A dedicated evidence-evaluation agent and explicit answer-bearing evidence validation are not implemented.

## Retrieval default change after this measurement

After the run above, GraphRAG's supporting-passage path was changed to retrieve up to 30 HNSW candidates, fuse them with BM25Plus via RRF, and rerank up to five passages. Agentic's prompt now prefers `hybrid_search` for document evidence and reserves dense-only search as a deliberate fallback; exact GSQL answers can still stop without unnecessary passage retrieval. RAG remains dense-only as the control. No post-change accuracy or latency claim is available yet.

## Evaluation limits and next work

- EM is strict equality against the dataset's short answer after repository normalization; token F1 is lexical overlap. Neither is a reviewed semantic-completeness/grounding score. Multi-athlete gold strings can differ from readable comma-separated answer formatting, and EM does not measure citation correctness.
- The stored GraphRAG and Agentic rank metrics are invalidated by unordered document-ID deduplication discovered after this run; only their answer/token/latency figures and RAG's ordered retrieval metrics remain usable. Rerun after the stable-order correction.
- Mean `context_tokens` is an estimate from text length, not provider-tokenizer measurement. The evaluator captures overall and per-call trace timing/token fields, but this report does not claim p50/p95/p99 distributions or complete per-operation accounting.
- This is one live run. Agentic is substantially better than the other two pipelines here and the temporal fix explains much of the gain, but answer accuracy remains below the 98% goal. The most important remaining Agentic categories are multi-hop (9 misses) and superlative (3 misses); lookup has 3 misses.
- Inspect the remaining errors and citation/evidence sufficiency, fix general entity resolution and tool-selection/answer-type errors, run a public answer-level BM25/RRF ablation, then regenerate all three full public results. Regenerate the 50-row hidden submission only after public tuning is frozen; do not tune on hidden answers.

## Trace-level audit of the 16 Agentic exact-match misses

This is a first-pass classification from stored questions, outputs, and tool traces, not a causal attribution. Several misses have overlapping causes.

| Failure pattern | Questions | Evidence in the trace |
|---|---|---|
| Multi-answer string representation differs from the gold encoding | `pub-015`, `pub-099` | Tool evidence contains the gold medal-winning teams; the generated readable comma-separated names do not equal the dataset's concatenated gold string. This is a metric/data representation mismatch, not a retrieval miss. |
| Event/entity resolution or query constraint mismatch | `pub-022`, `pub-028`, `pub-035`, `pub-054`, `pub-059`, `pub-064`, `pub-073`, `pub-074`, `pub-095` | Empty graph matches, malformed or over-specific date/event fragments, category labels passed in question form, or multiple venue/date candidates left unresolved. These are graph query construction and entity linking problems; some may also involve source-field quality. |
| Correct tool evidence not converted into the requested answer | `pub-053`, `pub-060`, `pub-084`, `pub-088` | The trace shows wrong tool selection or output-type mismatch for superlatives, and at least one malformed/repetitive synthesis. In `pub-088`, the result is the right event discriminator but omits the full event title required by exact match. |
| Source/ingestion representation issue | `pub-067` | The graph tool returns `Rosannagh Mac, Lennan`, while the answer key has `Rosannagh MacLennan`; this needs an ingestion/source audit before classifying it as a model error. |

The lookup and temporal GSQL filters compared gender values without normalizing the common possessive form from questions (for example, `Women's`) to the stored category (`Women`). This was a concrete code defect. A boundary normalizer and regression coverage now handle straight and curly apostrophes, and the lookup GSQL comparison is case-insensitive. A read-only live query check returned the relevant year/sport candidates when gender was omitted, but no candidates with the normalized gender argument. The source fix has not yet been installed into the live compiled GSQL query or measured in a fresh benchmark; do not count any expected recoveries until deployment and evaluation are complete.

The audit points to query grounding and answer construction as the next engineering priorities. It does not exonerate embeddings: the current document-level retrieval audit cannot tell whether the answer-bearing chunk reached context. Conversely, this Agentic run used the same Jina vectors and achieved 84% while selecting hybrid search zero times, so replacing Jina alone is not supported as the first fix.
