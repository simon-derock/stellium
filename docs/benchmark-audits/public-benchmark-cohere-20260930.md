# Matched public three-pipeline benchmark — 2026-09-30

## Configuration and reproducibility

- Dataset: `hackathon-resources/questions/eval_public.jsonl` (100 questions)
- Dataset SHA-256: `abddb7d18a6d8ed908f514a7e560fe4950ebe479ebb2cdb75a7456887c10c6e5`
- Chat provider/model: Cohere `command-a-03-2025`
- Passage and query embeddings: Cohere `embed-v4.0`, 1024 dimensions, live TigerGraph HNSW index
- Cluster: TigerGraph Savanna 4.2.5, `OlympicsGraph`
- Pipeline outputs: `results/rag_cohere_20260930.jsonl`, `results/graphrag_cohere_validated_20260930.jsonl`, `results/agentic_cohere_20260930.jsonl`

Run each pipeline with:

```bash
uv run python -m src.evaluate \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --pipeline rag --provider cohere \
  --output results/rag_cohere_20260930.jsonl

uv run python -m src.evaluate \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --pipeline graphrag --provider cohere \
  --output results/graphrag_cohere_validated_20260930.jsonl

uv run python -m src.evaluate \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --pipeline agentic --provider cohere \
  --output results/agentic_cohere_20260930.jsonl
```

The Cohere chat adapter spaces request starts by 3.25 seconds to stay below the trial request limit. Latencies below include this pacing. All outputs completed 100/100 rows.

| Pipeline | EM | Token F1 | Mean document MRR | Recall@5 | Precision@5 | Mean LLM tokens | Mean context tokens | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAG | 43/100 (43%) | 0.4608 | 0.7795 | 0.5814 | 0.4983 | 2,913.7 | 1,994.6 | 3,294.4 ms |
| GraphRAG | 53/100 (53%) | 0.5838 | 0.7709 | 0.6623 | 0.4332 | 3,008.9 | 2,034.8 | 6,523.1 ms |
| Agentic GraphRAG | 90/100 (90%) | 0.9242 | 0.9608 | 0.6503 | 0.9205 | 3,092.9 | 2,931.1 | 4,281.3 ms |

EM is strict normalized answer equality; token F1 is lexical token overlap. MRR/recall/precision evaluate retrieved documents, not answer completeness or evidence entailment. Context tokens are estimates based on text length; total LLM tokens use Cohere usage metadata.

## Exact match by question type

| Type | RAG | GraphRAG | Agentic GraphRAG |
|---|---:|---:|---:|
| Aggregation | 1/21 (4.8%) | 2/21 (9.5%) | 20/21 (95.2%) |
| Temporal | 11/22 (50.0%) | 18/22 (81.8%) | 21/22 (95.5%) |
| Superlative | 0/10 (0%) | 0/10 (0%) | 10/10 (100%) |
| Multi-hop | 12/28 (42.9%) | 15/28 (53.6%) | 20/28 (71.4%) |
| Lookup | 19/19 (100%) | 18/19 (94.7%) | 19/19 (100%) |

## Feature use and cost

- RAG uses dense TigerVector HNSW top-5 and one answer-generation call per question: 100 successful LLM calls total. It does not use graph operations, BM25, RRF, or reranking.
- GraphRAG uses one LLM operation-extraction call and one answer-synthesis call per question: 200 successful LLM calls. It runs a typed compiled GSQL operation and calls the hybrid passage path for every question; that path performs BM25Plus retrieval and RRF fusion. The optional cross-encoder is configured, but this run does not record whether it loaded or the fallback ranking was used, so reranker impact is unverified.
- Agentic records 136 successful LLM calls (1.36 per question). It called `gsql_aggregate` 21 times, `gsql_temporal` 22 times, `gsql_superlative` 10 times, `gsql_multihop` 28 times, `gsql_lookup` 20 times, and `hybrid_search` 3 times. BM25/RRF and optional cross-encoder logic run only when `hybrid_search` is selected; they were not used on the other 97 questions.

## GraphRAG extraction defect found and fixed

An earlier Cohere run scored 47/100 EM and averaged 10,428 LLM tokens. Some venue/date questions were extracted as `lookup` plans without an event-name filter. The lookup GSQL query ignored venue/date fields and matched all 2,210 Event vertices, producing contexts of about 64k estimated tokens.

The extraction layer now validates operation/field compatibility: lookup plans lacking an event name are routed to venue/date multi-hop when those fields exist, otherwise graph execution is skipped. Graph-client guards also skip unconstrained lookups and reject broad multi-hop results. The corrected run scored 53/100 EM, with multi-hop rising from 9/28 to 15/28; GraphRAG mean LLM tokens fell to 3,009. This isolated the oversized-context defect, but the remaining 13 GraphRAG errors still need trace-level analysis. Aggregation and superlative accuracy remain weak at 2/21 and 0/10.

The fresh Agentic result is 90/100 EM, compared with the historical 84/100 Cloudflare run. This is not a controlled model-only comparison: provider/model, embeddings, query fixes, and agent code differ. Agentic's 10 misses are 8 multi-hop, 1 aggregation, and 1 temporal.

## Retrieval context

The separate Cohere dense retrieval audit over the same 100 questions reports hit@30 93%, mean gold-document recall@30 90.21%, MRR@30 0.7853, and 96.63 ms mean TigerVector request latency. Multi-hop dense document hit@30 is 75% (21/28); other question categories are at 100%. Retrieval coverage is not answer accuracy. See `results/dense_retrieval_cohere_20260930_report.json`.

## Current limits

- This is one benchmark run per pipeline; no confidence intervals or repeated-run variance are reported.
- The 98% target has not been reached.
- Agentic hybrid search is available but selected on only 3/100 questions; answer-level BM25/RRF ablation remains incomplete.
- Cross-encoder load/fallback status is not included in pipeline traces.
- A same-provider hidden 50-question Agentic submission has not yet been generated.

## Artifact checksums

```text
ab572a766adcdc0b82b2aee3baaef92daa40d10d77bfee48729590792fc03df0  results/rag_cohere_20260930.jsonl
10ad13a557033800d5de14a0fcc07c2a5e3211e997ea05fd23d1e29e239ccd9c  results/graphrag_cohere_validated_20260930.jsonl
96aa72dae58829233b7fc50867275202e7489855b5c8e480eaae9a86528a4ad9  results/agentic_cohere_20260930.jsonl
```
