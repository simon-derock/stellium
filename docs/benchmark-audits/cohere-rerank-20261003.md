# Cohere Rerank replaces the local cross-encoder (2026-10-03)

## Change

Hybrid retrieval (TigerGraph vector search + BM25, fused by RRF) was reranked by a local int8 MiniLM cross-encoder. From `2fca5c8` it is reranked by Cohere `rerank-v3.5` in one API call per search. On the deployed 0.1-CPU instance the local model needed about 44 s per question; the API call takes about 0.6 s. If reranking is unavailable the fused order is kept and the trace records `reranker_executed: false`.

`fc6bfd6` also lets the agent ground a year that the question asks for in a passage that states it whole ("In which year was Hostel released?"). Counts remain graph-only.

## What was re-run

Only rows whose code path runs the reranker were re-measured; every other row is unchanged by construction.

| Set | Re-run | Not re-run, and why |
|---|---|---|
| Public 100 | RAG on `2fca5c8` | Agent and GraphRAG reranked 0 of 100 questions |
| Hidden 50 | RAG on `2fca5c8` | Agent and GraphRAG reranked 0 of 50 |
| Paraphrase 12 | RAG on `2fca5c8` | Agent and GraphRAG reranked 0 of 12 |
| Two-step 24 | RAG and GraphRAG on `2fca5c8` | Agent reranked 0 of 24 |
| Unanswerable 12 | RAG and GraphRAG on `2fca5c8`, agent on `fc6bfd6` | |
| Off-template 12 | RAG and GraphRAG on `2fca5c8`, agent on `fc6bfd6` | |

Model `command-a-03-2025`, temperature 0, response cache off. Raw rows and manifests (not committed): `results/*_2fca5c8.jsonl`, `results/*_agentic_fc6bfd6.jsonl`.

## Results

| Set | Pipeline | Before | After |
|---|---|---:|---:|
| Public 100 | RAG | 67 | 71 |
| Public 100 | RAG context recall | 83% | 97% |
| Public 100 | RAG grounded | 51/54 | 56/56 |
| Public 100 | RAG latency p50 | 4.4 s | 1.8 s |
| Hidden 50 vs oracle | RAG | 19/49 | 28/49 |
| Two-step 24 | RAG / GraphRAG | 15 / 19 | 18 / 19 |
| Unanswerable 12 | RAG / GraphRAG / agent | 12 / 12 / 12 | 11 / 12 / 12 |
| Off-template 12 | RAG / GraphRAG / agent | 8 / 7 / 10 | 12 / 11 / 12 |

RAG's one unanswerable miss declines correctly ("Not found in corpus.") but adds prose after it, which exact match counts as wrong. The graph pipelines stay at 99/100 public and 49/49 hidden.
