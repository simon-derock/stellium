# Dense retrieval audit — 2026-09-29

## Run configuration

- Dataset: `hackathon-resources/questions/eval_public.jsonl` (100 questions)
- Dataset SHA-256: `abddb7d18a6d8ed908f514a7e560fe4950ebe479ebb2cdb75a7456887c10c6e5`
- Passage and query model: `jina-embeddings-v5-text-small`, 1024 dimensions
- Passage task: `retrieval.passage`; query task: `retrieval.query`
- Retrieval: TigerVector HNSW only; top 30 chunks per query
- Run command: `uv run python scripts/evaluate_dense_retrieval.py --dataset hackathon-resources/questions/eval_public.jsonl --output results/dense_retrieval_20260929.jsonl --report results/dense_retrieval_20260929_report.json --top-k 30 --cutoffs 5 10 30`

## Result

| Metric | @5 | @10 | @30 |
| --- | ---: | ---: | ---: |
| Gold-document hit rate | 41% | 57% | 95% |
| Mean gold-document recall | 15.92% | 32.60% | 93.43% |
| Document MRR | 0.2735 | 0.2945 | 0.3166 |

All 100 queries returned 30 chunks. Mean vector-search latency was **95.86 ms**. Hit rate means at least one gold document was found; recall is the fraction of each question's gold documents found. These are retrieval measurements, not answer accuracy.

| Question type | Questions | Hit@30 | Recall@30 | MRR@30 |
| --- | ---: | ---: | ---: | ---: |
| Aggregation | 21 | 100% | 94.64% | 0.6246 |
| Temporal | 22 | 100% | 100% | 0.2711 |
| Superlative | 10 | 100% | 95.58% | 0.6869 |
| Multi-hop | 28 | 82.14% | 82.14% | 0.0695 |
| Lookup | 19 | 100% | 100% | 0.1981 |

## Correctness fix discovered during the audit

The live TigerGraph query returned each vertex ID in `v_id` and projected attributes in the nested `attributes` mapping, including `TopChunks.chunk_id`. The Python adapter expected a top-level `chunk_id`, silently converted IDs to empty strings, and consequently looked up missing distances as `1.0` (zero similarity). The adapter now accepts the actual TigerGraph response shape and preserves the prior flat shape for mocks and compatibility. The first audit run, which showed 0% retrieval, is invalid and superseded; the rerun above uses the fixed adapter.

## Artifacts

- Raw per-question output: `results/dense_retrieval_20260929.jsonl`
- Machine-readable report: `results/dense_retrieval_20260929_report.json`
