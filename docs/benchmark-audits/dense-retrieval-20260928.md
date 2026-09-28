# Live Dense Retrieval Audit — 2026-09-28

## Finding

TigerVector returned zero chunk results for all 100 public questions at top-30. Hit rate, gold-document recall, and MRR were therefore zero at cutoffs 5, 10, and 30 in every question category. The queries used `jina-embeddings-v5-text-small`, 1024 dimensions, and the `retrieval.query` task adapter.

The live graph reports 22,016 Chunk vertices and has the HNSW vector attribute in its schema. A read-only interpreted query of one Chunk vertex with `PRINT ... WITH VECTOR` returned its ordinary text/metadata attributes but no `embedding` value. The raw installed vector query returned empty `TopChunks` and `@@distances` collections. The evidence indicates that vector values are missing from, or unavailable to, the live vector index. It does not measure semantic embedding quality because no candidates were returned to rank.

The old RAG Exact Match result (5/100) is consequently not a valid assessment of Jina retrieval quality. The old result rows also recorded empty RAG document IDs and zero retrieval recall, consistent with the current live failure. The old Agentic answer score is independently contaminated by two public question/answer examples in its prompt.

## Verification gap closed

The previous live verifier searched with an all-zero vector and marked the vector query successful whenever the call returned without raising, even when it returned zero chunks. The verifier now requires an aligned Jina query vector and fails when the search returns no candidates. Ingestion now checks TigerGraph's accepted-record count so a zero/partial upsert cannot be reported as a successful full batch. Live data was not modified during this audit.

## Reproduction

```bash
uv run python -m scripts.evaluate_dense_retrieval \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --output results/dense_retrieval_20260928.jsonl \
  --report results/dense_retrieval_20260928_report.json
```

The run used the dataset SHA-256 `abddb7d18a6d8ed908f514a7e560fe4950ebe479ebb2cdb75a7456887c10c6e5`. Per-question JSONL and the detailed report are local ignored artifacts; this file records the aggregate result and conclusions for the repository history.

## Next work

1. Audit the cached chunk vectors for model, task, corpus-content, and dimension provenance; invalidate vectors whose provenance cannot be established.
2. Generate or reuse vectors from the exact indexed Jina model and passage adapter, then verify that TigerGraph accepts every vector upsert.
3. Verify live vector availability with the nonzero-result health check and rerun dense Recall@k/MRR.
4. Compare dense-only, BM25-only, and RRF retrieval on the same public set before running answer generation.
