# Local reranker public retrieval audit — 2026-09-30

## Scope

This audit runs the compact BM25Plus index over the 22,016-chunk corpus and measures whether a local int8 MiniLM cross-encoder improves gold-document ranking for the 100 public questions. It is retrieval-only: it does not use TigerVector candidates, RRF, answer generation, answer EM, or completeness. Do not treat these numbers as a three-pipeline benchmark or as proof of answer accuracy.

Inputs are pinned by SHA-256:

- Corpus: `hackathon-resources/corpus/corpus.jsonl`, `27aef30bbe32474df782f5164262d5d765af14efc9f0288320486cee585e191a`
- Questions: `hackathon-resources/questions/eval_public.jsonl`, `abddb7d18a6d8ed908f514a7e560fe4950ebe479ebb2cdb75a7456887c10c6e5`
- 22,016 chunks; 100 questions; final metric cutoff `k=5`.

## Results

| BM25 candidate pool | Candidate-pool document hit rate | Candidate-pool document MRR | Reranked hit@5 | Reranked gold-doc recall@5 | Reranked MRR@5 | Mean / median / p95 rerank latency | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 95% | 0.682 | 94% | 0.706 | 0.842 | 501 / 495 / 625 ms | 367 MiB |
| 30 | 96% | 0.682 | 94% | 0.720 | 0.842 | 1,895 / 1,846 / 2,528 ms | 379 MiB |

Candidate-pool hit/MRR are measured over the candidate list deduplicated to documents. Reranked metrics are computed over the final five ranked chunks, deduplicated by document ID.

With 30 candidates, reranking raises MRR from 0.682 before reranking to 0.842 and top-five gold-document hit rate from 87% (BM25's first five chunks) to 94%. Ten candidates preserve the same rounded reranked hit rate and MRR while reducing mean reranker time by 73.6%; it loses a small amount of gold-document recall (0.720 to 0.706). This is a promising sparse-only latency/quality tradeoff, but it does not establish that limiting the production dense-plus-sparse fused candidate set to ten is safe. The shipped hybrid path remains at up to 60 fused candidates until a matched full-hybrid answer-level ablation is run.

The 30-candidate batch-size comparison also supports the new default:

| Batch size | Reranked hit@5 | Reranked MRR@5 | Mean rerank latency | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 94% | 0.842 | 1,895 ms | 379 MiB |
| 8 | 93% | 0.835 | 2,427 ms | 782 MiB |

Batch size 1 is the default. These are one-host measurements and will vary with CPU, ONNX Runtime, and deployment memory limits. The Linux process high-water mark includes model load and inference.

## Reproduce

```bash
uv run python scripts/evaluate_local_reranker.py --candidate-k 30 --output /tmp/stellium-reranker-k30.json
uv run python scripts/evaluate_local_reranker.py --candidate-k 10 --output /tmp/stellium-reranker-k10.json
RERANKER_BATCH_SIZE=8 uv run python scripts/evaluate_local_reranker.py --candidate-k 30 --output /tmp/stellium-reranker-batch8.json
```

The generated JSON reports are intentionally kept outside the repository. The audit script records the source hashes, category breakdown, timing distribution, and process memory profile.
