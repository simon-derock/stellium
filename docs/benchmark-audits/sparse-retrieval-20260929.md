# Sparse Retrieval Audit — 2026-09-29

## Result

The public 100-question set was evaluated against the checked-in 22,016-chunk corpus with BM25Plus. The production tokenizer is punctuation-normalized Unicode/ASCII tokenization from `src.coprocessor._tokenize`; the legacy whitespace tokenizer is included as a comparison. No LLM, live graph, embeddings, cross-encoder, or answer generation was used.

| Tokenization | Gold-document hit@5 | Gold-document hit@10 | Gold-document hit@30 | Mean gold-document recall@5 | Mean gold-document recall@30 | Document MRR@30 |
|---|---:|---:|---:|---:|---:|---:|
| Legacy whitespace | 81% | 90% | 95% | 55.9% | 83.3% | 0.5733 |
| Punctuation-normalized (production) | 88% | 95% | 96% | 62.8% | 84.8% | 0.6824 |

Metrics are computed after deduplicating document IDs from the ranked chunk candidates. Hit@k measures whether at least one gold document appears in the first k unique documents. Recall@k measures the fraction of all gold documents found, averaged over questions. MRR uses the first gold document rank. The candidate pool is the top 30 BM25-ranked chunks.

## Findings

- The production tokenizer improves hit@5 by 7 percentage points and document MRR@30 by about 0.109 over the whitespace baseline on this dataset.
- Hit rate and recall are materially different: many aggregation and superlative questions have multiple gold documents. A retriever can find one useful document while missing much of the evidence required to calculate the answer.
- For superlative questions, production BM25 hit@30 is 80%, while mean gold-document recall@30 is 45.9%. The 10 superlative questions list 8–43 gold documents each. They require structured event candidate retrieval and aggregation through the graph/GSQL path; lexical ranking alone does not establish the maximum.
- Production BM25's misses at the top-30 unique-document cutoff are `pub-021`, `pub-051`, `pub-057`, and `pub-088`. These include two multi-source superlatives, an aggregation, and a temporal question. Full per-question ranks and gold IDs are in the generated JSON report.

## Scope and limitations

Gold-document IDs are weak retrieval supervision: a hit does not prove that the retrieved chunk contains the answer-bearing passage. The report measures candidate document coverage only. It does not establish chunk-level evidence recall, answer EM/F1, grounding, citation correctness, latency, or live TigerGraph performance. It must not be presented as the system's answer accuracy.

The evaluation is reproducible with:

```bash
uv run python -m scripts.evaluate_sparse_retrieval \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --corpus hackathon-resources/corpus/corpus.jsonl \
  --output results/sparse_retrieval_20260929.json
```

Dataset SHA-256: `abddb7d18a6d8ed908f514a7e560fe4950ebe479ebb2cdb75a7456887c10c6e5`  
Corpus SHA-256: `27aef30bbe32474df782f5164262d5d765af14efc9f0288320486cee585e191a`
