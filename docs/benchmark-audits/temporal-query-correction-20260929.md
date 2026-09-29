# Temporal query correction — 2026-09-29

## Root cause

The embedding model was not the primary cause of the low temporal score. The temporal GSQL path had two query-contract mismatches:

1. Ingestion writes `PRECEDES` from a current event to its prior edition, but `get_preceding_event` used an undirected traversal. The query could walk either end of the edge.
2. Event `gender` values are stored in title case (for example, `Men`); the ReAct model commonly supplied lowercase values (for example, `men`). The temporal query compared those values case-sensitively and returned no current event.

The query now follows the stored directed edge and compares gender case-insensitively. Both lowercase and title-case live calls for the 2016 men's 20 km walk return the 2012 event and Chen Ding. The model/dimension and the Jina vector index were not changed.

## Public temporal subset results

All runs used the same 22 temporal questions and the same configured Cloudflare model. Each row below is a single run; model variability means this is directional evidence, not a controlled multi-seed experiment.

| Version / run | Exact match | Token F1 | Mean tokens | Mean latency |
|---|---:|---:|---:|---:|
| Original full-run subset | 2/22 (9.1%) | 0.129 | 8,732 | 5,826 ms |
| Fallback synthesis grounding | 6/22 (27.3%) | 0.273 | 9,410 | 4,763 ms |
| Directed `PRECEDES` traversal | 9/22 (40.9%) | 0.409 | 10,589 | 4,636 ms |
| Directed traversal + case-insensitive gender | **20/22 (90.9%)** | **0.909** | **3,202** | 7,321 ms |

The last run contains one 121.7-second question outlier; therefore its mean latency is not a steady-state latency estimate. Its raw 22-row output is in `results/temporal_agentic_genderfix_20260929.jsonl`; it contains no TigerGraph authentication failures. The single-question live checks separately confirmed the corrected GSQL result.

## Remaining misses and limits

The two misses are `pub-072` and `pub-090`. Their ReAct calls passed non-contiguous/reordered fragments (`500 metres speed skating`; `96 kg Greco-Roman`) to a substring matcher, while corpus titles use `Speed skating … Women's 500 metres` and `Greco-Roman 96 kg`. The event phrase matcher therefore failed to identify the current event. This is an entity-linking/retrieval mismatch, not evidence that the Jina vector space is wrong.

The 20/22 result is a targeted public subset result; it does not replace the 100-question three-pipeline baseline. Rerun all 100 public questions through all three pipelines after the final retrieval/tool changes. Do not tune on hidden questions.
