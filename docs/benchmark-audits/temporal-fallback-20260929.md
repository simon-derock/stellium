# Temporal fallback follow-up — 2026-09-29

## Question

The full public run had 2/22 Agentic temporal exact matches. Trace inspection showed that 20/22 questions reached the final synthesis fallback after the four-action ReAct budget. That fallback previously received document text only; GSQL observations remained in the conversation history and were omitted. It also lacked the strict answer-value-only grounding instructions used by the ReAct prompt.

Commit `479715c` updated fallback synthesis to include accumulated tool observations and enforce a grounded, value-only response contract. This follow-up reran the same 22 public temporal questions using Cloudflare Workers AI `@cf/meta/llama-3.1-8b-instruct-fast`.

## Observed results

| Run | Exact match | Token F1 | Mean tokens | Mean latency | Questions ending in synthesis |
|---|---:|---:|---:|---:|---:|
| Full public baseline subset | 2/22 (9.1%) | 0.129 | 8,732 | 5,826 ms | 20/22 |
| Post-fix temporal rerun | 6/22 (27.3%) | 0.273 | 9,410 | 4,763 ms | 18/22 |

This is a directional paired-subset comparison, not proof that the code change alone caused the difference: model outputs can vary, and it is one run per version. Mean token use increased by 7.8%; mean latency decreased by 18.3%. The full 100-question three-pipeline baseline remains the published overall score; this targeted run does not replace it.

The rerun answers and traces are recorded in `results/temporal_agentic_20260929.jsonl`. The public input questions are selected from `hackathon-resources/questions/eval_public.jsonl`; no hidden questions or answers were used.

## Remaining failure signal

The fallback now grounds its response in GSQL observations, but 16/22 exact matches still fail. Several return `Not found in corpus`; others contain an incorrect athlete. This shows that prompt-only fallback correction is insufficient: event/year selection and evidence quality in the temporal graph path need inspection. Do not convert this result into a general accuracy claim or tune against the hidden set.

Reproduce by selecting the 22 `qtype == "temporal"` rows from the public dataset and running:

```bash
uv run python -m src.evaluate \
  --dataset /path/to/public_temporal_questions.jsonl \
  --pipeline agentic \
  --provider cloudflare \
  --output results/temporal_agentic_20260929.jsonl
```
