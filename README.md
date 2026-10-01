<div align="center">

# STELLIUM

**Agentic GraphRAG on TigerGraph, with a measured answer to when an agent is worth it.**

[![CI](https://github.com/simon-derock/stellium/actions/workflows/ci.yml/badge.svg)](https://github.com/simon-derock/stellium/actions)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![TigerGraph](https://img.shields.io/badge/TigerGraph-Savanna%204.2.5-orange)
![Typed](https://img.shields.io/badge/mypy-strict-blue)
![Style](https://img.shields.io/badge/ruff-checked-black)

TigerGraph Agentic GraphRAG Hackathon 2026 · Built by Philip Simon Derock

</div>

---

## At a glance

| | |
|---|---|
| **Accuracy** | 98% Agentic GraphRAG · 98% GraphRAG · 67% RAG on 100 public questions |
| **Cost** | 990 LLM tokens per agentic answer, about $0.32 per 100 questions |
| **Evidence** | Every graph-pipeline answer is stated in a source article it cites |
| **Efficiency** | 86 of 100 agentic answers need a single LLM call |
| **Stack** | TigerGraph Savanna (graph + native vector search) · Cohere Command A · local int8 reranker |

---

## Results

One model, Cohere `command-a-03-2025`, for every LLM call in every pipeline. Token counts are LLM tokens only; graph queries and retrieval count zero.

| Pipeline | Accuracy | Tokens / answer | LLM calls | Latency p50 | Cost / 100 q |
|---|---:|---:|---:|---:|---:|
| RAG | 67% | 2,581 | 1 | 4.4 s | $0.65 |
| GraphRAG | **98%** | 1,094 | 2 | 6.1 s | $0.35 |
| Agentic GraphRAG | **98%** | **990** | 1.16 | 4.5 s | **$0.32** |

<sub>Latency is from the matched three-pipeline run and includes client-side request pacing. Agentic accuracy, tokens, and cost are from the final agent run on the same code.</sub>

### By question type

| Type | n | RAG | GraphRAG | Agentic |
|---|---:|---:|---:|---:|
| Count: how many events meet a condition | 21 | 2 | 21 | 21 |
| Lookup: one attribute of one event | 19 | 19 | 19 | 19 |
| Multi-hop: venue and date to winner | 28 | 26 | 27 | 27 |
| Ranking: event with the most competitors | 10 | 3 | 9 | 9 |
| Temporal: winner at the previous Games | 22 | 17 | 22 | 22 |

### Evidence and retrieval

| Pipeline | Gold among answers | Grounded | Hit@1 | MRR | nDCG@5 | Context recall |
|---|---:|---:|---:|---:|---:|---:|
| RAG | 67% | 94% | 0.800 | 0.869 | 0.843 | 83% |
| GraphRAG | 100% | 100% | 0.990 | 0.995 | 0.830 | 100% |
| Agentic GraphRAG | 100% | 100% | 0.990 | 0.995 | 0.830 | 100% |

| Measure | Meaning |
|---|---|
| Gold among answers | The gold answer is in what the pipeline returned, so honest tie reports count |
| Grounded | Every claimed value appears in an article the pipeline cited |
| Context recall | The gold answer is stated somewhere in the retrieved articles |

The two graph-pipeline misses are genuine ties in the corpus: two events with 41 competitors each, and two events at one venue on one date. Both pipelines return every candidate instead of guessing.

Full method, commits, and caveats: [public benchmark audit](docs/benchmark-audits/public-final-20261002.md)

---

## When does a question need an agent?

| Question shape | Best choice | Why |
|---|---|---|
| Fact stated in one passage | RAG | Lookups and most venue-and-date questions resolve from the top passages |
| Count or ranking across many events | GraphRAG | Five passages hold about a third of the events; one graph query holds all |
| Ambiguous names or a failed first lookup | Agentic | Re-plans after an error or a tie (10 strategy changes on the public set) |
| Any structured question where cost matters | Agentic | Stops on the first verified value, below a fixed two-call pipeline |

> **Finding.** On template-shaped questions the agent is not more accurate than a well-built GraphRAG. It is cheaper and corrects itself. Its value is early stopping and recovery, not raw accuracy.

---

## How it works

```mermaid
flowchart LR
    Q([Question]) --> RAG & GR & AG

    subgraph RAG [RAG · 1 LLM call]
        R1[Vector + BM25 search] --> R2[Rank fusion + reranker] --> R3[Answer]
    end

    subgraph GR [GraphRAG · 2 LLM calls]
        G1[Extract lookup] --> G2[One graph operation] --> G3[Answer, checked against graph]
    end

    subgraph AG [Agentic GraphRAG · stops when verified]
        O[Orchestrator] -->|tool call| T[Specialists]
        T -->|observation| O
        O --> A[Answer + trace]
    end

    R1 & G2 & T --- TG[(TigerGraph Savanna<br/>events · venues · articles · vectors)]
```

| | RAG | GraphRAG | Agentic GraphRAG |
|---|---|---|---|
| **Plan** | none | fixed: extract, query, answer | adaptive, one step at a time |
| **Graph use** | vector index | one typed operation | any specialist, in any order |
| **On failure** | answers anyway | answers anyway | re-plans or reports the ambiguity |
| **Evidence** | top five passages | graph value + cited article | graph value checked against the source line |

### Agent specialists

| Specialist | Job | LLM cost |
|---|---|---|
| Orchestrator | Chooses the next step from the question, evidence, and gaps | 1 call per step |
| Entity linking | Maps free text to graph events, sports, venues, and dates | 0 |
| Graph traversal | Venue-and-date multi-hop, previous editions, event attributes | 0 |
| Aggregation | Counts with exact bounds; rankings with tie detection | 0 |
| Evidence evaluation | Confirms each value against the cited article | 0 |
| Document retrieval | Hybrid or dense passage search when the graph has no answer | 0 |
| Query generation | Guarded read-only GSQL for questions no tool covers | 0 |

---

## Engineering

| Area | What was built |
|---|---|
| Entity linking | Order-insensitive matching over all 2,210 graph events; "+100 kg" never matches "100 kg" |
| Honesty | Unsupported answers rejected, ties reported, run stopped if the graph is unavailable |
| Same-model rule | One locked model per run, with no silent provider fallback |
| Reproducibility | Each run writes a manifest: commit, model, dataset and corpus SHA-256, settings |
| Quality gate | pytest, ruff, mypy strict, vulture, bandit, and pip-audit on every push; 85%+ coverage |
| Cost control | Query-embedding cache, batched prefetch, local reranker, early-stopping agent |

### What the data taught us

| Finding | Fix |
|---|---|
| LLM-written GSQL reached 76% | Typed graph tools reach 98% |
| TigerGraph `ACCUM` lists drop `ORDER BY` order | Rankings read stored counts directly |
| 26 events had a wrong or missing Games year | Article title made authoritative; live graph reconciled |
| One infobox lists 41,000,000 competitors | Prose value used only when it confirms a zero-padded count |

---

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env      # TigerGraph and Cohere credentials

uv run pytest             # full test suite, no network

uv run python -m src.evaluate --provider cohere --pipeline all \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --output results/public.jsonl --resume

uv run python scripts/summarize_results.py results/public.jsonl \
  --dataset hackathon-resources/questions/eval_public.jsonl

uv run uvicorn src.api.main:app --port 8000
```

| Tool | Purpose |
|---|---|
| `scripts/export_submission.py` | Hidden-set JSON and CSV: answers, tokens, latency, citations, traces |
| `scripts/summarize_results.py` | Accuracy, retrieval, grounding, latency, and cost tables |
| `scripts/reconcile_graph_attributes.py` | Diff or repair live graph attributes against the parser |
| `--provider offline` | Mock graph with no network, for CI smoke runs |

---

## Repository

| Path | Contents |
|---|---|
| `src/pipelines/` | `rag.py`, `graphrag.py`, `agentic.py`, `toolkit.py` |
| `src/linking/` | Graph-loaded entity linker |
| `src/graph/` | TigerGraph client, schema, compiled queries |
| `src/coprocessor/` | BM25, rank fusion, int8 reranker |
| `src/evaluate.py` | Benchmark runner and metrics |
| `docs/benchmark-audits/` | Every measured run, with caveats |
| `tests/` | Unit, contract, chaos, security, and API tests |

---

## Limitations

| Limitation | Effect |
|---|---|
| One gold string per question | Honest tie reports score zero on exact match (2 public questions) |
| Client-side request pacing | Pipelines with more LLM calls look slower |
| Template-shaped benchmark | Passage search and generated GSQL are rarely exercised |
| Round 2 conflict handling | Schema and resolver exist, not yet wired into evidence evaluation |
