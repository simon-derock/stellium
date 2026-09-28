<div align="center">

# 🐯 STELLIUM
### Autonomous Agentic GraphRAG Intelligence Platform on TigerGraph Savanna

**Built by Philip Simon Derock**

[![CI](https://github.com/simon-derock/stellium/actions/workflows/ci.yml/badge.svg)](https://github.com/simon-derock/stellium/actions)
[![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TigerGraph Savanna](https://img.shields.io/badge/TigerGraph-Savanna-orange.svg)](https://tgcloud.io/)
[![TigerVector HNSW](https://img.shields.io/badge/TigerVector-HNSW%20Native-yellow.svg)](https://www.tigergraph.com/)
[![Managed by uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy%20strict-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<p align="center">
  <b>A production-grade Agentic GraphRAG system proving where multi-step autonomous reasoning delivers positive ROI over traditional RAG and GraphRAG — and where simpler retrieval suffices.</b>
</p>

[Key Features](#-key-features) •
[Architecture](#-system-architecture) •
[3-Way Benchmark](#-the-3-way-benchmark) •
[Quickstart](#-quickstart) •
[API & UI Surface](#-live-api--judging-surface) •
[Specifications](#-engineering-specifications)

---

</div>

## 💡 Executive Summary & Core Research Thesis

Retrieval-Augmented Generation (RAG) retrieves text chunks. GraphRAG adds relational structure. But complex real-world questions require an autonomous system that can **plan investigations**, **execute mathematical operations inside the database**, **evaluate evidence quality**, **backtrack on dead ends**, and **decide when sufficient evidence exists to terminate**.

**Stellium** was created and architected by **Philip Simon Derock** for the **TigerGraph Agentic GraphRAG Hackathon 2026**. It operates across **2,951 historical Olympic Wikipedia articles (~5.47M tokens)**. The repository includes deterministic graph tools, hybrid retrieval, and a ReAct agent; benchmark outcomes depend on the configured providers and corpus snapshot.

1. **Where Agentic GraphRAG Wins Decisively**: Multi-hop entity navigation, temporal succession (`PRECEDES`/`SUCCEEDS`), superlative rankings, and numerical aggregations where standard RAG suffers arithmetic hallucinations.
2. **The Zero-Token Fast Path**: Numerical aggregations and chronological edge traversals can execute through compiled GSQL without an LLM call. A recorded live run returned the biathlon count `5` in 1,742 ms end-to-end; a separate live query measured 38.59 ms end-to-end. Server-side execution and network latency are different measurements.
3. **Where Simpler Retrieval Suffices**: Demonstrating exact efficiency thresholds where low-complexity lookups can be routed to single-turn vector search without agentic orchestration overhead.

---

## 🚀 Key Features

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STELLIUM CORE ENGINE                            │
├──────────────────────────┬──────────────────────────┬──────────────────┤
│    LOCAL RETRIEVAL      │    TIGERGRAPH SAVANNA    │   REACT AGENT    │
│   Packed Integer Masks  │   Native TigerVector HNSW│  Bounded Tool Loop│
│   BM25Plus Inverted Index│   Compiled GSQL Queries  │  Observation Review│
│   Reciprocal Rank Fusion │   Graph Topological Edges│  Tool Strategy Pivot│
│ Optional Cross-Encoder  │   Compiled GSQL Queries │  Grounded Trace  │
└──────────────────────────┴──────────────────────────┴──────────────────┘
```

### 1. Local Retrieval Coprocessor
- **Packed year/season masks**: Integer masks filter requested year and season facets before candidate selection; sport names remain corpus-derived text.
- **BM25Plus Sparse Indexing (`rank-bm25`)**: Guarantees positive IDF scores across all corpus sizes, achieving exact token matching for athlete names (`Naim Süleymanoğlu`), numbers, and venue names.
- **Reciprocal Rank Fusion (RRF)**: Merges dense TigerVector semantic similarity ranks with sparse BM25 scores using $RRF(d)=\sum_{m\in\{dense,sparse\}}\frac{1}{60+r_m(d)}$.
- **Cross-Encoder Precision Reranker**: MiniLM-L6 cross-encoder scoring top-30 fused candidates down to high-precision top-1 and top-2 passages.

### 2. Native TigerGraph Savanna Integration
- **Zero Raw String Concatenation**: Injection-proof GSQL execution using compiled stored queries (`get_event_aggregates`, `get_preceding_event`, `get_superlative_event`, `get_event_by_venue_date`, `get_event_attribute`, `vector_search_chunks`).
- **TigerVector HNSW Embedding Space**: Native in-database vector index co-located with graph topology.
- **Graph-backed evidence**: Event, venue, document, and chunk relationships support structured traversal alongside dense retrieval.

### 3. Table-Aware Entity & Doubly-Linked Semantic Chunking
- **Zero Boundary Fragmentation**: Extracts structured Olympic infobox metadata (competitor counts, nation counts, medalists, venues, dates) intact into **Chunk 0**.
- **Doubly-Linked Pointers**: Chunks carry `prev_chunk_id` and `next_chunk_id` pointers, allowing $O(1)$ neighboring passage expansion without secondary vector searches.
- **Name Boundary Normalization**: Resolves concatenated Wikipedia athlete strings (e.g., `Rudolf DombiRoland Kökény` $\to$ `['Rudolf Dombi', 'Roland Kökény']`).

### 4. Resilient Session-Locked Multi-Provider LLM Router
- **Strict Compliance with Hackathon Binding Rulings**: Guarantees the **exact same model** is used across planner, tool dispatcher, and synthesis within a single pipeline execution.
- **Cloudflare Workers AI Primary**: FP8-optimized `@cf/meta/llama-3.1-8b-instruct-fast` (~9 neurons/call, comfortably within the 10,000 neurons/day free tier).
- **Provider Selection**: Cloudflare is the default; Gemini and Mistral can be selected for a fresh run. Retries remain pinned to the selected provider/model for that run.

### 5. Judge-Safe Security Guardrails
- **Conservative Query Guardrails**: Structural command and jailbreak defenses are tested against the public question suite and adversarial inputs.
- **Structural Command Isolation**: Blocks structural database mutations (`DROP GRAPH`, `DELETE FROM`, `ALTER VERTEX`) and multi-word jailbreaks while welcoming all valid sports queries.
- **Automated Trust Tiers**: Batch evaluation endpoints (`/api/v1/evaluate/batch`) bypass input guardrails for automated evaluation sets.

---

## 🔬 System Architecture

```mermaid
flowchart TD
    UserQuery["User / Evaluation Query"] --> Layer0{"Guardrails Filter"}
    Layer0 -->|"Safe"| Pipeline{"Pipeline selection"}
    Layer0 -->|"Blocked"| SecurityBlock["HTTP 400 Safety Policy Notice"]
    Pipeline -->|"RAG"| RAG["TigerVector HNSW"]
    Pipeline -->|"GraphRAG"| GraphRAG["Graph + hybrid retrieval"]
    Pipeline -->|"Agentic"| Agent["ReAct agent"]
    Agent --> Action{"LLM-selected action"}
    Action -->|"Structured query"| GSQL["Compiled GSQL"]
    Action -->|"Semantic retrieval"| Dense["TigerVector HNSW"]
    Action -->|"Hybrid retrieval"| Sparse["BM25Plus → RRF → rerank"]
    GSQL -->|"Observation"| Agent
    Dense -->|"Observation"| Agent
    Sparse -->|"Observation"| Agent
    Agent -->|"Final answer"| Output["Answer + agentic trace"]
    RAG --> Output
    GraphRAG --> Output
```

---

## 📊 The 3-Way Benchmark

Stellium runs a side-by-side benchmark comparing three distinct retrieval pipelines over the **100 public and 50 hidden questions**:

| Dimension | Pipeline 1: Baseline Vector RAG | Pipeline 2: Fixed GraphRAG | Pipeline 3: Autonomous Agentic GraphRAG (Stellium) |
| :--- | :--- | :--- | :--- |
| **Retrieval Strategy** | Vanilla TigerVector HNSW top-5 | Fixed GSQL lookup/venue traversal + TigerVector HNSW top-3 | **Dynamic tool dispatch (GSQL + BM25Plus/RRF + TigerVector)** |
| **Tool Calling** | None (Single vector retrieval) | Fixed retrieval flow | **Bounded cyclic ReAct; reviews each observation and can pivot tools** |
| **Aggregations** | LLM over retrieved passages | Graph plus retrieved passages | Compiled GSQL accumulators when the agent selects the matching tool |
| **Temporal Chains** | Passage retrieval | Graph expansion | `PRECEDES`/`SUCCEEDS` traversal when applicable |
| **Token Cost** | Measured per run | Measured per run | Deterministic tools consume 0 LLM tokens; LLM calls are reported per run |
| **Latency** | Measure with the evaluation runner | Measure with the evaluation runner | Live end-to-end GSQL example: 38.59 ms; includes network overhead |
| **Investigation Trace** | None | Fixed subgraph triples | **Full 10-field Agentic Trace (Judges Spec)** |

These are three independent benchmark pipelines, not sequential phases. BM25Plus/RRF is enabled through the Agentic `hybrid_search` tool; it is deliberately absent from the vector-only RAG baseline and is not currently called by the fixed GraphRAG pipeline.

### Historical public run data (not a current clean benchmark)

**Do not read this table as Stellium's current accuracy or a valid head-to-head result.** It is retained as historical diagnostic data. A later audit found that the Agentic system prompt contained fully worked examples copied from public questions `pub-001` and `pub-002`, including their answers, so its score is contaminated by benchmark leakage. The RAG and fixed GraphRAG arms did not use that prompt, but those historical scores also need a reproducible rerun before they can support a current comparison. The copied examples have been removed and a regression check now prevents public question text from being added to the Agentic prompt. Exact Match (EM) is strict normalized string equality. No clean post-fix three-pipeline score or 98% result has been established; the latest clean run attempt stopped before question 1 because the provider daily quota was exhausted.

| Pipeline | Exact Match | Token F1 | Mean latency | Mean LLM tokens |
| :--- | ---: | ---: | ---: | ---: |
| RAG | 5% (5/100) | 0.065 | 981 ms | 134 |
| GraphRAG | 34% (34/100) | 0.354 | 1,487 ms | 728 |
| Agentic GraphRAG | 80% (80/100) | 0.807 | 2,895 ms | 3,907 |

Historical Agentic category counts were 21/21 aggregation, 17/22 temporal, 10/10 superlative, 17/28 multi-hop, and 15/19 lookup. The local output `results/public_results_20260928_v3.jsonl` is ignored by Git. A later 100-question Agentic-only run reported 84% EM and 0.852 token F1; it is also contaminated and predates current aggregate-action validation and answer-verification changes. The evaluator checkpoints every completed question and supports `--resume` after provider interruption.

Post-v4 work further corrects case-insensitive graph matching and venue/date extraction: the agent now preserves the date order in the source/question and receives the full candidate list instead of only five results. Direct live Savanna checks verified the expected unique results for two previously missed questions. These changes have not yet been scored in a fresh full benchmark: the current Cloudflare account returns HTTP 429 with provider error 4006 (daily neuron allocation exhausted). The client now stops immediately on this non-recoverable quota response instead of retrying it five times.

A separate live dense-retrieval audit on 2026-09-28 used aligned Jina v5 1024-dimensional query embeddings for all 100 public questions. TigerVector returned **zero chunks for every question**, with zero hit/recall/MRR through top-30. The graph has 22,016 Chunk vertices and the schema declares the HNSW vector attribute, but an inspected Chunk vertex had no stored `embedding` value. This points to missing or unsearchable vector data, so the old 5% RAG score cannot be used to judge embedding relevance. See [the retrieval audit](docs/benchmark-audits/dense-retrieval-20260928.md) and run `uv run python -m scripts.evaluate_dense_retrieval` to reproduce the live retrieval check.

### Sparse Retrieval Ablation

On the 100 public questions and 22,016 chunks, BM25Plus candidate retrieval was compared using the previous whitespace tokenizer and a general punctuation-normalizing tokenizer. The ranking unit is a chunk; a hit means its document ID appears in the question's `gold_doc_ids`. This measures retrieval coverage only, not answer accuracy or hybrid-search quality.

| BM25Plus tokenization | Recall@5 | Recall@10 | Recall@30 | MRR@30 |
| :--- | ---: | ---: | ---: | ---: |
| Previous whitespace split | 81% | 90% | 95% | 0.571 |
| Punctuation normalized | **87%** | **95%** | **96%** | **0.681** |

The new tokenizer improved top-5 candidate recall by 6 percentage points and MRR by 0.110 on this fixed set. The question and corpus SHA-256 fingerprints are recorded by the reproducible runner; repeat the comparison with:

```bash
uv run python scripts/evaluate_sparse_retrieval.py \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --corpus hackathon-resources/corpus/corpus.jsonl \
  --output results/sparse_retrieval_ablation.json
```

---

## 🛠️ Quickstart

### 1. Prerequisites & Environment Setup
Stellium uses **`uv`** for reproducible sub-second environment management:

```bash
# Clone the repository
git clone https://github.com/simon-derock/stellium.git
cd stellium

# Install all dependencies (including dev and test suites)
uv sync --extra dev

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials (TigerGraph Savanna, Cloudflare / Gemini API keys)
```

### 2. Run the Verification Quality Gate
```bash
# Run the complete test suite
uv run pytest

# Run Ruff linter and formatter checks
uv run ruff check && uv run ruff format --check

# Run static type checking for application and tests
uv run mypy src/ tests/
```

### 3. Run the Evaluation Benchmark Suite
```bash
# Run all 3 pipelines across the public 100 questions
uv run python -m src.evaluate \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --pipeline all \
  --output results/public_results.jsonl

# Resume completed rows after a provider or network interruption
uv run python -m src.evaluate \
  --dataset hackathon-resources/questions/eval_public.jsonl \
  --pipeline all \
  --output results/public_results.jsonl \
  --resume

# Run the autonomous agent on hidden questions
uv run python -m src.evaluate \
  --dataset hackathon-resources/questions/eval_hidden.jsonl \
  --pipeline agentic \
  --output results/hidden_submission.jsonl
```

### 4. Launch the FastAPI Application
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🖥️ Live API & Judging Surface

Stellium provides a modern REST API with native Lunarbit `GraphSurface.tsx` visualization DTO support:

| Endpoint | Method | Purpose |
| :--- | :---: | :--- |
| `/health` | `GET` | System health check and readiness status |
| `/api/v1/query/compare` | `POST` | Executes RAG, GraphRAG, and Agentic side-by-side |
| `/api/v1/query/rag` | `POST` | Runs the vector RAG pipeline |
| `/api/v1/query/graphrag` | `POST` | Runs the hybrid GraphRAG pipeline |
| `/api/v1/query/agentic` | `POST` | Runs the full autonomous Agentic GraphRAG pipeline |
| `/api/v1/evaluate/batch` | `POST` | Batch evaluation runner bypassing input guardrails |
| `/api/v1/graph/snapshot` | `GET` | Serializes graph topology to Lunarbit `SnapshotDTO` |
| `/api/v1/sessions/{id}/history` | `GET` | Reads session history when supported by the configured graph |

---

## 📁 Repository Structure

```text
stellium/
├── PLAN_SPEC.md              # Canonical specification (Grep-first single source of truth)
├── BOARD.md                  # Live coordination board & architectural decision log
├── README.md                 # Public product documentation (this file)
├── pyproject.toml            # Project configuration & dependency specifications
├── .github/workflows/ci.yml  # Matrix CI/CD quality gate (Python 3.12 & 3.13)
├── hackathon-resources/      # Official Olympic corpus and evaluation question sets
│   ├── corpus/corpus.jsonl   # 2,951 documents, ~5.47M tokens
│   ├── questions/eval_public.jsonl
│   └── questions/eval_hidden.jsonl
├── samples/frontend/         # Lunarbit GraphSurface.tsx force-directed graph engine
├── src/
│   ├── models/               # Domain models, AgentState, trace contracts, Snapshot DTOs
│   ├── guardrails/           # Judge-safe structural injection detection & NFKD normalizer
│   ├── ingest/               # Table-aware infobox parser & doubly-linked chunking engine
│   ├── coprocessor/          # Packed integer masks, BM25Plus, RRF, optional cross-encoder
│   ├── graph/                # TigerGraph Savanna schema DDL, TigerVector, compiled GSQL
│   ├── llm/                  # Session-locked multi-provider router (Cloudflare, Gemini, Mistral)
│   ├── pipelines/            # Pipeline 1 (RAG), Pipeline 2 (GraphRAG), Pipeline 3 (Agentic)
│   ├── evaluate.py           # CLI benchmark runner with Tier-1 metrics (EM, Token F1, MRR)
│   └── api/main.py           # FastAPI backend server with CORS and snapshot serialization
└── tests/                    # Unit, integration, resilience, contract, and security tests
```

---

## 📜 Engineering Specifications

### Retrieval and Reasoning Formulations

Dense and sparse rankings are fused by Reciprocal Rank Fusion (RRF). The rankers are TigerVector HNSW and BM25Plus, with `k = 60`:

```text
RRF(d) = sum over rankers m of 1 / (k + rank_m(d))
rankers = {Dense HNSW, Sparse BM25Plus}; k = 60
```

The sparse ranker uses BM25Plus with document-length normalization and an additive `delta` term. `avgdl` is the average document length:

```text
Score_BM25Plus(D, Q) = sum over query terms q_i of:
  IDF(q_i) * [ f(q_i,D)*(k1+1) / (f(q_i,D) + k1*(1-b+b*|D|/avgdl)) + delta ]
k1 = 1.5; b = 0.75; delta = 1.0
```

Evaluation normalizes answer strings before exact match and computes token overlap for F1. `y_hat` is the generated answer, `y_star` is the reference answer, and `T(x)` is the token multiset of `x`:

```text
EM        = 1 when normalize(y_hat) == normalize(y_star), otherwise 0
Precision = |T(y_hat) intersect T(y_star)| / |T(y_hat)|
Recall    = |T(y_hat) intersect T(y_star)| / |T(y_star)|
F1        = 2 * Precision * Recall / (Precision + Recall)
```

The bitemporal module resolves conflicts by source authority, then validity timestamps, and reports competing versions when unresolved. This conceptual score describes authority and recency; the implementation uses ordered comparisons rather than evaluating the formula:

```text
S(f) = alpha * authority(source)
     + beta * exp(-lambda * (now - valid_from))
       * 1[ superseded_by(f) is empty ]
```

The packed mask applies requested year and season facets independently:

```text
mask(q) = year_mask(q) AND season_mask(q)
```

In code, requested year and season groups are checked independently, so a chunk must match every requested facet. Masks are packed integer fields; sport names are not encoded in a static taxonomy.

### Complexity, Memory, and Runtime Profile

| Operation | Cost / profile | Scope and qualification |
| :--- | :--- | :--- |
| Neighbor chunk expansion | Expected `O(1)` map lookup per pointer | Uses `prev_chunk_id` / `next_chunk_id` in the in-memory coprocessor map |
| TigerVector retrieval | Expected `O(log N)` graph traversal | HNSW is approximate nearest neighbor; this is expected scaling, not a worst-case guarantee |
| RRF fusion | `O(K log K)` | Sorts at most the union of the input lists; current hybrid path takes up to 30 dense and 30 sparse candidates, so `K <= 60` |
| ReAct stream parsing | `O(N)` | Balanced-brace scan and line lexer over response length `N` |
| Local coprocessor memory | Under 150 MB RSS in the reported full-corpus run | 22,016 chunks; host/runtime and measurement method affect RSS |
| Category filtering | Integer bitwise operations | Avoids allocating candidate bit arrays; implementation scans BM25 scores when filtering |

Live Savanna profiling used compiled native GSQL query installations. One `get_event_aggregates` request measured **38.59 ms end-to-end**; this includes client/network overhead, while the recorded internal execution target/measurement was **below 5 ms**. Do not compare the internal engine number directly with end-to-end API latency.

### Token Economics and Recovery

The configured Cloudflare Workers AI model has a recorded approximate consumption of **9 neurons per inference**. At that rate, 450 model inferences would use about **4,050 neurons**, or **40.5%** of a 10,000-neuron daily allowance. This is a planning estimate: retries, prompt length, and output length affect actual consumption. Deterministic GSQL steps report zero LLM tokens.

The router locks a provider/model for each pipeline session. Transient HTTP 429/500/502/503/504/524 responses retry with backoff on that same model; provider failover starts a fresh run rather than mixing models mid-answer. The ReAct agent can select deterministic GSQL tools directly, avoiding planner/synthesis calls when its response and route permit. Bitemporal conflicts are resolved by authority first, then recency, with dual-version reports for ties or unresolved conflicts.

### API and Operational Notes

- The agent consumes LLM-generated ReAct tool actions; its response parser accepts structured text and JSON tool-call payloads.
- GSQL calls use compiled named queries with parameter dictionaries.
- Public and hidden benchmark commands are shown in Quickstart. The hidden-set command is intentionally `--pipeline agentic`; run all pipelines only when the submission protocol requests it.
- Python application code follows the repository's no-docstring convention and uses `#` comments.
- Commit messages follow `<type>(<scope>): <summary> [committed by master-agent-001]` for coordinator changes.

### Architecture at a Glance

```text
Question
  │
  ├── Guardrails ── blocked request → HTTP 400
  │
  └── Pipeline selection
       ├── RAG ─────────────── TigerVector HNSW ── grounded answer
       ├── GraphRAG ────────── graph + hybrid retrieval ── grounded answer
       └── Agentic ReAct ───── LLM chooses tools and next step
              ├── compiled GSQL aggregations / graph traversal
              ├── TigerVector HNSW
              └── BM25Plus → RRF → optional cross-encoder
                       └── evidence → answer and trace
```

**Creator and Lead Architect:** Philip Simon Derock.

---

<div align="center">
  <sub>Developed by <b>Philip Simon Derock</b> for the TigerGraph Agentic GraphRAG Hackathon 2026.</sub>
</div>
