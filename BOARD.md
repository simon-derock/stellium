# grep -n 'TASK-XXX\|Decision\|Next Steps' BOARD.md  <-- Use this to jump to any section instantly
# STELLIUM: Agent Orchestra Coordination Board

## 1. Project Metadata
- Remote Repository: `https://github.com/simon-derock/stellium.git`
- Primary Branch: `main`
- Coordinator: `master-agent-001`
- Canonical Specification: `PLAN_SPEC.md`
- Quality Gate: `uv run pytest && uv run ruff check && uv run ruff format --check && uv run mypy --strict`

---

## 2. Active Task Queue

| Task ID | Task Scope | Assigned Agent | Status | Branch / Worktree | Green Tests | Commit Hash |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| **TASK-001** | Python Environment (`uv`), `pyproject.toml`, Quality Tooling (`ruff`, `mypy`, `pytest`) | `master-agent-001` | **DONE** | `main` | 2 passed | `01cef2a` |
| **TASK-002** | Ingestion Engine, Table-Aware Chunking & Doubly Linked Pointers | `master-agent-001` | **DONE** | `main` | 3 passed | `afa12b7` |
| **TASK-003** | TigerGraph Savanna DDL Schema & Stored GSQL Queries | `master-agent-001` | **DONE** | `main` | Type check pass | `074c61f` |
| **TASK-004** | High-Speed Coprocessor: Roaring Bitmasks, BM25Plus, RRF & Reranker | `master-agent-001` | **DONE** | `main` | 3 passed | `62d5dff` |
| **TASK-005** | Resilient Multi-Provider LLM Router & 3-Layer Guardrails | `master-agent-001` | **DONE** | `main` | 6 passed | `7534d93` |
| **TASK-006** | Pipeline 1 (Vector RAG) & Pipeline 2 (Hybrid GraphRAG) | `master-agent-001` | **DONE** | `main` | Type check pass | `1565f8d` |
| **TASK-007** | Pipeline 3: Autonomous Agentic GraphRAG Engine & GSQL Tools | `master-agent-001` | **DONE** | `main` | 14 passed | `1565f8d` |
| **TASK-008** | Evaluation Suite (100 Public + 50 Questions), Metrics & Benchmarks | `master-agent-001` | **IN_PROGRESS** | `main` | - | - |
| **TASK-009** | FastAPI Backend & Lunarbit `GraphSurface.tsx` Frontend | `master-agent-001` | **QUEUED** | `main` | - | - |
| **TASK-010** | Presentation, Architecture SVG & Final Submission Pack | `master-agent-001` | **QUEUED** | `main` | - | - |

---

## 3. Dynamic Elastic Agent Pool

```text
[MASTER] master-agent-001 : Supervising execution, quality gates, integration.
[WORKER] agent-001        : Assigned to Ingestion, Chunking & Savanna Schema.
[WORKER] agent-002        : Assigned to Hybrid Coprocessor (Bitmaps, BM25, RRF).
[WORKER] agent-003        : Assigned to Multi-LLM Router, Guardrails & LangGraph Engine.
[WORKER] agent-004        : Assigned to Evaluation Benchmark & Dashboard Integration.
```

---

## 4. Architectural Decision Log

1. **Deterministic Metadata Header Injection vs. Synthetic Questions**: Rejected synthetic question generation to eliminate semantic drift, save 15,000 API calls, and maintain strict corpus grounding.
2. **High-Speed Coprocessor Integration**: TigerGraph Savanna handles Graph topology, HNSW vector search, and GSQL aggregations; local coprocessor handles Roaring Bitmasks, BM25, and RRF for $<1\,\text{ms}$ hybrid search.
3. **Doubly Linked Chunk Pointers**: Chunks carry `prev_chunk_id` and `next_chunk_id` (`uint16` offsets); Olympic events carry `prev_event_id` and `next_event_id` (`PRECEDES` / `SUCCEEDS` edges) to resolve boundary overlaps and temporal sequences.
4. **Pragmatic 3-Layer Guardrails**: Medium/simple regex scanning for database commands and prompt injections, parameterized GSQL calls, and output API key leak redaction without over-restricting evaluation test sets.
5. **No Docstrings in Python Code**: Use `#` comments exclusively across all Python files per project specification.
6. **Lunarbit Visualizer Integration**: Direct backend serialization from TigerGraph traversed subgraphs to `Snapshot`, `GraphNode`, and `GraphEdge` DTOs rendered via `GraphSurface.tsx`.

---

## 5. Immediate Next Steps (Next 3 Atomic Steps)

1. **Step 1 (TASK-008)**: Build `src/evaluate.py` evaluation harness to benchmark all 3 pipelines across the evaluation questions, computing Tier-1 EM, Token-F1, Recall@k, and Pareto Frontier statistics.
2. **Step 2 (TASK-009)**: Build the FastAPI async API server (`/api/v1/query/compare`, `/api/v1/evaluate/batch`, `/api/v1/graph/snapshot`) with the Lunarbit `GraphSurface.tsx` payload generator.
3. **Step 3 (TASK-010)**: Generate architecture diagrams, live evaluation results, and submission package.
