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
| **TASK-008** | Evaluation Suite (100 Public + 50 Questions), Metrics & Benchmarks | `agent-004` | **DONE** | `agent/004/eval-dashboard` (merged) | 26 passed | `28ce75a` |
| **TASK-009** | FastAPI Backend & Interactive GraphSurface Frontend UI | `master-agent-001` | **DONE** | `main` | 26 passed | `daec8f6` |
| **TASK-010** | Round 2 Prep: Bitemporal Schema, Architecture Diagrams & Final Submission | `agent-003` | **DONE** | `agent/003/langgraph-agent` (merged) | 41 passed | `c2e232b` |

---

## 3. Dynamic Elastic Agent Pool

```text
[MASTER] master-agent-001 : Supervising execution, quality gates, integration (main: /media/simon/.../stellium)
[WORKER] agent-003        : Stream 3 Complete: Round 2 Bitemporal Schema & Conflict Resolution merged into main
[WORKER] agent-004        : Stream 4 Complete: 150-Question Benchmark Execution & Submission Pack merged into main
```

---

## 4. Architectural Decision Log

1. **Deterministic Metadata Header Injection vs. Synthetic Questions**: Rejected synthetic question generation to eliminate semantic drift, save 15,000 API calls, and maintain strict corpus grounding.
2. **High-Speed Coprocessor Integration**: TigerGraph Savanna handles Graph topology, HNSW vector search, and GSQL aggregations; local coprocessor handles Roaring Bitmasks, BM25, and RRF for $<1\,\text{ms}$ hybrid search.
3. **Doubly Linked Chunk Pointers**: Chunks carry `prev_chunk_id` and `next_chunk_id` (`uint16` offsets); Olympic events carry `prev_event_id` and `next_event_id` (`PRECEDES` / `SUCCEEDS` edges) to resolve boundary overlaps and temporal sequences.
4. **Pragmatic 3-Layer Guardrails**: Medium/simple regex scanning for database commands and prompt injections, parameterized GSQL calls, and output API key leak redaction without over-restricting evaluation test sets.
5. **No Docstrings in Python Code**: Use `#` comments exclusively across all Python files per project specification.
6. **Lunarbit Visualizer Integration**: Direct backend serialization from TigerGraph traversed subgraphs to `Snapshot`, `GraphNode`, and `GraphEdge` DTOs rendered via `GraphSurface.tsx`.
7. **Round 2 Bitemporal Schema & Conflict Resolution**: 4-step conflict resolution engine comparing source authority and timestamps, with dual reporting of bounded confidence intervals on ties and agentic trace logging.

---

## 5. Immediate Next Steps (Next 3 Atomic Steps)

1. **Step 1 (TigerGraph Savanna Deployment)**: Connect live cluster with database secret / credentials, run `setup_schema()`, and compile 5 GSQL stored queries.
2. **Step 2 (Corpus Ingestion)**: Batch upsert 22,016 chunks and graph edges into TigerGraph Savanna cloud instance.
3. **Step 3 (Final Hackathon Submission)**: Verify live cloud queries, generate visual artifacts, and finalize Devpost/GitHub submission pack.
