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
| **TASK-001** | Python Environment (`uv`), `pyproject.toml`, Quality Tooling (`ruff`, `mypy`, `pytest`) | `master-agent-001` | **IN_PROGRESS** | `main` | - | Pending |
| **TASK-002** | Ingestion Engine, Table-Aware Chunking & Doubly Linked Pointers | `agent-001` | **QUEUED** | `agent/001/ingest-schema` | - | - |
| **TASK-003** | TigerGraph Savanna DDL Schema & Offline Mock Adapter | `agent-001` | **QUEUED** | `agent/001/ingest-schema` | - | - |
| **TASK-004** | High-Speed Coprocessor: Roaring Bitmasks, BM25, RRF & Reranker | `agent-002` | **QUEUED** | `agent/002/hybrid-engine` | - | - |
| **TASK-005** | Resilient Multi-Provider LLM Router & 3-Layer Guardrails | `agent-003` | **QUEUED** | `agent/003/langgraph-agent` | - | - |
| **TASK-006** | Pipeline 1 (Vector RAG) & Pipeline 2 (Hybrid GraphRAG) | `agent-003` | **QUEUED** | `agent/003/langgraph-agent` | - | - |
| **TASK-007** | Pipeline 3: LangGraph Agentic Engine & GSQL Reasoning Tools | `agent-003` | **QUEUED** | `agent/003/langgraph-agent` | - | - |
| **TASK-008** | Evaluation Suite (100 Public + 50 Hidden), MRR & Ragas | `agent-004` | **QUEUED** | `agent/004/eval-dashboard` | - | - |
| **TASK-009** | FastAPI Backend & Lunarbit `GraphSurface.tsx` Frontend | `agent-004` | **QUEUED** | `agent/004/eval-dashboard` | - | - |
| **TASK-010** | GitHub Actions CI, Architecture Diagrams & Submission Pack | `master-agent-001` | **QUEUED** | `main` | - | - |

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

1. **Step 1 (TASK-001)**: Initialize `pyproject.toml` using `uv`, configure dependencies (`fastapi`, `pydantic`, `langgraph`, `pytigergraph`, `pyroaring`, `rank-bm25`, `sentence-transformers`, `ruff`, `mypy`, `pytest`), and verify `uv sync`.
2. **Step 2 (TASK-001)**: Setup quality configurations (`ruff.toml` / `pyproject.toml` tool sections), add initial smoke test, and execute the quality gate.
3. **Step 3 (TASK-001)**: Make the initial atomic micro-commit with signature `[committed by master-agent-001]`.
