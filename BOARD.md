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
| **TASK-011** | Pure ReAct Agent Harness, Dynamic LLM Linking, Disk Cache & 100-Test Suite | `master-agent-001` | **DONE** | `main` | 100 passed | `51b189a` |
| **TASK-012** | Clean Benchmark Integrity, Exact Embedding Alignment & Failure-Led Accuracy Work | `master-agent-001` | **IN PROGRESS** | `main` | 175 passed, 3 skipped | `1c305ba`, `f7d2aba`, `38a764c`, `ff6ea21`, `df167dd`, `aa9de94` |
| **TASK-013** | Restore and Verify Live TigerVector Evidence Retrieval | `master-agent-001` | **DONE** | `main` | 175 passed, 3 skipped | `1096c6d` |
| **TASK-014** | Correct Sparse Retrieval Metrics and Audit Public Gold-Document Coverage | `master-agent-001` | **DONE** | `main` | 175 passed, 3 skipped | `502df22` |
| **TASK-015** | Publish Current Three-Pipeline Baseline and Submission Artifacts | `master-agent-001` | **IN PROGRESS** | `main` | 177 passed, 3 skipped | `1096c6d` |

---

## 3. Dynamic Elastic Agent Pool

```text
[MASTER] master-agent-001 : Supervising execution, quality gates, integration (main: /media/simon/.../stellium)
[WORKER] agent-001        : Stream 1 Complete: Live Corpus Ingested into TigerGraph Savanna (27,493 vertices, 29,662 edges) (agent/001/ingest-schema)
[WORKER] agent-002        : Stream 2 Complete: Live Savanna Schema Deployed & GSQL Queries Compiled (agent/002/hybrid-engine)
[WORKER] agent-003        : Stream 3 Complete: Round 2 Bitemporal Schema & Conflict Resolution merged into main
[WORKER] agent-004        : Stream 4 Complete: 150-Question Benchmark Execution & Submission Pack merged into main
```

---

## 4. Architectural Decision Log

1. **Deterministic Metadata Header Injection vs. Synthetic Questions**: Rejected synthetic question generation to eliminate semantic drift, save 15,000 API calls, and maintain strict corpus grounding.
2. **High-Speed Coprocessor Integration**: TigerGraph Savanna handles graph topology, HNSW vector search, and GSQL aggregations; the local coprocessor handles BM25Plus, packed year/season integer masks, and RRF. Runtime claims require measured scope and must not imply the full BM25 scan is sub-millisecond.
3. **Doubly Linked Chunk Pointers**: Chunks carry `prev_chunk_id` and `next_chunk_id` IDs; Olympic events carry `prev_event_id` and `next_event_id` (`PRECEDES` / `SUCCEEDS` edges) to resolve boundary overlaps and temporal sequences.
4. **Pragmatic 3-Layer Guardrails**: Medium/simple regex scanning for database commands and prompt injections, parameterized GSQL calls, and output API key leak redaction without over-restricting evaluation test sets.
5. **No Docstrings in Python Code**: Use `#` comments exclusively across all Python files per project specification.
6. **Lunarbit Visualizer Integration Target**: The sample files under `samples/frontend/` are the visual reference. Live query-derived graph serialization and a maintained React frontend remain incomplete.
7. **Round 2 Bitemporal Schema & Conflict Resolution**: 4-step conflict resolution engine comparing source authority and timestamps, with dual reporting of bounded confidence intervals on ties and agentic trace logging.

---

## 5. Immediate Next Steps (Next 3 Atomic Steps)

1. **Finish the Current Benchmark Commit**: Publish the reproducible 2026-09-29 three-pipeline report and public raw output; keep hidden questions/results out of the public repository and preserve the fresh 50-row submission file locally as `results/hidden_submission.jsonl`. Re-run the complete quality gate, commit with the required attribution, and push.
2. **Prioritize Answer-Level Failure Analysis**: The current public baseline is RAG 43/100, GraphRAG 35/100, Agentic 60/100. Agentic temporal EM is 2/22, multi-hop 18/28, lookup 11/19. Analyze per-question evidence and traces, then fix general retrieval/reasoning failures without static question classifiers or hidden-set tuning; rerun the public benchmark after any material change.
3. **Complete Submission Surface**: Dense retrieval is restored and reports 95% gold-document hit@30 (document coverage, not answer accuracy). Remaining deliverables include richer completeness/grounding metrics, a live query-backed graph dashboard and maintained frontend, final architecture/demo/writeup, and an honest cost/accuracy comparison. The 98% objective is unmet; track evidence, not aspiration.
