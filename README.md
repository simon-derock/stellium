# STELLIUM 🐯
### Agentic GraphRAG System on TigerGraph Savanna

[![CI](https://github.com/simon-derock/stellium/actions/workflows/ci.yml/badge.svg)](https://github.com/simon-derock/stellium/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

Stellium is an enterprise-grade Agentic GraphRAG system designed for the **TigerGraph Agentic GraphRAG Hackathon**.
It benchmarks three distinct retrieval paradigms side-by-side:
1. **Standard Vector RAG**: Fixed-chunk semantic similarity search via TigerVector (HNSW).
2. **Hybrid GraphRAG**: Subgraph expansion fused with sparse & dense chunk retrieval.
3. **Autonomous Agentic GraphRAG**: LangGraph multi-step investigation, compiled GSQL mathematical aggregations, bitemporal precedence tracking, and adaptive strategy backtracking.

---

## Architecture & Documentation

- **Canonical Specification & Plan**: [PLAN_SPEC.md](PLAN_SPEC.md) (Grep-first single source of truth)
- **Live Coordination Board**: [BOARD.md](BOARD.md) (Dynamic task queue and agent pool)
- **Dataset**: Historical Olympic Wikipedia Corpus (2,951 documents, ~5.47M tokens)

---

## Quickstart

```bash
# Install dependencies using uv
uv sync --extra dev

# Run quality gate verification chain
uv run pytest && uv run ruff check && uv run ruff format --check && uv run mypy --strict

# Run the 3-way benchmark evaluation suite
uv run python -m src.evaluate --dataset hackathon-resources/questions/eval_public.jsonl --pipeline all

# Start the FastAPI server & Lunarbit visualizer
uv run uvicorn src.api.main:app --reload --port 8000
```

---

## Submission Artifacts

- **GitHub Repository**: `https://github.com/simon-derock/stellium.git`
- **Primary Branch**: `main`
- **Lead Architect**: `master-agent-001`
