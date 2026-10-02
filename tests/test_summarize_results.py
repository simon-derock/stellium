# Evidence and cost measures: candidate recall, grounding, abstention, latency, and dollars.
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_evidence_table_credits_ties_and_checks_cited_text(tmp_path: Path) -> None:
    corpus = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {"doc_id": "D1", "title": "Rowing – Men's eight", "text": "Gold: Ann LeeBo Kim"},
            {"doc_id": "D2", "title": "Rowing – Women's pair", "text": "Gold: Cy Day"},
        ],
    )
    dataset = _write_jsonl(
        tmp_path / "questions.jsonl",
        [
            {"qid": "q1", "question": "?", "qtype": "multi_hop", "answer": ["Ann LeeBo Kim"]},
            {
                "qid": "q2",
                "question": "?",
                "qtype": "superlative",
                "answer": ["Rowing – Women's pair"],
            },
            {"qid": "q3", "question": "?", "qtype": "lookup", "answer": ["4"]},
        ],
    )
    base = {"agentic_input_tokens": 1000, "agentic_output_tokens": 100, "agentic_tokens": 1100}
    results = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            # Team answer glued in the source; tie report that contains the gold title.
            {
                **base,
                "qid": "q1",
                "qtype": "multi_hop",
                "agentic_answer": "Ann Lee, Bo Kim",
                "agentic_em": 1.0,
                "agentic_latency_ms": 1000,
                "agentic_retrieved_doc_ids": ["D1"],
            },
            {
                **base,
                "qid": "q2",
                "qtype": "superlative",
                "agentic_answer": "Rowing – Men's eight; Rowing – Women's pair",
                "agentic_em": 0.0,
                "agentic_latency_ms": 3000,
                "agentic_retrieved_doc_ids": ["D1", "D2"],
            },
            {
                **base,
                "qid": "q3",
                "qtype": "lookup",
                "agentic_answer": "Not found in corpus",
                "agentic_em": 0.0,
                "agentic_latency_ms": 2000,
                "agentic_retrieved_doc_ids": [],
            },
        ],
    )

    output = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            str(results),
            "--dataset",
            str(dataset),
            "--corpus",
            str(corpus),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    row = next(line for line in output.splitlines() if line.startswith("| Agentic GraphRAG | 67%"))
    # 2/3 contain gold, 2/2 claims grounded, 1 abstention, 1.0 citations, $0.35 per 100 questions.
    assert "| 100% (2/2) | 1 | 1.0 | 2.0 s / 3.0 s | $0.35 |" in row


def test_retrieval_table_scores_ranked_citations_against_gold_documents(tmp_path: Path) -> None:
    corpus = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [{"doc_id": d, "title": d, "text": f"Gold: Winner {d}"} for d in ("D1", "D2", "D3")],
    )
    dataset = _write_jsonl(
        tmp_path / "questions.jsonl",
        [
            {
                "qid": "q1",
                "question": "?",
                "qtype": "multi_hop",
                "answer": ["Winner D2"],
                "gold_doc_ids": ["D2"],
            }
        ],
    )
    results = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            {
                "qid": "q1",
                "qtype": "multi_hop",
                "rag_answer": "Winner D2",
                "rag_em": 1.0,
                "rag_tokens": 10,
                "rag_latency_ms": 1000,
                "rag_retrieved_doc_ids": ["D1", "D2", "D3"],
            }
        ],
    )

    output = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            str(results),
            "--dataset",
            str(dataset),
            "--corpus",
            str(corpus),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    row = next(line for line in output.splitlines() if line.startswith("| RAG | 0.000"))
    # Gold at rank 2: hit@1 0, hit@5 1, MRR 0.5, nDCG 1/log2(3), recall 1, precision 1/3, AP 0.5.
    assert "| 0.000 | 1.000 | 0.500 | 0.631 | 1.000 | 0.333 | 100% (1 q) | 0.500 |" in row


def test_partial_rerun_overrides_only_the_rows_it_covers(tmp_path: Path) -> None:
    def row(qid: str, answer: str, em: float) -> dict[str, object]:
        return {
            "qid": qid,
            "qtype": "superlative",
            "graphrag_answer": answer,
            "graphrag_em": em,
            "graphrag_tokens": 100,
            "graphrag_latency_ms": 1000,
            "graphrag_retrieved_doc_ids": [],
        }

    base = _write_jsonl(tmp_path / "base.jsonl", [row("q1", "tie", 0.0), row("q2", "right", 1.0)])
    rerun = _write_jsonl(tmp_path / "rerun.jsonl", [row("q1", "right", 1.0)])
    out = tmp_path / "metrics.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            str(base),
            "--pipeline-from",
            f"graphrag={rerun}",
            "--json",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

    metrics = json.loads(out.read_text())
    assert metrics["pipelines"]["graphrag"]["overall"]["exact_match"] == 2
    assert [p["rows"] for p in metrics["provenance"]] == [2, 1]


def test_rows_with_estimated_token_counts_are_called_out(tmp_path: Path) -> None:
    def row(qid: str, rag_source: str, agent_source: str) -> dict[str, object]:
        return {
            "qid": qid,
            "qtype": "lookup",
            "rag_answer": "1",
            "rag_tokens": 100,
            "rag_latency_ms": 1000,
            "rag_retrieved_doc_ids": [],
            "rag_retrieval_metadata": {"token_sources": [rag_source]},
            "agentic_answer": "1",
            "agentic_tokens": 100,
            "agentic_latency_ms": 1000,
            "agentic_retrieved_doc_ids": [],
            "agentic_trace": {
                "llm_calls": [{"token_source": "billed"}, {"token_source": agent_source}]
            },
        }

    results = _write_jsonl(
        tmp_path / "r.jsonl", [row("q1", "billed", "estimated"), row("q2", "billed", "billed")]
    )
    out = tmp_path / "metrics.json"
    summary = subprocess.run(
        [sys.executable, "scripts/summarize_results.py", str(results), "--json", str(out)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    pipelines = json.loads(out.read_text())["pipelines"]
    assert pipelines["rag"]["overall"]["estimated_token_rows"] == 0
    assert pipelines["agentic"]["overall"]["estimated_token_rows"] == 1
    assert "estimated, not provider-reported, in: Agentic GraphRAG 1 rows" in summary
