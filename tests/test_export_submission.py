# Submission export: every pipeline's answer, tokens, latency, citations, and the agent trace.
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_export_writes_json_and_csv_with_traces(tmp_path: Path) -> None:
    row = {
        "qid": "eval-001",
        "qtype": "lookup",
        "question": "How many nations?",
        "rag_answer": "12",
        "rag_tokens": 900,
        "rag_input_tokens": 880,
        "rag_output_tokens": 20,
        "rag_latency_ms": 1500.0,
        "rag_retrieved_doc_ids": ["Q1"],
        "graphrag_answer": "12",
        "graphrag_tokens": 1100,
        "graphrag_retrieval_metadata": {"graph_operation": "event_attribute"},
        "agentic_answer": "12",
        "agentic_tokens": 700,
        "agentic_trace": {
            "step_count": 2,
            "llm_calls": [{"prompt_tokens": 690, "credential_alias": "LOCAL_ALIAS"}],
            "retrieval_methods": ["event_attribute"],
            "agents_invoked": ["OrchestratorAgent", "EntityLinkingAgent"],
            "strategy_changed": False,
            "stopping_reason": "verified",
        },
    }
    results = tmp_path / "hidden.jsonl"
    results.write_text(json.dumps(row) + "\n")

    subprocess.run(
        [
            sys.executable,
            "scripts/export_submission.py",
            "--results",
            str(results),
            "--out",
            str(tmp_path / "out" / "hidden"),
        ],
        check=True,
        capture_output=True,
    )

    document = json.loads((tmp_path / "out" / "hidden.json").read_text())
    question = document["questions"][0]
    assert question["rag"]["llm_tokens"]["total"] == 900
    assert question["rag"]["latency_s"] == 1.5
    assert question["graphrag"]["graph"]["graph_operation"] == "event_attribute"
    assert question["agentic"]["trace"]["step_count"] == 2
    assert question["agentic"]["trace"]["llm_calls"] == [{"prompt_tokens": 690}]
    assert "LOCAL_ALIAS" not in (tmp_path / "out" / "hidden.json").read_text()
    with (tmp_path / "out" / "hidden.csv").open() as csv_file:
        flat = next(csv.DictReader(csv_file))
    assert flat["agentic_agents"] == "OrchestratorAgent;EntityLinkingAgent"
    assert flat["agentic_llm_calls"] == "1"
