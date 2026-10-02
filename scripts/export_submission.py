# Turn a benchmark results JSONL into the submission files judges read: one JSON document with
# every pipeline's answer, LLM token counts, latency, citations, and the full agentic trace, plus
# a flat CSV of the same rows. Token counts are LLM tokens only; graph and retrieval steps are 0.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

_PIPELINES = ("rag", "graphrag", "agentic")
# Per-call fields that describe the local run setup, not the answer; older traces still carry them.
_PRIVATE_CALL_FIELDS = frozenset({"credential_alias"})


def _public_trace(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trace or "llm_calls" not in trace:
        return trace
    calls = [
        {key: value for key, value in call.items() if key not in _PRIVATE_CALL_FIELDS}
        for call in trace["llm_calls"]
    ]
    return {**trace, "llm_calls": calls}


def _pipeline_entry(row: dict[str, Any], pipeline: str) -> dict[str, Any] | None:
    if f"{pipeline}_answer" not in row:
        return None
    metadata = row.get(f"{pipeline}_retrieval_metadata") or {}
    entry: dict[str, Any] = {
        "answer": row[f"{pipeline}_answer"],
        "llm_tokens": {
            "input": row.get(f"{pipeline}_input_tokens", 0),
            "output": row.get(f"{pipeline}_output_tokens", 0),
            "total": row.get(f"{pipeline}_tokens", 0),
        },
        "latency_s": round(float(row.get(f"{pipeline}_latency_ms", 0.0)) / 1000, 3),
        "citations": row.get(f"{pipeline}_retrieved_doc_ids", []),
        "model": row.get(f"{pipeline}_model"),
        "provider": row.get(f"{pipeline}_provider"),
    }
    if pipeline == "graphrag":
        entry["graph"] = {
            key: metadata.get(key)
            for key in ("graph_operation", "graph_plan", "graph_status", "answer_source")
        }
    if pipeline == "agentic":
        entry["trace"] = _public_trace(row.get("agentic_trace"))
    if f"{pipeline}_em" in row:
        entry["scores"] = {
            "exact_match": row[f"{pipeline}_em"],
            "exact_match_strict": row.get(f"{pipeline}_em_strict"),
            "token_f1": row.get(f"{pipeline}_f1"),
        }
    return entry


def _csv_row(question: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {key: question[key] for key in ("qid", "qtype", "question")}
    for pipeline in _PIPELINES:
        entry = question.get(pipeline)
        if entry is None:
            continue
        flat[f"{pipeline}_answer"] = entry["answer"]
        flat[f"{pipeline}_llm_tokens"] = entry["llm_tokens"]["total"]
        flat[f"{pipeline}_latency_s"] = entry["latency_s"]
        flat[f"{pipeline}_citations"] = ";".join(entry["citations"])
        if pipeline == "agentic" and entry.get("trace"):
            trace = entry["trace"]
            flat["agentic_steps"] = trace.get("step_count")
            flat["agentic_llm_calls"] = len(trace.get("llm_calls", []))
            flat["agentic_tools"] = ";".join(trace.get("retrieval_methods", []))
            flat["agentic_agents"] = ";".join(trace.get("agents_invoked", []))
            flat["agentic_strategy_changed"] = trace.get("strategy_changed")
            flat["agentic_stopping_reason"] = trace.get("stopping_reason")
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark results for submission")
    parser.add_argument("--results", required=True, help="results JSONL from src.evaluate")
    parser.add_argument("--out", required=True, help="output path stem, e.g. submission/hidden")
    args = parser.parse_args()

    results_path = Path(args.results)
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    manifest_path = results_path.with_suffix(results_path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    questions = []
    for row in rows:
        question: dict[str, Any] = {key: row[key] for key in ("qid", "qtype", "question")}
        for pipeline in _PIPELINES:
            entry = _pipeline_entry(row, pipeline)
            if entry is not None:
                question[pipeline] = entry
        questions.append(question)

    document = {
        "run": {
            key: manifest.get(key)
            for key in ("model", "provider", "git_commit", "dataset", "started_at", "finished_at")
        },
        "token_accounting": "LLM input+output tokens only; graph queries and retrieval count 0",
        "questions": questions,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")

    flat_rows = [_csv_row(question) for question in questions]
    columns = list(dict.fromkeys(key for row in flat_rows for key in row))
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote {len(questions)} questions to {json_path} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
