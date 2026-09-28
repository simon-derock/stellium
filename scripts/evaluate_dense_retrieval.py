# Measure live TigerVector retrieval against public gold document evidence.
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.embeddings import JinaEmbeddingClient
from src.graph import GraphClient, connect
from src.models import EvalQuestion


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> dict[str, float]:
    gold = set(gold_doc_ids)
    top = retrieved_doc_ids[:k]
    hits = set(top) & gold
    first_rank = next((rank for rank, doc_id in enumerate(top, 1) if doc_id in gold), 0)
    return {
        "hit": float(bool(hits)),
        "recall": len(hits) / len(gold) if gold else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def _summarize(rows: list[dict[str, Any]], cutoffs: tuple[int, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_question_type": {}}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"].extend(rows)
    for row in rows:
        groups[row["qtype"]].append(row)

    for group_name, group_rows in groups.items():
        metrics: dict[str, float] = {}
        for cutoff in cutoffs:
            for name in ("hit", "recall", "mrr"):
                field = f"{name}@{cutoff}"
                metrics[field] = sum(row[field] for row in group_rows) / len(group_rows)
        target = summary["overall"] if group_name == "overall" else summary["by_question_type"]
        if group_name == "overall":
            target.update(metrics)
            target["questions"] = len(group_rows)
        else:
            target[group_name] = {**metrics, "questions": len(group_rows)}
    return summary


def evaluate_dense_retrieval(
    questions: list[EvalQuestion],
    embedding_client: JinaEmbeddingClient,
    graph: GraphClient,
    top_k: int,
    output_path: Path,
    cutoffs: tuple[int, ...],
    dataset_path: Path,
) -> dict[str, Any]:
    eligible = [question for question in questions if question.gold_doc_ids]
    if not eligible:
        raise ValueError("The selected dataset has no gold_doc_ids for retrieval evaluation")

    embeddings = embedding_client.embed_queries([question.question for question in eligible])
    if len(embeddings) != len(eligible):
        raise RuntimeError("Embedding response count did not match the question count")

    rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as result_file:
        for index, (question, vector) in enumerate(zip(eligible, embeddings, strict=True), 1):
            started = time.perf_counter()
            results = graph.vector_search(vector, top_k=top_k)
            latency_ms = (time.perf_counter() - started) * 1000
            retrieved = [chunk_id.rsplit("#", 1)[0] for chunk_id, _ in results]
            row: dict[str, Any] = {
                "qid": question.qid,
                "qtype": question.qtype,
                "question": question.question,
                "gold_doc_ids": question.gold_doc_ids,
                "retrieved_chunks": [
                    {"chunk_id": chunk_id, "doc_id": chunk_id.rsplit("#", 1)[0], "score": score}
                    for chunk_id, score in results
                ],
                "retrieved_doc_ids": retrieved,
                "latency_ms": latency_ms,
            }
            for cutoff in cutoffs:
                metric = _metrics(retrieved, question.gold_doc_ids or [], cutoff)
                row[f"hit@{cutoff}"] = metric["hit"]
                row[f"recall@{cutoff}"] = metric["recall"]
                row[f"mrr@{cutoff}"] = metric["mrr"]
            rows.append(row)
            result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            result_file.flush()
            print(
                f"[{index}/{len(eligible)}] {question.qid}: {len(results)} chunks "
                f"in {latency_ms:.1f} ms",
                file=sys.stderr,
            )

    summary = _summarize(rows, cutoffs)
    summary["overall"]["empty_result_queries"] = sum(not row["retrieved_chunks"] for row in rows)
    summary["overall"]["mean_returned_chunks"] = sum(
        len(row["retrieved_chunks"]) for row in rows
    ) / len(rows)
    summary["overall"]["mean_search_latency_ms"] = sum(row["latency_ms"] for row in rows) / len(
        rows
    )
    return {
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "question_count": len(eligible),
        "embedding_model": embedding_client.model,
        "embedding_dimension": embedding_client.dimension,
        "embedding_task": "retrieval.query",
        "retrieval": "TigerVector HNSW only",
        "top_k": top_k,
        "cutoffs": list(cutoffs),
        "summary": summary,
        "results_jsonl": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure live TigerVector dense retrieval on public gold evidence."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("hackathon-resources/questions/eval_public.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("results/dense_retrieval.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("results/dense_retrieval_report.json"))
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 30])
    args = parser.parse_args()
    cutoffs = tuple(sorted(set(args.cutoffs)))
    if args.top_k < 1 or not cutoffs or cutoffs[0] < 1 or cutoffs[-1] > args.top_k:
        parser.error("top-k and cutoffs must be positive, with every cutoff <= top-k")

    load_dotenv()
    embedding_client = JinaEmbeddingClient.from_env()
    if not embedding_client.is_configured:
        raise RuntimeError("JINA_API_KEY is required for live dense retrieval evaluation")

    questions = [
        EvalQuestion.model_validate_json(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate_dense_retrieval(
        questions=questions,
        embedding_client=embedding_client,
        graph=GraphClient(conn=connect()),
        top_k=args.top_k,
        output_path=args.output,
        cutoffs=cutoffs,
        dataset_path=args.dataset,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["summary"]["overall"]["empty_result_queries"] == report["question_count"]:
        raise SystemExit(
            "TigerVector returned zero chunks for every query; dense retrieval is unavailable"
        )


if __name__ == "__main__":
    main()
