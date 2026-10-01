# Measure local cross-encoder impact on BM25 candidate ordering for public gold-document coverage.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.coprocessor import BM25Index, LocalCrossEncoder
from src.ingest import load_all_chunks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _memory_profile() -> dict[str, float] | None:
    # Linux reports current and high-water RSS directly in procfs.
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    memory = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _unit = line.split()
            memory[key.rstrip(":")] = round(int(value) / 1024, 1)
    return memory


def _record(metrics: dict[str, float], docs: list[str], gold: set[str]) -> None:
    relevant = set(docs) & gold
    metrics["hit_count"] += bool(relevant)
    metrics["recall_sum"] += len(relevant) / len(gold) if gold else 0.0
    first_rank = next((rank for rank, doc_id in enumerate(docs, start=1) if doc_id in gold), None)
    metrics["reciprocal_rank_sum"] += 1 / first_rank if first_rank is not None else 0.0


def _summarize(metrics: dict[str, float], question_count: int) -> dict[str, float]:
    return {
        "questions": question_count,
        "hit_rate": round(metrics["hit_count"] / question_count, 4),
        "mean_gold_document_recall": round(metrics["recall_sum"] / question_count, 4),
        "mean_reciprocal_rank": round(metrics["reciprocal_rank_sum"] / question_count, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BM25 top-k with local cross-encoder reranking on public gold documents."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("hackathon-resources/questions/eval_public.jsonl")
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("hackathon-resources/corpus/corpus.jsonl")
    )
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.candidate_k < 1 or args.top_k < 1 or args.top_k > args.candidate_k:
        parser.error("require 1 <= --top-k <= --candidate-k")

    questions = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_questions is not None:
        if args.max_questions < 1:
            parser.error("--max-questions must be positive")
        questions = questions[: args.max_questions]
    chunks = load_all_chunks(args.corpus)
    index = BM25Index()
    index.build(chunks)
    index_memory = _memory_profile()
    reranker = LocalCrossEncoder.load()
    model_memory = _memory_profile()

    overall: dict[str, dict[str, float]] = defaultdict(
        lambda: {"hit_count": 0.0, "recall_sum": 0.0, "reciprocal_rank_sum": 0.0}
    )
    by_category: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(
            lambda: {"hit_count": 0.0, "recall_sum": 0.0, "reciprocal_rank_sum": 0.0}
        )
    )
    latencies: list[float] = []
    started = time.perf_counter()
    for position, question in enumerate(questions, start=1):
        text = str(question["question"])
        gold = set(question.get("gold_doc_ids") or [])
        category = str(question.get("qtype", "unknown"))
        candidates = [chunk for chunk, _score in index.search(text, top_k=args.candidate_k)]
        ranked = candidates[: args.top_k]
        before = list(dict.fromkeys(chunk.doc_id for chunk in ranked))
        rerank_started = time.perf_counter()
        scores = reranker.predict([(text, chunk.raw_text) for chunk in candidates])
        latencies.append((time.perf_counter() - rerank_started) * 1000)
        reranked = [
            candidates[index]
            for index in sorted(range(len(scores)), key=lambda item: (-scores[item], item))[
                : args.top_k
            ]
        ]
        candidate_docs = list(dict.fromkeys(chunk.doc_id for chunk in candidates))
        after = list(dict.fromkeys(chunk.doc_id for chunk in reranked))
        for name, docs in (
            ("candidate_pool", candidate_docs),
            ("bm25_top_k", before),
            ("reranked_top_k", after),
        ):
            _record(overall[name], docs, gold)
            _record(by_category[category][name], docs, gold)
        if position % 20 == 0:
            print(f"Processed {position}/{len(questions)} public questions", flush=True)

    sorted_latencies = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "retrieval_unit": "chunks ranked, then deduplicated by gold document ID",
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "reranker_model": reranker.model_name,
        "reranker_batch_size": max(1, min(16, int(os.environ.get("RERANKER_BATCH_SIZE", "1")))),
        "reranker_cpu_threads": max(1, int(os.environ.get("RERANKER_CPU_THREADS", "2"))),
        "retrieval_metrics": {
            name: _summarize(metrics, len(questions)) for name, metrics in sorted(overall.items())
        },
        "metrics_by_question_type": {
            category: {
                name: _summarize(
                    metrics, sum(1 for item in questions if item.get("qtype") == category)
                )
                for name, metrics in sorted(groups.items())
            }
            for category, groups in sorted(by_category.items())
        },
        "reranker_latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "p95": round(sorted_latencies[p95_index], 1),
            "total_seconds": round(sum(latencies) / 1000, 2),
        },
        "memory_mib": {
            "after_corpus_and_bm25_build": index_memory,
            "after_model_load_and_queries": _memory_profile(),
            "after_model_load_before_reranking": model_memory,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "scope_note": (
            "Gold-document retrieval coverage only; this is not answer EM, completeness, or a "
            "full dense-plus-sparse hybrid answer benchmark."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
