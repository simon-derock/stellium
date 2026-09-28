# Compare sparse tokenization variants on the public question set using gold-document hits.
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Plus

from src.coprocessor import _tokenize
from src.guardrails import normalize
from src.ingest import load_all_chunks
from src.models import Chunk

Tokenize = Callable[[str], list[str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluate(
    chunks: list[Chunk],
    questions: list[dict[str, Any]],
    tokenize: Tokenize,
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    index = BM25Plus([tokenize(chunk.raw_text) for chunk in chunks])
    max_k = max(cutoffs)
    results: list[dict[str, Any]] = []

    for question in questions:
        scores = index.get_scores(tokenize(question["question"]))
        candidates = heapq.nlargest(
            max_k,
            ((chunk_index, float(score)) for chunk_index, score in enumerate(scores) if score > 0),
            key=lambda item: item[1],
        )
        gold = set(question.get("gold_doc_ids") or [])
        ranked_docs = list(
            dict.fromkeys(chunks[chunk_index].doc_id for chunk_index, _ in candidates)
        )
        first_gold_rank = next(
            (rank for rank, doc_id in enumerate(ranked_docs, start=1) if doc_id in gold), None
        )
        results.append(
            {
                "qid": str(question.get("qid", "unknown")),
                "qtype": str(question.get("qtype", "unknown")),
                "gold_doc_ids": gold,
                "ranked_chunk_doc_ids": [
                    chunks[chunk_index].doc_id for chunk_index, _ in candidates
                ],
                "ranked_unique_doc_ids": ranked_docs,
                "first_gold_doc_rank": first_gold_rank,
            }
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["qtype"]].append(result)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {"questions": len(group)}
        for cutoff in cutoffs:
            hit_count = 0
            recall_sum = 0.0
            reciprocal_rank_sum = 0.0
            for result in group:
                top_docs = set(result["ranked_unique_doc_ids"][:cutoff])
                gold = result["gold_doc_ids"]
                hit_count += bool(top_docs & gold)
                recall_sum += len(top_docs & gold) / len(gold) if gold else 0.0
                rank = result["first_gold_doc_rank"]
                reciprocal_rank_sum += 1 / rank if rank is not None and rank <= cutoff else 0.0
            metrics[f"hit_rate@{cutoff}"] = hit_count / len(group)
            metrics[f"doc_recall@{cutoff}"] = recall_sum / len(group)
            metrics[f"doc_mrr@{cutoff}"] = reciprocal_rank_sum / len(group)
        return metrics

    return {
        "overall": summarize(results),
        "by_question_type": {qtype: summarize(group) for qtype, group in sorted(by_type.items())},
        "per_question": [
            {
                "qid": result["qid"],
                "qtype": result["qtype"],
                "gold_doc_ids": sorted(result["gold_doc_ids"]),
                "top_ranked_chunk_doc_ids": result["ranked_chunk_doc_ids"][:max_k],
                "top_ranked_unique_doc_ids": result["ranked_unique_doc_ids"][:max_k],
                "first_gold_doc_rank": result["first_gold_doc_rank"],
                "doc_recall_by_cutoff": {
                    str(cutoff): (
                        len(set(result["ranked_unique_doc_ids"][:cutoff]) & result["gold_doc_ids"])
                        / len(result["gold_doc_ids"])
                        if result["gold_doc_ids"]
                        else 0.0
                    )
                    for cutoff in cutoffs
                },
            }
            for result in results
        ],
        "misses_at_k": [
            result["qid"]
            for result in results
            if not set(result["ranked_unique_doc_ids"][:max_k]) & result["gold_doc_ids"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure BM25 sparse candidate recall on a JSONL question set."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("hackathon-resources/questions/eval_public.jsonl")
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("hackathon-resources/corpus/corpus.jsonl")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 30])
    args = parser.parse_args()

    cutoffs = tuple(sorted(set(args.cutoffs)))
    if not cutoffs or cutoffs[0] < 1:
        parser.error("--cutoffs must contain positive integers")

    questions = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines()]
    chunks = load_all_chunks(args.corpus)
    methods: dict[str, Tokenize] = {
        "legacy_whitespace": lambda text: normalize(text).lower().split(),
        "punctuation_normalized": _tokenize,
    }
    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "retrieval_unit": (
            "BM25 ranks chunks; metrics deduplicate their document IDs before doc-level cutoffs"
        ),
        "cutoffs": list(cutoffs),
        "methods": {
            name: _evaluate(chunks, questions, tokenize, cutoffs)
            for name, tokenize in methods.items()
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
