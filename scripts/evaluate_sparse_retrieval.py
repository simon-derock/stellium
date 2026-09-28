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
    ranks: list[tuple[str, int | None]] = []

    for question in questions:
        scores = index.get_scores(tokenize(question["question"]))
        candidates = heapq.nlargest(
            max_k,
            ((chunk_index, float(score)) for chunk_index, score in enumerate(scores) if score > 0),
            key=lambda item: item[1],
        )
        gold = set(question.get("gold_doc_ids", []))
        rank = next(
            (
                candidate_rank
                for candidate_rank, (chunk_index, _) in enumerate(candidates, start=1)
                if chunks[chunk_index].doc_id in gold
            ),
            None,
        )
        ranks.append((str(question.get("qtype", "unknown")), rank))

    by_type: dict[str, list[int | None]] = defaultdict(list)
    for qtype, rank in ranks:
        by_type[qtype].append(rank)

    def summarize(group: list[int | None]) -> dict[str, Any]:
        result: dict[str, Any] = {
            f"recall@{cutoff}": sum(rank is not None and rank <= cutoff for rank in group)
            / len(group)
            for cutoff in cutoffs
        }
        result[f"mrr@{max_k}"] = sum(1 / rank if rank is not None else 0 for rank in group) / len(
            group
        )
        result["questions"] = len(group)
        return result

    return {
        "overall": summarize([rank for _, rank in ranks]),
        "by_question_type": {qtype: summarize(group) for qtype, group in sorted(by_type.items())},
        "misses_at_k": [
            str(questions[index]["qid"])
            for index, (_, rank) in enumerate(ranks)
            if rank is None or rank > max_k
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
        "retrieval_unit": "ranked chunks; a hit means chunk.doc_id is in gold_doc_ids",
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
