# Evaluation and benchmarking engine.
# Runs evaluation sets (public 100 + hidden 50) across RAG, GraphRAG, and Agentic GraphRAG.
# Computes Tier-1 deterministic metrics: Exact Match, Token F1, Gold Doc Recall/Precision/MRR.
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.coprocessor import Coprocessor
from src.graph import GraphClient, connect, create_mock_graph_client
from src.guardrails import normalize
from src.ingest import load_all_chunks
from src.llm import make_session
from src.models import EvalQuestion, PipelineResult
from src.pipelines.agentic import AgenticPipeline
from src.pipelines.graphrag import GraphRAGPipeline
from src.pipelines.rag import RAGPipeline

# ---------------------------------------------------------------------------
# Tier-1 Deterministic Metric Computation
# ---------------------------------------------------------------------------


def compute_exact_match(prediction: str, ground_truths: Sequence[str]) -> float:
    # Normalized exact match: diacritics stripped, lowercased, whitespace normalized.
    pred_norm = normalize(prediction).lower().strip()
    for gt in ground_truths:
        gt_norm = normalize(gt).lower().strip()
        if pred_norm == gt_norm:
            return 1.0
    return 0.0


def compute_token_f1(prediction: str, ground_truths: Sequence[str]) -> float:
    # Token-level F1: handles team events and rephrased names.
    pred_tokens = set(normalize(prediction).lower().replace(",", " ").split())
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = set(normalize(gt).lower().replace(",", " ").split())
        if not gt_tokens:
            continue
        common = pred_tokens & gt_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def compute_mrr(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> float:
    # Mean Reciprocal Rank: rank of the first gold document in retrieved list.
    gold_set = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(
    retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int = 5
) -> float:
    if not gold_doc_ids:
        return 0.0
    gold_set = set(gold_doc_ids)
    retrieved_k = set(retrieved_doc_ids[:k])
    matched = retrieved_k & gold_set
    return len(matched) / len(gold_set)


def compute_precision_at_k(
    retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int = 5
) -> float:
    if not retrieved_doc_ids or k <= 0:
        return 0.0
    gold_set = set(gold_doc_ids)
    retrieved_k = retrieved_doc_ids[:k]
    matched = set(retrieved_k) & gold_set
    return len(matched) / len(retrieved_k)


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------


class EvaluationHarness:
    def __init__(
        self,
        corpus_path: str = "hackathon-resources/corpus/corpus.jsonl",
        provider: str = "cloudflare",
        use_mock: bool = False,
    ) -> None:
        self.corpus_path = corpus_path
        self.provider = provider
        self.coprocessor = Coprocessor()

        # Initialize offline coprocessor by parsing corpus
        corpus_file = Path(corpus_path)
        if corpus_file.exists():
            print(f"Loading corpus from {corpus_path} into coprocessor...", file=sys.stderr)
            t0 = time.perf_counter()
            chunks = load_all_chunks(corpus_file)
            self.coprocessor.build(chunks)
            print(
                f"Coprocessor built with {len(chunks)} chunks in {time.perf_counter() - t0:.2f}s",
                file=sys.stderr,
            )

        # Graph client setup (offline mock or live TigerGraph cluster)
        has_tg = (
            bool(os.environ.get("TG_HOST"))
            and not bool(os.environ.get("TG_USE_MOCK"))
            and not use_mock
        )
        if has_tg:
            try:
                self.graph = GraphClient(conn=connect())
            except Exception as e:
                print(
                    f"Warning: Live TigerGraph connection failed ({e}), using mock graph",
                    file=sys.stderr,
                )
                self.graph = self._make_mock_graph()
        else:
            self.graph = self._make_mock_graph()

    def _make_mock_graph(self) -> GraphClient:
        # High-fidelity offline mock graph client for deterministic offline testing
        return create_mock_graph_client()

    async def evaluate_question(
        self,
        q: EvalQuestion,
        pipelines: list[str],
    ) -> dict[str, PipelineResult]:
        results: dict[str, PipelineResult] = {}
        session = make_session(self.provider)
        async with session:
            if "rag" in pipelines:
                rag_pipe = RAGPipeline(graph=self.graph, llm=session, coprocessor=self.coprocessor)
                results["rag"] = await rag_pipe.run(q.qid, q.question)

            if "graphrag" in pipelines:
                graphrag_pipe = GraphRAGPipeline(graph=self.graph, llm=session)
                results["graphrag"] = await graphrag_pipe.run(q.qid, q.question)

            if "agentic" in pipelines:
                agentic_pipe = AgenticPipeline(
                    graph=self.graph, coprocessor=self.coprocessor, llm=session
                )
                results["agentic"] = await agentic_pipe.run(q.qid, q.question)

        return results

    async def run_benchmark(
        self,
        dataset_path: str,
        pipeline_names: list[str],
        output_path: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        questions: list[EvalQuestion] = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(EvalQuestion.model_validate_json(line))

        if limit:
            questions = questions[:limit]

        print(
            f"Benchmarking {len(questions)} questions across pipelines: {pipeline_names}...",
            file=sys.stderr,
        )

        all_records: list[dict[str, Any]] = []
        stats: dict[str, dict[str, float]] = {
            p: {
                "em": 0.0,
                "f1": 0.0,
                "mrr": 0.0,
                "rec5": 0.0,
                "prec5": 0.0,
                "tokens": 0.0,
                "zero_tokens": 0.0,
                "latency_ms": 0.0,
            }
            for p in pipeline_names
        }

        # Track metrics by question category
        qtypes = ["aggregation", "temporal", "superlative", "multi_hop", "lookup"]
        qtype_counts: dict[str, int] = {qt: 0 for qt in qtypes}
        qtype_stats: dict[str, dict[str, dict[str, float]]] = {
            qt: {p: {"em": 0.0, "tokens": 0.0} for p in pipeline_names} for qt in qtypes
        }

        for idx, q in enumerate(questions, start=1):
            t0 = time.perf_counter()
            pipe_results = await self.evaluate_question(q, pipeline_names)
            elapsed = (time.perf_counter() - t0) * 1000

            qt_norm = q.qtype if q.qtype in qtypes else "lookup"
            qtype_counts[qt_norm] += 1

            record: dict[str, Any] = {
                "qid": q.qid,
                "question": q.question,
                "qtype": q.qtype,
            }

            for p in pipeline_names:
                res = pipe_results[p]
                record[f"{p}_answer"] = res.answer
                record[f"{p}_tokens"] = res.total_llm_tokens
                record[f"{p}_latency_ms"] = res.latency_ms

                if res.agentic_trace:
                    record["agentic_trace"] = res.agentic_trace

                if res.total_llm_tokens == 0:
                    stats[p]["zero_tokens"] += 1.0

                # Compute metrics if ground truth is present
                if q.answer:
                    em = compute_exact_match(res.answer, q.answer)
                    f1 = compute_token_f1(res.answer, q.answer)
                    mrr = compute_mrr(res.retrieved_doc_ids, q.gold_doc_ids or [])
                    rec5 = compute_recall_at_k(res.retrieved_doc_ids, q.gold_doc_ids or [], k=5)
                    prec5 = compute_precision_at_k(res.retrieved_doc_ids, q.gold_doc_ids or [], k=5)

                    record[f"{p}_em"] = em
                    record[f"{p}_f1"] = f1
                    record[f"{p}_mrr"] = mrr
                    record[f"{p}_recall@5"] = rec5
                    record[f"{p}_prec@5"] = prec5

                    stats[p]["em"] += em
                    stats[p]["f1"] += f1
                    stats[p]["mrr"] += mrr
                    stats[p]["rec5"] += rec5
                    stats[p]["prec5"] += prec5

                    qtype_stats[qt_norm][p]["em"] += em
                    qtype_stats[qt_norm][p]["tokens"] += res.total_llm_tokens

                stats[p]["tokens"] += res.total_llm_tokens
                stats[p]["latency_ms"] += res.latency_ms

            all_records.append(record)
            print(
                f"[{idx}/{len(questions)}] {q.qid} ({q.qtype}) in {elapsed:.1f}ms",
                file=sys.stderr,
            )

        n = max(1, len(questions))

        # Overall Matrix Report
        print("\n" + "=" * 90, file=sys.stderr)
        print("  STELLIUM: 3-WAY COMPARATIVE BENCHMARK MATRIX", file=sys.stderr)
        print("=" * 90, file=sys.stderr)
        print(
            f"{'Pipeline':<18} {'EM':<8} {'F1':<8} {'MRR':<8} {'Rec@5':<8} {'Prec@5':<8} {'Avg Tokens':<12} {'0-Tok %':<9} {'Avg Latency'}",
            file=sys.stderr,
        )
        print("-" * 90, file=sys.stderr)
        for p in pipeline_names:
            em_avg = stats[p]["em"] / n
            f1_avg = stats[p]["f1"] / n
            mrr_avg = stats[p]["mrr"] / n
            rec_avg = stats[p]["rec5"] / n
            prec_avg = stats[p]["prec5"] / n
            tok_avg = stats[p]["tokens"] / n
            zero_tok_pct = (stats[p]["zero_tokens"] / n) * 100
            lat_avg = stats[p]["latency_ms"] / n
            print(
                f"{p:<18} {em_avg:<8.3f} {f1_avg:<8.3f} {mrr_avg:<8.3f} {rec_avg:<8.3f} {prec_avg:<8.3f} {tok_avg:<12.1f} {zero_tok_pct:<8.1f}% {lat_avg:.1f}ms",
                file=sys.stderr,
            )
        print("=" * 90, file=sys.stderr)

        # Breakdown by Question Type
        print("\n" + "-" * 90, file=sys.stderr)
        print("  EXACT MATCH & TOKEN CONSUMPTION BY QUESTION CATEGORY", file=sys.stderr)
        print("-" * 90, file=sys.stderr)
        print(
            f"{'Category':<15} {'Count':<7} "
            + " | ".join(f"{p.upper()} (EM / Tok)" for p in pipeline_names),
            file=sys.stderr,
        )
        print("-" * 90, file=sys.stderr)
        for qt in qtypes:
            cnt = qtype_counts[qt]
            if cnt == 0:
                continue
            cols = []
            for p in pipeline_names:
                em_qt = qtype_stats[qt][p]["em"] / cnt
                tok_qt = qtype_stats[qt][p]["tokens"] / cnt
                cols.append(f"{em_qt:.2f} / {tok_qt:4.0f}")
            print(f"{qt.capitalize():<15} {cnt:<7} " + " | ".join(cols), file=sys.stderr)
        print("-" * 90, file=sys.stderr)

        # Comparative ROI Summary
        if "rag" in pipeline_names and "agentic" in pipeline_names:
            em_rag = stats["rag"]["em"] / n
            em_agentic = stats["agentic"]["em"] / n
            delta_acc = (em_agentic - em_rag) * 100
            tok_rag = stats["rag"]["tokens"] / n
            tok_agentic = stats["agentic"]["tokens"] / n
            efficiency = tok_agentic / max(1.0, tok_rag)
            print(
                f"ROI Summary: Agentic Accuracy Delta: +{delta_acc:.1f}% | Token Efficiency Ratio: {efficiency:.2f}x",
                file=sys.stderr,
            )
            print("=" * 90 + "\n", file=sys.stderr)

        # Write output file if specified
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                for r in all_records:
                    f.write(json.dumps(r) + "\n")
            print(f"Results written to {output_path}", file=sys.stderr)

        return all_records


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stellium Agentic GraphRAG Benchmark Runner")
    parser.add_argument(
        "--dataset",
        default="hackathon-resources/questions/eval_public.jsonl",
        help="Path to questions JSONL file",
    )
    parser.add_argument(
        "--pipeline",
        default="all",
        choices=["rag", "graphrag", "agentic", "all"],
        help="Pipeline(s) to benchmark",
    )
    parser.add_argument(
        "--output",
        default="results/benchmark_results.jsonl",
        help="Output JSONL path for benchmark results",
    )
    parser.add_argument(
        "--provider",
        default="cloudflare",
        choices=["cloudflare", "gemini", "mistral"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of questions to evaluate",
    )
    args = parser.parse_args()

    pipelines = ["rag", "graphrag", "agentic"] if args.pipeline == "all" else [args.pipeline]

    harness = EvaluationHarness(provider=args.provider)
    asyncio.run(
        harness.run_benchmark(
            dataset_path=args.dataset,
            pipeline_names=pipelines,
            output_path=args.output,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
