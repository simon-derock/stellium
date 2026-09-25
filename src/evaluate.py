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
from src.graph import GraphClient, connect
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


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------


class EvaluationHarness:
    def __init__(
        self,
        corpus_path: str = "hackathon-resources/corpus/corpus.jsonl",
        provider: str = "cloudflare",
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
        has_tg = bool(os.environ.get("TG_HOST"))
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
        class MockConnection:
            def runInstalledQuery(
                self, query_name: str, params: dict[str, Any] | None = None, **_: Any
            ) -> list[dict[str, Any]]:
                params = params or {}
                if query_name == "get_event_aggregates":
                    return [
                        {"count": 5, "events": ["Biathlon 10km"], "gold_doc_ids": ["Q47091419"]}
                    ]
                elif query_name == "get_preceding_event":
                    return [
                        {
                            "prev_events": ["Athletics 20km walk 2012"],
                            "gold_athletes": ["Chen Ding"],
                            "gold_doc_ids": ["Q1050909"],
                        }
                    ]
                elif query_name == "get_superlative_event":
                    return [
                        {
                            "events": ["Athletics at the 2008 Summer Olympics – Men's marathon"],
                            "competitor_counts": [98],
                            "gold_doc_ids": ["Q1005784"],
                        }
                    ]
                elif query_name == "get_event_by_venue_date":
                    return [
                        {
                            "events": ["Weightlifting 60kg"],
                            "gold_athletes": ["Naim Süleymanoğlu"],
                            "gold_doc_ids": ["Q25239316"],
                        }
                    ]
                elif query_name == "get_event_attribute":
                    return [
                        {
                            "events": ["Men's foil"],
                            "competitor_counts": [68],
                            "nation_counts": [26],
                            "gold_athletes": ["Stefano Cerioni"],
                            "venues": ["Fencing Gymnasium"],
                            "gold_doc_ids": ["Q12345"],
                        }
                    ]
                elif query_name == "vector_search_chunks":
                    return [
                        {
                            "TopChunks": [
                                {
                                    "chunk_id": "c1",
                                    "doc_id": "d1",
                                    "text": "sample",
                                    "raw_text": "sample",
                                    "prev_chunk_id": "",
                                    "next_chunk_id": "",
                                }
                            ]
                        },
                        {"@@distances": {"c1": 0.1}},
                    ]
                return [{}]

        client = GraphClient.__new__(GraphClient)
        client.conn = MockConnection()
        return client

    async def evaluate_question(
        self,
        q: EvalQuestion,
        pipelines: list[str],
    ) -> dict[str, PipelineResult]:
        results: dict[str, PipelineResult] = {}
        session = make_session(self.provider)
        async with session:
            if "rag" in pipelines:
                rag_pipe = RAGPipeline(graph=self.graph, llm=session)
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
            p: {"em": 0.0, "f1": 0.0, "mrr": 0.0, "tokens": 0.0, "latency_ms": 0.0}
            for p in pipeline_names
        }

        for idx, q in enumerate(questions, start=1):
            t0 = time.perf_counter()
            pipe_results = await self.evaluate_question(q, pipeline_names)
            elapsed = (time.perf_counter() - t0) * 1000

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

                # Compute metrics if ground truth is present
                if q.answer:
                    em = compute_exact_match(res.answer, q.answer)
                    f1 = compute_token_f1(res.answer, q.answer)
                    mrr = compute_mrr(res.retrieved_doc_ids, q.gold_doc_ids or [])
                    record[f"{p}_em"] = em
                    record[f"{p}_f1"] = f1
                    record[f"{p}_mrr"] = mrr

                    stats[p]["em"] += em
                    stats[p]["f1"] += f1
                    stats[p]["mrr"] += mrr
                stats[p]["tokens"] += res.total_llm_tokens
                stats[p]["latency_ms"] += res.latency_ms

            all_records.append(record)
            print(
                f"[{idx}/{len(questions)}] {q.qid} ({q.qtype}) completed in {elapsed:.1f}ms",
                file=sys.stderr,
            )

        # Average statistics
        n = max(1, len(questions))
        print("\n" + "=" * 65, file=sys.stderr)
        print(
            f"{'Pipeline':<15} {'EM':<8} {'F1':<8} {'MRR':<8} {'Avg Tokens':<12} {'Avg Latency'}",
            file=sys.stderr,
        )
        print("-" * 65, file=sys.stderr)
        for p in pipeline_names:
            em_avg = stats[p]["em"] / n
            f1_avg = stats[p]["f1"] / n
            mrr_avg = stats[p]["mrr"] / n
            tok_avg = stats[p]["tokens"] / n
            lat_avg = stats[p]["latency_ms"] / n
            print(
                f"{p:<15} {em_avg:<8.3f} {f1_avg:<8.3f} {mrr_avg:<8.3f} {tok_avg:<12.1f} {lat_avg:.1f}ms",
                file=sys.stderr,
            )
        print("=" * 65 + "\n", file=sys.stderr)

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
