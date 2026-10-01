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
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.coprocessor import Coprocessor
from src.embeddings import (
    GRAPH_EMBEDDING_DIMENSION,
    embedding_client_from_env,
    is_graph_embedding_compatible,
)
from src.graph import GraphClient, connect, create_mock_graph_client, wait_for_graph_ready
from src.guardrails import normalize
from src.ingest import load_all_chunks
from src.llm import LockedLLMSession, make_session, mistral_api_keys_configured
from src.models import EvalQuestion, PipelineResult
from src.pipelines.agentic import AgenticPipeline
from src.pipelines.graphrag import GraphRAGPipeline
from src.pipelines.rag import RAGPipeline

# ---------------------------------------------------------------------------
# Tier-1 Deterministic Metric Computation
# ---------------------------------------------------------------------------


_ANSWER_PUNCTUATION_MAP = str.maketrans({"‘": "'", "’": "'", "–": "-", "—": "-", "−": "-"})


def normalize_answer(text: str) -> str:
    # Compare answer content, not typography: diacritics, case, punctuation, and spacing are
    # ignored, so "Men’s épée." equals "Men's epee" and gold team strings written without
    # separators ("Dani KingLaura Trott") equal comma-separated predictions.
    folded = normalize(text.translate(_ANSWER_PUNCTUATION_MAP)).casefold()
    return "".join(char for char in folded if char.isalnum())


def compute_exact_match(prediction: str, ground_truths: Sequence[str]) -> float:
    # Normalized exact match over answer content (see normalize_answer).
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return 0.0
    return float(any(pred_norm == normalize_answer(gt) for gt in ground_truths))


def compute_strict_exact_match(prediction: str, ground_truths: Sequence[str]) -> float:
    # Legacy metric kept for comparison with earlier audits: punctuation-sensitive equality.
    pred_norm = normalize(prediction).lower().strip()
    return float(any(pred_norm == normalize(gt).lower().strip() for gt in ground_truths))


def compute_token_f1(prediction: str, ground_truths: Sequence[str]) -> float:
    # Multiset token F1: repeated terms count once per occurrence, not as a set.
    pred_tokens = Counter(normalize(prediction).lower().replace(",", " ").split())
    pred_count = sum(pred_tokens.values())
    if pred_count == 0:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = Counter(normalize(gt).lower().replace(",", " ").split())
        gt_count = sum(gt_tokens.values())
        if gt_count == 0:
            continue
        overlap = sum((pred_tokens & gt_tokens).values())
        if overlap == 0:
            continue
        precision = overlap / pred_count
        recall = overlap / gt_count
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


def _load_checkpoint(
    out_file: Path,
    questions: list[EvalQuestion],
    pipeline_names: list[str],
) -> dict[str, dict[str, Any]]:
    # Reject stale or malformed files rather than mixing results from different runs.
    question_by_id = {question.qid: question for question in questions}
    completed: dict[str, dict[str, Any]] = {}
    with out_file.open(encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, start=1):
            if not line.strip():
                continue
            try:
                saved_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot resume malformed JSONL record at {out_file}:{line_number}"
                ) from exc
            qid = saved_record.get("qid")
            question = question_by_id.get(qid)
            if question is None or saved_record.get("question") != question.question:
                raise ValueError(f"Resume artifact {out_file} does not match the selected dataset")
            if qid in completed:
                raise ValueError(f"Duplicate question {qid} in resume artifact {out_file}")
            if any(f"{pipeline}_answer" not in saved_record for pipeline in pipeline_names):
                raise ValueError(f"Resume artifact {out_file} lacks a requested pipeline result")
            completed[qid] = saved_record
    return completed


def _summarize_records(
    questions: list[EvalQuestion],
    records: list[dict[str, Any]],
    pipeline_names: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, int], dict[str, dict[str, dict[str, float]]]]:
    # Recompute report metrics from the durable result rows after fresh or resumed runs.
    qtypes = ["aggregation", "temporal", "superlative", "multi_hop", "lookup"]
    stats = {
        pipeline: {
            metric: 0.0
            for metric in (
                "em",
                "f1",
                "mrr",
                "rec5",
                "prec5",
                "tokens",
                "zero_tokens",
                "latency_ms",
            )
        }
        for pipeline in pipeline_names
    }
    qtype_counts = {qtype: 0 for qtype in qtypes}
    qtype_stats = {
        qtype: {pipeline: {"em": 0.0, "tokens": 0.0} for pipeline in pipeline_names}
        for qtype in qtypes
    }
    metric_fields = {
        "em": "em",
        "f1": "f1",
        "mrr": "mrr",
        "rec5": "recall@5",
        "prec5": "prec@5",
    }
    for question, record in zip(questions, records, strict=True):
        qtype = question.qtype if question.qtype in qtypes else "lookup"
        qtype_counts[qtype] += 1
        for pipeline in pipeline_names:
            tokens = float(record.get(f"{pipeline}_tokens", 0.0))
            stats[pipeline]["tokens"] += tokens
            stats[pipeline]["latency_ms"] += float(record.get(f"{pipeline}_latency_ms", 0.0))
            stats[pipeline]["zero_tokens"] += float(tokens == 0)
            for metric, field_name in metric_fields.items():
                value = record.get(f"{pipeline}_{field_name}")
                if value is not None:
                    stats[pipeline][metric] += float(value)
            if f"{pipeline}_em" in record:
                qtype_stats[qtype][pipeline]["em"] += float(record[f"{pipeline}_em"])
                qtype_stats[qtype][pipeline]["tokens"] += tokens
    return stats, qtype_counts, qtype_stats


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------


class EvaluationHarness:
    def __init__(
        self,
        corpus_path: str = "hackathon-resources/corpus/corpus.jsonl",
        provider: str = "cloudflare",
        use_mock: bool = False,
        session_factory: Callable[[str], LockedLLMSession] | None = None,
    ) -> None:
        # Load local credentials before selecting the graph client or creating LLM sessions.
        load_dotenv()
        if not use_mock:
            provider_credentials = {
                "cloudflare": ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"),
                "gemini": ("GEMINI_API_KEY",),
                "cohere": (
                    "COHERE_CHAT_API_KEY",
                    "COHERE",
                    "COHERE_BACKUP",
                    "COHERE_KEY",
                    "COHERE_KEYS",
                ),
            }
            if provider == "mistral":
                provider_is_configured = mistral_api_keys_configured()
                names = "KEY_ONE..KEY_TWENTYTHREE or MISTRAL_API_KEY"
            else:
                required_credentials = provider_credentials[provider]
                provider_is_configured = any(
                    os.environ.get(name, "").strip() for name in required_credentials
                )
                names = " or ".join(required_credentials)
            if not provider_is_configured:
                raise RuntimeError(
                    f"Live evaluation requires one credential for provider {provider}: {names}"
                )
            if not os.environ.get("TG_HOST"):
                raise RuntimeError("Live evaluation requires TG_HOST; use_mock is for tests only")
            embedding_client = embedding_client_from_env()
            if not embedding_client.is_configured:
                raise RuntimeError(
                    "Live evaluation requires the embedding API key used by the indexed embeddings"
                )
            if not is_graph_embedding_compatible(embedding_client):
                raise RuntimeError(
                    "Live evaluation query embeddings must match the TigerGraph index model "
                    "and dimension: "
                    f"{GRAPH_EMBEDDING_DIMENSION}"
                )

        self.corpus_path = corpus_path
        self.provider = provider
        self.session_factory = session_factory or make_session
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

        # Offline mocks are opt-in for tests; evaluation never silently scores a mock graph.
        if use_mock:
            self.graph = self._make_mock_graph()
        else:
            waited_s = wait_for_graph_ready(os.environ["TG_HOST"])
            print(f"TigerGraph workspace ready after {waited_s:.1f}s", file=sys.stderr)
            self.graph = GraphClient(conn=connect())

    def _make_mock_graph(self) -> GraphClient:
        # High-fidelity offline mock graph client for deterministic offline testing
        return create_mock_graph_client()

    async def evaluate_question(
        self,
        q: EvalQuestion,
        pipelines: list[str],
    ) -> dict[str, PipelineResult]:
        results: dict[str, PipelineResult] = {}
        session = self.session_factory(self.provider)
        async with session:
            if "rag" in pipelines:
                rag_pipe = RAGPipeline(graph=self.graph, llm=session, coprocessor=self.coprocessor)
                results["rag"] = await rag_pipe.run(q.qid, q.question)

            if "graphrag" in pipelines:
                graphrag_pipe = GraphRAGPipeline(
                    graph=self.graph, llm=session, coprocessor=self.coprocessor
                )
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
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        questions: list[EvalQuestion] = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(EvalQuestion.model_validate_json(line))

        if limit:
            questions = questions[:limit]

        out_file = Path(output_path) if output_path else None
        completed_records: dict[str, dict[str, Any]] = {}
        if out_file:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if resume and out_file.exists():
                completed_records = _load_checkpoint(out_file, questions, pipeline_names)
            else:
                out_file.write_text("", encoding="utf-8")

        print(
            f"Benchmarking {len(questions)} questions across pipelines: {pipeline_names}...",
            file=sys.stderr,
        )

        all_records: list[dict[str, Any]] = []

        for idx, q in enumerate(questions, start=1):
            if q.qid in completed_records:
                restored_record = completed_records[q.qid]
                all_records.append(restored_record)
                print(
                    f"[{idx}/{len(questions)}] {q.qid} restored from checkpoint",
                    file=sys.stderr,
                )
                continue

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
                record[f"{p}_input_tokens"] = res.llm_input_tokens
                record[f"{p}_output_tokens"] = res.llm_output_tokens
                record[f"{p}_context_tokens"] = res.context_tokens
                record[f"{p}_latency_ms"] = res.latency_ms
                record[f"{p}_model"] = res.model_name
                record[f"{p}_provider"] = res.provider
                record[f"{p}_retrieved_doc_ids"] = res.retrieved_doc_ids

                if res.agentic_trace:
                    record["agentic_trace"] = res.agentic_trace

                # Compute metrics if ground truth is present
                if q.answer:
                    em = compute_exact_match(res.answer, q.answer)
                    record[f"{p}_em_strict"] = compute_strict_exact_match(res.answer, q.answer)
                    f1 = compute_token_f1(res.answer, q.answer)
                    mrr = compute_mrr(res.retrieved_doc_ids, q.gold_doc_ids or [])
                    rec5 = compute_recall_at_k(res.retrieved_doc_ids, q.gold_doc_ids or [], k=5)
                    prec5 = compute_precision_at_k(res.retrieved_doc_ids, q.gold_doc_ids or [], k=5)

                    record[f"{p}_em"] = em
                    record[f"{p}_f1"] = f1
                    record[f"{p}_mrr"] = mrr
                    record[f"{p}_recall@5"] = rec5
                    record[f"{p}_prec@5"] = prec5

            all_records.append(record)
            if out_file:
                with out_file.open("a", encoding="utf-8") as result_file:
                    result_file.write(json.dumps(record) + "\n")
                    result_file.flush()
            print(
                f"[{idx}/{len(questions)}] {q.qid} ({q.qtype}) in {elapsed:.1f}ms",
                file=sys.stderr,
            )

        stats, qtype_counts, qtype_stats = _summarize_records(
            questions, all_records, pipeline_names
        )
        qtypes = list(qtype_counts)

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

        if output_path:
            print(f"Results checkpointed to {output_path}", file=sys.stderr)

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
        choices=["cloudflare", "gemini", "mistral", "cohere", "offline"],
        help="LLM provider; 'offline' runs on the mock graph with no network (CI smoke only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of questions to evaluate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume missing questions from existing JSONL results at --output",
    )
    args = parser.parse_args()

    pipelines = ["rag", "graphrag", "agentic"] if args.pipeline == "all" else [args.pipeline]

    harness = EvaluationHarness(provider=args.provider, use_mock=args.provider == "offline")
    asyncio.run(
        harness.run_benchmark(
            dataset_path=args.dataset,
            pipeline_names=pipelines,
            output_path=args.output,
            limit=args.limit,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
