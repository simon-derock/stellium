# Summarise benchmark results. Metrics are computed once into a plain document; the markdown
# tables and the dashboard JSON are two renderings of it. Covers accuracy, tokens, latency, and
# cost per pipeline and question type; retrieval quality (Hit@k, MRR, nDCG, context recall and
# precision); grounding and candidate recall; and agent behaviour.
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import log2
from pathlib import Path
from statistics import mean, median
from typing import Any

_PIPELINES = ("rag", "graphrag", "agentic")
_LABELS = {"rag": "RAG", "graphrag": "GraphRAG", "agentic": "Agentic GraphRAG"}
# Cohere Command A list price per million tokens (input, output); an estimate, not a bill.
_PRICE_PER_MILLION = (2.50, 10.00)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _token_sources(row: dict[str, Any], p: str) -> list[str]:
    # Where each LLM call's counts came from; rows older than this field read as "reported".
    if p == "agentic":
        calls = (row.get("agentic_trace") or {}).get("llm_calls", [])
        return [str(call.get("token_source", "reported")) for call in calls]
    metadata = row.get(f"{p}_retrieval_metadata") or {}
    return [str(source) for source in metadata.get("token_sources", ["reported"])]


def _overall(rows: list[dict[str, Any]], p: str) -> dict[str, Any]:
    scored = [row for row in rows if f"{p}_em" in row]
    latencies = [row[f"{p}_latency_ms"] / 1000 for row in rows]
    cost = (
        sum(
            row.get(f"{p}_input_tokens", 0) * _PRICE_PER_MILLION[0]
            + row.get(f"{p}_output_tokens", 0) * _PRICE_PER_MILLION[1]
            for row in rows
        )
        / 1e6
    )
    return {
        "questions": len(rows),
        "scored": len(scored),
        "exact_match": sum(row[f"{p}_em"] for row in scored),
        "exact_match_strict": sum(row.get(f"{p}_em_strict", 0) for row in scored),
        "token_f1": _mean([row.get(f"{p}_f1", 0.0) for row in scored]),
        "tokens_mean": mean(row[f"{p}_tokens"] for row in rows),
        "latency_mean_s": mean(latencies),
        "latency_p50_s": _percentile(latencies, 0.5),
        "latency_p95_s": _percentile(latencies, 0.95),
        "cost_per_100_usd": cost * 100 / len(rows),
        "citations_mean": mean(len(row[f"{p}_retrieved_doc_ids"]) for row in rows),
        "estimated_token_rows": sum("estimated" in _token_sources(row, p) for row in rows),
    }


def _by_type(rows: list[dict[str, Any]], p: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if f"{p}_em" in row:
            groups[row["qtype"]].append(row)
    return {
        qtype: {
            "questions": len(group),
            "exact_match": sum(row[f"{p}_em"] for row in group),
            "tokens_mean": mean(row[f"{p}_tokens"] for row in group),
        }
        for qtype, group in sorted(groups.items())
    }


def _evidence(
    rows: list[dict[str, Any]], p: str, gold: dict[str, list[str]], articles: dict[str, str]
) -> dict[str, Any]:
    # Candidate recall credits an honest tie report that contains the gold answer; grounding asks
    # whether every non-numeric claim is stated in an article that pipeline cited (title
    # included, spacing ignored, since team names appear glued together in the source).
    from src.evaluate import normalize_answer
    from src.linking import compact

    recall = grounded = checkable = abstained = 0
    for row in rows:
        answer = str(row[f"{p}_answer"])
        if normalize_answer(answer) == "notfoundincorpus":
            abstained += 1
            continue
        parts = [part.strip() for part in answer.split(";") if part.strip()]
        golds = {normalize_answer(item) for item in gold.get(row["qid"], [])}
        recall += any(normalize_answer(part) in golds for part in parts)
        if parts and not all(part.replace(",", "").isdigit() for part in parts):
            cited = "".join(compact(articles.get(doc, "")) for doc in row[f"{p}_retrieved_doc_ids"])
            checkable += 1
            grounded += all(compact(part) in cited for part in parts)
    return {
        "gold_among_answers": recall,
        "grounded": grounded,
        "grounding_checked": checkable,
        "abstained": abstained,
    }


def _retrieval(
    rows: list[dict[str, Any]],
    p: str,
    gold: dict[str, list[str]],
    gold_docs: dict[str, list[str]],
    articles: dict[str, str],
) -> dict[str, Any]:
    # IR metrics over the pipeline's ranked citations, plus context recall (gold answer stated in
    # the retrieved articles) and context precision (average precision of gold documents in top 5).
    from src.linking import compact

    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        relevant = set(gold_docs.get(row["qid"], []))
        if not relevant:
            continue
        ranked = list(dict.fromkeys(row[f"{p}_retrieved_doc_ids"]))
        top = ranked[:5]
        hits = [doc in relevant for doc in top]
        first = next((rank for rank, doc in enumerate(ranked, 1) if doc in relevant), None)
        ideal = sum(1 / log2(rank + 1) for rank in range(1, min(len(relevant), 5) + 1))
        precisions = [sum(hits[:k]) / k for k in range(1, len(top) + 1) if hits[k - 1]]
        scores["hit_at_1"].append(float(bool(top) and hits[0]))
        scores["hit_at_5"].append(float(any(hits)))
        scores["mrr"].append(1 / first if first else 0.0)
        scores["ndcg_at_5"].append(
            sum(1 / log2(rank + 1) for rank, hit in enumerate(hits, 1) if hit) / ideal
        )
        scores["recall_at_5"].append(len(relevant & set(top)) / len(relevant))
        scores["precision_at_5"].append(sum(hits) / len(top) if top else 0.0)
        scores["context_precision_at_5"].append(
            sum(precisions) / min(len(relevant), 5) if precisions else 0.0
        )
        answers = [a for a in gold.get(row["qid"], []) if not a.replace(",", "").isdigit()]
        if answers:
            context = "".join(compact(articles.get(doc, "")) for doc in ranked)
            scores["context_recall"].append(
                float(any(compact(answer) in context for answer in answers))
            )
    result: dict[str, Any] = {key: _mean(values) for key, values in scores.items()}
    result["context_recall_questions"] = len(scores["context_recall"])
    return result


def _agent(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    traces = [row["agentic_trace"] for row in rows if row.get("agentic_trace")]
    if not traces:
        return None
    tools: Counter[str] = Counter()
    agents: Counter[str] = Counter()
    stops: Counter[str] = Counter()
    for trace in traces:
        tools.update(trace.get("retrieval_methods", []))
        agents.update(trace.get("agents_invoked", []))
        stops[str(trace.get("stopping_reason", "")).split(" stated")[0]] += 1
    llm_calls = [len(trace.get("llm_calls", [])) for trace in traces]
    steps = [trace.get("step_count", 0) for trace in traces]
    return {
        "questions": len(traces),
        "steps_mean": mean(steps),
        "steps_median": median(steps),
        "steps_max": max(steps),
        "llm_calls_mean": mean(llm_calls),
        "single_call_answers": sum(calls == 1 for calls in llm_calls),
        "strategy_changes": sum(bool(trace.get("strategy_changed")) for trace in traces),
        "tools": dict(tools.most_common()),
        "specialists": dict(agents.most_common()),
        "stopping_reasons": dict(stops.most_common()),
    }


def collect_metrics(
    rows: list[dict[str, Any]],
    gold: dict[str, list[str]] | None = None,
    gold_docs: dict[str, list[str]] | None = None,
    articles: dict[str, str] | None = None,
) -> dict[str, Any]:
    pipelines = [p for p in _PIPELINES if any(f"{p}_answer" in row for row in rows)]
    document: dict[str, Any] = {"pipelines": {}, "agent": _agent(rows)}
    for p in pipelines:
        entry: dict[str, Any] = {
            "label": _LABELS[p],
            "overall": _overall(rows, p),
            "by_type": _by_type(rows, p),
        }
        if gold is not None and articles is not None:
            entry["evidence"] = _evidence(rows, p, gold, articles)
            if gold_docs:
                entry["retrieval"] = _retrieval(rows, p, gold, gold_docs, articles)
        document["pipelines"][p] = entry
    misses = [
        {"qid": row["qid"], "qtype": row["qtype"], "pipeline": p, "answer": row[f"{p}_answer"]}
        for row in rows
        for p in pipelines
        if p != "rag" and f"{p}_em" in row and not row[f"{p}_em"]
    ]
    document["graph_pipeline_misses"] = misses
    return document


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _pct(numerator: float, denominator: int) -> str:
    return f"{100 * numerator / denominator:.0f}%" if denominator else "n/a"


def _fmt(value: float | None, pattern: str = "{:.3f}") -> str:
    return pattern.format(value) if value is not None else "n/a"


def render_markdown(metrics: dict[str, Any]) -> str:
    pipelines: dict[str, dict[str, Any]] = metrics["pipelines"]
    lines = ["| Pipeline | Exact match | Strict EM | Token F1 | LLM tokens / q | Latency / q |"]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for entry in pipelines.values():
        o = entry["overall"]
        lines.append(
            f"| {entry['label']} | {o['exact_match']:.0f}/{o['scored']} "
            f"({_pct(o['exact_match'], o['scored'])}) | {_pct(o['exact_match_strict'], o['scored'])} "
            f"| {_fmt(o['token_f1'])} | {o['tokens_mean']:,.0f} | {o['latency_mean_s']:.2f} s |"
        )

    estimated = {
        entry["label"]: entry["overall"].get("estimated_token_rows", 0)
        for entry in pipelines.values()
    }
    if any(estimated.values()):
        lines.append(
            "\nToken counts estimated, not provider-reported, in: "
            + ", ".join(f"{label} {count} rows" for label, count in estimated.items() if count)
        )

    qtypes = sorted({qtype for entry in pipelines.values() for qtype in entry["by_type"]})
    lines += [
        "",
        "| Question type | n | " + " | ".join(e["label"] for e in pipelines.values()) + " |",
    ]
    lines.append("|---|---:|" + "---:|" * len(pipelines))
    for qtype in qtypes:
        cells = []
        count = 0
        for entry in pipelines.values():
            group = entry["by_type"].get(qtype)
            if group:
                count = group["questions"]
                cells.append(
                    f"{group['exact_match']:.0f}/{group['questions']} · "
                    f"{group['tokens_mean']:,.0f} tok"
                )
            else:
                cells.append("n/a")
        lines.append(f"| {qtype} | {count} | " + " | ".join(cells) + " |")

    if all("evidence" in entry for entry in pipelines.values()):
        lines += [
            "",
            "| Pipeline | Gold among candidates | Grounded in cited articles | Abstained "
            "| Citations / answer | Latency p50 / p95 | Cost / 100 questions |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for entry in pipelines.values():
            e, o = entry["evidence"], entry["overall"]
            lines.append(
                f"| {entry['label']} | {_pct(e['gold_among_answers'], o['questions'])} | "
                f"{_pct(e['grounded'], e['grounding_checked'])} "
                f"({e['grounded']}/{e['grounding_checked']}) | {e['abstained']} | "
                f"{o['citations_mean']:.1f} | {o['latency_p50_s']:.1f} s / "
                f"{o['latency_p95_s']:.1f} s | ${o['cost_per_100_usd']:.2f} |"
            )

    if all("retrieval" in entry for entry in pipelines.values()):
        lines += [
            "",
            "| Pipeline | Hit@1 | Hit@5 | MRR | nDCG@5 | Recall@5 | Precision@5 "
            "| Context recall | Context precision@5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for entry in pipelines.values():
            r = entry["retrieval"]
            keys = ("hit_at_1", "hit_at_5", "mrr", "ndcg_at_5", "recall_at_5", "precision_at_5")
            recall = (
                f"{r['context_recall']:.0%} ({r['context_recall_questions']} q)"
                if r.get("context_recall") is not None
                else "n/a"
            )
            lines.append(
                f"| {entry['label']} | "
                + " | ".join(_fmt(r.get(key)) for key in keys)
                + f" | {recall} | {_fmt(r.get('context_precision_at_5'))} |"
            )

    agent = metrics.get("agent")
    if agent:
        lines += [
            "",
            "**Agentic behaviour**",
            "",
            f"- Steps per question (LLM calls + tool calls): mean {agent['steps_mean']:.2f}, "
            f"median {agent['steps_median']:.0f}, max {agent['steps_max']}",
            f"- LLM calls per question: mean {agent['llm_calls_mean']:.2f}; single-call answers "
            f"{agent['single_call_answers']}/{agent['questions']}",
            f"- Strategy changes: {agent['strategy_changes']}/{agent['questions']}",
            "- Tools called: " + ", ".join(f"{k} {v}" for k, v in agent["tools"].items()),
            "- Specialists invoked (questions): "
            + ", ".join(f"{k} {v}" for k, v in agent["specialists"].items()),
            "- Stopping reasons: "
            + "; ".join(f"{k} ({v})" for k, v in agent["stopping_reasons"].items()),
        ]
    misses = metrics.get("graph_pipeline_misses") or []
    if misses:
        lines += ["", "**Graph pipeline misses**", ""]
        lines += [
            f"- {m['qid']} ({m['qtype']}) {m['pipeline']}: {str(m['answer'])[:90]!r}"
            for m in misses
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _merge_pipeline(rows: list[dict[str, Any]], pipeline: str, path: str) -> int:
    # Take one pipeline's fields from another run (an agent-only rerun on newer code, or a rerun
    # of just the questions a change can affect). Rows absent from that run are left as they are.
    replacement = {row["qid"]: row for row in _read_jsonl(path)}
    prefix = f"{pipeline}_"
    replaced = 0
    for row in rows:
        source = replacement.get(row["qid"])
        if source is None:
            continue
        for key in [key for key in row if key.startswith(prefix)]:
            del row[key]
        row.update({key: value for key, value in source.items() if key.startswith(prefix)})
        if pipeline == "agentic" and "agentic_trace" in source:
            row["agentic_trace"] = source["agentic_trace"]
        replaced += 1
    return replaced


def _manifest_summary(results_path: str, pipelines: str, rows: int | None = None) -> dict[str, Any]:
    # Where each pipeline's numbers came from: the run's commit, model, and dataset hash.
    manifest_path = Path(f"{results_path}.manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {
        "pipelines": pipelines if pipelines != "base" else manifest.get("pipelines"),
        "commit": str(manifest.get("git_commit", ""))[:7],
        "model": manifest.get("model"),
        "dataset_sha256": (manifest.get("dataset") or {}).get("sha256"),
        "finished_at": manifest.get("finished_at"),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise benchmark results")
    parser.add_argument("results", help="results JSONL from src.evaluate")
    parser.add_argument("--dataset", help="questions JSONL with gold answers (enables evidence)")
    parser.add_argument("--corpus", default="hackathon-resources/corpus/corpus.jsonl")
    parser.add_argument(
        "--pipeline-from",
        action="append",
        default=[],
        metavar="PIPELINE=RESULTS",
        help="take one pipeline's rows from another results file",
    )
    parser.add_argument("--json", dest="json_out", help="also write the metrics document here")
    args = parser.parse_args()

    rows = _read_jsonl(args.results)
    merged = []
    for spec in args.pipeline_from:
        pipeline, _, path = spec.partition("=")
        merged.append((path, pipeline, _merge_pipeline(rows, pipeline, path)))
    gold = gold_docs = articles = None
    if args.dataset:
        questions = _read_jsonl(args.dataset)
        gold = {q["qid"]: q.get("answer") or [] for q in questions}
        gold_docs = {q["qid"]: q.get("gold_doc_ids") or [] for q in questions}
        articles = {
            doc["doc_id"]: f"{doc['title']}\n{doc['text']}" for doc in _read_jsonl(args.corpus)
        }
    metrics = collect_metrics(rows, gold, gold_docs, articles)
    metrics["provenance"] = [_manifest_summary(args.results, "base", len(rows))] + [
        _manifest_summary(path, pipeline, count) for path, pipeline, count in merged
    ]
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(render_markdown(metrics))


if __name__ == "__main__":
    main()
