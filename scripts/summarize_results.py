# Summarise a benchmark results JSONL into markdown tables and the submission form's numbers:
# accuracy, LLM tokens, and latency per pipeline and question type, plus agent behaviour (steps,
# LLM calls, tools, specialists, strategy changes, stopping reasons), and evidence/cost measures:
# candidate recall, grounding in cited articles, latency percentiles, citations, and dollar cost.
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

_PIPELINES = ("rag", "graphrag", "agentic")
# Cohere Command A list price per million tokens (input, output); an assumption, not a bill.
_PRICE_PER_MILLION = (2.50, 10.00)
_LABELS = {"rag": "RAG", "graphrag": "GraphRAG", "agentic": "Agentic GraphRAG"}


def _pct(numerator: float, denominator: int) -> str:
    return f"{100 * numerator / denominator:.0f}%" if denominator else "n/a"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def evidence_and_cost(
    rows: list[dict[str, Any]],
    gold: dict[str, list[str]],
    articles: dict[str, str],
    pipelines: list[str],
) -> list[str]:
    # Candidate recall credits an honest tie report that contains the gold answer; grounding asks
    # whether a non-numeric answer is stated in the articles that pipeline cited.
    from src.evaluate import normalize_answer
    from src.linking import compact

    lines = [
        "",
        "| Pipeline | Gold among candidates | Grounded in cited articles | Abstained "
        "| Citations / answer | Latency p50 / p95 | Cost / 100 questions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in pipelines:
        recall = grounded = checkable = abstained = 0
        for row in rows:
            parts = [part.strip() for part in str(row[f"{p}_answer"]).split(";") if part.strip()]
            if normalize_answer(str(row[f"{p}_answer"])) == "notfoundincorpus":
                # An abstention makes no claim, so it is counted, not grounding-checked.
                abstained += 1
                continue
            golds = {normalize_answer(answer) for answer in gold.get(row["qid"], [])}
            recall += any(normalize_answer(part) in golds for part in parts)
            if parts and not all(part.replace(",", "").isdigit() for part in parts):
                # Spacing-insensitive, title included: event answers are article titles, and
                # team names appear glued together in the source ("Dani KingLaura Trott").
                cited = "".join(
                    compact(articles.get(doc, "")) for doc in row[f"{p}_retrieved_doc_ids"]
                )
                checkable += 1
                grounded += all(compact(part) in cited for part in parts)
        latencies = [row[f"{p}_latency_ms"] / 1000 for row in rows]
        citations = mean(len(row[f"{p}_retrieved_doc_ids"]) for row in rows)
        cost = (
            sum(
                row.get(f"{p}_input_tokens", 0) * _PRICE_PER_MILLION[0]
                + row.get(f"{p}_output_tokens", 0) * _PRICE_PER_MILLION[1]
                for row in rows
            )
            / 1e6
            * (100 / len(rows))
        )
        lines.append(
            f"| {_LABELS[p]} | {_pct(recall, len(rows))} | {_pct(grounded, checkable)} "
            f"({grounded}/{checkable}) | {abstained} | {citations:.1f} | "
            f"{_percentile(latencies, 0.5):.1f} s / {_percentile(latencies, 0.95):.1f} s | "
            f"${cost:.2f} |"
        )
    return lines


def summarize(
    rows: list[dict[str, Any]],
    gold: dict[str, list[str]] | None = None,
    articles: dict[str, str] | None = None,
) -> str:
    pipelines = [p for p in _PIPELINES if any(f"{p}_answer" in row for row in rows)]
    scored = [row for row in rows if any(f"{p}_em" in row for p in pipelines)]
    lines = ["| Pipeline | Exact match | Strict EM | Token F1 | LLM tokens / q | Latency / q |"]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for p in pipelines:
        em = sum(row.get(f"{p}_em", 0) for row in scored)
        strict = sum(row.get(f"{p}_em_strict", 0) for row in scored)
        f1 = mean(row.get(f"{p}_f1", 0) for row in scored) if scored else 0.0
        tokens = mean(row[f"{p}_tokens"] for row in rows)
        latency = mean(row[f"{p}_latency_ms"] for row in rows) / 1000
        lines.append(
            f"| {_LABELS[p]} | {em:.0f}/{len(scored)} ({_pct(em, len(scored))}) | "
            f"{_pct(strict, len(scored))} | {f1:.3f} | {tokens:,.0f} | {latency:.2f} s |"
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_type[row["qtype"]].append(row)
    lines += ["", "| Question type | n | " + " | ".join(_LABELS[p] for p in pipelines) + " |"]
    lines.append("|---|---:|" + "---:|" * len(pipelines))
    for qtype, group in sorted(by_type.items()):
        cells = []
        for p in pipelines:
            em = sum(row.get(f"{p}_em", 0) for row in group)
            tokens = mean(row[f"{p}_tokens"] for row in group)
            cells.append(f"{em:.0f}/{len(group)} · {tokens:,.0f} tok")
        lines.append(f"| {qtype} | {len(group)} | " + " | ".join(cells) + " |")

    if gold is not None and articles is not None:
        lines += evidence_and_cost(rows, gold, articles, pipelines)

    traces = [row["agentic_trace"] for row in rows if row.get("agentic_trace")]
    if traces:
        tools: Counter[str] = Counter()
        agents: Counter[str] = Counter()
        stops: Counter[str] = Counter()
        for trace in traces:
            tools.update(trace.get("retrieval_methods", []))
            agents.update(trace.get("agents_invoked", []))
            stops[str(trace.get("stopping_reason", "")).split(" stated")[0]] += 1
        llm_calls = [len(trace.get("llm_calls", [])) for trace in traces]
        steps = [trace.get("step_count", 0) for trace in traces]
        changed = sum(bool(trace.get("strategy_changed")) for trace in traces)
        lines += [
            "",
            "**Agentic behaviour**",
            "",
            f"- Steps per question (LLM calls + tool calls): mean {mean(steps):.2f}, "
            f"median {median(steps):.0f}, max {max(steps)}",
            f"- LLM calls per question: mean {mean(llm_calls):.2f}; "
            f"single-call answers {sum(c == 1 for c in llm_calls)}/{len(traces)}",
            f"- Strategy changes: {changed}/{len(traces)}",
            "- Tools called: "
            + ", ".join(f"{name} {count}" for name, count in tools.most_common()),
            "- Specialists invoked (questions): "
            + ", ".join(f"{name} {count}" for name, count in agents.most_common()),
            "- Stopping reasons: "
            + "; ".join(f"{reason} ({count})" for reason, count in stops.most_common()),
        ]
    misses = [
        f"- {row['qid']} ({row['qtype']}) {p}: {str(row[f'{p}_answer'])[:90]!r}"
        for row in scored
        for p in pipelines
        if p != "rag" and not row.get(f"{p}_em")
    ]
    if misses:
        lines += ["", "**Graph pipeline misses**", "", *misses]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise benchmark results")
    parser.add_argument("results", help="results JSONL from src.evaluate")
    parser.add_argument("--dataset", help="questions JSONL with gold answers (enables evidence)")
    parser.add_argument("--corpus", default="hackathon-resources/corpus/corpus.jsonl")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line]
    gold = articles = None
    if args.dataset:
        questions = [
            json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line
        ]
        gold = {q["qid"]: q.get("answer") or [] for q in questions}
        articles = {
            doc["doc_id"]: f"{doc['title']}\n{doc['text']}"
            for doc in map(json.loads, Path(args.corpus).read_text().splitlines())
        }
    print(summarize(rows, gold, articles))


if __name__ == "__main__":
    main()
