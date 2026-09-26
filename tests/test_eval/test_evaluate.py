# Unit tests for evaluation metrics: Exact Match, Token F1, MRR, Recall@k.
import pytest

from src.evaluate import (
    EvaluationHarness,
    compute_exact_match,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_token_f1,
)
from src.models import EvalQuestion


def test_exact_match() -> None:
    assert compute_exact_match("Chen Ding", ["Chen Ding"]) == 1.0
    # Case insensitivity
    assert compute_exact_match("chen ding", ["Chen Ding"]) == 1.0
    # Diacritics normalization
    assert compute_exact_match("Naim Suleymanoglu", ["Naim Süleymanoğlu"]) == 1.0
    # Whitespace normalization
    assert compute_exact_match("  5  ", ["5"]) == 1.0
    # Mismatch
    assert compute_exact_match("Usain Bolt", ["Chen Ding"]) == 0.0


def test_token_f1() -> None:
    # Exact overlap
    assert compute_token_f1("Chen Ding", ["Chen Ding"]) == 1.0

    # Team event set matching: "Rudolf Dombi, Roland Kokeny" vs "Roland Kökény, Rudolf Dombi"
    f1 = compute_token_f1("Rudolf Dombi, Roland Kokeny", ["Roland Kökény, Rudolf Dombi"])
    assert f1 == 1.0

    # Partial overlap
    f1_partial = compute_token_f1(
        "Men's marathon", ["Athletics at the 2008 Summer Olympics – Men's marathon"]
    )
    assert f1_partial > 0.0

    # Zero overlap
    assert compute_token_f1("Archery", ["Swimming"]) == 0.0


def test_mrr() -> None:
    gold = ["Q100", "Q200"]
    # First candidate is gold -> MRR = 1.0
    assert compute_mrr(["Q100", "Q300", "Q400"], gold) == 1.0
    # Second candidate is gold -> MRR = 0.5
    assert compute_mrr(["Q300", "Q200", "Q400"], gold) == 0.5
    # Third candidate is gold -> MRR = 0.333...
    assert pytest.approx(compute_mrr(["Q300", "Q400", "Q100"], gold), 0.01) == 0.333
    # No match
    assert compute_mrr(["Q500", "Q600"], gold) == 0.0


def test_recall_at_k() -> None:
    gold = ["Q1", "Q2", "Q3", "Q4"]
    retrieved = ["Q1", "Q2", "Q99", "Q100", "Q101"]
    # 2 out of 4 retrieved in top-5 -> Recall@5 = 0.5
    assert compute_recall_at_k(retrieved, gold, k=5) == 0.5


def test_precision_at_k() -> None:
    gold = ["Q1", "Q2"]
    retrieved = ["Q1", "Q2", "Q99", "Q100", "Q101"]
    # 2 out of 5 retrieved are gold -> Precision@5 = 0.4
    assert compute_precision_at_k(retrieved, gold, k=5) == 0.4


@pytest.mark.asyncio
async def test_evaluation_harness_offline_mock() -> None:
    harness = EvaluationHarness(corpus_path="nonexistent.jsonl", use_mock=True)
    q = EvalQuestion(
        qid="test-001",
        question="According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
        qtype="aggregation",
        answer=["5"],
        gold_doc_ids=["Q47091419"],
    )
    # Evaluate agentic pipeline in offline mock mode
    results = await harness.evaluate_question(q, ["agentic"])
    assert "agentic" in results
    assert results["agentic"].answer == "5"
    assert results["agentic"].total_llm_tokens == 0  # Deterministic GSQL path = 0 tokens
    assert results["agentic"].agentic_trace is not None
    assert results["agentic"].agentic_trace["stopping_reason"] != ""
