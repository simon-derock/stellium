# Unit tests for evaluation metrics: Exact Match, Token F1, MRR, Recall@k.
import hashlib
import json
from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock

import pytest

import src.evaluate as evaluate_module
from src.evaluate import (
    EvaluationHarness,
    compute_exact_match,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_strict_exact_match,
    compute_token_f1,
)
from src.models import EvalQuestion, PipelineResult


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


def test_exact_match_ignores_typography_but_not_content() -> None:
    title = "Fencing at the 2008 Summer Olympics – Men's épée"
    # Curly apostrophe, ASCII hyphen, trailing period, and missing accent are typography only.
    assert compute_exact_match("Fencing at the 2008 Summer Olympics - Men’s epee.", [title]) == 1.0
    # Gold team answers concatenate names; a comma-separated prediction has the same content.
    team = ["Dani KingLaura TrottJoanna Rowsell"]
    assert compute_exact_match("Dani King, Laura Trott, Joanna Rowsell", team) == 1.0
    # Different content still fails: a suffix is not the canonical title, and numbers differ.
    assert compute_exact_match("Men's épée", [title]) == 0.0
    assert compute_exact_match("41", ["4"]) == 0.0
    assert compute_exact_match("", [""]) == 0.0
    # The legacy metric keeps its punctuation sensitivity for audit comparability.
    assert compute_strict_exact_match("Chen Ding.", ["Chen Ding"]) == 0.0
    assert compute_strict_exact_match("chen ding", ["Chen Ding"]) == 1.0


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

    # Repeated terms count with multiplicity in the token intersection.
    assert compute_token_f1("Alice Alice", ["Alice Bob"]) == pytest.approx(0.5)
    assert compute_token_f1("Alice Alice", ["Alice"]) == pytest.approx(2 / 3)

    # Empty predictions and references have no token overlap score.
    assert compute_token_f1("", ["Alice"]) == 0.0
    assert compute_token_f1("Alice", [""]) == 0.0


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


def test_evaluation_harness_loads_environment_before_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(evaluate_module, "load_dotenv", lambda: calls.append(True))

    EvaluationHarness(corpus_path="missing-corpus.jsonl", use_mock=True)

    assert calls == [True]


def test_live_evaluation_fails_closed_without_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_module, "load_dotenv", lambda: None)
    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "TG_HOST",
        "JINA_API_KEY",
        "jina_embedding_api_key",
        "JINA_EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        EvaluationHarness(corpus_path="missing-corpus.jsonl")


def test_live_evaluation_does_not_replace_graph_failure_with_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")
    monkeypatch.setenv("TG_HOST", "https://graph.invalid")
    monkeypatch.setenv("JINA_API_KEY", "test-jina-key")

    def fail_connect() -> object:
        raise ConnectionError("test connection failure")

    monkeypatch.setattr(evaluate_module, "connect", fail_connect)
    monkeypatch.setattr(evaluate_module, "wait_for_graph_ready", lambda host: 0.0)
    with pytest.raises(ConnectionError, match="test connection failure"):
        EvaluationHarness(corpus_path="missing-corpus.jsonl")


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("JINA_EMBEDDING_MODEL", "jina-embeddings-v4"),
        ("EMBEDDING_DIMENSION", "768"),
    ],
)
def test_live_evaluation_rejects_embedding_index_mismatch(
    monkeypatch: pytest.MonkeyPatch, setting: str, value: str
) -> None:
    for name in ("COHERE", "COHERE_BACKUP", "COHERE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(evaluate_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")
    monkeypatch.setenv("TG_HOST", "https://graph.invalid")
    monkeypatch.setenv("JINA_API_KEY", "test-jina-key")
    monkeypatch.setenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv(setting, value)

    with pytest.raises(RuntimeError, match="must match the TigerGraph index"):
        EvaluationHarness(corpus_path="missing-corpus.jsonl")


@pytest.mark.asyncio
async def test_evaluation_harness_offline_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.llm import LLMCallResult, LockedLLMSession
    from tests.graph_fixtures import seeded_graph

    q = EvalQuestion(
        qid="test-001",
        question="How many rowing events at the 2004 Summer Olympics had more than 25 competitors?",
        qtype="aggregation",
        answer=["1"],
        gold_doc_ids=["R04M"],
    )
    plan = LLMCallResult(
        content=(
            "Thought: Count rowing events.\nAction: count_events\n"
            'Action Input: {"sport": "Rowing", "year": 2004, "threshold": 25}'
        ),
        input_tokens=50,
        output_tokens=30,
        model_name="mock-model",
        provider="mock",
        latency_ms=12.0,
    )
    session = LockedLLMSession(provider="cloudflare", model="mock-model")
    chat = AsyncMock(return_value=plan)
    monkeypatch.setattr(session, "chat", chat)

    harness = EvaluationHarness(
        corpus_path="nonexistent.jsonl",
        use_mock=True,
        session_factory=lambda provider: session,
    )
    harness.graph, _, harness.coprocessor = seeded_graph()

    results = await harness.evaluate_question(q, ["agentic"])
    assert results["agentic"].answer == "1"
    assert results["agentic"].total_llm_tokens == 80
    assert results["agentic"].agentic_trace is not None
    assert results["agentic"].agentic_trace["stopping_reason"] != ""
    assert chat.await_count == 1


@pytest.mark.asyncio
async def test_benchmark_runner_writes_public_and_hidden_records(tmp_path: Path) -> None:
    # Exercises dataset loading, per-pipeline result fields, public metrics, and JSONL output.
    class DeterministicHarness(EvaluationHarness):
        async def evaluate_question(
            self, question: EvalQuestion, pipelines: list[str]
        ) -> dict[str, PipelineResult]:
            expected = question.answer[0] if question.answer else "Not found in corpus"
            return {
                pipeline: PipelineResult(
                    qid=question.qid,
                    pipeline=cast(Literal["rag", "graphrag", "agentic"], pipeline),
                    question=question.question,
                    answer=expected,
                    llm_input_tokens=12,
                    llm_output_tokens=3,
                    total_llm_tokens=15,
                    latency_ms=7.5,
                    retrieved_doc_ids=["Q1"],
                    model_name="mock-model",
                    provider="mock",
                    agentic_trace={"step_count": 1} if pipeline == "agentic" else None,
                )
                for pipeline in pipelines
            }

    public_question = EvalQuestion(
        qid="public-001",
        question="How many events?",
        qtype="aggregation",
        answer=["5"],
        gold_doc_ids=["Q1"],
    )
    hidden_question = EvalQuestion(
        qid="hidden-001",
        question="Who won?",
        qtype="lookup",
    )
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text(
        "\n".join([public_question.model_dump_json(), hidden_question.model_dump_json()]) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "results" / "benchmark.jsonl"
    harness = DeterministicHarness(corpus_path="missing-corpus.jsonl", use_mock=True)

    records = await harness.run_benchmark(
        str(dataset_path), ["rag", "graphrag", "agentic"], str(output_path)
    )

    assert len(records) == 2
    manifest = json.loads((tmp_path / "results" / "benchmark.jsonl.manifest.json").read_text())
    assert manifest["pipelines"] == ["rag", "graphrag", "agentic"]
    assert manifest["dataset"]["sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert "finished_at" in manifest and "git_commit" in manifest
    assert records[0]["rag_em"] == 1.0
    assert records[0]["agentic_recall@5"] == 1.0
    assert records[0]["agentic_trace"] == {"step_count": 1}
    assert records[0]["rag_input_tokens"] == 12
    assert records[0]["graphrag_model"] == "mock-model"
    assert records[0]["agentic_provider"] == "mock"
    assert records[0]["rag_retrieved_doc_ids"] == ["Q1"]
    assert "rag_em" not in records[1]
    assert records[1]["graphrag_answer"] == "Not found in corpus"
    written_records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert written_records == records

    # A partial JSONL checkpoint restores completed work and only evaluates missing questions.
    output_path.write_text(json.dumps(written_records[0]) + "\n", encoding="utf-8")

    class ResumeHarness(DeterministicHarness):
        evaluated_qids: list[str] = []

        async def evaluate_question(
            self, question: EvalQuestion, pipelines: list[str]
        ) -> dict[str, PipelineResult]:
            self.evaluated_qids.append(question.qid)
            return await super().evaluate_question(question, pipelines)

    resume_harness = ResumeHarness(corpus_path="missing-corpus.jsonl", use_mock=True)
    resumed_records = await resume_harness.run_benchmark(
        str(dataset_path), ["rag", "graphrag", "agentic"], str(output_path), resume=True
    )

    assert [record["qid"] for record in resumed_records] == ["public-001", "hidden-001"]
    assert resume_harness.evaluated_qids == ["hidden-001"]
    assert [json.loads(line)["qid"] for line in output_path.read_text().splitlines()] == [
        "public-001",
        "hidden-001",
    ]
