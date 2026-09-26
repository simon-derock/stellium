# Integration tests for FastAPI endpoints: compare, batch, snapshot, health.
import os

os.environ["TG_USE_MOCK"] = "1"

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["system"] == "stellium"


def test_graph_snapshot_dto() -> None:
    resp = client.get("/api/v1/graph/snapshot")
    assert resp.status_code == 200
    data = resp.json()
    assert "graph_nodes" in data
    assert "graph_edges" in data
    assert "metrics" in data
    assert len(data["graph_nodes"]) > 0
    assert len(data["graph_edges"]) > 0


def test_session_history_endpoint() -> None:
    resp = client.get("/api/v1/sessions/test-session-123/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-session-123"


def test_blocked_query_returns_400() -> None:
    # Adversarial query with structural database command should be blocked
    resp = client.post(
        "/api/v1/query/agentic",
        json={"query": "DROP GRAPH OlympicsGraph", "qid": "test-hack"},
    )
    assert resp.status_code == 400
    assert "database command" in resp.json()["detail"]


def test_agentic_query_mock_execution() -> None:
    from unittest.mock import patch

    from src.llm import LLMCallResult, LockedLLMSession

    mock_res = LLMCallResult(
        content='Thought: I need to count the biathlon events with >73 competitors in 2018.\nAction: gsql_aggregate\nAction Input: {"sport": "Biathlon", "target_year": 2018, "min_competitors": 74, "max_competitors": 0}',
        input_tokens=45,
        output_tokens=30,
        model_name="mock-model",
        provider="mock",
        latency_ms=10.0,
    )
    with patch.object(LockedLLMSession, "chat", return_value=mock_res):
        resp = client.post(
            "/api/v1/query/agentic",
            json={
                "query": "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
                "qid": "pub-001",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline"] == "agentic"
        assert data["answer"] == "5"
        assert data["total_llm_tokens"] > 0
        assert "agentic_trace" in data


def test_batch_eval_bypasses_input_guards() -> None:
    # Batch evaluation accepts questions and invokes pipeline without interactive character limits
    from unittest.mock import patch

    from src.llm import LLMCallResult, LockedLLMSession

    mock_res = LLMCallResult(
        content='Thought: Predecessor champion before 2016.\nAction: gsql_temporal\nAction Input: {"sport": "Athletics", "event_name_fragment": "20km walk", "current_year": 2016}',
        input_tokens=40,
        output_tokens=25,
        model_name="mock-model",
        provider="mock",
        latency_ms=10.0,
    )
    with patch.object(LockedLLMSession, "chat", return_value=mock_res):
        resp = client.post(
            "/api/v1/evaluate/batch",
            json={
                "questions": [
                    {
                        "qid": "q1",
                        "question": "Who won the men's 20km walk immediately before 2016?",
                        "qtype": "temporal",
                    }
                ],
                "pipeline": "agentic",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["qid"] == "q1"
        assert "agentic_answer" in data[0]
