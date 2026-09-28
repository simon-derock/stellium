# Integration tests for FastAPI endpoints: compare, batch, snapshot, health.
import os
from unittest.mock import AsyncMock, MagicMock

os.environ["TG_USE_MOCK"] = "1"

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import app
from src.coprocessor import Coprocessor
from src.llm import LLMCallResult, LockedLLMSession
from src.models import Chunk

client = TestClient(app)


def _wire_retrieval_api(
    monkeypatch: pytest.MonkeyPatch, chat_results: list[LLMCallResult]
) -> tuple[MagicMock, AsyncMock]:
    # Provides a local graph and source-text index to exercise complete API retrieval wiring.
    chunk = Chunk(
        chunk_id="Q123#0",
        doc_id="Q123",
        chunk_index=0,
        section_title="Results",
        text="The 2008 Olympic men's marathon was won by Samuel Wanjiru.",
        raw_text="The 2008 Olympic men's marathon was won by Samuel Wanjiru.",
    )
    coprocessor = Coprocessor()
    coprocessor.build([chunk])
    graph = MagicMock()
    graph.vector_search.return_value = [("Q123#0", 0.91)]
    graph.run_multihop.return_value = {
        "events": ["Men's marathon"],
        "gold_athletes": ["Samuel Wanjiru"],
        "gold_doc_ids": ["Q123"],
    }
    graph.run_lookup.return_value = {"events": [], "gold_athletes": [], "gold_doc_ids": []}
    session = LockedLLMSession(provider="cloudflare", model="test-model")
    chat = AsyncMock(side_effect=chat_results)
    monkeypatch.setattr(api_main, "get_graph", lambda: graph)
    monkeypatch.setattr(api_main, "get_coprocessor", lambda: coprocessor)
    monkeypatch.setattr(api_main, "make_session", lambda provider: session)
    monkeypatch.setattr(session, "embed", AsyncMock(return_value=[[0.0] * 1024]))
    monkeypatch.setattr(session, "chat", chat)
    return graph, chat


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
    assert data["graph_nodes"] == []
    assert data["graph_edges"] == []
    assert data["metrics"] == []
    assert "not available yet" in data["disclosure"]


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
    mock_final = LLMCallResult(
        content="Thought: The event count matches the requested threshold.\nFinal Answer: 5",
        input_tokens=60,
        output_tokens=12,
        model_name="mock-model",
        provider="mock",
        latency_ms=8.0,
    )
    with patch.object(LockedLLMSession, "chat", side_effect=[mock_res, mock_final]):
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


def test_rag_api_sends_retrieved_text_to_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    _, chat = _wire_retrieval_api(
        monkeypatch,
        [
            LLMCallResult(
                content="Samuel Wanjiru",
                input_tokens=20,
                output_tokens=5,
                model_name="test-model",
                provider="test",
                latency_ms=1.0,
            )
        ],
    )

    response = client.post("/api/v1/query/rag", json={"query": "Who won the marathon in 2008?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Samuel Wanjiru"
    call = chat.await_args
    assert call is not None
    assert "Samuel Wanjiru" in call.args[0][1]["content"]


def test_graphrag_api_sends_retrieved_text_to_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, chat = _wire_retrieval_api(
        monkeypatch,
        [
            LLMCallResult(
                content='{"year": 2008, "sport": "Athletics", "venue": "Olympic Stadium"}',
                input_tokens=20,
                output_tokens=10,
                model_name="test-model",
                provider="test",
                latency_ms=1.0,
            ),
            LLMCallResult(
                content="Samuel Wanjiru",
                input_tokens=20,
                output_tokens=5,
                model_name="test-model",
                provider="test",
                latency_ms=1.0,
            ),
        ],
    )

    response = client.post(
        "/api/v1/query/graphrag", json={"query": "Who won the marathon in 2008?"}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Samuel Wanjiru"
    call = chat.await_args
    assert call is not None
    assert (
        "The 2008 Olympic men's marathon was won by Samuel Wanjiru." in call.args[0][1]["content"]
    )


def test_compare_api_runs_all_pipelines_with_context_and_agentic_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, chat = _wire_retrieval_api(
        monkeypatch,
        [
            LLMCallResult(
                content="Samuel Wanjiru",
                input_tokens=20,
                output_tokens=5,
                model_name="test-model",
                provider="test",
                latency_ms=1.0,
            ),
            LLMCallResult(
                content='{"year": 2008, "sport": "Athletics", "venue": null}',
                input_tokens=12,
                output_tokens=8,
                model_name="test-model",
                provider="test",
                latency_ms=1.0,
            ),
            LLMCallResult(
                content="Samuel Wanjiru",
                input_tokens=24,
                output_tokens=5,
                model_name="test-model",
                provider="test",
                latency_ms=1.0,
            ),
            LLMCallResult(
                content=(
                    "Thought: A deterministic event query can answer this count.\n"
                    "Action: gsql_aggregate\n"
                    'Action Input: {"sport":"Athletics","target_year":2008,'
                    '"min_competitors":0,"max_competitors":0}'
                ),
                input_tokens=30,
                output_tokens=18,
                model_name="test-model",
                provider="test",
                latency_ms=1.0,
            ),
            LLMCallResult(
                content=(
                    "Thought: The count does not answer who won; query the gold medalist.\n"
                    "Action: gsql_lookup\n"
                    'Action Input: {"event_name_fragment":"men\'s marathon",'
                    '"target_year":2008,"sport":"Athletics",'
                    '"gender":"Men","attribute":"gold_athlete"}'
                ),
                input_tokens=42,
                output_tokens=24,
                model_name="test-model",
                provider="test",
                latency_ms=1.0,
            ),
        ],
    )
    graph.run_aggregation.return_value = {
        "count": 7,
        "events": ["Men's marathon"],
        "gold_doc_ids": ["Q123"],
    }
    graph.run_lookup.return_value = {
        "events": ["Athletics at the 2008 Summer Olympics – Men's marathon"],
        "competitor_counts": [98],
        "nation_counts": [59],
        "gold_athletes": ["Samuel Wanjiru"],
        "venues": ["Beijing National Stadium"],
        "gold_doc_ids": ["Q123"],
    }

    response = client.post(
        "/api/v1/query/compare",
        json={"query": "Who won the 2008 Olympic men's marathon?", "qid": "compare-001"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["qid"] == "compare-001"
    assert payload["rag"]["answer"] == "Samuel Wanjiru"
    assert payload["graphrag"]["answer"] == "Samuel Wanjiru"
    assert payload["agentic"]["answer"] == "Samuel Wanjiru"
    assert payload["rag"]["retrieved_doc_ids"] == ["Q123"]
    assert payload["graphrag"]["retrieved_doc_ids"] == ["Q123"]
    trace = payload["agentic"]["agentic_trace"]
    assert trace["agents_invoked"] == ["ReActOrchestrator"]
    assert trace["tools_called"][0]["tool_name"] == "gsql_aggregate"
    assert trace["tools_called"][0]["llm_tokens"] == 0
    assert chat.await_count == 5


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
