# Integration tests for FastAPI endpoints: compare, batch, snapshot, health.
import os
from unittest.mock import AsyncMock, MagicMock

os.environ["TG_USE_MOCK"] = "1"

import pytest

import src.api.main as api_main
import src.pipelines.agentic as agentic_module
from src.api.main import app
from src.coprocessor import Coprocessor
from src.linking import EventCatalog, EventRecord
from src.llm import LLMCallResult, LockedLLMSession
from src.models import Chunk
from tests.asgi_client import InProcessASGIClient

client = InProcessASGIClient(app)


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
    graph.conn.getVerticesById.return_value = [
        {"v_id": "Q123", "attributes": {"gold_athlete": "Samuel Wanjiru"}}
    ]
    marathon = EventRecord(
        "Q123",
        "Athletics at the 2008 Summer Olympics – Men's marathon",
        2008,
        "Summer",
        "Athletics",
        "Men",
        "Beijing National Stadium",
        "24 August",
    )
    monkeypatch.setattr(agentic_module, "catalog_for", lambda graph: EventCatalog([marathon]))
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


def test_agentic_query_runs_typed_graph_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.graph_fixtures import seeded_graph

    graph, _, coprocessor = seeded_graph()
    plan = LLMCallResult(
        content=(
            "Thought: Count the matching events in the graph.\nAction: count_events\n"
            'Action Input: {"sport": "Rowing", "year": 2004, "comparison": "more_than", '
            '"threshold": 25}'
        ),
        input_tokens=45,
        output_tokens=30,
        model_name="mock-model",
        provider="mock",
        latency_ms=10.0,
    )
    session = LockedLLMSession(provider="cloudflare", model="mock-model")
    monkeypatch.setattr(api_main, "get_graph", lambda: graph)
    monkeypatch.setattr(api_main, "get_coprocessor", lambda: coprocessor)
    monkeypatch.setattr(api_main, "make_session", lambda provider: session)
    monkeypatch.setattr(session, "chat", AsyncMock(return_value=plan))

    resp = client.post(
        "/api/v1/query/agentic",
        json={"query": "How many rowing events in 2004 had more than 25 competitors?"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline"] == "agentic"
    assert data["answer"] == "1"
    assert data["total_llm_tokens"] == 75
    assert data["agentic_trace"]["tools_called"][0]["tool_name"] == "count_events"


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
                content=(
                    "INTERPRET QUERY () FOR GRAPH OlympicsGraph {\n"
                    " Events = {Event.*};\n"
                    " Matched = SELECT e FROM Events:e\n"
                    ' WHERE e.year == 2008 AND lower(e.sport) == "athletics"\n'
                    " LIMIT 10;\n"
                    " PRINT Matched[Matched.gold_athlete];\n"
                    "}"
                ),
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
                    "Thought: Look up the gold medallist of the named event.\n"
                    "Action: event_attribute\n"
                    'Action Input: {"event":"men\'s marathon","year":2008,'
                    '"sport":"Athletics","attribute":"gold_athlete"}'
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
    assert trace["agents_invoked"][:3] == [
        "OrchestratorAgent",
        "EntityLinkingAgent",
        "GraphTraversalAgent",
    ]
    assert trace["tools_called"][0]["tool_name"] == "event_attribute"
    assert trace["tools_called"][0]["llm_tokens"] == 0
    assert chat.await_count == 4


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


def _reset_graph_selection(monkeypatch: pytest.MonkeyPatch, **env: str) -> list[str]:
    # Isolate graph selection from the developer's .env and cached clients.
    loaded: list[str] = []
    monkeypatch.setattr(api_main, "_graph", None)
    monkeypatch.setattr(api_main, "_load_graph_environment", lambda: loaded.append("dotenv"))
    for name in ("TG_USE_MOCK", "TG_HOST"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return loaded


def test_query_without_graph_configuration_returns_503_not_mock_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _reset_graph_selection(monkeypatch)
    resp = client.post("/api/v1/query/rag", json={"query": "Who won the 2008 men's marathon?"})
    assert resp.status_code == 503
    assert "TG_HOST is not configured" in resp.json()["detail"]
    # Environment files load before TG_HOST is checked.
    assert loaded == ["dotenv"]


def test_graph_connection_failure_returns_503_without_mock_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_graph_selection(monkeypatch, TG_HOST="https://graph.invalid")
    monkeypatch.setattr(api_main, "wait_for_graph_ready", lambda host, timeout_s: 0.0)

    def fail_connect() -> object:
        raise ConnectionError("unreachable")

    monkeypatch.setattr(api_main, "connect", fail_connect)
    mock_factory = MagicMock()
    monkeypatch.setattr(api_main, "_get_mock_graph", mock_factory)

    resp = client.post("/api/v1/query/agentic", json={"query": "Who won the marathon in 2008?"})

    assert resp.status_code == 503
    assert "refusing to substitute mock graph data" in resp.json()["detail"]
    mock_factory.assert_not_called()


def test_explicit_mock_mode_selects_mock_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_graph_selection(monkeypatch, TG_USE_MOCK="true")
    sentinel = MagicMock()
    monkeypatch.setattr(api_main, "_get_mock_graph", lambda: sentinel)
    assert api_main.get_graph() is sentinel


def test_live_graph_is_created_once_after_workspace_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_graph_selection(monkeypatch, TG_HOST="https://graph.example")
    readiness: list[str] = []
    monkeypatch.setattr(
        api_main, "wait_for_graph_ready", lambda host, timeout_s: readiness.append(host) or 0.0
    )
    connection = MagicMock()
    monkeypatch.setattr(api_main, "connect", lambda: connection)

    first = api_main.get_graph()
    second = api_main.get_graph()

    assert first is second
    assert first.conn is connection
    assert readiness == ["https://graph.example"]
