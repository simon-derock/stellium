# Integration tests for FastAPI endpoints: compare, batch, snapshot, health.
import os
from unittest.mock import AsyncMock, MagicMock

os.environ["TG_USE_MOCK"] = "1"

import pytest

import src.api.main as api_main
import src.pipelines.agentic as agentic_module
import src.pipelines.graphrag as graphrag_module
from src.api.limits import QuestionBudget
from src.api.main import app
from src.coprocessor import Coprocessor
from src.linking import EventCatalog, EventRecord
from src.llm import LLMCallResult, LockedLLMSession
from src.models import Chunk, PipelineResult
from src.pipelines.intent import Intent
from tests.asgi_client import InProcessASGIClient
from tests.graph_fixtures import seeded_graph

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
    graph.vertices_by_id.return_value = [
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
    monkeypatch.setattr(graphrag_module, "catalog_for", lambda graph: EventCatalog([marathon]))
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


def test_graph_snapshot_projects_the_loaded_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "get_graph", MagicMock)
    monkeypatch.setattr(api_main, "catalog_for", lambda graph: seeded_graph()[1])

    overview = client.get("/api/v1/graph/snapshot").json()
    focused = client.get(
        "/api/v1/graph/snapshot", params={"view": "investigation", "focus": "R08W, ,R12W"}
    ).json()

    assert {node["type"] for node in overview["graph_nodes"]} == {"Games", "Sport"}
    layers = {node["id"]: node["layer"] for node in focused["graph_nodes"]}
    assert layers["R08W"] == layers["R12W"] == "evidence"
    assert client.get("/api/v1/graph/snapshot", params={"view": "bogus"}).status_code == 422


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
                content=(
                    '{"operation": "event_attribute", "event": "men\'s marathon", '
                    '"sport": "Athletics", "year": 2008, "attribute": "gold_athlete"}'
                ),
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
                    '{"operation": "event_attribute", "event": "men\'s marathon", '
                    '"sport": "Athletics", "year": 2008, "attribute": "gold_athlete"}'
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

    # The pipelines run concurrently, so each gets its own session and scripted replies.
    replies = list(chat.side_effect)
    scripted = {"rag": replies[:1], "graphrag": replies[1:3], "agentic": replies[3:]}
    sessions = []
    chats = []
    for name in ("rag", "graphrag", "agentic"):
        session = LockedLLMSession(provider="cohere", model="test-model")
        scripted_chat = AsyncMock(side_effect=scripted[name])
        monkeypatch.setattr(session, "embed", AsyncMock(return_value=[[0.0] * 1024]))
        monkeypatch.setattr(session, "chat", scripted_chat)
        sessions.append(session)
        chats.append(scripted_chat)
    created = iter(sessions)
    monkeypatch.setattr(api_main, "make_session", lambda provider: next(created))
    monkeypatch.setattr(api_main, "classify_intent", AsyncMock(return_value=Intent("ask", 60, 1.0)))

    response = client.post(
        "/api/v1/query/compare",
        json={"query": "Who won the 2008 Olympic men's marathon?", "qid": "compare-001"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["qid"] == "compare-001"
    assert payload["intent"] == "ask"
    assert payload["intent_tokens"] == 60
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
    assert [scripted_chat.await_count for scripted_chat in chats] == [1, 2, 1]


def test_batch_eval_bypasses_input_guards(monkeypatch: pytest.MonkeyPatch) -> None:
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
        monkeypatch.setenv("STELLIUM_ADMIN_TOKEN", "judge-token")
        resp = client.post(
            "/api/v1/evaluate/batch",
            headers={"Authorization": "Bearer judge-token"},
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

    def record_readiness(host: str, timeout_s: float) -> float:
        readiness.append(host)
        return 0.0

    monkeypatch.setattr(api_main, "wait_for_graph_ready", record_readiness)
    connection = MagicMock()
    monkeypatch.setattr(api_main, "connect", lambda: connection)

    first = api_main.get_graph()
    second = api_main.get_graph()

    assert first is second
    assert first.conn is connection
    assert readiness == ["https://graph.example"]


def test_queries_default_to_the_benchmark_provider_and_reject_unknown_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STELLIUM_LLM_PROVIDER", raising=False)
    assert api_main.QueryRequest(query="q").provider == "cohere"
    resp = client.post("/api/v1/query/rag", json={"query": "Who won?", "provider": "nope"})
    assert resp.status_code == 400


def test_exhausted_language_model_returns_503_not_a_fabricated_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_retrieval_api(monkeypatch, [])
    session = LockedLLMSession(provider="cohere", model="test-model")
    monkeypatch.setattr(session, "embed", AsyncMock(return_value=[[0.0] * 1024]))
    monkeypatch.setattr(
        session, "chat", AsyncMock(side_effect=RuntimeError("monthly call limit reached"))
    )
    monkeypatch.setattr(api_main, "make_session", lambda provider: session)

    resp = client.post("/api/v1/query/rag", json={"query": "Who won the marathon in 2008?"})

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"
    # The cause is logged, never shown: no account or key details reach a visitor.
    assert "monthly" not in resp.json()["detail"]
    assert "unavailable" in resp.json()["detail"]


def test_metrics_endpoint_serves_the_committed_benchmark_document() -> None:
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    pipelines = resp.json()["pipelines"]
    assert set(pipelines) == {"rag", "graphrag", "agentic"}
    assert pipelines["agentic"]["overall"]["questions"] == 100


def test_presets_list_public_questions_without_answers() -> None:
    resp = client.get("/api/v1/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert len(presets) == 100
    assert set(presets[0]) == {"qid", "qtype", "question"}


def test_deep_health_runs_a_real_graph_read(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = MagicMock()
    graph.ping.return_value = 316
    monkeypatch.setattr(api_main, "get_graph", lambda: graph)

    shallow = client.get("/health").json()
    deep = client.get("/health", params={"deep": "true"}).json()

    assert "graph" not in shallow
    assert deep["graph"]["venues"] == 316
    graph.ping.assert_called_once()


def test_graph_outage_mid_question_returns_503_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyTigerGraph.common.exception import TigerGraphException

    graph, _ = _wire_retrieval_api(monkeypatch, [])
    graph.vector_search.side_effect = TigerGraphException("Bad gateway while resuming", None)

    resp = client.post("/api/v1/query/rag", json={"query": "Who won the marathon in 2008?"})

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "15"
    assert "waking up" in resp.json()["detail"]


def test_named_metrics_serve_only_published_documents() -> None:
    assert client.get("/api/v1/metrics/compositional").json()["pipelines"]["agentic"]
    assert client.get("/api/v1/metrics/hidden_oracle").json()["hidden"]["pipelines"]
    assert client.get("/api/v1/metrics/..%2Fsecrets").status_code == 404
    assert client.get("/api/v1/metrics/unknown").status_code == 404
    published = client.get("/api/v1/metrics/index").json()
    assert {"public", "compositional", "hidden_oracle"} <= set(published)
    assert all(client.get(f"/api/v1/metrics/{name}").status_code == 200 for name in published)


def test_graph_stats_count_the_loaded_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "get_graph", MagicMock)
    monkeypatch.setattr(api_main, "catalog_for", lambda graph: seeded_graph()[1])

    stats = client.get("/api/v1/graph/stats").json()

    assert (stats["events"], stats["games"], stats["sports"], stats["venues"]) == (6, 3, 1, 3)
    # Men's single sculls 2004 -> 2008 and women's 2004 -> 2008 -> 2012.
    assert stats["previous_edition_links"] == 3
    assert {"chunks", "documents", "bm25_terms"} <= set(stats)


def test_small_talk_stops_at_the_intent_check_without_running_a_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_main, "classify_intent", AsyncMock(return_value=Intent("chat", 58, 1.0))
    )
    pipelines = MagicMock(side_effect=AssertionError("no pipeline runs for small talk"))
    monkeypatch.setattr(api_main, "make_session", pipelines)

    payload = client.post("/api/v1/query/compare", json={"query": "hi"}).json()

    assert payload["intent"] == "chat"
    assert payload["intent_tokens"] == 58
    assert payload["rag"] is payload["graphrag"] is payload["agentic"] is None


@pytest.mark.asyncio
async def test_keep_alive_reads_the_graph_and_survives_a_failed_beat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    graph = MagicMock()
    graph.ping.side_effect = [RuntimeError("Starting workspace"), 316, 316]
    monkeypatch.setattr(api_main, "get_graph", lambda: graph)

    beat = asyncio.create_task(api_main._keep_graph_awake(0.01))
    while graph.ping.call_count < 3:
        await asyncio.sleep(0.01)
    beat.cancel()

    assert graph.ping.call_count >= 3


def test_batch_eval_is_hidden_without_a_token_and_refuses_a_wrong_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"questions": [{"qid": "q1", "question": "Who won?", "qtype": "lookup"}]}
    monkeypatch.delenv("STELLIUM_ADMIN_TOKEN", raising=False)
    assert client.post("/api/v1/evaluate/batch", json=body).status_code == 404
    monkeypatch.setenv("STELLIUM_ADMIN_TOKEN", "judge-token")
    wrong = client.post(
        "/api/v1/evaluate/batch", json=body, headers={"Authorization": "Bearer guess"}
    )
    assert wrong.status_code == 401


def test_visitors_cannot_pick_an_unlisted_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STELLIUM_ALLOWED_PROVIDERS", raising=False)
    resp = client.post("/api/v1/query/rag", json={"query": "Who won?", "provider": "offline"})
    assert resp.status_code == 400


def test_daily_cap_stops_live_questions_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "_budget", QuestionBudget())
    monkeypatch.setenv("STELLIUM_DAILY_QUESTION_CAP", "1")
    monkeypatch.setattr(api_main, "classify_intent", AsyncMock(return_value=Intent("chat", 5, 1.0)))
    assert client.post("/api/v1/query/compare", json={"query": "hi"}).status_code == 200
    resp = client.post("/api/v1/query/compare", json={"query": "hi"})
    assert resp.status_code == 429
    assert int(resp.headers["retry-after"]) > 0
    assert "00:00 UTC" in resp.json()["detail"]


def test_oversized_bodies_are_refused_before_parsing() -> None:
    resp = client.post(
        "/api/v1/query/compare",
        content=b"x" * (65 * 1024),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413


def test_browsers_on_other_sites_get_no_cors_grant() -> None:
    resp = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_operator_status_needs_the_admin_token_and_shows_no_key_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STELLIUM_ADMIN_TOKEN", raising=False)
    assert client.get("/api/v1/ops/status").status_code == 404
    monkeypatch.setenv("STELLIUM_ADMIN_TOKEN", "owner")
    assert client.get("/api/v1/ops/status").status_code == 401
    monkeypatch.setenv("COHERE_KEY", "secret-value")
    graph = MagicMock()
    graph.ping.return_value = 316
    monkeypatch.setattr(api_main, "_get_request_graph", lambda: graph)

    resp = client.get("/api/v1/ops/status", headers={"Authorization": "Bearer owner"})

    body = resp.json()
    assert resp.status_code == 200
    assert body["graph"]["venues"] == 316
    assert body["questions"]["daily_cap"] == 0
    assert [row["alias"] for row in body["keys"]] == ["COHERE_KEY"]
    assert "secret-value" not in resp.text


def test_alerts_turn_503_when_the_graph_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STELLIUM_ADMIN_TOKEN", "owner")
    monkeypatch.setenv("STELLIUM_LLM_PROVIDER", "offline")
    graph = MagicMock()
    graph.ping.side_effect = [316, RuntimeError("suspended")]
    monkeypatch.setattr(api_main, "_get_request_graph", lambda: graph)
    owner = {"Authorization": "Bearer owner"}

    healthy = client.get("/api/v1/ops/alerts", headers=owner)
    broken = client.get("/api/v1/ops/alerts", headers=owner)

    assert healthy.status_code == 200 and healthy.json() == {"ok": True, "issues": []}
    assert broken.status_code == 503
    assert broken.json()["issues"] == ["graph unreachable"]


def test_compare_still_answers_when_one_pipeline_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "classify_intent", AsyncMock(return_value=Intent("ask", 5, 1.0)))
    answer = PipelineResult(
        qid="custom-001", question="q", pipeline="agentic", answer="Samuel Wanjiru"
    )
    monkeypatch.setattr(api_main, "_get_request_graph", MagicMock)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(api_main, "make_session", lambda provider: session)
    for name, outcome in (
        ("RAGPipeline", RuntimeError("model hiccup")),
        ("GraphRAGPipeline", answer.model_copy(update={"pipeline": "graphrag"})),
        ("AgenticPipeline", answer),
    ):
        pipeline = MagicMock()
        if isinstance(outcome, Exception):
            pipeline.return_value.run = AsyncMock(side_effect=outcome)
        else:
            pipeline.return_value.run = AsyncMock(return_value=outcome)
        monkeypatch.setattr(api_main, name, pipeline)

    resp = client.post("/api/v1/query/compare", json={"query": "Who won the 2008 marathon?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["rag"] is None
    assert body["agentic"]["answer"] == body["graphrag"]["answer"] == "Samuel Wanjiru"
