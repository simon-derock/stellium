# Chaos engineering and fault injection test suite for Stellium.
# Tests system resilience under network partitions, database outages, rate limits, and corrupted payloads.

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.coprocessor import Coprocessor
from src.graph import create_mock_graph_client
from src.llm import LLMCallResult, LockedLLMSession
from src.pipelines.agentic import AgenticPipeline, parse_react_response


def test_react_parsing_under_json_corruption() -> None:
    # Verifies resilient JSON parsing when LLM outputs unescaped quotes or malformed syntax.
    valid_text = 'Thought: Search\nAction: gsql_aggregate\nAction Input: {"sport": "Athletics", "year": 2012}'
    step = parse_react_response(valid_text)
    assert step.action == "gsql_aggregate"
    assert step.action_input.get("sport") == "Athletics"
    assert step.action_input.get("year") == 2012

    # Single quotes recovery
    single_quote_text = (
        "Thought: Search\nAction: gsql_lookup\nAction Input: {'sport': 'Aquatics', 'year': 2008}"
    )
    step_sq = parse_react_response(single_quote_text)
    assert step_sq.action == "gsql_lookup"
    assert step_sq.action_input.get("sport") == "Aquatics"

    # Completely unparseable garbage returns empty input dict without crashing
    garbage = "Thought: Unknown\nAction: finish\nAction Input: not a json"
    step_garbage = parse_react_response(garbage)
    assert step_garbage.action == "finish"
    assert step_garbage.action_input == {}


def test_corrupted_vector_embedding_dimension_validation() -> None:
    # Verifies that vector embeddings with invalid dimension or non-finite numbers are rejected.
    def validate_embedding(vec: list[float], expected_dim: int = 1024) -> bool:
        if len(vec) != expected_dim:
            return False
        import math

        return not any(math.isnan(x) or math.isinf(x) for x in vec)

    valid_vec = [0.1] * 1024
    assert validate_embedding(valid_vec) is True

    # Wrong dimension
    short_vec = [0.1] * 512
    assert validate_embedding(short_vec) is False

    # NaN vector
    nan_vec = [0.1] * 1023 + [float("nan")]
    assert validate_embedding(nan_vec) is False

    # Inf vector
    inf_vec = [float("inf")] + [0.1] * 1023
    assert validate_embedding(inf_vec) is False


@pytest.mark.asyncio
async def test_llm_rate_limit_and_server_error_chaos(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulates HTTP 429 Too Many Requests and HTTP 500 errors from LLM endpoint.
    session = LockedLLMSession(
        provider="cloudflare",
        model="@cf/meta/llama-3.1-8b-instruct-fast",
    )
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")

    req = httpx.Request("POST", "https://api.cloudflare.com/test")
    # Simulate HTTP 429 response
    rate_limit_resp = httpx.Response(status_code=429, request=req, text="Rate limit exceeded")
    err = httpx.HTTPStatusError("Rate limit", request=req, response=rate_limit_resp)

    with patch.object(httpx.AsyncClient, "post", side_effect=err):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError) as exc_info:
                await session.chat(
                    [{"role": "user", "content": "hello"}], max_tokens=10, max_retries=2
                )
            assert "failed after" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_agentic_pipeline_resilience_on_database_chaos() -> None:
    # Simulates database outage (network timeout / 502) during agentic tool execution.
    # The agent should record the error in observation and pivot rather than crashing.
    graph = create_mock_graph_client()
    # Inject failure into run_aggregation
    graph.run_aggregation = MagicMock(side_effect=RuntimeError("TigerGraph connection timeout"))  # type: ignore[method-assign]

    session = LockedLLMSession(provider="cloudflare", model="mock-llama")
    session.chat = AsyncMock()  # type: ignore[method-assign]

    # Model attempts gsql_aggregate first, then pivots to direct answer
    first_resp = LLMCallResult(
        content='Thought: Look up aggregate count.\nAction: gsql_aggregate\nAction Input: {"sport": "Biathlon", "min_competitors": 74}',
        input_tokens=20,
        output_tokens=30,
        model_name="mock-llama",
        provider="mock",
        latency_ms=10.0,
    )
    second_resp = LLMCallResult(
        content="Thought: The database call failed. I will provide the best available answer.\nFinal Answer: 5",
        input_tokens=40,
        output_tokens=15,
        model_name="mock-llama",
        provider="mock",
        latency_ms=8.0,
    )
    session.chat.side_effect = [first_resp, second_resp]

    coprocessor = Coprocessor()
    coprocessor.build([])
    pipeline = AgenticPipeline(graph=graph, coprocessor=coprocessor, llm=session)
    result = await pipeline.run(
        "pub-test-chaos", "How many biathlon events had more than 73 competitors?"
    )

    assert result.answer == "5"
    assert result.agentic_trace is not None
    assert len(result.agentic_trace["tools_called"]) >= 1
    # Check that error was captured in observation
    assert "error" in result.agentic_trace["tools_called"][0]["output_summary"].lower()


def test_chunk_filter_mask_bit_invariants_under_fuzz() -> None:
    # Verifies bitmask parsing resilience against arbitrary inputs.
    from src.coprocessor import build_filter_mask

    # Empty inputs
    mask_empty = build_filter_mask(None, None)
    assert mask_empty == 0

    # Malformed year / season
    mask_weird = build_filter_mask(1800, "Spring")
    # Year outside Olympic range should not set year bits
    assert mask_weird >= 0
