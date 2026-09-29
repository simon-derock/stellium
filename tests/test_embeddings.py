# Tests for Jina Embeddings v3 client, rate limiter, and fallbacks.
# Strictly zero triple-quote docstrings per project coding standards.
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.embeddings import (
    CohereEmbeddingClient,
    JinaEmbeddingClient,
    RateLimiter,
    _retry_after_seconds,
    embedding_client_from_env,
)


# Test RateLimiter enforces minimum delay between successive calls
def test_rate_limiter_throttling() -> None:
    limiter = RateLimiter(min_interval_s=0.05)
    t0 = time.monotonic()
    limiter.wait_turn()
    limiter.wait_turn()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.045


def test_rate_limiter_waits_for_token_window_when_next_batch_exceeds_budget() -> None:
    limiter = RateLimiter(min_interval_s=0)
    limiter.remaining_tokens = 100
    limiter.token_reset_at = time.monotonic() + 30

    with patch("src.embeddings.time.sleep") as sleep:
        limiter.wait_turn(estimated_tokens=200)

    assert sleep.call_count == 1
    assert sleep.call_args.args[0] >= 29.9
    assert limiter.remaining_tokens == 99_800
    assert limiter.token_reset_at is not None


# Test client creation from environment with various naming variants
def test_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit generic key takes precedence over legacy provider-specific aliases.
    monkeypatch.setenv("EMBEDDING_KEY", "new-embedding-key")
    monkeypatch.setenv("JINA_API_KEY", "legacy-jina-key")
    client = JinaEmbeddingClient.from_env()
    assert client.api_key == "new-embedding-key"

    # Test lowercase user variant
    monkeypatch.delenv("EMBEDDING_KEY", raising=False)
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.setenv("jina_embedding_api_key", "test-jina-key-123")
    client = JinaEmbeddingClient.from_env()
    assert client.api_key == "test-jina-key-123"
    assert client.is_configured is True
    assert client.dimension == 1024

    # Test unconfigured fallback
    monkeypatch.delenv("jina_embedding_api_key", raising=False)
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.delenv("JINA_EMBEDDING_API_KEY", raising=False)
    unconf_client = JinaEmbeddingClient.from_env()
    assert unconf_client.is_configured is False


def test_cohere_key_selects_cohere_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_KEY", "cohere-test-key")
    client = embedding_client_from_env()
    assert isinstance(client, CohereEmbeddingClient)
    assert client.model == "embed-v4.0"
    assert client.dimension == 1024
    assert client.batch_size == 96


def test_cohere_batch_uses_query_and_document_modes() -> None:
    client = CohereEmbeddingClient(api_key="test-key", batch_size=2, request_interval_s=0)
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.headers = {}
    fake_response.json.return_value = {"embeddings": {"float": [[0.25] * 1024] * 2}}

    with patch("httpx.Client.post", return_value=fake_response) as post:
        passages = client.embed_passages(["one", "two"])
        queries = client.embed_queries(["first", "second"])

    assert len(passages) == len(queries) == 2
    assert all(len(vector) == 1024 for vector in passages + queries)
    assert post.call_count == 2
    assert post.call_args_list[0].kwargs["json"]["input_type"] == "search_document"
    assert post.call_args_list[1].kwargs["json"]["input_type"] == "search_query"


def test_cohere_429_waits_for_rate_window() -> None:
    client = CohereEmbeddingClient(api_key="test-key", request_interval_s=0)
    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.headers = {}
    successful = MagicMock()
    successful.status_code = 200
    successful.json.return_value = {"embeddings": {"float": [[0.25] * 1024]}}

    with (
        patch("httpx.Client.post", side_effect=[rate_limited, successful]),
        patch("src.embeddings.time.sleep") as sleep,
    ):
        vectors = client.embed_passages(["retry after the rate window"])

    assert len(vectors) == 1
    sleep.assert_called_once_with(60.0)


# Unconfigured embeddings fail closed instead of feeding zero vectors to retrieval.
def test_unconfigured_client_rejects_embedding_requests() -> None:
    client = JinaEmbeddingClient(api_key="", dimension=1024)
    with pytest.raises(RuntimeError, match="JINA_API_KEY is required"):
        client.embed_passages(["Passage"])
    with pytest.raises(RuntimeError, match="JINA_API_KEY is required"):
        client.embed_query("Question")


# Test mock HTTP response parsing for batch passages
def test_embed_passages_mock() -> None:
    client = JinaEmbeddingClient(api_key="mock-key", dimension=1024, batch_size=2)
    fake_data = {
        "data": [
            {"index": 0, "embedding": [0.5] * 1024},
            {"index": 1, "embedding": [0.8] * 1024},
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_data

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.embed_passages(["A", "B"])
        assert len(res) == 2
        assert len(res[0]) == 1024
        assert res[0][0] == 0.5
        assert res[1][0] == 0.8


def test_embed_queries_batches_and_preserves_input_order() -> None:
    client = JinaEmbeddingClient(api_key="mock-key", dimension=2, batch_size=2)
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.2, 0.3]},
            {"index": 0, "embedding": [0.4, 0.5]},
        ]
    }

    with patch("httpx.Client.post", return_value=response) as post:
        vectors = client.embed_queries(["first", "second"])

    assert vectors == [[0.4, 0.5], [0.2, 0.3]]
    assert post.call_args.kwargs["json"]["task"] == "retrieval.query"
    assert post.call_args.kwargs["json"]["input"] == ["first", "second"]


def test_embed_rejects_invalid_vector_shape() -> None:
    client = JinaEmbeddingClient(api_key="mock-key", dimension=2)
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = {"data": [{"index": 0, "embedding": [0.2]}]}

    with (
        patch("httpx.Client.post", return_value=response),
        pytest.raises(RuntimeError, match="invalid vector record"),
    ):
        client.embed_query("question")


def test_embed_rejects_zero_vector() -> None:
    client = JinaEmbeddingClient(api_key="mock-key", dimension=2)
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = {"data": [{"index": 0, "embedding": [0.0, 0.0]}]}

    with (
        patch("httpx.Client.post", return_value=response),
        pytest.raises(RuntimeError, match="zero vector"),
    ):
        client.embed_query("question")


def test_embed_raises_after_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    client = JinaEmbeddingClient(api_key="mock-key", dimension=2, max_retries=2)
    response = MagicMock()
    response.status_code = 503
    response.headers = {}
    monkeypatch.setattr("src.embeddings.time.sleep", lambda _: None)

    with (
        patch("httpx.Client.post", return_value=response),
        pytest.raises(RuntimeError, match="failed after 2 attempts"),
    ):
        client.embed_query("question")


def test_embed_reports_insufficient_jina_balance() -> None:
    client = JinaEmbeddingClient(api_key="mock-key", dimension=2)
    response = MagicMock()
    response.status_code = 403
    response.headers = {}
    response.json.return_value = {"code": "AUTHZ_INSUFFICIENT_BALANCE"}

    with (
        patch("httpx.Client.post", return_value=response),
        pytest.raises(RuntimeError, match="account balance is insufficient"),
    ):
        client.embed_query("question")


# Test rate limit 429 retry logic with backoff
def test_embed_retry_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    client = JinaEmbeddingClient(api_key="mock-key", dimension=1024)
    monkeypatch.setattr("src.embeddings.time.sleep", lambda _: None)

    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.headers = {"retry-after": "0.01"}

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1024}]}

    with patch("httpx.Client.post", side_effect=[mock_429, mock_200]):
        res = client.embed_query("Test query")
        assert len(res) == 1024
        assert res[0] == 0.1


def test_retry_after_accepts_seconds_and_http_date() -> None:
    assert _retry_after_seconds("12.5") == 12.5
    assert _retry_after_seconds("invalid") is None
    assert (_retry_after_seconds("Wed, 21 Oct 2099 07:28:00 GMT") or 0) > 0


# Test arbitrary new Jina embedding models can be configured dynamically
def test_custom_jina_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test instantiating with any new Jina model (v5, v4, v3, etc.)
    client_v5 = JinaEmbeddingClient(
        api_key="mock-key",
        model="jina-embeddings-v5-text-small",
        dimension=1024,
    )
    assert client_v5.model == "jina-embeddings-v5-text-small"
    assert client_v5.dimension == 1024

    # Test environment override
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v4")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    client_env = JinaEmbeddingClient.from_env()
    assert client_env.model == "jina-embeddings-v4"
    assert client_env.dimension == 1024

    monkeypatch.setenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
    assert JinaEmbeddingClient.from_env().model == "jina-embeddings-v5-text-small"
