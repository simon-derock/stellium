# Tests for Jina Embeddings v3 client, rate limiter, and fallbacks.
# Strictly zero triple-quote docstrings per project coding standards.
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.embeddings import JinaEmbeddingClient, RateLimiter


# Test RateLimiter enforces minimum delay between successive calls
def test_rate_limiter_throttling() -> None:
    limiter = RateLimiter(min_interval_s=0.05)
    t0 = time.monotonic()
    limiter.wait_turn()
    limiter.wait_turn()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.045


# Test client creation from environment with various naming variants
def test_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test lowercase user variant
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


# Test offline fallback returns dimension-compliant zero vectors without crashing
def test_offline_fallback_zero_vectors() -> None:
    client = JinaEmbeddingClient(api_key="", dimension=1024)
    passages = ["Test passage 1", "Test passage 2"]
    vectors = client.embed_passages(passages)
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
    assert len(vectors[1]) == 1024
    assert all(v == 0.0 for v in vectors[0])

    query_vec = client.embed_query("Who won gold in 2012?")
    assert len(query_vec) == 1024
    assert all(v == 0.0 for v in query_vec)


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


# Test rate limit 429 retry logic with backoff
def test_embed_retry_on_429() -> None:
    client = JinaEmbeddingClient(api_key="mock-key", dimension=1024)

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
