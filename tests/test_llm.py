# Tests for Cloudflare Workers AI router, session model lock, and retry mechanisms.
# Strictly zero triple-quote docstrings per project coding standards.
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm import (
    CLOUDFLARE_PRIMARY_MODEL,
    GEMINI_MODEL,
    MISTRAL_MODEL,
    LockedLLMSession,
    ProviderQuotaExceededError,
    make_session,
)


@pytest.mark.asyncio
async def test_query_embeddings_fail_closed_without_index_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.delenv("jina_embedding_api_key", raising=False)
    monkeypatch.delenv("JINA_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock-token")

    with pytest.raises(RuntimeError, match="JINA_API_KEY is required"):
        await make_session("cloudflare").embed(["query"])


@pytest.mark.asyncio
async def test_query_embeddings_reject_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "mock-jina-key")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")

    with pytest.raises(RuntimeError, match="must match the TigerGraph index"):
        await make_session("cloudflare").embed(["query"])


# Test session factory initialization and model assignment
def test_make_session_defaults() -> None:
    session = make_session("cloudflare")
    assert session.provider == "cloudflare"
    assert session.model == CLOUDFLARE_PRIMARY_MODEL


# Test offline mock fallback when Cloudflare credentials are unset
@pytest.mark.asyncio
async def test_cloudflare_offline_mock_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    session = make_session("cloudflare")
    messages = [{"role": "user", "content": "Who won the 200m sprint in 2012?"}]
    result = await session.chat(messages)

    assert result.provider == "offline_mock"
    assert "Not found in corpus" in result.content
    assert result.input_tokens > 0
    assert result.output_tokens > 0


# Test OpenAI-compatible endpoint returns exact ground-truth token usage
@pytest.mark.asyncio
async def test_cloudflare_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock-account-id")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock-api-token")

    fake_response_data = {
        "id": "chatcmpl-mock",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Usain Bolt won the gold medal."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 12,
            "total_tokens": 54,
        },
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response_data

    session = make_session("cloudflare")
    messages = [{"role": "user", "content": "Who won the 200m sprint in 2012?"}]

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        res = await session.chat(messages)
        assert res.provider == "cloudflare"
        assert res.content == "Usain Bolt won the gold medal."
        assert res.input_tokens == 42
        assert res.output_tokens == 12


# Test resilient fallback to native REST model endpoint when OpenAI route returns non-200
@pytest.mark.asyncio
async def test_cloudflare_native_endpoint_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock-account-id")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock-api-token")

    # First call (OpenAI endpoint) returns 404
    mock_openai_fail = MagicMock()
    mock_openai_fail.status_code = 404

    # Second call (Native REST endpoint) returns 200 with result.response
    mock_native_success = MagicMock()
    mock_native_success.status_code = 200
    mock_native_success.raise_for_status.return_value = None
    mock_native_success.json.return_value = {
        "result": {"response": "Usain Bolt (Jamaica)"},
        "success": True,
        "errors": [],
    }

    session = make_session("cloudflare")
    messages = [{"role": "user", "content": "Who won the 200m sprint in 2012?"}]

    with patch("httpx.AsyncClient.post", side_effect=[mock_openai_fail, mock_native_success]):
        res = await session.chat(messages)
        assert res.provider == "cloudflare"
        assert res.content == "Usain Bolt (Jamaica)"
        assert res.output_tokens > 0


# Test retry with backoff on transient edge gateway status codes (429, 503)
@pytest.mark.asyncio
async def test_cloudflare_transient_error_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock-account-id")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock-api-token")

    # Create mock 503 HTTPStatusError
    mock_req = httpx.Request("POST", "https://api.cloudflare.com")
    resp_503 = httpx.Response(503, request=mock_req, headers={"retry-after": "0.01"})
    err_503 = httpx.HTTPStatusError("Service Unavailable", request=mock_req, response=resp_503)

    # Success response on retry
    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Chen Ding"}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 5},
    }

    session = LockedLLMSession(provider="cloudflare")
    messages = [{"role": "user", "content": "Who won men 20km walk in 2012?"}]

    with patch("httpx.AsyncClient.post", side_effect=[err_503, mock_success]):
        res = await session.chat(messages, max_retries=3)
        assert res.content == "Chen Ding"
        assert res.input_tokens == 30


@pytest.mark.asyncio
async def test_cloudflare_daily_neuron_quota_error_fails_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock-account-id")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock-api-token")
    request = httpx.Request("POST", "https://api.cloudflare.com")
    quota_body = {
        "success": False,
        "errors": [
            {
                "code": 4006,
                "message": "daily neuron quota exceeded",
            }
        ],
    }
    openai_response = httpx.Response(429, json=quota_body, request=request)
    native_response = httpx.Response(429, json=quota_body, request=request)
    session = LockedLLMSession(provider="cloudflare")

    with (
        patch("httpx.AsyncClient.post", side_effect=[openai_response, native_response]) as post,
        patch("src.llm.asyncio.sleep", new_callable=AsyncMock) as sleep,
        pytest.raises(ProviderQuotaExceededError, match="allocation is exhausted"),
    ):
        await session.chat([{"role": "user", "content": "question"}], max_retries=5)

    assert post.await_count == 2
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_gemini_provider_request_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Chen Ding"}]}}],
        "usageMetadata": {"promptTokenCount": 31, "candidatesTokenCount": 6},
    }

    session = make_session("gemini")
    with patch("httpx.AsyncClient.post", return_value=response) as post:
        result = await session.chat([{"role": "system", "content": "Use evidence."}])

    assert result.provider == "gemini"
    assert result.model_name == GEMINI_MODEL
    assert (result.input_tokens, result.output_tokens) == (31, 6)
    request_url = post.call_args.args[0]
    request_payload = post.call_args.kwargs["json"]
    assert GEMINI_MODEL in request_url
    assert request_payload["contents"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_mistral_provider_request_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "Chen Ding"}}],
        "usage": {"prompt_tokens": 29, "completion_tokens": 5},
    }

    session = make_session("mistral")
    with patch("httpx.AsyncClient.post", return_value=response) as post:
        result = await session.chat([{"role": "user", "content": "Who won?"}])

    assert result.provider == "mistral"
    assert result.model_name == MISTRAL_MODEL
    assert (result.input_tokens, result.output_tokens) == (29, 5)
    assert post.call_args.args[0] == "https://api.mistral.ai/v1/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-mistral-key"
