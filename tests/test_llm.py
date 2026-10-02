# Tests for Cloudflare Workers AI router, session model lock, and retry mechanisms.
# Strictly zero triple-quote docstrings per project coding standards.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src import llm as llm_module
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
    for name in ("COHERE", "COHERE_BACKUP", "COHERE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_KEY", raising=False)
    monkeypatch.delenv("jina_embedding_api_key", raising=False)
    monkeypatch.delenv("JINA_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock-token")

    with pytest.raises(RuntimeError, match="embedding API key is required"):
        await make_session("cloudflare").embed(["query"])


@pytest.mark.asyncio
async def test_query_embeddings_reject_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("COHERE", "COHERE_BACKUP", "COHERE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JINA_API_KEY", "mock-jina-key")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")

    with pytest.raises(RuntimeError, match="must match the TigerGraph index"):
        await make_session("cloudflare").embed(["query"])


# Test session factory initialization and model assignment
def test_make_session_defaults() -> None:
    session = make_session("cloudflare")
    assert session.provider == "cloudflare"
    assert session.model == CLOUDFLARE_PRIMARY_MODEL


@pytest.mark.asyncio
async def test_cloudflare_without_credentials_fails_instead_of_switching_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "would-have-been-used-before")

    with pytest.raises(RuntimeError, match="CLOUDFLARE_ACCOUNT_ID"):
        await make_session("cloudflare").chat([{"role": "user", "content": "Who won?"}])


@pytest.mark.asyncio
async def test_offline_provider_is_explicit_and_never_answers() -> None:
    session = make_session("offline")
    result = await session.chat([{"role": "user", "content": "Who won the 200m in 2012?"}])

    assert (result.provider, result.model_name) == ("offline", "offline-deterministic")
    assert result.content.endswith("Final Answer: Not found in corpus")
    assert await session.embed(["q"]) == [[0.0] * 1024]


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


@pytest.mark.asyncio
async def test_persistent_response_cache_survives_cache_registry_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("STELLIUM_LLM_RESPONSE_CACHE", str(cache_path))
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "mock-account-id")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "mock-api-token")
    llm_module._response_cache_registry.clear()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": "cached answer"}}],
        "usage": {"prompt_tokens": 31, "completion_tokens": 4},
    }

    with patch("httpx.AsyncClient.post", return_value=response) as post:
        first = await make_session("cloudflare").chat(
            [{"role": "user", "content": "same prompt"}], max_tokens=32
        )
        llm_module._response_cache_registry.clear()
        second = await make_session("cloudflare").chat(
            [{"role": "user", "content": "same prompt"}], max_tokens=32
        )

    assert first.content == second.content == "cached answer"
    assert first.input_tokens == second.input_tokens == 31
    assert first.output_tokens == second.output_tokens == 4
    assert post.await_count == 1
    cache_record = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])
    assert cache_record["content"] == "cached answer"
    assert len(cache_record["cache_key"]) == 64


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
    request_headers = post.call_args.kwargs["headers"]
    assert GEMINI_MODEL in request_url
    assert "test-gemini-key" not in request_url
    assert request_headers["x-goog-api-key"] == "test-gemini-key"
    assert request_payload["contents"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_mistral_provider_request_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in llm_module._MISTRAL_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.setattr(llm_module, "_mistral_key_cursor", 0)
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
    assert result.credential_alias == "MISTRAL_API_KEY"


@pytest.mark.asyncio
async def test_mistral_key_pool_rotates_distinct_numbered_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in llm_module._MISTRAL_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY_1", "mistral-account-one")
    monkeypatch.setenv("MISTRAL_API_KEY_2", "mistral-account-two")
    monkeypatch.setenv("MISTRAL_API_KEY_3", "mistral-account-one")
    monkeypatch.setattr(llm_module, "_mistral_key_cursor", 0)
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }

    with patch("httpx.AsyncClient.post", return_value=response) as post:
        results = [
            await make_session("mistral").chat([{"role": "user", "content": "question"}])
            for _ in range(2)
        ]

    assert [result.credential_alias for result in results] == [
        "MISTRAL_API_KEY_1",
        "MISTRAL_API_KEY_2",
    ]
    assert [call.kwargs["headers"]["Authorization"] for call in post.call_args_list] == [
        "Bearer mistral-account-one",
        "Bearer mistral-account-two",
    ]


def test_mistral_key_pool_recognizes_user_named_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in llm_module._MISTRAL_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KEY_ONE", "mistral-one")
    monkeypatch.setenv("KEY_TWENTYTHREE", "mistral-twenty-three")
    monkeypatch.setenv("MISTRAL_API_KEY", "legacy-key")

    assert llm_module._configured_mistral_api_keys() == [
        ("KEY_ONE", "mistral-one"),
        ("KEY_TWENTYTHREE", "mistral-twenty-three"),
    ]


@pytest.mark.asyncio
async def test_cohere_chat_round_robins_distinct_configured_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_CHAT_API_KEY", raising=False)
    monkeypatch.setenv("COHERE", "cohere-primary-test-key")
    monkeypatch.setenv("COHERE_BACKUP", "cohere-backup-test-key")
    monkeypatch.setenv("COHERE_KEY", "cohere-third-test-key")
    monkeypatch.setattr(llm_module, "COHERE_CHAT_REQUEST_INTERVAL_S", 0.0)
    monkeypatch.setattr(llm_module, "_cohere_key_cursor", 0)
    llm_module._cohere_last_start_by_key.clear()
    llm_module._cohere_key_locks.clear()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "message": {"content": [{"type": "text", "text": "ok"}]},
        "usage": {"billed_units": {"input_tokens": 2, "output_tokens": 1}},
    }

    with patch("httpx.AsyncClient.post", return_value=response) as post:
        results = [
            await make_session("cohere").chat([{"role": "user", "content": "question"}])
            for _ in range(3)
        ]

    assert [result.credential_alias for result in results] == [
        "COHERE",
        "COHERE_BACKUP",
        "COHERE_KEY",
    ]
    assert [call.kwargs["headers"]["Authorization"] for call in post.call_args_list] == [
        "Bearer cohere-primary-test-key",
        "Bearer cohere-backup-test-key",
        "Bearer cohere-third-test-key",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {
                "billed_units": {"input_tokens": 700, "output_tokens": 60},
                "tokens": {"input_tokens": 760, "output_tokens": 60},
            },
            (700, 60, "billed"),
        ),
        ({"tokens": {"input_tokens": 760, "output_tokens": 60}}, (760, 60, "processed")),
        ({}, (2, 1, "estimated")),
    ],
)
async def test_cohere_token_counts_say_where_they_came_from(
    monkeypatch: pytest.MonkeyPatch, usage: dict[str, Any], expected: tuple[int, int, str]
) -> None:
    monkeypatch.setenv("COHERE", "cohere-test-key")
    monkeypatch.setattr(llm_module, "COHERE_CHAT_REQUEST_INTERVAL_S", 0.0)
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "message": {"content": [{"type": "text", "text": "ok"}]},
        "usage": usage,
    }

    with patch("httpx.AsyncClient.post", return_value=response):
        result = await make_session("cohere").chat([{"role": "user", "content": "question"}])

    assert (result.input_tokens, result.output_tokens, result.token_source) == expected


@pytest.mark.asyncio
async def test_cohere_key_pool_deduplicates_shared_secret_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_CHAT_API_KEY", raising=False)
    monkeypatch.setenv("COHERE", "shared-test-key")
    monkeypatch.setenv("COHERE_BACKUP", "shared-test-key")
    monkeypatch.setenv("COHERE_KEY", "distinct-test-key")

    assert llm_module._configured_cohere_api_keys() == [
        ("COHERE", "shared-test-key"),
        ("COHERE_KEY", "distinct-test-key"),
    ]


@pytest.mark.asyncio
async def test_mistral_retry_error_reports_status_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in llm_module._MISTRAL_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.setattr(llm_module, "_mistral_key_cursor", 0)
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={"retry-after": "0"})
    session = make_session("mistral")

    with (
        patch("httpx.AsyncClient.post", return_value=response) as post,
        patch("src.llm.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(RuntimeError, match="HTTP status 429") as error,
    ):
        await session.chat([{"role": "user", "content": "question"}], max_retries=2)

    assert post.await_count == 1
    assert "test-mistral-key" not in str(error.value)


@pytest.mark.asyncio
async def test_mistral_rate_limit_moves_to_distinct_configured_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in llm_module._MISTRAL_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY_1", "mistral-first-account")
    monkeypatch.setenv("MISTRAL_API_KEY_2", "mistral-second-account")
    monkeypatch.setattr(llm_module, "_mistral_key_cursor", 0)
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    limited = httpx.Response(429, request=request)
    successful = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
        request=request,
    )
    session = make_session("mistral")

    with (
        patch("httpx.AsyncClient.post", side_effect=[limited, successful]) as post,
        patch("src.llm.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        result = await session.chat([{"role": "user", "content": "question"}])

    assert result.content == "ok"
    assert result.credential_alias == "MISTRAL_API_KEY_2"
    assert [call.kwargs["headers"]["Authorization"] for call in post.call_args_list] == [
        "Bearer mistral-first-account",
        "Bearer mistral-second-account",
    ]
    sleep.assert_not_awaited()
