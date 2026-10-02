# Cohere key pool: discovery, deduplication, and parking keys that hit their monthly cap.
from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import credentials
from src.llm import make_session

_CAP_BODY = '{"message":"You are using a Trial key, which is limited to 1000 API calls / month."}'


@pytest.fixture(autouse=True)
def _isolated_pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "COHERE_CHAT_API_KEY",
        "COHERE",
        "COHERE_BACKUP",
        "COHERE_KEY",
        "COHERE_KEYS",
        "COHERE_KEY_TIERS",
    ):
        monkeypatch.delenv(name, raising=False)
    credentials.reset_exhausted_keys()
    yield
    credentials.reset_exhausted_keys()


def test_pool_reads_named_numbered_and_listed_keys_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE", "k1")
    monkeypatch.setenv("COHERE_KEY_TWO", "k2")
    monkeypatch.setenv("COHERE_KEYS", "k3, k1 ,")
    assert [key for _, key in credentials.cohere_keys()] == ["k1", "k2", "k3"]


def test_only_monthly_cap_messages_park_a_key() -> None:
    assert credentials.is_monthly_cap(429, _CAP_BODY)
    assert not credentials.is_monthly_cap(429, '{"message":"too many requests per minute"}')
    assert not credentials.is_monthly_cap(500, _CAP_BODY)


@pytest.mark.asyncio
async def test_chat_moves_to_the_next_key_when_one_is_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE", "spent-key")
    monkeypatch.setenv("COHERE_BACKUP", "live-key")
    monkeypatch.setattr("src.llm.COHERE_CHAT_REQUEST_INTERVAL_S", 0.0)
    capped = MagicMock(status_code=429, text=_CAP_BODY)
    ok = MagicMock(status_code=200, text="{}")
    ok.json.return_value = {
        "message": {"content": [{"type": "text", "text": "Chen Ding"}]},
        "usage": {"tokens": {"input_tokens": 9, "output_tokens": 2}},
    }
    used: list[str] = []

    async def post(url: str, json: object, headers: dict[str, str]) -> MagicMock:
        used.append(headers["Authorization"])
        return capped if headers["Authorization"].endswith("spent-key") else ok

    client = MagicMock(post=AsyncMock(side_effect=post), aclose=AsyncMock())
    monkeypatch.setattr("httpx.AsyncClient", lambda **_: client)

    session = make_session("cohere")
    first = await session.chat([{"role": "user", "content": "q"}])
    second = await session.chat([{"role": "user", "content": "q2"}])

    assert first.content == second.content == "Chen Ding"
    # The spent key is tried once, then never again in this process.
    assert sum(header.endswith("spent-key") for header in used) == 1
    assert [key for _, key in credentials.available_cohere_keys()] == ["live-key"]


def test_embedding_batch_moves_to_the_next_key_when_one_is_capped() -> None:
    from unittest.mock import patch

    from src.embeddings import CohereEmbeddingClient

    client = CohereEmbeddingClient(api_key="spent", backup_api_key="live", request_interval_s=0)
    capped = MagicMock(status_code=429, text=_CAP_BODY, headers={})
    ok = MagicMock(status_code=200)
    ok.json.return_value = {"embeddings": {"float": [[0.5] * 1024]}}

    with (
        patch("httpx.Client.post", side_effect=[capped, ok]) as post,
        patch("src.embeddings.time.sleep") as sleep,
    ):
        vectors = client.embed_queries(["question"])

    assert len(vectors) == 1
    sleep.assert_not_called()
    assert post.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer live"


def test_tiers_spread_over_the_first_tier_and_fall_through_when_it_is_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRESH_A", "f1")
    monkeypatch.setenv("FRESH_B", "f2")
    monkeypatch.setenv("COHERE", "old")
    monkeypatch.setenv("COHERE_KEY_TIERS", "FRESH_A,FRESH_B|COHERE")
    assert [[key for _, key in tier] for tier in credentials.cohere_key_tiers()] == [
        ["f1", "f2"],
        ["old"],
    ]
    assert [key for _, key in credentials.available_cohere_keys()] == ["f1", "f2"]
    credentials.park_exhausted_key("f1")
    credentials.park_exhausted_key("f2")
    assert [key for _, key in credentials.available_cohere_keys()] == ["old"]


def test_a_resting_key_yields_to_an_awake_one_but_never_empties_the_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE", "k1")
    monkeypatch.setenv("COHERE_BACKUP", "k2")
    credentials.rest_key("k1", 60)
    assert [key for _, key in credentials.available_cohere_keys()] == ["k2"]
    credentials.rest_key("k2", 60)
    assert [key for _, key in credentials.available_cohere_keys()] == ["k1", "k2"]


@pytest.mark.asyncio
async def test_chat_hands_a_throttled_or_rejected_call_to_another_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE", "bad-key")
    monkeypatch.setenv("COHERE_BACKUP", "busy-key")
    monkeypatch.setenv("COHERE_KEY", "good-key")
    monkeypatch.setattr("src.llm.COHERE_CHAT_REQUEST_INTERVAL_S", 0.0)
    replies = {
        "bad-key": MagicMock(status_code=401, text='{"message":"invalid api token"}'),
        "busy-key": MagicMock(status_code=429, text='{"message":"too many requests"}'),
    }
    ok = MagicMock(status_code=200, text="{}")
    ok.json.return_value = {
        "message": {"content": [{"type": "text", "text": "Chen Ding"}]},
        "usage": {"tokens": {"input_tokens": 9, "output_tokens": 2}},
    }

    async def post(url: str, json: object, headers: dict[str, str]) -> MagicMock:
        return replies.get(headers["Authorization"].removeprefix("Bearer "), ok)

    client = MagicMock(post=AsyncMock(side_effect=post), aclose=AsyncMock())
    monkeypatch.setattr("httpx.AsyncClient", lambda **_: client)

    for _ in range(3):
        result = await make_session("cohere").chat([{"role": "user", "content": "q"}])
        assert result.content == "Chen Ding"
    assert "bad-key" not in [
        key for _, key in credentials.cohere_keys() if key not in credentials.parked_keys([key])
    ]
