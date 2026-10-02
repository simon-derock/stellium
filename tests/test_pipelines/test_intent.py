# The intent gate: one tiny call, "chat" only on a clear CHAT label, and never a blocker.
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm import LLMCallResult
from src.pipelines import intent as intent_module
from src.pipelines.intent import classify_intent


def _session(reply: str | Exception) -> MagicMock:
    llm = MagicMock()
    if isinstance(reply, Exception):
        llm.chat = AsyncMock(side_effect=reply)
    else:
        llm.chat = AsyncMock(
            return_value=LLMCallResult(
                content=reply,
                input_tokens=52,
                output_tokens=1,
                model_name="m",
                provider="p",
                latency_ms=3.0,
            )
        )
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=llm)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "label"), [("CHAT", "chat"), ("chat.", "chat"), ("ASK", "ask"), ("Maybe", "ask")]
)
async def test_only_a_clear_chat_label_skips_the_pipelines(
    monkeypatch: pytest.MonkeyPatch, reply: str, label: str
) -> None:
    session = _session(reply)
    monkeypatch.setattr(intent_module, "make_session", lambda provider: session)

    result = await classify_intent("hi", "cohere")

    assert result.label == label
    assert result.tokens == 53
    llm = await session.__aenter__()
    assert llm.chat.await_args.kwargs["max_tokens"] == 3


@pytest.mark.asyncio
async def test_a_failed_check_still_answers_the_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        intent_module, "make_session", lambda provider: _session(RuntimeError("down"))
    )

    result = await classify_intent("Who won the men's marathon in 2008?", "cohere")

    assert (result.label, result.tokens) == ("ask", 0)
