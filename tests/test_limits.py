# Question allowances: per visitor per hour, and per day for the whole demo.
import pytest
from fastapi import HTTPException

from src.api.limits import QuestionBudget

_NOON = 1_790_000_000.0


def test_a_visitor_is_held_to_the_hourly_cap_while_others_still_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STELLIUM_CLIENT_HOURLY_CAP", "2")
    budget = QuestionBudget()
    budget.spend("a", _NOON)
    budget.spend("a", _NOON + 1)
    with pytest.raises(HTTPException) as refused:
        budget.spend("a", _NOON + 2)
    assert refused.value.status_code == 429
    budget.spend("b", _NOON + 3)
    # An hour after the first question, one slot opens again.
    budget.spend("a", _NOON + 3601)


def test_the_daily_cap_resets_on_the_next_utc_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STELLIUM_DAILY_QUESTION_CAP", "1")
    budget = QuestionBudget()
    budget.spend("a", _NOON)
    with pytest.raises(HTTPException):
        budget.spend("b", _NOON + 60)
    budget.spend("b", _NOON + 86_400)


def test_refused_questions_do_not_spend_the_allowance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STELLIUM_CLIENT_HOURLY_CAP", "1")
    monkeypatch.setenv("STELLIUM_DAILY_QUESTION_CAP", "2")
    budget = QuestionBudget()
    budget.spend("a", _NOON)
    for _ in range(5):
        with pytest.raises(HTTPException):
            budget.spend("a", _NOON + 1)
    budget.spend("b", _NOON + 2)
