# Evaluation oracle on invented events: one answer per template, every answer when ambiguous.
from __future__ import annotations

from typing import Any

from scripts.oracle_check import oracle
from src.linking import compact, fold


def _event(
    title: str, competitors: int, gold: str, venue: str, dates: str, nations: int = 10
) -> dict[str, Any]:
    sport, rest = title.split(" at the ", 1)
    return {
        "title": title,
        "sport": fold(sport),
        "year": int(rest[:4]),
        "season": rest[5:].split(" ", 1)[0],
        "label": title.rsplit(" – ", 1)[-1],
        "competitors": competitors,
        "nations": nations,
        "gold": gold,
        "venue": compact(venue),
        "dates": compact(dates),
    }


EVENTS = [
    _event(
        "Rowing at the 2004 Summer Olympics – Men's eight", 40, "Ada Stone", "Lake A", "15 August"
    ),
    _event(
        "Rowing at the 2004 Summer Olympics – Women's eight", 30, "Bea Moss", "Lake A", "15 August"
    ),
    _event(
        "Rowing at the 2008 Summer Olympics – Men's eight", 41, "Cal Reed", "Lake B", "9 August", 21
    ),
    _event(
        "Rowing at the 2008 Summer Olympics – Women's eight", 41, "Dee Lake", "Lake B", "10 August"
    ),
]


def test_counts_rankings_and_lookups() -> None:
    assert oracle(
        "According to the provided corpus, how many rowing events at the 2004 Summer Olympics had more than 35 competitors?",
        EVENTS,
    ) == ["1"]
    assert oracle(
        "According to the provided corpus, which rowing event at the 2004 Summer Olympics had the highest number of competitors?",
        EVENTS,
    ) == [EVENTS[0]["title"]]
    assert oracle(
        "How many nations competed in Rowing at the 2008 Summer Olympics – Men's eight?", EVENTS
    ) == ["21"]


def test_previous_games_and_venue_dates() -> None:
    assert oracle(
        "Who won the gold medal in the men's eight rowing event at the Summer Olympics held immediately before 2008?",
        EVENTS,
    ) == ["Ada Stone"]
    assert oracle("Who won the gold medal in the event held at Lake B on 9 August?", EVENTS) == [
        "Cal Reed"
    ]


def test_ambiguity_returns_every_supported_answer() -> None:
    # A competitor-count tie and two events on one venue and date both stay ambiguous.
    tie = oracle(
        "According to the provided corpus, which rowing event at the 2008 Summer Olympics had the highest number of competitors?",
        EVENTS,
    )
    assert len(tie) == 2
    assert set(
        oracle("Who won the gold medal in the event held at Lake A on 15 August?", EVENTS)
    ) == {"Ada Stone", "Bea Moss"}
