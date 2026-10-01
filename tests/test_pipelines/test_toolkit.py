# Typed graph tools against an in-memory graph seeded with invented events.
from __future__ import annotations

import pytest

from src.pipelines.toolkit import AGGREGATOR, ENTITY_LINKER, EVALUATOR
from tests.graph_fixtures import EVENTS, seeded_toolkit


@pytest.mark.parametrize(
    ("comparison", "threshold", "expected"),
    [("more_than", 30, 2), ("at_least", 30, 3), ("fewer_than", 30, 2), ("at_most", 24, 1)],
)
def test_count_events_applies_strict_and_inclusive_bounds(
    comparison: str, threshold: int, expected: int
) -> None:
    outcome = seeded_toolkit().count_events("rowing", 0, comparison=comparison, threshold=threshold)
    assert outcome.answer == str(expected)
    assert AGGREGATOR in outcome.agents


def test_count_events_excludes_unknown_counts_from_upper_bounds() -> None:
    # The 2008 men's eight has no recorded count (0) and must not count as "fewer than 30".
    outcome = seeded_toolkit().count_events("Rowing", 2008, comparison="fewer_than", threshold=30)
    assert outcome.answer == "0"


def test_count_events_narrows_by_gender_from_returned_rows() -> None:
    outcome = seeded_toolkit().count_events(
        "rowing", 2008, gender="women's", comparison="more_than", threshold=1
    )
    assert outcome.answer == "1"
    assert outcome.observation["matching_events"] == [EVENTS[3]["name"]]


def test_count_events_reports_unknown_sport_with_graph_vocabulary() -> None:
    outcome = seeded_toolkit().count_events("curling", 2008, threshold=1)
    assert outcome.answer is None
    assert outcome.observation["graph_sports"] == ["Rowing"]


def test_rank_events_reports_ties_and_ignores_missing_counts() -> None:
    toolkit = seeded_toolkit()
    tie = toolkit.rank_events("rowing", 2008, "summer")
    assert tie.answer is None
    assert tie.ambiguous
    assert set(tie.candidates) == {EVENTS[2]["name"], EVENTS[3]["name"]}

    fewest = toolkit.rank_events("rowing", 2008, order="fewest")
    # The 0-competitor (unknown) eight is skipped; the tie at 33 is the only ranked value.
    assert fewest.ambiguous

    single = toolkit.rank_events("rowing", 2004)
    assert single.answer == EVENTS[0]["name"]


def test_event_attribute_links_and_verifies_source_line() -> None:
    outcome = seeded_toolkit().event_attribute(
        "women's single sculls", "gold_athlete", "rowing", 2012
    )
    assert outcome.answer == "Eve Wren"
    assert outcome.agents == [ENTITY_LINKER, "GraphTraversalAgent", EVALUATOR]
    assert outcome.evidence[0].text == "Gold: Eve Wren"
    assert outcome.citations == ["R12W"]


def test_previous_edition_follows_precedes_edge() -> None:
    outcome = seeded_toolkit().previous_edition("men's single sculls", 2008, sport="rowing")
    assert outcome.answer == "Ada Stone"
    assert outcome.observation["traversal"] == "PRECEDES"


def test_previous_edition_falls_back_to_previous_games_without_current_edition() -> None:
    # No 2016 event exists, so the tool links the same event at the 2012 Games directly.
    outcome = seeded_toolkit().previous_edition("women's single sculls", 2016, season="Summer")
    assert outcome.answer == "Eve Wren"
    assert outcome.observation["traversal"] == "previous_games"


def test_event_at_venue_date_returns_every_same_day_event() -> None:
    toolkit = seeded_toolkit()
    shared = toolkit.event_at_venue_date("Lake A", "15 to 21 August", 2004)
    assert shared.answer is None
    assert set(shared.candidates) == {"Ada Stone", "Bea Moss"}

    unique = toolkit.event_at_venue_date("lake b", "August 9-16, 2008")
    assert unique.answer == "Cal Reed"


def test_find_events_lists_canonical_names() -> None:
    outcome = seeded_toolkit().find_events(sport="rowing", year=2004)
    assert outcome.observation["total"] == 2
