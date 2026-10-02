# Typed graph tools against an in-memory graph seeded with invented events.
from __future__ import annotations

import pytest

from src.coprocessor import Coprocessor
from src.models import Chunk
from src.pipelines.toolkit import (
    AGGREGATOR,
    ENTITY_LINKER,
    EVALUATOR,
    GraphToolkit,
    ToolOutcome,
    asked_kind,
)
from tests.graph_fixtures import EVENTS, seeded_graph, seeded_toolkit


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


def test_invoke_normalises_model_arguments() -> None:
    # Synonyms, numeric strings, empty values, and unknown keys from the model are tolerated.
    outcome = seeded_toolkit().invoke(
        "event_attribute",
        {"event_name": "men's single sculls", "target_year": "2008", "sport": "", "note": "x"},
    )
    assert outcome.answer == "Cal Reed"


def _toolkit_with_article_text(texts: dict[str, str]) -> GraphToolkit:
    # Replace the opening chunk of selected articles to control what their prose says.
    graph, catalog, _ = seeded_graph()
    chunks = []
    for event in EVENTS:
        raw = texts.get(
            event["id"], f"Gold: {event['gold']}\nThere were {event['competitors']} competitors."
        )
        chunks.append(
            Chunk(
                chunk_id=f"{event['id']}#0",
                doc_id=event["id"],
                chunk_index=0,
                section_title=event["name"],
                text=raw,
                raw_text=raw,
            )
        )
    rebuilt = Coprocessor()
    rebuilt.build(chunks)
    return GraphToolkit(graph, catalog, rebuilt)


def test_rank_events_breaks_a_tie_when_only_one_article_restates_the_count() -> None:
    # Both 2008 single sculls events list 33 competitors; only the men's article says so in prose.
    toolkit = _toolkit_with_article_text(
        {"R08W": "  competitors: 33\nThe event was held over five days."}
    )
    outcome = toolkit.rank_events("rowing", 2008)

    assert outcome.answer == EVENTS[2]["name"]
    tie = outcome.observation["tie"]
    assert tie["resolved_by"] == "count restated in the article's prose"
    assert tie["evidence"]["sentence"] == "There were 33 competitors."
    assert set(tie["events"]) == {EVENTS[2]["name"], EVENTS[3]["name"]}


def test_rank_events_keeps_the_tie_when_every_article_restates_the_count() -> None:
    outcome = _toolkit_with_article_text({}).rank_events("rowing", 2008)
    assert outcome.answer is None
    assert "resolved_by" not in outcome.observation["tie"]


def test_infobox_lines_are_not_prose_confirmation() -> None:
    toolkit = _toolkit_with_article_text(
        {"R08M": "  competitors: 33", "R08W": "  competitors: 33\nNations: 16"}
    )
    assert toolkit.rank_events("rowing", 2008).ambiguous


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        ("How many nations competed in the event with the most competitors?", "number"),
        ("Count the archery events with over 40 archers.", "number"),
        ("Who won gold in the archery event with the most competitors?", "person"),
        ("Name the pole vault gold medallist from 2012.", "person"),
        ("At which venue was the diving event with the most competitors held?", "venue"),
        ("Which cross-country skiing event had the highest number of competitors?", "event"),
        ("Which nation won the team event?", "nation"),
        ("When was the final held?", "date"),
        ("Of the archery events, which one had the most entrants?", None),
        ("Tell me about rowing.", None),
    ],
)
def test_asked_kind_reads_the_head_of_the_wh_phrase(question: str, kind: str | None) -> None:
    assert asked_kind(question) == kind


@pytest.mark.parametrize(
    ("tool", "attribute", "kind"),
    [
        ("rank_events", "competitor_count", "event"),
        ("count_events", "", "number"),
        ("event_attribute", "gold_athlete", "person"),
        ("event_attribute", "nation_count", "number"),
        ("event_at_venue_date", "venue", "venue"),
        ("gsql_query", "", None),
    ],
)
def test_answer_kind_follows_the_tool_then_the_attribute(
    tool: str, attribute: str, kind: str | None
) -> None:
    assert ToolOutcome(tool=tool, observation={"attribute": attribute}).answer_kind == kind
