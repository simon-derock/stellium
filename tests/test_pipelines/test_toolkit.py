# Typed graph tools against an in-memory graph seeded with invented events.
from __future__ import annotations

from typing import Any

import pytest

from src.coprocessor import Coprocessor
from src.graph.mock import MockTigerGraphConnection, create_mock_graph_client
from src.linking import EventCatalog
from src.models import Chunk
from src.pipelines.toolkit import AGGREGATOR, ENTITY_LINKER, EVALUATOR, GraphToolkit

_EVENTS: list[dict[str, Any]] = [
    # id, title, competitors, gold, venue, dates, previous edition id
    {
        "id": "R04M",
        "name": "Rowing at the 2004 Summer Olympics – Men's single sculls",
        "competitors": 30,
        "gold": "Ada Stone",
        "venue": "Lake A",
        "dates": "15 to 21 August",
    },
    {
        "id": "R04W",
        "name": "Rowing at the 2004 Summer Olympics – Women's single sculls",
        "competitors": 24,
        "gold": "Bea Moss",
        "venue": "Lake A",
        "dates": "15 to 21 August",
    },
    {
        "id": "R08M",
        "name": "Rowing at the 2008 Summer Olympics – Men's single sculls",
        "competitors": 33,
        "gold": "Cal Reed",
        "venue": "Lake B",
        "dates": "9–16 August",
        "prev": "R04M",
    },
    {
        "id": "R08W",
        "name": "Rowing at the 2008 Summer Olympics – Women's single sculls",
        "competitors": 33,
        "gold": "Dee Lake",
        "venue": "Lake B",
        "dates": "9–17 August",
        "prev": "R04W",
    },
    {
        "id": "R08X",
        "name": "Rowing at the 2008 Summer Olympics – Men's eight",
        "competitors": 0,
        "gold": "Team Nine",
        "venue": "Lake B",
        "dates": "10–17 August",
    },
    {
        "id": "R12W",
        "name": "Rowing at the 2012 Summer Olympics – Women's single sculls",
        "competitors": 28,
        "gold": "Eve Wren",
        "venue": "Lake C",
        "dates": "28 July – 3 August",
    },
]


def _toolkit() -> GraphToolkit:
    conn = MockTigerGraphConnection()
    chunks = []
    for event in _EVENTS:
        year = int(event["name"].split(" at the ")[1][:4])
        gender = "Women" if "Women" in event["name"] else "Men"
        conn.upsertVertex(
            "Event",
            event["id"],
            {
                "name": event["name"],
                "year": year,
                "season": "Summer",
                "sport": "Rowing",
                "gender": gender,
                "venue": event["venue"],
                "competitor_count": event["competitors"],
                "nation_count": event["competitors"] // 2,
                "gold_athlete": event["gold"],
            },
        )
        conn.upsertVertex("Document", event["id"], {"wikidata_qid": event["id"]})
        conn.upsertEdge("Event", event["id"], "DOCUMENTED_IN", "Document", event["id"])
        conn.upsertEdge(
            "Event", event["id"], "HELD_AT", "Venue", "v", {"start_date": event["dates"]}
        )
        if "prev" in event:
            conn.upsertEdge("Event", event["id"], "PRECEDES", "Event", event["prev"])
        text = f"Gold: {event['gold']}\nThere were {event['competitors']} competitors."
        chunks.append(
            Chunk(
                chunk_id=f"{event['id']}#0",
                doc_id=event["id"],
                chunk_index=0,
                section_title=event["name"],
                text=f"Title: {event['name']} | Gold: {event['gold']}\n\n{text}",
                raw_text=text,
            )
        )
    coprocessor = Coprocessor()
    coprocessor.build(chunks)
    graph = create_mock_graph_client(conn)
    return GraphToolkit(graph, EventCatalog.from_graph(conn), coprocessor)


@pytest.mark.parametrize(
    ("comparison", "threshold", "expected"),
    [("more_than", 30, 2), ("at_least", 30, 3), ("fewer_than", 30, 2), ("at_most", 24, 1)],
)
def test_count_events_applies_strict_and_inclusive_bounds(
    comparison: str, threshold: int, expected: int
) -> None:
    outcome = _toolkit().count_events("rowing", 0, comparison=comparison, threshold=threshold)
    assert outcome.answer == str(expected)
    assert AGGREGATOR in outcome.agents


def test_count_events_excludes_unknown_counts_from_upper_bounds() -> None:
    # The 2008 men's eight has no recorded count (0) and must not count as "fewer than 30".
    outcome = _toolkit().count_events("Rowing", 2008, comparison="fewer_than", threshold=30)
    assert outcome.answer == "0"


def test_count_events_narrows_by_gender_from_returned_rows() -> None:
    outcome = _toolkit().count_events(
        "rowing", 2008, gender="women's", comparison="more_than", threshold=1
    )
    assert outcome.answer == "1"
    assert outcome.observation["matching_events"] == [_EVENTS[3]["name"]]


def test_count_events_reports_unknown_sport_with_graph_vocabulary() -> None:
    outcome = _toolkit().count_events("curling", 2008, threshold=1)
    assert outcome.answer is None
    assert outcome.observation["graph_sports"] == ["Rowing"]


def test_rank_events_reports_ties_and_ignores_missing_counts() -> None:
    toolkit = _toolkit()
    tie = toolkit.rank_events("rowing", 2008, "summer")
    assert tie.answer is None
    assert tie.ambiguous
    assert set(tie.candidates) == {_EVENTS[2]["name"], _EVENTS[3]["name"]}

    fewest = toolkit.rank_events("rowing", 2008, order="fewest")
    # The 0-competitor (unknown) eight is skipped; the tie at 33 is the only ranked value.
    assert fewest.ambiguous

    single = toolkit.rank_events("rowing", 2004)
    assert single.answer == _EVENTS[0]["name"]


def test_event_attribute_links_and_verifies_source_line() -> None:
    outcome = _toolkit().event_attribute("women's single sculls", "gold_athlete", "rowing", 2012)
    assert outcome.answer == "Eve Wren"
    assert outcome.agents == [ENTITY_LINKER, "GraphTraversalAgent", EVALUATOR]
    assert outcome.evidence[0].text == "Gold: Eve Wren"
    assert outcome.citations == ["R12W"]


def test_previous_edition_follows_precedes_edge() -> None:
    outcome = _toolkit().previous_edition("men's single sculls", 2008, sport="rowing")
    assert outcome.answer == "Ada Stone"
    assert outcome.observation["traversal"] == "PRECEDES"


def test_previous_edition_falls_back_to_previous_games_without_current_edition() -> None:
    # No 2016 event exists, so the tool links the same event at the 2012 Games directly.
    outcome = _toolkit().previous_edition("women's single sculls", 2016, season="Summer")
    assert outcome.answer == "Eve Wren"
    assert outcome.observation["traversal"] == "previous_games"


def test_event_at_venue_date_returns_every_same_day_event() -> None:
    toolkit = _toolkit()
    shared = toolkit.event_at_venue_date("Lake A", "15 to 21 August", 2004)
    assert shared.answer is None
    assert set(shared.candidates) == {"Ada Stone", "Bea Moss"}

    unique = toolkit.event_at_venue_date("lake b", "August 9-16, 2008")
    assert unique.answer == "Cal Reed"


def test_find_events_lists_canonical_names() -> None:
    outcome = _toolkit().find_events(sport="rowing", year=2004)
    assert outcome.observation["total"] == 2
