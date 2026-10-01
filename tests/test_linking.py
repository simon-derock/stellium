# Entity linking over synthetic events; fixtures are invented so tests cannot memorise benchmarks.
from __future__ import annotations

from src.graph.mock import MockTigerGraphConnection
from src.linking import EventCatalog, EventRecord, compact, fold, tokens


def _event(event_id: str, name: str, venue: str = "", dates: str = "") -> EventRecord:
    sport, rest = name.split(" at the ", 1)
    year, season = int(rest[:4]), rest[5:].split(" ", 1)[0]
    label = name.rsplit(" – ", 1)[-1]
    gender = "Women" if label.startswith("Women") else "Men" if label.startswith("Men") else ""
    return EventRecord(event_id, name, year, season, sport, gender, venue, dates)


CATALOG = EventCatalog(
    [
        _event(
            "e1",
            "Rowing at the 2004 Summer Olympics – Men's single sculls",
            "Lake A",
            "15 to 21 August",
        ),
        _event(
            "e2",
            "Rowing at the 2004 Summer Olympics – Women's single sculls",
            "Lake A",
            "15 to 21 August",
        ),
        _event(
            "e3",
            "Rowing at the 2008 Summer Olympics – Men's single sculls",
            "Lake B",
            "9–16 August",
        ),
        _event(
            "e4",
            "Speed skating at the 2006 Winter Olympics – Women's 500 metres",
            "Oval C",
            "14 February",
        ),
        _event("e5", "Judo at the 2008 Summer Olympics – Men's +100 kg", "Hall D", "15 August"),
        _event("e6", "Judo at the 2008 Summer Olympics – Men's 100 kg", "Hall D", "14 August"),
        _event(
            "e7",
            "Athletics at the 2008 Summer Olympics – Men's 400 metres",
            "Stadium E",
            "17–21 August",
        ),
        _event(
            "e8",
            "Athletics at the 2008 Summer Olympics – Men's 400 metres hurdles",
            "Stadium E",
            "15–18 August",
        ),
        _event(
            "e9",
            "Athletics at the 2008 Summer Olympics – Men's 4 × 400 metres relay",
            "Stadium E",
            "22–23 August",
        ),
        _event(
            "e10",
            "Speed skating at the 2006 Winter Olympics – Men's 500 metres",
            "Oval C",
            "13 February",
        ),
    ]
)


def test_fold_and_compact_ignore_typography() -> None:
    assert fold("Women’s épée – Final") == fold("womens epee - final") == "womens epee final"
    assert compact("Science and TechnologyUniversity Gym") == compact(
        "Science and Technology University Gym"
    )
    assert tokens("200 metres") == tokens("200 metre")


def test_link_event_is_word_order_insensitive_and_infers_sport() -> None:
    result = CATALOG.link_event("women's 500 metres speed skating", year=2006, season="Winter")
    assert result.resolved is not None
    assert result.resolved.event_id == "e4"


def test_link_event_keeps_numbers_and_plus_classes_distinct() -> None:
    assert CATALOG.link_event("men's 100 kg", sport="Judo", year=2008).resolved == CATALOG.get("e6")
    assert CATALOG.link_event("men's +100 kg", sport="Judo", year=2008).resolved == CATALOG.get(
        "e5"
    )
    # A perfect label match wins even when a longer title shares most words.
    assert CATALOG.link_event("Men's 400 metres", sport="Athletics", year=2008).resolved == (
        CATALOG.get("e7")
    )


def test_link_event_reports_ambiguity_instead_of_guessing() -> None:
    result = CATALOG.link_event("single sculls", sport="Rowing", year=2004)
    assert result.resolved is None
    assert result.ambiguous
    assert {match.record.event_id for match in result.matches} == {"e1", "e2"}


def test_link_venue_date_exact_and_equivalent_formats() -> None:
    exact = CATALOG.link_venue_date("Oval C", "14 February", year=2006)
    assert exact.method == "exact_venue_date"
    assert exact.resolved == CATALOG.get("e4")
    reordered = CATALOG.link_venue_date("Oval C", "February 13, 2006")
    assert reordered.method == "venue_and_equivalent_date"
    assert reordered.resolved == CATALOG.get("e10")


def test_link_venue_date_surfaces_same_day_events() -> None:
    result = CATALOG.link_venue_date("Lake A", "15 to 21 August")
    assert result.ambiguous
    assert {match.record.event_id for match in result.matches} == {"e1", "e2"}


def test_resolvers_and_previous_games_use_catalog_values() -> None:
    assert CATALOG.resolve_sport("speed skating") == "Speed skating"
    assert CATALOG.resolve_sport("curling") is None
    assert CATALOG.resolve_gender("women's") == "Women"
    assert CATALOG.resolve_season("summer") == "Summer"
    assert CATALOG.previous_games_year(2008, "Summer") == 2004
    assert CATALOG.previous_games_year(2004, "Summer") is None


def test_catalog_loads_events_and_dates_from_graph_connection() -> None:
    conn = MockTigerGraphConnection()
    conn.upsertVertex(
        "Event",
        "Q1",
        {
            "name": "Judo at the 2008 Summer Olympics – Men's 100 kg",
            "year": 2008,
            "season": "Summer",
            "sport": "Judo",
            "gender": "Men",
            "venue": "Hall D",
        },
    )
    conn.upsertEdge("Event", "Q1", "HELD_AT", "Venue", "v_hall_d", {"start_date": "14 August"})

    catalog = EventCatalog.from_graph(conn)

    record = catalog.get("Q1")
    assert record is not None
    assert (record.dates, record.label, record.year) == ("14 August", "Men's 100 kg", 2008)
