# Comprehensive unit tests for high-fidelity offline MockTigerGraphConnection.
# Strictly zero docstrings per project coding standards.
from __future__ import annotations

from src.graph.mock import MockTigerGraphConnection


# Test initialization and default state
def test_mock_connection_init() -> None:
    conn = MockTigerGraphConnection()
    assert conn.graphname == "OlympicsGraph"
    assert "Document" in conn.vertices
    assert "Chunk" in conn.vertices
    assert "Event" in conn.vertices
    assert "Venue" in conn.vertices
    assert len(conn.edges) == 0


# Test token retrieval and GSQL execution logging
def test_mock_auth_and_gsql() -> None:
    conn = MockTigerGraphConnection()
    token, lifetime = conn.getToken(secret="test-secret")
    assert bool(token)
    assert lifetime == 86400

    msg = conn.gsql("USE GLOBAL; CREATE VERTEX Sample (PRIMARY_ID id STRING);")
    assert "SUCCESS" in msg
    assert len(conn.gsql_history) == 1


# Test vertex upserts
def test_mock_upsert_vertex() -> None:
    conn = MockTigerGraphConnection()
    res1 = conn.upsertVertex(
        "Event",
        "e1",
        {
            "name": "Athletics Men's 100m",
            "year": 2012,
            "sport": "Athletics",
            "competitor_count": 80,
        },
    )
    assert res1 == 1
    assert "e1" in conn.vertices["Event"]
    assert conn.vertices["Event"]["e1"]["year"] == 2012

    # Batch upsert
    batch = [
        ("c1", {"text": "Passage 1", "filter_mask": 1}),
        ("c2", {"text": "Passage 2", "filter_mask": 2}),
    ]
    res2 = conn.upsertVertices("Chunk", batch)
    assert res2 == 2
    assert len(conn.vertices["Chunk"]) == 2


# Test edge upserts
def test_mock_upsert_edge() -> None:
    conn = MockTigerGraphConnection()
    res1 = conn.upsertEdge("Event", "e1", "HELD_AT", "Venue", "v1", {"start_date": "2012-08-01"})
    assert res1 == 1
    assert len(conn.edges) == 1

    batch_edges = [
        ("e1", "e0", {"time_diff": 4}),
        ("e2", "e1", {"time_diff": 4}),
    ]
    res2 = conn.upsertEdges("Event", "PRECEDES", "Event", batch_edges)
    assert res2 == 2
    assert len(conn.edges) == 3


def test_mock_multihop_filters_event_year_and_yearless_edge_date() -> None:
    conn = MockTigerGraphConnection()
    conn.upsertVertex(
        "Event", "e2012", {"name": "2012 event", "year": 2012, "gold_athlete": "Athlete A"}
    )
    conn.upsertVertex(
        "Event", "e2016", {"name": "2016 event", "year": 2016, "gold_athlete": "Athlete B"}
    )
    conn.upsertVertex("Venue", "venue", {"name": "Test Arena"})
    conn.upsertEdge("Event", "e2012", "HELD_AT", "Venue", "venue", {"start_date": "11–19 August"})
    conn.upsertEdge("Event", "e2016", "HELD_AT", "Venue", "venue", {"start_date": "11–19 August"})

    result = conn.runInstalledQuery(
        "get_event_by_venue_date",
        {
            "venue_name_fragment": "Test Arena",
            "target_date_fragment": "11–19 August",
            "target_year": 2012,
        },
    )[0]

    assert result["events"] == ["2012 event"]
    assert result["gold_athletes"] == ["Athlete A"]


def test_mock_multihop_matches_corpus_month_first_date_fragment() -> None:
    conn = MockTigerGraphConnection()
    conn.upsertVertex(
        "Event",
        "swimming-2004",
        {
            "name": "Men's 400 metre individual medley",
            "year": 2004,
            "gold_athlete": "Michael Phelps",
        },
    )
    conn.upsertVertex("Venue", "aquatic-centre", {"name": "Olympic Aquatic Centre"})
    conn.upsertEdge(
        "Event",
        "swimming-2004",
        "HELD_AT",
        "Venue",
        "aquatic-centre",
        {"start_date": "August 14, 2004 (heats & final)"},
    )

    result = conn.runInstalledQuery(
        "get_event_by_venue_date",
        {
            "venue_name_fragment": "Olympic Aquatic Centre",
            "target_date_fragment": "August 14",
            "target_year": 2004,
        },
    )[0]

    assert result["events"] == ["Men's 400 metre individual medley"]
    assert result["gold_athletes"] == ["Michael Phelps"]


def test_mock_temporal_matches_event_name_and_sport_case_insensitively() -> None:
    conn = MockTigerGraphConnection()
    conn.upsertVertex(
        "Event",
        "current",
        {
            "name": "Nordic combined at the 2014 Winter Olympics – Individual normal hill/10 km",
            "year": 2014,
            "sport": "Nordic combined",
            "gender": "Men",
        },
    )
    conn.upsertVertex(
        "Event",
        "prior",
        {
            "name": "Nordic combined at the 2010 Winter Olympics – Individual normal hill/10 km",
            "year": 2010,
            "sport": "Nordic combined",
            "gender": "Men",
            "gold_athlete": "Jason Lamy Chappuis",
        },
    )
    conn.upsertEdge("Event", "current", "PRECEDES", "Event", "prior")

    result = conn.runInstalledQuery(
        "get_preceding_event",
        {
            "sport": "NORDIC COMBINED",
            "gender": "Men",
            "event_name_fragment": "individual normal hill/10 km",
            "current_year": 2014,
        },
    )[0]

    assert result["prev_events"] == [
        "Nordic combined at the 2010 Winter Olympics – Individual normal hill/10 km"
    ]
    assert result["gold_athletes"] == ["Jason Lamy Chappuis"]


# Test installed query simulation for event aggregates
def test_mock_query_event_aggregates() -> None:
    conn = MockTigerGraphConnection()
    conn.upsertVertex(
        "Event",
        "e1",
        {
            "name": "Biathlon Men's 10km",
            "year": 2018,
            "sport": "Biathlon",
            "competitor_count": 87,
        },
    )
    conn.upsertVertex(
        "Event",
        "e2",
        {
            "name": "Biathlon Women's 7.5km",
            "year": 2018,
            "sport": "Biathlon",
            "competitor_count": 85,
        },
    )
    conn.upsertVertex(
        "Event",
        "e3",
        {
            "name": "Biathlon Mixed Relay",
            "year": 2018,
            "sport": "Biathlon",
            "competitor_count": 60,
        },
    )

    params = {"sport": "Biathlon", "year": 2018, "min_competitors": 70}
    res = conn.runInstalledQuery("get_event_aggregates", params)
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]["count"] == 2


# Test installed query simulation for superlatives
def test_mock_query_superlative() -> None:
    conn = MockTigerGraphConnection()
    conn.upsertVertex(
        "Event",
        "e1",
        {
            "name": "Athletics Men's Marathon",
            "year": 2008,
            "sport": "Athletics",
            "competitor_count": 98,
        },
    )
    conn.upsertVertex(
        "Event",
        "e2",
        {
            "name": "Athletics Men's 100m",
            "year": 2008,
            "sport": "Athletics",
            "competitor_count": 80,
        },
    )

    params = {"sport": "Athletics", "year": 2008, "attribute": "competitors", "direction": "MAX"}
    res = conn.runInstalledQuery("get_superlative_event", params)
    assert isinstance(res, list)
    assert len(res) == 1
    assert "Marathon" in res[0]["events"][0]
