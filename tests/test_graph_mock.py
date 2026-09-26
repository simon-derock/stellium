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
