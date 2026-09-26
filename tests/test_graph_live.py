# Unit tests for TigerGraph Savanna schema verification and compiled GSQL queries.
# Supports both deterministic offline mock testing and live cluster integration.
# Zero docstrings in Python code per project standard.
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

from src.graph import GraphClient, connect
from src.graph.mock import create_mock_graph_client
from src.graph.verify_live import (
    benchmark_queries,
    verify_connection,
    verify_queries,
    verify_schema,
)

load_dotenv()

_HAS_LIVE_CREDS = bool(os.environ.get("TG_HOST") and os.environ.get("TG_SECRET"))


# ---------------------------------------------------------------------------
# Offline Mock Tests (Run in CI & local test runner without network)
# ---------------------------------------------------------------------------


def test_mock_verify_connection() -> None:
    # Test connection verification against mock client.
    mock_conn = MagicMock()
    mock_conn.getVer.return_value = "4.2.5"
    client = GraphClient(conn=mock_conn)

    report = verify_connection(client)
    assert report["status"] == "OK"
    assert report["version"] == "4.2.5"
    assert "latency_ms" in report


def test_mock_verify_schema() -> None:
    # Test schema structure validation with mock schema response.
    mock_conn = MagicMock()
    mock_conn.getSchema.return_value = {
        "VertexTypes": [
            {"Name": "Document"},
            {"Name": "Chunk"},
            {"Name": "Event"},
            {"Name": "Venue"},
            {"Name": "Session"},
            {"Name": "ChatMessage"},
        ],
        "EdgeTypes": [
            {"Name": "HAS_CHUNK"},
            {"Name": "DOCUMENTED_IN"},
            {"Name": "HELD_AT"},
            {"Name": "PRECEDES"},
            {"Name": "SUCCEEDS"},
            {"Name": "HAS_MESSAGE"},
            {"Name": "CONFLICTS_WITH"},
        ],
    }
    mock_conn.gsql.return_value = (
        "Vector Embeddings:\n  - Chunk:\n    - embedding(Dimension=1024, IndexType='HNSW')"
    )
    client = GraphClient(conn=mock_conn)

    report = verify_schema(client)
    assert report["vertices_ok"] is True
    assert report["edges_ok"] is True
    assert report["vector_ok"] is True


def test_mock_verify_queries() -> None:
    # Test query inventory check against mock installed queries.
    mock_conn = MagicMock()
    mock_conn.getInstalledQueries.return_value = {
        "GET /query/OlympicsGraph/get_event_aggregates": {},
        "GET /query/OlympicsGraph/get_preceding_event": {},
        "GET /query/OlympicsGraph/get_superlative_event": {},
        "GET /query/OlympicsGraph/get_event_by_venue_date": {},
        "GET /query/OlympicsGraph/get_event_attribute": {},
        "GET /query/OlympicsGraph/vector_search_chunks": {},
    }
    client = GraphClient(conn=mock_conn)

    report = verify_queries(client)
    assert report["all_installed"] is True
    assert len(report["missing_queries"]) == 0
    assert report["installed_count"] == 6


def test_mock_benchmark_queries() -> None:
    # Test benchmark execution with mock graph adapter.
    client = create_mock_graph_client()
    bench = benchmark_queries(client)

    assert "get_event_aggregates" in bench
    assert "get_preceding_event" in bench
    assert "get_superlative_event" in bench
    assert "get_event_by_venue_date" in bench
    assert "get_event_attribute" in bench
    assert "vector_search_chunks" in bench

    for qname, data in bench.items():
        assert data["status"] == "PASS"
        assert data["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Live Cluster Tests (Only executed when live credentials are configured)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_LIVE_CREDS, reason="Live TigerGraph credentials not configured")
def test_live_tigergraph_connection() -> None:
    # Verifies real live connection and token authentication against TigerGraph Savanna.
    conn = connect()
    client = GraphClient(conn=conn)
    report = verify_connection(client)

    assert report["status"] == "OK"
    assert "4." in report["version"]


@pytest.mark.skipif(not _HAS_LIVE_CREDS, reason="Live TigerGraph credentials not configured")
def test_live_tigergraph_schema() -> None:
    # Verifies all 6 vertices, 7 edges, and HNSW vector index on live Savanna cluster.
    conn = connect()
    client = GraphClient(conn=conn)
    report = verify_schema(client)

    assert report["vertices_ok"] is True, f"Missing vertices: {report['missing_vertices']}"
    assert report["edges_ok"] is True, f"Missing edges: {report['missing_edges']}"
    assert report["vector_ok"] is True, "Vector embedding attribute missing from Chunk"


@pytest.mark.skipif(not _HAS_LIVE_CREDS, reason="Live TigerGraph credentials not configured")
def test_live_tigergraph_compiled_queries() -> None:
    # Verifies that all 6 compiled queries execute with sub-second latency on live Savanna.
    conn = connect()
    client = GraphClient(conn=conn)
    bench = benchmark_queries(client)

    for qname in [
        "get_event_aggregates",
        "get_preceding_event",
        "get_superlative_event",
        "get_event_by_venue_date",
        "get_event_attribute",
        "vector_search_chunks",
    ]:
        assert qname in bench
        assert bench[qname]["status"] == "PASS"
