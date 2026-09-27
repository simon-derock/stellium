# Contract testing, consumer-driven contracts, and schema drift validation.
# Validates OpenAPI specifications, TigerGraph REST envelopes, and bitemporal schema evolution.

from fastapi.testclient import TestClient

from src.api.main import app
from src.graph.bitemporal import BitemporalFact, parse_temporal_datetime
from src.models import (
    GraphEdgeDTO,
    GraphNodeDTO,
    SnapshotDTO,
)


def test_openapi_specification_contract() -> None:
    # Verifies the live OpenAPI schema conforms to contract expectations.
    schema = app.openapi()
    assert schema["openapi"].startswith("3.")
    assert "paths" in schema

    paths = schema["paths"]
    # Check essential endpoints exist
    assert "/health" in paths
    assert "/api/v1/query/rag" in paths
    assert "/api/v1/query/graphrag" in paths
    assert "/api/v1/query/agentic" in paths
    assert "/api/v1/query/compare" in paths
    assert "/api/v1/evaluate/batch" in paths
    assert "/api/v1/graph/snapshot" in paths
    assert "/api/v1/sessions/{session_id}/history" in paths


def test_health_endpoint_contract() -> None:
    # Verifies health check response format.
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert data.get("system") == "stellium"
    assert "version" in data


def test_snapshot_dto_schema_contract() -> None:
    # Verifies GraphSurface visualization DTO schema invariants.
    node = GraphNodeDTO(
        id="test_node_1",
        type="Event",
        layer="events",
        label="Men's 100m",
        weight=1.5,
    )
    edge = GraphEdgeDTO(
        id="test_edge_1",
        source="test_node_1",
        target="test_node_2",
        relationship_type="PRECEDES",
        confidence=1.0,
    )
    snapshot = SnapshotDTO(
        metrics=[],
        graph_nodes=[node],
        graph_edges=[edge],
        findings=[],
        disclosure="Stellium Agentic GraphRAG",
    )

    dumped = snapshot.model_dump()
    assert dumped["disclosure"] == "Stellium Agentic GraphRAG"
    assert len(dumped["graph_nodes"]) == 1
    assert dumped["graph_nodes"][0]["id"] == "test_node_1"
    assert dumped["graph_edges"][0]["source"] == "test_node_1"


def test_bitemporal_fact_schema_drift_tolerance() -> None:
    # Tests backward compatibility and schema drift resilience of bitemporal facts.
    base_fact = BitemporalFact(
        fact_id="fact_001",
        subject="Event_001",
        predicate="gold_medalist",
        value="Usain Bolt",
        source_authority=1.0,
        valid_from=parse_temporal_datetime("2012-08-05T21:00:00Z"),
    )
    assert base_fact.subject == "Event_001"
    assert base_fact.superseded_by is None

    # Simulating forward schema drift: metadata dictionary handles unseen telemetry fields
    base_fact.metadata["future_experimental_field"] = "unseen_metadata"
    base_fact.metadata["cluster_node_id"] = "node-42"
    dumped = base_fact.to_dict()
    assert dumped["value"] == "Usain Bolt"
    assert base_fact.metadata["cluster_node_id"] == "node-42"


def test_vector_space_mathematical_contracts() -> None:
    # Asserts cosine similarity contracts for 1024-dimensional normalized vectors.
    import math

    dim = 1024
    vec_a = [1.0 / math.sqrt(dim)] * dim
    vec_b = [1.0 / math.sqrt(dim)] * dim

    def dot_product(v1: list[float], v2: list[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2))

    # Identical vectors must have cosine similarity == 1.0
    sim_self = dot_product(vec_a, vec_b)
    assert abs(sim_self - 1.0) < 1e-6

    # Orthogonal vectors: first half positive, second half flipped
    half = dim // 2
    vec_c = [1.0 / math.sqrt(dim)] * half + [-1.0 / math.sqrt(dim)] * half
    sim_ortho = dot_product(vec_a, vec_c)
    assert abs(sim_ortho) < 1e-6
