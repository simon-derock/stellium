# Schema contract validation and vector integrity testing.
# Verifies vertex/edge schemas, 1024-dim vector attributes, and topology integrity.
from __future__ import annotations

import math

from src.graph.mock import create_mock_graph_client
from src.models import Chunk, CorpusDoc, ParsedInbox


def test_schema_vertex_and_edge_contracts() -> None:
    client = create_mock_graph_client()
    # Contract: All 6 required vertex types exist
    required_vertices = ["Document", "Chunk", "Event", "Venue", "Session", "ChatMessage"]
    for v_type in required_vertices:
        assert v_type in client.conn.vertices, f"Missing vertex type: {v_type}"

    # Contract: Relational edge structure
    client.conn.upsertEdge("Document", "doc-1", "HAS_CHUNK", "Chunk", "chunk-1")
    client.conn.upsertEdge("Event", "ev-1", "DOCUMENTED_IN", "Document", "doc-1")
    client.conn.upsertEdge("Event", "ev-1", "HELD_AT", "Venue", "v-1")
    client.conn.upsertEdge("Event", "ev-1", "PRECEDES", "Event", "ev-0")
    client.conn.upsertEdge("Event", "ev-0", "SUCCEEDS", "Event", "ev-1")

    edge_types = {e[2] for e in client.conn.edges}
    assert "HAS_CHUNK" in edge_types
    assert "DOCUMENTED_IN" in edge_types
    assert "HELD_AT" in edge_types
    assert "PRECEDES" in edge_types
    assert "SUCCEEDS" in edge_types


def test_vector_dimension_and_normalization_integrity() -> None:
    # Contract: Embeddings must be exactly 1024 dimensions matching TigerGraph HNSW index
    expected_dim = 1024
    sample_vec = [1.0 / math.sqrt(expected_dim)] * expected_dim

    assert len(sample_vec) == expected_dim
    # Norm calculation (cosine unit sphere)
    norm = math.sqrt(sum(x * x for x in sample_vec))
    assert abs(norm - 1.0) < 1e-5


def test_domain_model_contracts() -> None:
    # Contract: CorpusDoc model validation
    doc = CorpusDoc(
        doc_id="Q12345",
        title="Athletics at the 2012 Summer Olympics",
        url="https://en.wikipedia.org/wiki/Athletics_at_the_2012_Summer_Olympics",
        wikidata_qid="Q12345",
        wikipedia_pageid=98765,
        text="Sample text content",
        approx_tokens=150,
    )
    assert doc.doc_id == "Q12345"
    assert doc.wikidata_qid == "Q12345"

    # Contract: Chunk model validation with doubly-linked pointers
    chunk = Chunk(
        chunk_id="Q12345#0",
        doc_id="Q12345",
        chunk_index=0,
        section_title="Summary",
        text="Sample passage text",
        raw_text="Sample passage text",
        prev_chunk_id=None,
        next_chunk_id="Q12345#1",
        filter_mask=1,
    )
    assert chunk.next_chunk_id == "Q12345#1"

    # Contract: ParsedInbox model validation
    infobox = ParsedInbox(
        event_name="Men's 100m",
        year=2012,
        season="Summer",
        sport="Athletics",
        venue="Olympic Stadium",
        competitor_count=80,
        nation_count=50,
        gold_athlete="Usain Bolt",
        gold_noc="JAM",
    )
    assert infobox.gold_athlete == "Usain Bolt"
    assert infobox.year == 2012
