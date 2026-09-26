# Comprehensive unit tests for batch upsert engine and partitioner.
# Strictly zero docstrings per project coding standards.
from __future__ import annotations

from pathlib import Path

from src.graph.mock import MockTigerGraphConnection, create_mock_graph_client
from src.ingest import (
    build_event_chronology,
    chunk_document,
    extract_sport_from_title,
    parse_infobox,
)
from src.ingest.batch_upsert import (
    IngestionPlan,
    VertexBatch,
    execute_ingestion,
    partition_items,
    prepare_ingestion_plan,
)
from src.models import CorpusDoc


# Test list partitioning helper
def test_partition_items() -> None:
    items = [1, 2, 3, 4, 5, 6, 7]
    assert partition_items(items, 3) == [[1, 2, 3], [4, 5, 6], [7]]
    assert partition_items(items, 10) == [[1, 2, 3, 4, 5, 6, 7]]
    assert partition_items([], 5) == []
    assert partition_items(items, 0) == [items]


# Test sport extraction from Wikipedia article titles
def test_extract_sport_from_title() -> None:
    t1 = "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres"
    assert extract_sport_from_title(t1) == "Canoeing"

    t2 = "Speed skating at the 2010 Winter Olympics – Men's 5000 metres"
    assert extract_sport_from_title(t2) == "Speed skating"

    t3 = "Baadshah (1999 film)"
    assert extract_sport_from_title(t3) is None

    t4 = "Brad Bird"
    assert extract_sport_from_title(t4) is None


# Test event chronology generation across multiple editions
def test_build_event_chronology() -> None:
    docs = [
        CorpusDoc(
            doc_id="ev_2008",
            wikidata_qid="Q2008",
            wikipedia_pageid=1,
            title="Athletics at the 2008 Summer Olympics – Men's 100 metres",
            url="https://en.wikipedia.org/wiki/2008",
            text="[Infobox Olympic event]\n  games: 2008 Summer\n  event: Men's 100 metres\n  next: 2012\n\nFinals.",
            approx_tokens=50,
        ),
        CorpusDoc(
            doc_id="ev_2012",
            wikidata_qid="Q2012",
            wikipedia_pageid=2,
            title="Athletics at the 2012 Summer Olympics – Men's 100 metres",
            url="https://en.wikipedia.org/wiki/2012",
            text="[Infobox Olympic event]\n  games: 2012 Summer\n  event: Men's 100 metres\n  prev: 2008\n  next: 2016\n\nFinals.",
            approx_tokens=50,
        ),
        CorpusDoc(
            doc_id="ev_2016",
            wikidata_qid="Q2016",
            wikipedia_pageid=3,
            title="Athletics at the 2016 Summer Olympics – Men's 100 metres",
            url="https://en.wikipedia.org/wiki/2016",
            text="[Infobox Olympic event]\n  games: 2016 Summer\n  event: Men's 100 metres\n  prev: 2012\n\nFinals.",
            approx_tokens=50,
        ),
    ]

    chrono = build_event_chronology(docs)
    assert len(chrono) == 3

    # 2008 edition has no predecessor, successor is 2012
    p08, n08, d08 = chrono["ev_2008"]
    assert p08 is None
    assert n08 == "ev_2012"
    assert d08 == 4

    # 2012 edition has predecessor 2008, successor 2016
    p12, n12, d12 = chrono["ev_2012"]
    assert p12 == "ev_2008"
    assert n12 == "ev_2016"
    assert d12 == 4

    # 2016 edition has predecessor 2012, no successor
    p16, n16, d16 = chrono["ev_2016"]
    assert p16 == "ev_2012"
    assert n16 is None
    assert d16 == 4


# Test infobox parsing and filter mask population on chunks
def test_chunking_with_filter_mask() -> None:
    doc = CorpusDoc(
        doc_id="Qtest",
        wikidata_qid="Qtest",
        wikipedia_pageid=99,
        title="Swimming at the 2016 Summer Olympics – Women's 100m freestyle",
        url="https://en.wikipedia.org/wiki/swim",
        text="[Infobox Olympic event]\n  games: 2016 Summer\n  event: Women's 100m freestyle\n  venue: Olympic Aquatics Stadium\n\nPenny Oleksiak won gold.",
        approx_tokens=80,
    )
    chunks = chunk_document(doc)
    assert len(chunks) >= 1
    # Check that filter_mask is non-zero
    assert chunks[0].filter_mask != 0
    # Infobox sport should be populated from title
    ib = parse_infobox(doc.text, title=doc.title)
    assert ib.sport == "Swimming"
    assert "Sport: Swimming" in chunks[0].text


# Test execution of batch ingestion plan with dry-run
def test_execute_ingestion_dry_run() -> None:
    plan = IngestionPlan(
        document_batches=[VertexBatch("Document", [("d1", {"title": "Doc 1"})])],
        chunk_batches=[VertexBatch("Chunk", [("c1", {"text": "Chunk 1"})])],
        total_documents=1,
        total_chunks=1,
        total_edges=1,
    )
    client = create_mock_graph_client()
    stats = execute_ingestion(client, plan, dry_run=True)
    assert stats.dry_run is True
    assert stats.total_vertices_upserted == 2
    assert stats.total_edges_upserted == 1
    assert stats.batches_processed == 2


# Test execution into mock TigerGraph connection
def test_execute_ingestion_into_mock() -> None:
    mock_conn = MockTigerGraphConnection()
    client = create_mock_graph_client(conn=mock_conn)

    plan = IngestionPlan(
        document_batches=[VertexBatch("Document", [("doc_1", {"title": "Olympic 1"})])],
        chunk_batches=[VertexBatch("Chunk", [("c1", {"text": "Text 1", "filter_mask": 4})])],
        event_batches=[
            VertexBatch(
                "Event",
                [
                    (
                        "ev_1",
                        {
                            "name": "Athletics",
                            "year": 2012,
                            "sport": "Athletics",
                            "competitor_count": 50,
                        },
                    )
                ],
            )
        ],
        total_documents=1,
        total_chunks=1,
        total_events=1,
        total_edges=0,
    )

    stats = execute_ingestion(client, plan, dry_run=False)
    assert stats.total_vertices_upserted == 3
    assert stats.batches_processed == 3
    assert len(stats.errors) == 0

    # Verify mock in-memory storage has the records
    assert "doc_1" in mock_conn.vertices["Document"]
    assert "c1" in mock_conn.vertices["Chunk"]
    assert "ev_1" in mock_conn.vertices["Event"]
    assert mock_conn.vertices["Event"]["ev_1"]["year"] == 2012


# Test prepare_ingestion_plan with temporary corpus file
def test_prepare_ingestion_plan(tmp_path: Path) -> None:
    p = tmp_path / "corpus_sample.jsonl"
    sample_line = (
        '{"doc_id": "Q100", "wikidata_qid": "Q100", "wikipedia_pageid": 100, '
        '"title": "Judo at the 2012 Summer Olympics – Men\'s 60 kg", '
        '"url": "https://en.wikipedia.org/wiki/judo", '
        '"text": "[Infobox Olympic event]\\n  games: 2012 Summer\\n  venue: ExCeL London\\n  event: Men\'s 60 kg\\n\\nArsen Galstyan won gold.", '
        '"approx_tokens": 60}\n'
    )
    p.write_text(sample_line, encoding="utf-8")

    plan = prepare_ingestion_plan(p, batch_size=2)
    assert plan.total_documents == 1
    assert plan.total_chunks >= 1
    assert plan.total_events == 1
    assert plan.total_venues == 1
    assert len(plan.document_batches) == 1
    assert len(plan.chunk_batches) >= 1
    assert len(plan.event_batches) == 1
    assert len(plan.venue_batches) == 1
    assert len(plan.has_chunk_batches) >= 1
    assert len(plan.documented_in_batches) == 1
    assert len(plan.held_at_batches) == 1
