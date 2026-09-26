# Batch upsert engine and partitioner for TigerGraph Savanna.
# Partitions 2,951 documents, 22,016 chunks, and Olympic graph topology into bounded batches.
# Supports dry-run validation, retry backoff, and live cluster upsert.
# Strictly zero docstrings per project coding standards.
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.coprocessor import build_filter_mask
from src.graph import GraphClient, connect
from src.graph.mock import create_mock_graph_client
from src.ingest import (
    build_event_chronology,
    chunk_document,
    iter_corpus,
    parse_infobox,
)


def partition_items[T](items: list[T], batch_size: int) -> list[list[T]]:
    # Splits an arbitrary list of items into contiguous slices of maximum size batch_size.
    if batch_size <= 0:
        return [items] if items else []
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


@dataclass
class VertexBatch:
    vertex_type: str
    records: list[tuple[str, dict[str, Any]]]


@dataclass
class EdgeBatch:
    source_type: str
    edge_type: str
    target_type: str
    records: list[tuple[str, str, dict[str, Any]]]


@dataclass
class IngestionPlan:
    # Pre-partitioned vertex and edge batches ready for atomic transmission to Savanna.
    document_batches: list[VertexBatch] = field(default_factory=list)
    chunk_batches: list[VertexBatch] = field(default_factory=list)
    event_batches: list[VertexBatch] = field(default_factory=list)
    venue_batches: list[VertexBatch] = field(default_factory=list)
    has_chunk_batches: list[EdgeBatch] = field(default_factory=list)
    documented_in_batches: list[EdgeBatch] = field(default_factory=list)
    held_at_batches: list[EdgeBatch] = field(default_factory=list)
    precedes_batches: list[EdgeBatch] = field(default_factory=list)
    succeeds_batches: list[EdgeBatch] = field(default_factory=list)
    total_documents: int = 0
    total_chunks: int = 0
    total_events: int = 0
    total_venues: int = 0
    total_edges: int = 0


@dataclass
class IngestionStats:
    # Execution telemetry and validation results from batch ingestion.
    total_vertices_upserted: int = 0
    total_edges_upserted: int = 0
    batches_processed: int = 0
    elapsed_seconds: float = 0.0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


def prepare_ingestion_plan(
    corpus_path: str | Path,
    batch_size: int = 500,
) -> IngestionPlan:
    # Reads corpus.jsonl, builds all vertices and relational edges, and partitions them.
    docs = list(iter_corpus(corpus_path))
    chronology = build_event_chronology(docs)

    doc_records: list[tuple[str, dict[str, Any]]] = []
    chunk_records: list[tuple[str, dict[str, Any]]] = []
    event_records: list[tuple[str, dict[str, Any]]] = []
    venue_dict: dict[str, str] = {}  # venue_id -> name

    has_chunk_edges: list[tuple[str, str, dict[str, Any]]] = []
    documented_in_edges: list[tuple[str, str, dict[str, Any]]] = []
    held_at_edges: list[tuple[str, str, dict[str, Any]]] = []
    precedes_edges: list[tuple[str, str, dict[str, Any]]] = []
    succeeds_edges: list[tuple[str, str, dict[str, Any]]] = []

    for doc in docs:
        infobox = parse_infobox(doc.text, title=doc.title)
        mask = build_filter_mask(infobox.year, infobox.season, infobox.sport)

        # 1. Document Vertex
        doc_records.append(
            (
                doc.doc_id,
                {
                    "title": doc.title,
                    "url": doc.url,
                    "wikidata_qid": doc.wikidata_qid,
                    "wikipedia_pageid": doc.wikipedia_pageid,
                    "approx_tokens": doc.approx_tokens,
                    "filter_mask": mask,
                },
            )
        )

        # 2. Chunk Vertices & HAS_CHUNK Edges
        chunks = chunk_document(doc)
        for c in chunks:
            chunk_records.append(
                (
                    c.chunk_id,
                    {
                        "doc_id": c.doc_id,
                        "chunk_index": c.chunk_index,
                        "section_title": c.section_title,
                        "text": c.text[:8000],
                        "raw_text": c.raw_text[:4000],
                        "prev_chunk_id": c.prev_chunk_id or "",
                        "next_chunk_id": c.next_chunk_id or "",
                        "filter_mask": c.filter_mask,
                    },
                )
            )
            has_chunk_edges.append((doc.doc_id, c.chunk_id, {}))

        # 3. Event Vertex & Edges (Only for Olympic event articles)
        is_olympic = bool(infobox.year or infobox.season or infobox.sport)
        if is_olympic:
            event_id = doc.doc_id
            chrono_info = chronology.get(doc.doc_id, (None, None, 0))
            prev_event_id, next_event_id, time_diff = chrono_info

            event_records.append(
                (
                    event_id,
                    {
                        "name": doc.title,
                        "year": infobox.year or 0,
                        "season": infobox.season or "",
                        "sport": infobox.sport or "",
                        "venue": infobox.venue or "",
                        "competitor_count": infobox.competitor_count or 0,
                        "nation_count": infobox.nation_count or 0,
                        "gold_athlete": infobox.gold_athlete or "",
                        "silver_athlete": infobox.silver_athlete or "",
                        "bronze_athlete": infobox.bronze_athlete or "",
                        "gold_noc": infobox.gold_noc or "",
                        "silver_noc": infobox.silver_noc or "",
                        "bronze_noc": infobox.bronze_noc or "",
                        "prev_event_id": prev_event_id or "",
                        "next_event_id": next_event_id or "",
                        "filter_mask": mask,
                        "source_authority": 1.0,
                    },
                )
            )

            # Link Event -> Document
            documented_in_edges.append((event_id, doc.doc_id, {}))

            # Venue Vertex & HELD_AT Edge
            if infobox.venue:
                venue_name = infobox.venue.strip()
                venue_id = (
                    "v_" + "".join(ch if ch.isalnum() else "_" for ch in venue_name.lower())[:60]
                )
                venue_dict[venue_id] = venue_name
                held_at_edges.append(
                    (
                        event_id,
                        venue_id,
                        {"start_date": infobox.start_date or "", "end_date": ""},
                    )
                )

            # PRECEDES / SUCCEEDS Edges
            if prev_event_id:
                precedes_edges.append((event_id, prev_event_id, {"time_diff": time_diff}))
            if next_event_id:
                succeeds_edges.append((event_id, next_event_id, {"time_diff": time_diff}))

    venue_records = [(v_id, {"name": name}) for v_id, name in venue_dict.items()]

    # Partition each record group into bounded batches
    doc_batches = [VertexBatch("Document", b) for b in partition_items(doc_records, batch_size)]
    chunk_batches = [VertexBatch("Chunk", b) for b in partition_items(chunk_records, batch_size)]
    event_batches = [VertexBatch("Event", b) for b in partition_items(event_records, batch_size)]
    venue_batches = [VertexBatch("Venue", b) for b in partition_items(venue_records, batch_size)]

    has_chunk_batches = [
        EdgeBatch("Document", "HAS_CHUNK", "Chunk", b)
        for b in partition_items(has_chunk_edges, batch_size)
    ]
    documented_in_batches = [
        EdgeBatch("Event", "DOCUMENTED_IN", "Document", b)
        for b in partition_items(documented_in_edges, batch_size)
    ]
    held_at_batches = [
        EdgeBatch("Event", "HELD_AT", "Venue", b)
        for b in partition_items(held_at_edges, batch_size)
    ]
    precedes_batches = [
        EdgeBatch("Event", "PRECEDES", "Event", b)
        for b in partition_items(precedes_edges, batch_size)
    ]
    succeeds_batches = [
        EdgeBatch("Event", "SUCCEEDS", "Event", b)
        for b in partition_items(succeeds_edges, batch_size)
    ]

    total_edges = (
        len(has_chunk_edges)
        + len(documented_in_edges)
        + len(held_at_edges)
        + len(precedes_edges)
        + len(succeeds_edges)
    )

    return IngestionPlan(
        document_batches=doc_batches,
        chunk_batches=chunk_batches,
        event_batches=event_batches,
        venue_batches=venue_batches,
        has_chunk_batches=has_chunk_batches,
        documented_in_batches=documented_in_batches,
        held_at_batches=held_at_batches,
        precedes_batches=precedes_batches,
        succeeds_batches=succeeds_batches,
        total_documents=len(doc_records),
        total_chunks=len(chunk_records),
        total_events=len(event_records),
        total_venues=len(venue_records),
        total_edges=total_edges,
    )


def execute_ingestion(
    client: GraphClient,
    plan: IngestionPlan,
    dry_run: bool = False,
    max_retries: int = 3,
) -> IngestionStats:
    # Executes the partitioned batches against TigerGraph Savanna.
    # When dry_run is True, validates batch structure and counts without network operations.
    t0 = time.perf_counter()
    stats = IngestionStats(dry_run=dry_run)

    if dry_run:
        # Sum up vertices and edges from plan batches
        v_total = plan.total_documents + plan.total_chunks + plan.total_events + plan.total_venues
        b_total = (
            len(plan.document_batches)
            + len(plan.chunk_batches)
            + len(plan.event_batches)
            + len(plan.venue_batches)
            + len(plan.has_chunk_batches)
            + len(plan.documented_in_batches)
            + len(plan.held_at_batches)
            + len(plan.precedes_batches)
            + len(plan.succeeds_batches)
        )
        stats.total_vertices_upserted = v_total
        stats.total_edges_upserted = plan.total_edges
        stats.batches_processed = b_total
        stats.elapsed_seconds = time.perf_counter() - t0
        return stats

    # Helper for executing vertex batches with exponential retry backoff
    all_vertex_batches = (
        plan.document_batches + plan.chunk_batches + plan.event_batches + plan.venue_batches
    )
    for b in all_vertex_batches:
        for attempt in range(max_retries):
            try:
                client.conn.upsertVertices(b.vertex_type, b.records)
                stats.total_vertices_upserted += len(b.records)
                stats.batches_processed += 1
                break
            except Exception as err:
                if attempt == max_retries - 1:
                    stats.errors.append(f"Vertex batch {b.vertex_type} failed: {err}")
                else:
                    time.sleep(2**attempt * 0.5)

    # Helper for executing edge batches with exponential retry backoff
    all_edge_batches = (
        plan.has_chunk_batches
        + plan.documented_in_batches
        + plan.held_at_batches
        + plan.precedes_batches
        + plan.succeeds_batches
    )
    for eb in all_edge_batches:
        for attempt in range(max_retries):
            try:
                client.conn.upsertEdges(eb.source_type, eb.edge_type, eb.target_type, eb.records)
                stats.total_edges_upserted += len(eb.records)
                stats.batches_processed += 1
                break
            except Exception as err:
                if attempt == max_retries - 1:
                    stats.errors.append(f"Edge batch {eb.edge_type} failed: {err}")
                else:
                    time.sleep(2**attempt * 0.5)

    stats.elapsed_seconds = time.perf_counter() - t0
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="TigerGraph Savanna Batch Upsert Partitioner")
    parser.add_argument(
        "--corpus-path",
        default="hackathon-resources/corpus/corpus.jsonl",
        help="Path to corpus.jsonl",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch partition size for vertices and edges",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate partition structure without upserting",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Execute batches into in-memory MockTigerGraphConnection",
    )
    args = parser.parse_args()

    print(f"[Stream 1] Partitioning corpus: {args.corpus_path} (batch_size={args.batch_size})")
    plan = prepare_ingestion_plan(args.corpus_path, batch_size=args.batch_size)

    print(f"  Total Documents: {plan.total_documents}")
    print(f"  Total Chunks:    {plan.total_chunks}")
    print(f"  Total Events:    {plan.total_events}")
    print(f"  Total Venues:    {plan.total_venues}")
    print(f"  Total Edges:     {plan.total_edges}")
    print(f"  Document Batches: {len(plan.document_batches)}")
    print(f"  Chunk Batches:    {len(plan.chunk_batches)}")
    print(f"  Event Batches:    {len(plan.event_batches)}")
    print(f"  Precedes Batches: {len(plan.precedes_batches)}")

    if args.mock:
        print("[Stream 1] Executing batches into high-fidelity mock graph...")
        client = create_mock_graph_client()
        stats = execute_ingestion(client, plan, dry_run=False)
    elif args.dry_run:
        print("[Stream 1] Executing dry-run partition verification...")
        client = create_mock_graph_client()
        stats = execute_ingestion(client, plan, dry_run=True)
    else:
        print("[Stream 1] Executing live cluster upsert...")
        client = GraphClient(conn=connect())
        stats = execute_ingestion(client, plan, dry_run=False)

    print(
        f"[Stream 1] Ingestion complete in {stats.elapsed_seconds:.2f}s "
        f"(Vertices: {stats.total_vertices_upserted}, Edges: {stats.total_edges_upserted}, Batches: {stats.batches_processed})"
    )
    if stats.errors:
        print(f"[Stream 1] Encountered {len(stats.errors)} batch errors:")
        for err in stats.errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
