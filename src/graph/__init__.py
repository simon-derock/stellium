# TigerGraph Savanna client + schema setup + compiled GSQL queries.
# Uses pyTigerGraph for all operations. Zero raw string concatenation in queries.
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import pyTigerGraph as tg

from src.graph.bitemporal import (
    BitemporalFact,
    BitemporalGraphResolver,
    ConflictEdge,
    ConflictResolutionResult,
    ConflictType,
    ResolutionStrategy,
    ResolvedBy,
    TemporalInterval,
    VersionReport,
    apply_conflict_to_agent_state,
    calculate_confidence_interval,
    create_strategy_shift_event,
    event_dict_to_facts,
    parse_temporal_datetime,
    resolve_conflicts_for_facts,
    resolve_fact_conflict,
)
from src.graph.mock import MockTigerGraphConnection, create_mock_graph_client
from src.models import Chunk, ParsedInbox

__all__ = [
    "BITEMPORAL_DDL",
    "BITEMPORAL_SCHEMA_CHANGE_DDL",
    "BitemporalFact",
    "BitemporalGraphResolver",
    "ConflictEdge",
    "ConflictResolutionResult",
    "ConflictType",
    "GraphClient",
    "MockTigerGraphConnection",
    "ResolutionStrategy",
    "ResolvedBy",
    "TemporalInterval",
    "VersionReport",
    "_BITEMPORAL_DDL",
    "_BITEMPORAL_SCHEMA_CHANGE_DDL",
    "_SCHEMA_DDL",
    "_VECTOR_DDL",
    "apply_conflict_to_agent_state",
    "calculate_confidence_interval",
    "connect",
    "create_mock_graph_client",
    "create_strategy_shift_event",
    "event_dict_to_facts",
    "parse_temporal_datetime",
    "resolve_conflicts_for_facts",
    "resolve_fact_conflict",
]

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_GRAPH_NAME = "OlympicsGraph"


def connect() -> tg.TigerGraphConnection:
    # Initialize connection to TigerGraph Savanna cluster from environment vars.
    conn = tg.TigerGraphConnection(
        host=os.environ["TG_HOST"],
        graphname=_GRAPH_NAME,
        username=os.environ.get("TG_USERNAME", "tigergraph"),
        password=os.environ.get("TG_PASSWORD", ""),
        useCert=True,
    )
    # Authenticate with secret token if provided
    secret = os.environ.get("TG_SECRET", "")
    if secret:
        token = conn.getToken(secret=secret, setToken=True, lifetime=86400)
        if isinstance(token, tuple):
            conn.apiToken = token[0]
    return conn


# ---------------------------------------------------------------------------
# Schema DDL (runs once during setup)
# ---------------------------------------------------------------------------

_SCHEMA_DDL = """
USE GLOBAL

CREATE VERTEX Document (
    PRIMARY_ID doc_id STRING,
    title STRING,
    url STRING,
    wikidata_qid STRING,
    wikipedia_pageid INT,
    approx_tokens INT,
    filter_mask UINT
) WITH PRIMARY_ID_AS_ATTRIBUTE="true", STATS="OUTDEGREE"

CREATE VERTEX Chunk (
    PRIMARY_ID chunk_id STRING,
    doc_id STRING,
    chunk_index INT,
    section_title STRING,
    text STRING,
    raw_text STRING,
    prev_chunk_id STRING,
    next_chunk_id STRING,
    filter_mask UINT
) WITH PRIMARY_ID_AS_ATTRIBUTE="true", STATS="OUTDEGREE"

CREATE VERTEX Event (
    PRIMARY_ID event_id STRING,
    name STRING,
    year INT,
    season STRING,
    sport STRING,
    gender STRING,
    venue STRING,
    competitor_count INT,
    nation_count INT,
    gold_athlete STRING,
    silver_athlete STRING,
    bronze_athlete STRING,
    gold_noc STRING,
    silver_noc STRING,
    bronze_noc STRING,
    prev_event_id STRING,
    next_event_id STRING,
    filter_mask UINT,
    valid_from DATETIME,
    valid_to DATETIME,
    superseded_by STRING,
    source_authority FLOAT
) WITH PRIMARY_ID_AS_ATTRIBUTE="true", STATS="OUTDEGREE"

CREATE VERTEX Venue (
    PRIMARY_ID venue_id STRING,
    name STRING
) WITH PRIMARY_ID_AS_ATTRIBUTE="true"

CREATE VERTEX Session (
    PRIMARY_ID session_id STRING,
    created_at DATETIME,
    last_active DATETIME
) WITH PRIMARY_ID_AS_ATTRIBUTE="true"

CREATE VERTEX ChatMessage (
    PRIMARY_ID msg_id STRING,
    role STRING,
    content STRING,
    pipeline STRING,
    created_at DATETIME,
    token_count INT
) WITH PRIMARY_ID_AS_ATTRIBUTE="true"

CREATE DIRECTED EDGE HAS_CHUNK (FROM Document, TO Chunk)
CREATE DIRECTED EDGE DOCUMENTED_IN (FROM Event, TO Document)
CREATE DIRECTED EDGE HELD_AT (FROM Event, TO Venue, start_date STRING, end_date STRING)
CREATE DIRECTED EDGE PRECEDES (FROM Event, TO Event, time_diff INT)
CREATE DIRECTED EDGE SUCCEEDS (FROM Event, TO Event, time_diff INT)
CREATE DIRECTED EDGE HAS_MESSAGE (FROM Session, TO ChatMessage, msg_order INT)
CREATE DIRECTED EDGE CONFLICTS_WITH (FROM Event, TO Event, conflict_type STRING, resolution STRING, resolved_by STRING)

CREATE GRAPH OlympicsGraph (
    Document, Chunk, Event, Venue, Session, ChatMessage,
    HAS_CHUNK, DOCUMENTED_IN, HELD_AT, PRECEDES, SUCCEEDS, HAS_MESSAGE,
    CONFLICTS_WITH
)
"""

_VECTOR_DDL = """
USE GRAPH OlympicsGraph
CREATE SCHEMA_CHANGE JOB add_chunk_vector FOR GRAPH OlympicsGraph {
    ALTER VERTEX Chunk ADD VECTOR ATTRIBUTE embedding (
        DIMENSION = 1024,
        METRIC = "COSINE",
        INDEXTYPE = "HNSW"
    );
}
RUN SCHEMA_CHANGE JOB add_chunk_vector
DROP JOB add_chunk_vector
"""

_BITEMPORAL_DDL = """
# Extended Event attributes for Round 2
ALTER VERTEX Event ADD ATTRIBUTE (
    valid_from DATETIME,
    valid_to DATETIME,
    superseded_by STRING,
    source_authority FLOAT
)

# Conflict Detection Edge
CREATE DIRECTED EDGE CONFLICTS_WITH (
    FROM Event, TO Event,
    conflict_type STRING,
    resolution STRING,
    resolved_by STRING
)
"""

_BITEMPORAL_SCHEMA_CHANGE_DDL = """
USE GRAPH OlympicsGraph
CREATE SCHEMA_CHANGE JOB alter_bitemporal_schema FOR GRAPH OlympicsGraph {
    ALTER VERTEX Event ADD ATTRIBUTE (
        valid_from DATETIME,
        valid_to DATETIME,
        superseded_by STRING,
        source_authority FLOAT
    );
    ADD DIRECTED EDGE CONFLICTS_WITH (
        FROM Event, TO Event,
        conflict_type STRING,
        resolution STRING,
        resolved_by STRING
    );
}
RUN SCHEMA_CHANGE JOB alter_bitemporal_schema
DROP JOB alter_bitemporal_schema
"""

BITEMPORAL_DDL = _BITEMPORAL_DDL
BITEMPORAL_SCHEMA_CHANGE_DDL = _BITEMPORAL_SCHEMA_CHANGE_DDL

# ---------------------------------------------------------------------------
# Compiled GSQL Queries
# ---------------------------------------------------------------------------

# Query 1: Deterministic aggregation — COUNT events matching sport/year/threshold.
# Zero LLM tokens. Returns exact count from graph.
_QUERY_AGGREGATION = """
CREATE OR REPLACE QUERY get_event_aggregates (
    STRING sport,
    INT target_year,
    INT min_competitors,
    INT max_competitors
) FOR GRAPH OlympicsGraph SYNTAX v3 {
    SumAccum<INT> @@match_count = 0;
    ListAccum<STRING> @@event_names;
    ListAccum<STRING> @@gold_doc_ids;

    Events = {Event.*};
    Matched = SELECT e FROM Events:e
              WHERE (sport == "" OR e.sport == sport)
                AND (target_year == 0 OR e.year == target_year)
                AND (min_competitors == 0 OR e.competitor_count >= min_competitors)
                AND (max_competitors == 0 OR e.competitor_count <= max_competitors)
              ACCUM @@match_count += 1, @@event_names += e.name;

    Docs = SELECT doc FROM Matched:e -(DOCUMENTED_IN)-> Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@match_count AS count, @@event_names AS events, @@gold_doc_ids AS gold_doc_ids;
}
INSTALL QUERY get_event_aggregates
"""

# Query 2: Temporal predecessor — get the winner of the preceding edition.
_QUERY_TEMPORAL = """
CREATE OR REPLACE QUERY get_preceding_event (
    STRING sport,
    STRING event_name_fragment,
    INT current_year
) FOR GRAPH OlympicsGraph SYNTAX v3 {
    ListAccum<STRING> @@prev_event_names;
    ListAccum<STRING> @@gold_doc_ids;
    ListAccum<STRING> @@gold_athletes;

    Events = {Event.*};
    Current = SELECT e FROM Events:e
              WHERE (sport == "" OR e.sport == sport)
                AND (event_name_fragment == "" OR e.name LIKE "%" + event_name_fragment + "%")
                AND e.year == current_year;

    PriorEvents = SELECT prior FROM Current:curr -(PRECEDES)-> Event:prior
                  ACCUM @@prev_event_names += prior.name,
                        @@gold_athletes += prior.gold_athlete;

    Docs = SELECT doc FROM PriorEvents:prior -(DOCUMENTED_IN)-> Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@prev_event_names AS prev_events, @@gold_athletes AS gold_athletes,
          @@gold_doc_ids AS gold_doc_ids;
}
INSTALL QUERY get_preceding_event
"""

# Query 3: Superlative — find event with max/min competitor count.
_QUERY_SUPERLATIVE = """
CREATE OR REPLACE QUERY get_superlative_event (
    STRING sport,
    INT target_year,
    STRING season,
    STRING order_by,
    INT result_limit
) FOR GRAPH OlympicsGraph SYNTAX v3 {
    ListAccum<STRING> @@event_names;
    ListAccum<INT> @@competitor_counts;
    ListAccum<STRING> @@gold_doc_ids;

    Events = {Event.*};
    Filtered = SELECT e FROM Events:e
               WHERE (sport == "" OR e.sport == sport)
                 AND (target_year == 0 OR e.year == target_year)
                 AND (season == "" OR e.season == season)
               ORDER BY (order_by == "desc" ? e.competitor_count : -e.competitor_count) DESC
               LIMIT result_limit;

    x = SELECT e FROM Filtered:e
        ACCUM @@event_names += e.name, @@competitor_counts += e.competitor_count;

    Docs = SELECT doc FROM Filtered:e -(DOCUMENTED_IN)-> Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@event_names AS events, @@competitor_counts AS competitor_counts,
          @@gold_doc_ids AS gold_doc_ids;
}
INSTALL QUERY get_superlative_event
"""

# Query 4: Multi-hop — find event by venue + date, return gold medalist.
_QUERY_MULTIHOP = """
CREATE OR REPLACE QUERY get_event_by_venue_date (
    STRING venue_name_fragment,
    STRING target_date_fragment
) FOR GRAPH OlympicsGraph SYNTAX v3 {
    ListAccum<STRING> @@event_names;
    ListAccum<STRING> @@gold_athletes;
    ListAccum<STRING> @@gold_doc_ids;

    Venues = {Venue.*};
    MatchedVenues = SELECT v FROM Venues:v
                    WHERE v.name LIKE "%" + venue_name_fragment + "%";

    Events = SELECT e FROM MatchedVenues:v -(HELD_AT)- Event:e
             WHERE (target_date_fragment == "" OR e.start_date LIKE "%" + target_date_fragment + "%");

    x = SELECT e FROM Events:e
        ACCUM @@event_names += e.name, @@gold_athletes += e.gold_athlete;

    Docs = SELECT doc FROM Events:e -(DOCUMENTED_IN)-> Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@event_names AS events, @@gold_athletes AS gold_athletes,
          @@gold_doc_ids AS gold_doc_ids;
}
INSTALL QUERY get_event_by_venue_date
"""

# Query 5: Lookup — get specific attribute of an event by name.
_QUERY_LOOKUP = """
CREATE OR REPLACE QUERY get_event_attribute (
    STRING event_name_fragment,
    INT target_year,
    STRING sport
) FOR GRAPH OlympicsGraph SYNTAX v3 {
    ListAccum<STRING> @@event_names;
    ListAccum<INT> @@competitor_counts;
    ListAccum<INT> @@nation_counts;
    ListAccum<STRING> @@gold_athletes;
    ListAccum<STRING> @@venues;
    ListAccum<STRING> @@gold_doc_ids;

    Events = {Event.*};
    Matched = SELECT e FROM Events:e
              WHERE (event_name_fragment == "" OR e.name LIKE "%" + event_name_fragment + "%")
                AND (target_year == 0 OR e.year == target_year)
                AND (sport == "" OR e.sport == sport);

    x = SELECT e FROM Matched:e
        ACCUM @@event_names += e.name,
              @@competitor_counts += e.competitor_count,
              @@nation_counts += e.nation_count,
              @@gold_athletes += e.gold_athlete,
              @@venues += e.venue;

    Docs = SELECT doc FROM Matched:e -(DOCUMENTED_IN)-> Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@event_names AS events,
          @@competitor_counts AS competitor_counts,
          @@nation_counts AS nation_counts,
          @@gold_athletes AS gold_athletes,
          @@venues AS venues,
          @@gold_doc_ids AS gold_doc_ids;
}
INSTALL QUERY get_event_attribute
"""

# Vector search query using TigerVector HNSW
_QUERY_VECTOR_SEARCH = """
CREATE OR REPLACE QUERY vector_search_chunks (
    LIST<FLOAT> query_vector,
    INT top_k
) FOR GRAPH OlympicsGraph SYNTAX v3 {
    MapAccum<VERTEX, FLOAT> @@distances;
    TopChunks = vectorSearch({Chunk.embedding}, query_vector, top_k, {distance_map: @@distances, ef: 64});
    PRINT TopChunks[TopChunks.chunk_id, TopChunks.doc_id, TopChunks.text, TopChunks.raw_text,
                    TopChunks.prev_chunk_id, TopChunks.next_chunk_id];
    PRINT @@distances;
}
INSTALL QUERY vector_search_chunks
"""


# ---------------------------------------------------------------------------
# Graph Client
# ---------------------------------------------------------------------------


@dataclass
class GraphClient:
    conn: tg.TigerGraphConnection = field(default_factory=connect)

    # ------------------------------------------------------------------
    # Schema setup (one-time)
    # ------------------------------------------------------------------

    def setup_schema(self) -> None:
        # Run DDL scripts to create graph schema and vector index.
        # Safe to call multiple times — uses CREATE OR REPLACE patterns.
        self.conn.gsql(_SCHEMA_DDL)
        self.conn.gsql(_VECTOR_DDL)

    def setup_bitemporal_schema(self) -> None:
        # Run schema change job to add bitemporal attributes and CONFLICTS_WITH edge.
        self.conn.gsql(_BITEMPORAL_SCHEMA_CHANGE_DDL)

    def install_queries(self) -> None:
        # Compile and install all GSQL stored queries.
        # Installed queries execute in ~2-4ms (C++ compiled).
        for query_ddl in [
            _QUERY_AGGREGATION,
            _QUERY_TEMPORAL,
            _QUERY_SUPERLATIVE,
            _QUERY_MULTIHOP,
            _QUERY_LOOKUP,
            _QUERY_VECTOR_SEARCH,
        ]:
            self.conn.gsql(f"USE GRAPH {_GRAPH_NAME}\n{query_ddl}")

    # ------------------------------------------------------------------
    # Ingestion (batch upserts)
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        # Batch upsert Chunk vertices with filter masks and text.
        vertices = [
            (
                c.chunk_id,
                {
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "section_title": c.section_title,
                    "text": c.text[:8000],  # TigerGraph STRING limit
                    "raw_text": c.raw_text[:4000],
                    "prev_chunk_id": c.prev_chunk_id or "",
                    "next_chunk_id": c.next_chunk_id or "",
                    "filter_mask": c.filter_mask,
                },
            )
            for c in chunks
        ]
        self.conn.upsertVertices("Chunk", vertices)

    def upsert_chunk_embeddings(self, chunk_id: str, embedding: list[float]) -> None:
        # Update the embedding attribute on an existing Chunk vertex.
        self.conn.upsertVertex("Chunk", chunk_id, {"embedding": embedding})

    def upsert_event(
        self,
        doc_id: str,
        infobox: ParsedInbox,
        title: str,
        valid_from: str | None = None,
        valid_to: str | None = None,
        superseded_by: str | None = None,
        source_authority: float = 1.0,
    ) -> None:
        # Upsert Event vertex from parsed infobox fields.
        event_id = doc_id  # 1-to-1 mapping: one event per document
        attributes: dict[str, Any] = {
            "name": title,
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
            "prev_event_id": "",  # linked separately via upsert_temporal_edges
            "next_event_id": "",
            "filter_mask": 0,
            "superseded_by": superseded_by or "",
            "source_authority": source_authority,
        }
        if valid_from:
            attributes["valid_from"] = valid_from
        if valid_to:
            attributes["valid_to"] = valid_to
        self.conn.upsertVertex(
            "Event",
            event_id,
            attributes,
        )
        # Link event → document
        self.conn.upsertEdge("Event", event_id, "DOCUMENTED_IN", "Document", doc_id)

    def upsert_conflict_edge(
        self,
        from_event_id: str,
        to_event_id: str,
        conflict_type: str = "contradiction",
        resolution: str = "unresolved",
        resolved_by: str = "source_authority",
    ) -> None:
        # Upsert a CONFLICTS_WITH directed edge between conflicting Event vertices.
        self.conn.upsertEdge(
            "Event",
            from_event_id,
            "CONFLICTS_WITH",
            "Event",
            to_event_id,
            {
                "conflict_type": conflict_type,
                "resolution": resolution,
                "resolved_by": resolved_by,
            },
        )

    def upsert_temporal_edges(
        self, event_id: str, prev_event_id: str | None, next_event_id: str | None
    ) -> None:
        if prev_event_id:
            self.conn.upsertEdge(
                "Event", event_id, "PRECEDES", "Event", prev_event_id, {"time_diff": 0}
            )
            self.conn.upsertEdge(
                "Event", prev_event_id, "SUCCEEDS", "Event", event_id, {"time_diff": 0}
            )

    # ------------------------------------------------------------------
    # Query Execution (parameterized — zero injection risk)
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        # Returns [(chunk_id, cosine_score)] from TigerVector HNSW.
        results = self.conn.runInstalledQuery(
            "vector_search_chunks",
            params={"query_vector": query_vector, "top_k": top_k},
            timeout=10000,
        )

        chunks_data = results[0].get("TopChunks", [])
        distances = results[1].get("@@distances", {})

        output: list[tuple[str, float]] = []
        for chunk in chunks_data:
            cid = chunk.get("chunk_id", "")
            dist = distances.get(cid, 1.0)
            score = 1.0 - float(dist)  # cosine similarity from distance
            output.append((cid, score))
        return output

    def run_aggregation(
        self,
        sport: str = "",
        year: int = 0,
        min_competitors: int = 0,
        max_competitors: int = 0,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = self.conn.runInstalledQuery(
            "get_event_aggregates",
            params={
                "sport": sport,
                "target_year": year,
                "min_competitors": min_competitors,
                "max_competitors": max_competitors,
            },
            timeout=10000,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        r = results[0] if results else {}
        return {
            "count": r.get("count", 0),
            "events": r.get("events", []),
            "gold_doc_ids": r.get("gold_doc_ids", []),
            "latency_ms": latency_ms,
        }

    def run_temporal(
        self,
        sport: str = "",
        event_name_fragment: str = "",
        current_year: int = 0,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = self.conn.runInstalledQuery(
            "get_preceding_event",
            params={
                "sport": sport,
                "event_name_fragment": event_name_fragment,
                "current_year": current_year,
            },
            timeout=10000,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        r = results[0] if results else {}
        return {
            "prev_events": r.get("prev_events", []),
            "gold_athletes": r.get("gold_athletes", []),
            "gold_doc_ids": r.get("gold_doc_ids", []),
            "latency_ms": latency_ms,
        }

    def run_superlative(
        self,
        sport: str = "",
        year: int = 0,
        season: str = "",
        order: str = "desc",
        limit: int = 1,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = self.conn.runInstalledQuery(
            "get_superlative_event",
            params={
                "sport": sport,
                "target_year": year,
                "season": season,
                "order_by": order,
                "result_limit": limit,
            },
            timeout=10000,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        r = results[0] if results else {}
        return {
            "events": r.get("events", []),
            "competitor_counts": r.get("competitor_counts", []),
            "gold_doc_ids": r.get("gold_doc_ids", []),
            "latency_ms": latency_ms,
        }

    def run_multihop(
        self,
        venue_fragment: str = "",
        date_fragment: str = "",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = self.conn.runInstalledQuery(
            "get_event_by_venue_date",
            params={
                "venue_name_fragment": venue_fragment,
                "target_date_fragment": date_fragment,
            },
            timeout=10000,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        r = results[0] if results else {}
        return {
            "events": r.get("events", []),
            "gold_athletes": r.get("gold_athletes", []),
            "gold_doc_ids": r.get("gold_doc_ids", []),
            "latency_ms": latency_ms,
        }

    def run_lookup(
        self,
        event_fragment: str = "",
        year: int = 0,
        sport: str = "",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = self.conn.runInstalledQuery(
            "get_event_attribute",
            params={
                "event_name_fragment": event_fragment,
                "target_year": year,
                "sport": sport,
            },
            timeout=10000,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        r = results[0] if results else {}
        return {
            "events": r.get("events", []),
            "competitor_counts": r.get("competitor_counts", []),
            "nation_counts": r.get("nation_counts", []),
            "gold_athletes": r.get("gold_athletes", []),
            "venues": r.get("venues", []),
            "gold_doc_ids": r.get("gold_doc_ids", []),
            "latency_ms": latency_ms,
        }
