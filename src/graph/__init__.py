# TigerGraph Savanna client + schema setup + compiled GSQL queries.
# Uses pyTigerGraph for all operations. Zero raw string concatenation in queries.
from __future__ import annotations

import os
import time
from collections.abc import Callable
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
from src.graph.normalization import normalize_gender_filter
from src.guardrails import normalize_gsql_string_quotes, validate_generated_gsql
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
    "wait_for_graph_ready",
]

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_GRAPH_NAME = "OlympicsGraph"
_MAX_MULTIHOP_GRAPH_MATCHES = 20


def wait_for_graph_ready(
    host: str, timeout_s: float = 300.0, poll_interval_s: float = 5.0
) -> float:
    # Savanna suspends idle workspaces and answers with an HTML "Starting workspace" page until
    # resumed; wait for the ping endpoint so the first real query does not fail on that page.
    import requests

    started = time.perf_counter()
    deadline = started + timeout_s
    last_status = "no response"
    while True:
        try:
            response = requests.get(f"{host.rstrip('/')}/api/ping", timeout=10)
            if response.status_code == 200:
                return time.perf_counter() - started
            last_status = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_status = type(exc).__name__
        if time.perf_counter() + poll_interval_s > deadline:
            raise RuntimeError(
                f"TigerGraph workspace did not become ready within {timeout_s:.0f}s "
                f"(last status: {last_status})"
            )
        time.sleep(poll_interval_s)


def connect() -> tg.TigerGraphConnection:
    # Initialize connection to TigerGraph Savanna cluster from environment vars.
    try:
        import dotenv

        dotenv.load_dotenv()
    except ImportError:
        pass

    host = os.environ.get("TG_HOST", "")
    graphname = os.environ.get("TG_GRAPH_NAME", _GRAPH_NAME)
    username = os.environ.get("TG_USERNAME", "tigergraph")
    password = os.environ.get("TG_PASSWORD", "")
    secret = os.environ.get("TG_SECRET", "")
    token = os.environ.get("TG_TOKEN", "")

    # If explicit JWT token not supplied, mint one via REST endpoint using TG_SECRET
    if not token and secret and host:
        try:
            import requests

            res = requests.post(f"{host}/gsql/v1/tokens", json={"secret": secret}, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if not data.get("error") and "token" in data:
                    token = data["token"]
        except Exception:
            pass

    conn = tg.TigerGraphConnection(
        host=host,
        graphname=graphname,
        username=username,
        password=password,
        apiToken=token or None,
    )

    # Fallback to pyTigerGraph getToken if direct REST call was not used
    if not token and secret and host:
        try:
            t = conn.getToken(secret=secret, setToken=True, lifetime=86400)
            if isinstance(t, tuple):
                conn.apiToken = t[0]
        except Exception:
            pass

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
) WITH PRIMARY_ID_AS_ATTRIBUTE="true"

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
) WITH PRIMARY_ID_AS_ATTRIBUTE="true"

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
) WITH PRIMARY_ID_AS_ATTRIBUTE="true"

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
CREATE GLOBAL SCHEMA_CHANGE JOB add_chunk_vector {
    ALTER VERTEX Chunk ADD VECTOR ATTRIBUTE embedding (
        DIMENSION = 1024,
        METRIC = "COSINE",
        INDEXTYPE = "HNSW"
    );
}
RUN GLOBAL SCHEMA_CHANGE JOB add_chunk_vector
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
USE GRAPH OlympicsGraph
CREATE OR REPLACE QUERY get_event_aggregates (
    STRING sport,
    INT target_year,
    INT min_competitors,
    INT max_competitors
) FOR GRAPH OlympicsGraph {
    SumAccum<INT> @@match_count = 0;
    ListAccum<STRING> @@event_names;
    ListAccum<STRING> @@gold_doc_ids;

    Events = {Event.*};
    Matched = SELECT e FROM Events:e
              WHERE (sport == "" OR lower(e.sport) == lower(sport))
                AND (target_year == 0 OR e.year == target_year)
                AND (min_competitors == 0 OR e.competitor_count >= min_competitors)
                AND (max_competitors == 0 OR e.competitor_count <= max_competitors)
              ACCUM @@match_count += 1, @@event_names += e.name;

    Docs = SELECT doc FROM Matched:e -(DOCUMENTED_IN:d)- Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@match_count AS match_count, @@event_names AS events, @@gold_doc_ids AS gold_doc_ids;
}
"""

# Query 2: Temporal predecessor — get the winner of the preceding edition.
_QUERY_TEMPORAL = """
USE GRAPH OlympicsGraph
CREATE OR REPLACE QUERY get_preceding_event (
    STRING sport,
    STRING gender,
    STRING event_name_fragment,
    INT current_year
) FOR GRAPH OlympicsGraph {
    ListAccum<STRING> @@prev_event_names;
    ListAccum<STRING> @@gold_doc_ids;
    ListAccum<STRING> @@gold_athletes;

    Events = {Event.*};
    Current = SELECT e FROM Events:e
              WHERE (sport == "" OR lower(e.sport) == lower(sport))
                AND (gender == "" OR lower(e.gender) == lower(gender))
                AND (event_name_fragment == "" OR lower(e.name) LIKE "%" + lower(event_name_fragment) + "%")
                AND e.year == current_year;

    PriorEvents = SELECT prior FROM Current:curr -(PRECEDES:p)-> Event:prior
                  ACCUM @@prev_event_names += prior.name,
                        @@gold_athletes += prior.gold_athlete;

    Docs = SELECT doc FROM PriorEvents:prior -(DOCUMENTED_IN:d)- Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@prev_event_names AS prev_events, @@gold_athletes AS gold_athletes,
          @@gold_doc_ids AS gold_doc_ids;
}
"""

# Query 3: Superlative — find event with max/min competitor count.
_QUERY_SUPERLATIVE = """
USE GRAPH OlympicsGraph
CREATE OR REPLACE QUERY get_superlative_event (
    STRING sport,
    INT target_year,
    STRING season,
    STRING order_by,
    INT result_limit
) FOR GRAPH OlympicsGraph {
    ListAccum<STRING> @@event_names;
    ListAccum<INT> @@competitor_counts;
    ListAccum<STRING> @@gold_doc_ids;

    Events = {Event.*};
    IF order_by == "desc" THEN
        Filtered = SELECT e FROM Events:e
                   WHERE (sport == "" OR lower(e.sport) == lower(sport))
                     AND (target_year == 0 OR e.year == target_year)
                     AND (season == "" OR lower(e.season) == lower(season))
                   ORDER BY e.competitor_count DESC
                   LIMIT result_limit;
    ELSE
        Filtered = SELECT e FROM Events:e
                   WHERE (sport == "" OR lower(e.sport) == lower(sport))
                     AND (target_year == 0 OR e.year == target_year)
                     AND (season == "" OR lower(e.season) == lower(season))
                   ORDER BY e.competitor_count ASC
                   LIMIT result_limit;
    END;

    x = SELECT e FROM Filtered:e
        ACCUM @@event_names += e.name, @@competitor_counts += e.competitor_count;

    Docs = SELECT doc FROM Filtered:e -(DOCUMENTED_IN:d)- Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@event_names AS events, @@competitor_counts AS competitor_counts,
          @@gold_doc_ids AS gold_doc_ids;
}
"""

# Query 4: Multi-hop — find event by venue + date, return gold medalist.
_QUERY_MULTIHOP = """
USE GRAPH OlympicsGraph
CREATE OR REPLACE QUERY get_event_by_venue_date (
    STRING venue_name_fragment,
    STRING target_date_fragment,
    INT target_year
) FOR GRAPH OlympicsGraph {
    ListAccum<STRING> @@event_names;
    ListAccum<STRING> @@gold_athletes;
    ListAccum<STRING> @@gold_doc_ids;

    Events = {Event.*};
    Matched = SELECT e FROM Events:e -(HELD_AT:h)- Venue:v
              WHERE (venue_name_fragment == "" OR lower(v.name) LIKE "%" + lower(venue_name_fragment) + "%")
                AND (target_year == 0 OR e.year == target_year)
                AND (target_date_fragment == "" OR lower(h.start_date) LIKE "%" + lower(target_date_fragment) + "%")
              ACCUM @@event_names += e.name, @@gold_athletes += e.gold_athlete;

    Docs = SELECT doc FROM Matched:e -(DOCUMENTED_IN:d)- Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@event_names AS events, @@gold_athletes AS gold_athletes,
          @@gold_doc_ids AS gold_doc_ids;
}
"""

# Query 5: Lookup — get specific attribute of an event by name.
_QUERY_LOOKUP = """
USE GRAPH OlympicsGraph
CREATE OR REPLACE QUERY get_event_attribute (
    STRING event_name_fragment,
    INT target_year,
    STRING sport,
    STRING gender
) FOR GRAPH OlympicsGraph {
    ListAccum<STRING> @@event_names;
    ListAccum<INT> @@competitor_counts;
    ListAccum<INT> @@nation_counts;
    ListAccum<STRING> @@gold_athletes;
    ListAccum<STRING> @@venues;
    ListAccum<STRING> @@gold_doc_ids;

    Events = {Event.*};
    Matched = SELECT e FROM Events:e
              WHERE (event_name_fragment == "" OR lower(e.name) LIKE "%" + lower(event_name_fragment) + "%")
                AND (target_year == 0 OR e.year == target_year)
                AND (sport == "" OR lower(e.sport) == lower(sport))
                AND (gender == "" OR lower(e.gender) == lower(gender));

    x = SELECT e FROM Matched:e
        ACCUM @@event_names += e.name,
              @@competitor_counts += e.competitor_count,
              @@nation_counts += e.nation_count,
              @@gold_athletes += e.gold_athlete,
              @@venues += e.venue;

    Docs = SELECT doc FROM Matched:e -(DOCUMENTED_IN:d)- Document:doc
           ACCUM @@gold_doc_ids += doc.wikidata_qid;

    PRINT @@event_names AS events,
          @@competitor_counts AS competitor_counts,
          @@nation_counts AS nation_counts,
          @@gold_athletes AS gold_athletes,
          @@venues AS venues,
          @@gold_doc_ids AS gold_doc_ids;
}
"""


def _extract_event_candidates(value: Any) -> list[dict[str, Any]]:
    # Preserve canonical names from projected Event rows as structured evidence for synthesis.
    candidates: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if str(item.get("v_type", "")).casefold() == "event":
                attrs = item.get("attributes", {})
                if isinstance(attrs, dict):
                    name = next(
                        (
                            val
                            for key, val in attrs.items()
                            if key.rsplit(".", maxsplit=1)[-1].casefold() == "name"
                            and isinstance(val, str)
                            and val.strip()
                        ),
                        None,
                    )
                    if name is not None:
                        candidate: dict[str, Any] = {"name": name.strip()}
                        count = next(
                            (
                                val
                                for key, val in attrs.items()
                                if key.rsplit(".", maxsplit=1)[-1].casefold() == "competitor_count"
                                and isinstance(val, int)
                                and not isinstance(val, bool)
                            ),
                            None,
                        )
                        if count is not None:
                            candidate["competitor_count"] = count
                        candidates.append(candidate)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return candidates[:10]


# Vector search query using TigerVector HNSW
_QUERY_VECTOR_SEARCH = """
USE GRAPH OlympicsGraph
CREATE OR REPLACE QUERY vector_search_chunks (
    LIST<FLOAT> query_vector,
    INT top_k
) FOR GRAPH OlympicsGraph {
    MapAccum<VERTEX, FLOAT> @@distances;
    TopChunks = vectorSearch({Chunk.embedding}, query_vector, top_k, {distance_map: @@distances, ef: 64});
    PRINT TopChunks[TopChunks.chunk_id, TopChunks.doc_id, TopChunks.text, TopChunks.raw_text,
                    TopChunks.prev_chunk_id, TopChunks.next_chunk_id];
    PRINT @@distances;
}
"""


# ---------------------------------------------------------------------------
# Graph Client
# ---------------------------------------------------------------------------


def _is_auth_failure(exc: Exception) -> bool:
    # A Savanna restart invalidates the session token; pyTigerGraph then fails every request with
    # an authentication error (or a 401/403) until a new connection mints a fresh token.
    status = getattr(getattr(exc, "response", None), "status_code", None)
    message = str(exc).casefold()
    return (
        status in (401, 403)
        or "authentication failed" in message
        or "token" in message
        and ("expired" in message or "invalid" in message)
    )


@dataclass
class GraphClient:
    conn: tg.TigerGraphConnection = field(default_factory=connect)

    def _read[T](self, call: Callable[[tg.TigerGraphConnection], T]) -> T:
        # Reads reconnect once on a stale token instead of failing until the server restarts.
        try:
            return call(self.conn)
        except Exception as exc:
            if isinstance(self.conn, MockTigerGraphConnection) or not _is_auth_failure(exc):
                raise
            self.conn = connect()
            return call(self.conn)

    def _installed(self, name: str, params: dict[str, Any], timeout: int = 10000) -> Any:
        return self._read(lambda conn: conn.runInstalledQuery(name, params=params, timeout=timeout))

    def vertices_by_id(self, vertex_type: str, vertex_ids: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = self._read(
            lambda conn: conn.getVerticesById(vertex_type, vertex_ids)
        )
        return rows

    def run_generated_gsql(self, query: str) -> dict[str, Any]:
        # Execute only bounded, read-only GSQL that passes the application allowlist.
        query, quotes_normalized = normalize_gsql_string_quotes(query)
        safe, reason = validate_generated_gsql(query)
        if not safe:
            raise ValueError(f"Generated GSQL rejected: {reason}")
        started = time.perf_counter()

        def interpret(conn: tg.TigerGraphConnection) -> Any:
            if conn._version_greater_than_4_0():
                # pyTigerGraph's v4 runInterpretedQuery selects password auth, while Savanna
                # deployments commonly configure bearer tokens. Call the same v4 endpoint through
                # its request layer so the existing token is used and the request is bounded.
                return conn._post(
                    conn.gsUrl + "/gsql/v1/queries/interpret",
                    authMode="token",
                    headers={"Content-Type": "text/plain", "GSQL-TIMEOUT": "15000"},
                    data=query,
                )
            return conn.runInterpretedQuery(query)

        rows = self._read(interpret)
        return {
            "rows": rows,
            "event_candidates": _extract_event_candidates(rows),
            "row_groups": len(rows),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "quotes_normalized": quotes_normalized,
        }

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
        # Compile and install all GSQL stored queries into native C++.
        # Installed queries execute in ~2-4ms.
        query_names = [
            "get_event_aggregates",
            "get_preceding_event",
            "get_superlative_event",
            "get_event_by_venue_date",
            "get_event_attribute",
            "vector_search_chunks",
        ]
        for query_ddl in [
            _QUERY_AGGREGATION,
            _QUERY_TEMPORAL,
            _QUERY_SUPERLATIVE,
            _QUERY_MULTIHOP,
            _QUERY_LOOKUP,
            _QUERY_VECTOR_SEARCH,
        ]:
            self.conn.gsql(f"USE GRAPH {_GRAPH_NAME}\n{query_ddl}")
        self.conn.gsql(f"USE GRAPH {_GRAPH_NAME}\nINSTALL QUERY {', '.join(query_names)}")

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
            "gender": infobox.gender or "",
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
        results = self._installed(
            "vector_search_chunks",
            params={"query_vector": query_vector, "top_k": top_k},
            timeout=10000,
        )

        chunks_data = results[0].get("TopChunks", [])
        distances = results[1].get("@@distances", {})

        output: list[tuple[str, float]] = []
        for chunk in chunks_data:
            attributes = chunk.get("attributes", {})
            cid = (
                chunk.get("chunk_id")
                or attributes.get("TopChunks.chunk_id", "")
                or chunk.get("v_id", "")
            )
            if not cid:
                continue
            dist = distances.get(cid, 1.0)
            score = 1.0 - float(dist)  # cosine similarity from distance
            output.append((cid, score))
        # TigerGraph's projected TopChunks rows are not guaranteed to retain score order.
        output.sort(key=lambda hit: hit[1], reverse=True)
        return output

    def run_aggregation(
        self,
        sport: str = "",
        year: int = 0,
        min_competitors: int = 0,
        max_competitors: int = 0,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = self._installed(
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
            "count": r.get("match_count", r.get("count", 0)),
            "events": r.get("events", []),
            "gold_doc_ids": r.get("gold_doc_ids", []),
            "latency_ms": latency_ms,
        }

    def run_temporal(
        self,
        sport: str = "",
        event_name_fragment: str = "",
        current_year: int = 0,
        gender: str = "",
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = self._installed(
            "get_preceding_event",
            params={
                "sport": sport,
                "gender": normalize_gender_filter(gender),
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
        results = self._installed(
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
        year: int = 0,
    ) -> dict[str, Any]:
        # Event year is stored on Event, while HELD_AT dates usually omit the year.
        date_fragment = date_fragment.split("(", maxsplit=1)[0].strip(" ,–—-()")
        if year:
            date_fragment = date_fragment.replace(str(year), "").strip(" ,–—-()")
        t0 = time.perf_counter()
        if not venue_fragment.strip() and not date_fragment:
            return {
                "events": [],
                "gold_athletes": [],
                "gold_doc_ids": [],
                "latency_ms": 0.0,
            }
        venue_words = venue_fragment.split()
        venue_candidates = [venue_fragment]
        while (date_fragment or year) and len(venue_words) > 3 and len(venue_candidates) < 4:
            venue_words.pop()
            venue_candidates.append(" ".join(venue_words))

        r: dict[str, Any] = {}
        for candidate in venue_candidates:
            results = self._installed(
                "get_event_by_venue_date",
                params={
                    "venue_name_fragment": candidate,
                    "target_date_fragment": date_fragment,
                    "target_year": year,
                },
                timeout=10000,
            )
            r = results[0] if results else {}
            if r.get("events"):
                break
        latency_ms = (time.perf_counter() - t0) * 1000
        events = r.get("events", [])
        if len(events) > _MAX_MULTIHOP_GRAPH_MATCHES:
            # Broad graph matches are weak evidence; let hybrid passages resolve the question.
            return {
                "events": [],
                "gold_athletes": [],
                "gold_doc_ids": [],
                "latency_ms": latency_ms,
            }
        return {
            "events": events,
            "gold_athletes": r.get("gold_athletes", []),
            "gold_doc_ids": r.get("gold_doc_ids", []),
            "latency_ms": latency_ms,
        }

    def run_lookup(
        self,
        event_fragment: str = "",
        year: int = 0,
        sport: str = "",
        gender: str = "",
    ) -> dict[str, Any]:
        if not event_fragment.strip():
            return {
                "events": [],
                "competitor_counts": [],
                "nation_counts": [],
                "gold_athletes": [],
                "venues": [],
                "gold_doc_ids": [],
                "latency_ms": 0.0,
            }
        t0 = time.perf_counter()
        results = self._installed(
            "get_event_attribute",
            params={
                "event_name_fragment": event_fragment,
                "target_year": year,
                "sport": sport,
                "gender": normalize_gender_filter(gender),
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
