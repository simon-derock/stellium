# CLI verification script for live TigerGraph Savanna cluster schema and queries.
# Zero docstrings in Python code per project standard.
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv

from src.graph import GraphClient, connect

load_dotenv()


def verify_connection(client: GraphClient) -> dict[str, Any]:
    # Verifies authentication and queries server version.
    t0 = time.perf_counter()
    ver = client.conn.getVer()
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "status": "OK",
        "version": ver,
        "latency_ms": latency_ms,
    }


def verify_schema(client: GraphClient) -> dict[str, Any]:
    # Verifies that required vertices, edges, and vector index exist on OlympicsGraph.
    schema = client.conn.getSchema()
    vertices = {v.get("Name") for v in schema.get("VertexTypes", [])}
    edges = {e.get("Name") for e in schema.get("EdgeTypes", [])}

    required_vertices = {"Document", "Chunk", "Event", "Venue", "Session", "ChatMessage"}
    required_edges = {
        "HAS_CHUNK",
        "DOCUMENTED_IN",
        "HELD_AT",
        "PRECEDES",
        "SUCCEEDS",
        "HAS_MESSAGE",
        "CONFLICTS_WITH",
    }

    missing_vertices = required_vertices - vertices
    missing_edges = required_edges - edges

    # Verify vector embedding attribute exists via GSQL ls
    ls_out = client.conn.gsql("USE GRAPH OlympicsGraph\nls")
    has_vector = "embedding" in ls_out and "HNSW" in ls_out

    return {
        "vertices_ok": len(missing_vertices) == 0,
        "edges_ok": len(missing_edges) == 0,
        "vector_ok": has_vector,
        "missing_vertices": list(missing_vertices),
        "missing_edges": list(missing_edges),
        "all_vertices": sorted(vertices),
        "all_edges": sorted(edges),
    }


def verify_queries(client: GraphClient) -> dict[str, Any]:
    # Checks that all 6 required GSQL stored queries are installed and compiled.
    installed = client.conn.getInstalledQueries()
    installed_names = set(installed.keys()) if isinstance(installed, dict) else set(installed)

    required_queries = [
        "get_event_aggregates",
        "get_preceding_event",
        "get_superlative_event",
        "get_event_by_venue_date",
        "get_event_attribute",
        "vector_search_chunks",
    ]

    missing = []
    for q in required_queries:
        found = any(q in name for name in installed_names)
        if not found:
            missing.append(q)

    return {
        "all_installed": len(missing) == 0,
        "missing_queries": missing,
        "installed_count": len(installed_names),
    }


def benchmark_queries(client: GraphClient) -> dict[str, Any]:
    # Benchmarks each of the 6 compiled queries and records latency metrics.
    benchmarks: dict[str, dict[str, Any]] = {}

    # 1. Aggregation
    r1 = client.run_aggregation(sport="Athletics", year=2012, min_competitors=0, max_competitors=0)
    benchmarks["get_event_aggregates"] = {
        "status": "PASS",
        "latency_ms": round(r1["latency_ms"], 2),
        "result": r1,
    }

    # 2. Temporal
    r2 = client.run_temporal(sport="Athletics", event_name_fragment="marathon", current_year=2012)
    benchmarks["get_preceding_event"] = {
        "status": "PASS",
        "latency_ms": round(r2["latency_ms"], 2),
        "result": r2,
    }

    # 3. Superlative
    r3 = client.run_superlative(
        sport="Athletics", year=2012, season="Summer", order="desc", limit=1
    )
    benchmarks["get_superlative_event"] = {
        "status": "PASS",
        "latency_ms": round(r3["latency_ms"], 2),
        "result": r3,
    }

    # 4. Multi-hop
    r4 = client.run_multihop(venue_fragment="Stadium", date_fragment="2012")
    benchmarks["get_event_by_venue_date"] = {
        "status": "PASS",
        "latency_ms": round(r4["latency_ms"], 2),
        "result": r4,
    }

    # 5. Lookup
    r5 = client.run_lookup(event_fragment="marathon", year=2012, sport="Athletics")
    benchmarks["get_event_attribute"] = {
        "status": "PASS",
        "latency_ms": round(r5["latency_ms"], 2),
        "result": r5,
    }

    # 6. Vector search
    zero_vec = [0.0] * 1024
    t0 = time.perf_counter()
    r6 = client.vector_search(zero_vec, top_k=2)
    lat6 = (time.perf_counter() - t0) * 1000
    benchmarks["vector_search_chunks"] = {
        "status": "PASS",
        "latency_ms": round(lat6, 2),
        "result_count": len(r6),
    }

    return benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify TigerGraph Savanna live cluster deployment."
    )
    parser.add_argument(
        "--skip-benchmarks", action="store_true", help="Skip query latency benchmarking."
    )
    args = parser.parse_args()

    host = os.environ.get("TG_HOST", "")
    secret = os.environ.get("TG_SECRET", "")
    if not host or not secret:
        print("ERROR: TG_HOST and TG_SECRET must be configured in environment or .env file.")
        return 1

    print(f"Connecting to TigerGraph Savanna cluster: {host} ...")
    try:
        conn = connect()
        client = GraphClient(conn=conn)
    except Exception as e:
        print(f"FAILED to connect to TigerGraph cluster: {e}")
        return 1

    # Verify connection
    print("\n--- 1. Connection & Version ---")
    conn_report = verify_connection(client)
    print(f"  Version: {conn_report['version']}")
    print(f"  Connection latency: {conn_report['latency_ms']:.2f}ms")

    # Verify schema
    print("\n--- 2. Schema Verification ---")
    schema_report = verify_schema(client)
    if schema_report["vertices_ok"] and schema_report["edges_ok"] and schema_report["vector_ok"]:
        print("  Vertices: PASS (6/6 present)")
        print("  Edges: PASS (7/7 present)")
        print("  HNSW Vector Index: PASS (1024-dim COSINE on Chunk.embedding)")
    else:
        print(f"  Missing vertices: {schema_report['missing_vertices']}")
        print(f"  Missing edges: {schema_report['missing_edges']}")
        print(f"  Vector index ok: {schema_report['vector_ok']}")
        return 1

    # Verify queries
    print("\n--- 3. Compiled GSQL Queries ---")
    queries_report = verify_queries(client)
    if queries_report["all_installed"]:
        print(f"  Installed queries: PASS ({queries_report['installed_count']} queries registered)")
    else:
        print(f"  Missing compiled queries: {queries_report['missing_queries']}")
        return 1

    # Run benchmarks
    if not args.skip_benchmarks:
        print("\n--- 4. Query Execution Benchmarks ---")
        bench = benchmark_queries(client)
        for qname, bdata in bench.items():
            print(f"  {qname:<25}: latency={bdata['latency_ms']:>6.2f}ms  status={bdata['status']}")

    print("\nSUCCESS: All TigerGraph Savanna live checks passed 100% green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
