# Reconcile live Event/Document attributes with the current corpus parser.
# Dry run by default: prints every attribute whose live value differs from what ingestion would
# write today. With --apply, upserts only the differing attributes (no deletes, no embeddings).
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from src.graph import connect, wait_for_graph_ready
from src.ingest.batch_upsert import prepare_ingestion_plan

_COMPARED_ATTRIBUTES = {
    "Event": (
        "name",
        "year",
        "season",
        "sport",
        "gender",
        "venue",
        "competitor_count",
        "nation_count",
        "gold_athlete",
        "silver_athlete",
        "bronze_athlete",
        "filter_mask",
    ),
    "Document": ("title", "filter_mask"),
}


def _expected_records(corpus_path: str) -> dict[str, dict[str, dict[str, Any]]]:
    # A missing cache path keeps plan preparation from loading or generating any embeddings.
    plan = prepare_ingestion_plan(corpus_path, cache_path=Path("/nonexistent/embedding-cache"))
    expected: dict[str, dict[str, dict[str, Any]]] = {"Event": {}, "Document": {}}
    for batch in plan.event_batches + plan.document_batches:
        for vertex_id, attributes in batch.records:
            expected[batch.vertex_type][vertex_id] = attributes
    return expected


def _live_records(conn: Any, vertex_type: str) -> dict[str, dict[str, Any]]:
    fields = ",".join(_COMPARED_ATTRIBUTES[vertex_type])
    rows = conn.getVertices(vertex_type, select=fields, timeout=60_000)
    return {str(row["v_id"]): dict(row.get("attributes", {})) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff or repair live graph attributes")
    parser.add_argument("--corpus", default="hackathon-resources/corpus/corpus.jsonl")
    parser.add_argument("--apply", action="store_true", help="upsert the differing attributes")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    wait_for_graph_ready(os.environ["TG_HOST"])
    conn = connect()
    expected = _expected_records(args.corpus)

    pending: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    changed_fields: Counter[str] = Counter()
    for vertex_type, compared in _COMPARED_ATTRIBUTES.items():
        live = _live_records(conn, vertex_type)
        missing = sorted(set(expected[vertex_type]) - set(live))
        if missing:
            print(
                f"{vertex_type}: {len(missing)} expected vertices missing live, e.g. {missing[:3]}"
            )
        for vertex_id, want in expected[vertex_type].items():
            have = live.get(vertex_id)
            if have is None:
                continue
            diff = {name: want[name] for name in compared if have.get(name) != want[name]}
            if not diff:
                continue
            for name in diff:
                changed_fields[f"{vertex_type}.{name}"] += 1
                print(
                    json.dumps(
                        {
                            "type": vertex_type,
                            "id": vertex_id,
                            "field": name,
                            "live": have.get(name),
                            "parsed": want[name],
                        },
                        ensure_ascii=False,
                    )
                )
            pending.setdefault(vertex_type, []).append((vertex_id, diff))

    print(f"Differences by attribute: {dict(changed_fields)}", file=sys.stderr)
    if not args.apply:
        print("Dry run; re-run with --apply to upsert these attributes.", file=sys.stderr)
        return
    for vertex_type, records in pending.items():
        accepted = conn.upsertVertices(vertex_type, records)
        print(f"Upserted {accepted}/{len(records)} {vertex_type} vertices", file=sys.stderr)
        if accepted != len(records):
            raise SystemExit(f"TigerGraph accepted {accepted} of {len(records)} {vertex_type} rows")


if __name__ == "__main__":
    main()
