# Mock-based contract tests for graph DDL, upsert mapping, and parameterized queries.
from __future__ import annotations

from unittest.mock import MagicMock

from src.graph import GraphClient
from src.models import Chunk, ParsedInbox


def test_graph_schema_setup_and_query_installation() -> None:
    connection = MagicMock()
    client = GraphClient(conn=connection)

    client.setup_schema()
    client.setup_bitemporal_schema()
    client.install_queries()

    assert connection.gsql.call_count == 10
    ddl_statements = [call.args[0] for call in connection.gsql.call_args_list]
    assert "CREATE GRAPH OlympicsGraph" in ddl_statements[0]
    assert "VECTOR ATTRIBUTE" in ddl_statements[1]
    assert "CONFLICTS_WITH" in ddl_statements[2]
    assert any("lower(e.name) LIKE" in statement for statement in ddl_statements)
    assert any("lower(e.sport) == lower(sport)" in statement for statement in ddl_statements)
    assert any("lower(e.gender) == lower(gender)" in statement for statement in ddl_statements)
    assert any("-(PRECEDES:p)-> Event:prior" in statement for statement in ddl_statements)
    assert "INSTALL QUERY" in ddl_statements[-1]


def test_graph_upserts_preserve_schema_fields_and_limits() -> None:
    connection = MagicMock()
    client = GraphClient(conn=connection)
    chunk = Chunk(
        chunk_id="Q1#0",
        doc_id="Q1",
        chunk_index=0,
        section_title="Results",
        text="t" * 8_001,
        raw_text="r" * 4_001,
    )
    infobox = ParsedInbox(
        event_name="Men's Marathon",
        year=2008,
        season="Summer",
        sport="Athletics",
        gender="Women",
        competitor_count=95,
    )

    client.upsert_chunks([chunk])
    client.upsert_chunk_embeddings("Q1#0", [0.25] * 1024)
    client.upsert_event("Q1", infobox, "Athletics at the 2008 Summer Olympics")
    client.upsert_conflict_edge("Q1", "Q2")
    client.upsert_temporal_edges("Q2", "Q1", None)

    chunk_attrs = connection.upsertVertices.call_args.args[1][0][1]
    assert len(chunk_attrs["text"]) == 8_000
    assert len(chunk_attrs["raw_text"]) == 4_000
    assert chunk_attrs["prev_chunk_id"] == ""
    assert connection.upsertVertex.call_args_list[0].args[2] == {"embedding": [0.25] * 1024}
    event_attrs = connection.upsertVertex.call_args_list[-1].args[2]
    assert event_attrs["year"] == 2008
    assert event_attrs["gender"] == "Women"
    assert event_attrs["competitor_count"] == 95
    assert connection.upsertEdge.call_count == 4


def test_graph_queries_pass_untrusted_values_only_as_parameters() -> None:
    connection = MagicMock()
    connection.runInstalledQuery.return_value = [{}]
    client = GraphClient(conn=connection)
    hostile_value = "Athletics' OR 1=1 --"

    client.run_aggregation(sport=hostile_value, year=2012, min_competitors=74)
    client.run_temporal(
        sport=hostile_value,
        event_name_fragment="marathon",
        current_year=2016,
        gender="Men",
    )
    client.run_superlative(sport=hostile_value, year=2008, order="desc", limit=2)
    client.run_multihop(venue_fragment=hostile_value, date_fragment="12 August 2012", year=2012)
    client.run_lookup(event_fragment=hostile_value, year=2004, gender="Women")

    calls = connection.runInstalledQuery.call_args_list
    assert [call.args[0] for call in calls] == [
        "get_event_aggregates",
        "get_preceding_event",
        "get_superlative_event",
        "get_event_by_venue_date",
        "get_event_attribute",
    ]
    assert calls[0].kwargs["params"]["sport"] == hostile_value
    assert calls[1].kwargs["params"]["sport"] == hostile_value
    assert calls[1].kwargs["params"]["gender"] == "men"
    assert calls[2].kwargs["params"]["sport"] == hostile_value
    assert calls[3].kwargs["params"]["venue_name_fragment"] == hostile_value
    assert calls[3].kwargs["params"]["target_date_fragment"] == "12 August"
    assert calls[3].kwargs["params"]["target_year"] == 2012
    assert calls[4].kwargs["params"]["event_name_fragment"] == hostile_value
    assert calls[4].kwargs["params"]["gender"] == "women"
    assert all("timeout" in call.kwargs for call in calls)


def test_graph_gender_filters_normalize_possessive_question_forms() -> None:
    connection = MagicMock()
    connection.runInstalledQuery.return_value = [{}]
    client = GraphClient(conn=connection)

    client.run_temporal(gender="Women’s")
    client.run_lookup(gender="Men's")

    calls = connection.runInstalledQuery.call_args_list
    assert calls[0].kwargs["params"]["gender"] == "women"
    assert calls[1].kwargs["params"]["gender"] == "men"


def test_vector_search_converts_distance_to_similarity() -> None:
    connection = MagicMock()
    connection.runInstalledQuery.return_value = [
        {
            "TopChunks": [
                {
                    "v_id": "Q1#0",
                    "attributes": {"TopChunks.chunk_id": "Q1#0"},
                },
                {"chunk_id": "Q2#1"},
            ]
        },
        {"@@distances": {"Q1#0": 0.2, "Q2#1": 0.7}},
    ]
    client = GraphClient(conn=connection)

    results = client.vector_search([0.1] * 1024, top_k=2)

    assert results[0] == ("Q1#0", 0.8)
    assert results[1] == ("Q2#1", 0.30000000000000004)
    assert connection.runInstalledQuery.call_args.kwargs["params"]["top_k"] == 2
