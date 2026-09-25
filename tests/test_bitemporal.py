# Tests for Bitemporal Schema Extension & Conflict Resolution (Round 2 Prep)
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from src.graph import (
    _BITEMPORAL_DDL,
    _BITEMPORAL_SCHEMA_CHANGE_DDL,
    _SCHEMA_DDL,
    BITEMPORAL_DDL,
    BITEMPORAL_SCHEMA_CHANGE_DDL,
    BitemporalFact,
    BitemporalGraphResolver,
    ConflictType,
    GraphClient,
    ResolutionStrategy,
    ResolvedBy,
    TemporalInterval,
    apply_conflict_to_agent_state,
    calculate_confidence_interval,
    create_strategy_shift_event,
    event_dict_to_facts,
    parse_temporal_datetime,
    resolve_conflicts_for_facts,
    resolve_fact_conflict,
)
from src.models import AgentState, ParsedInbox

# ---------------------------------------------------------------------------
# 1. Schema DDL & Extension Tests
# ---------------------------------------------------------------------------


def test_schema_ddl_contains_bitemporal_attributes() -> None:
    # Verify the Event vertex DDL in _SCHEMA_DDL contains all required bitemporal attributes
    assert "valid_from DATETIME" in _SCHEMA_DDL
    assert "valid_to DATETIME" in _SCHEMA_DDL
    assert "superseded_by STRING" in _SCHEMA_DDL
    assert "source_authority FLOAT" in _SCHEMA_DDL


def test_schema_ddl_contains_conflict_edge() -> None:
    # Verify CONFLICTS_WITH directed edge is defined and included in the graph
    assert (
        "CREATE DIRECTED EDGE CONFLICTS_WITH (FROM Event, TO Event, conflict_type STRING, resolution STRING, resolved_by STRING)"
        in _SCHEMA_DDL
    )
    assert "CONFLICTS_WITH" in _SCHEMA_DDL


def test_bitemporal_ddl_constants() -> None:
    # Verify dedicated alter DDL and schema change jobs are exposed
    assert "ALTER VERTEX Event ADD ATTRIBUTE" in _BITEMPORAL_DDL
    assert "valid_from DATETIME" in _BITEMPORAL_DDL
    assert "CREATE DIRECTED EDGE CONFLICTS_WITH" in _BITEMPORAL_DDL

    assert "CREATE SCHEMA_CHANGE JOB alter_bitemporal_schema" in _BITEMPORAL_SCHEMA_CHANGE_DDL
    assert BITEMPORAL_DDL == _BITEMPORAL_DDL
    assert BITEMPORAL_SCHEMA_CHANGE_DDL == _BITEMPORAL_SCHEMA_CHANGE_DDL


# ---------------------------------------------------------------------------
# 2. Datetime Utilities & TemporalInterval Tests
# ---------------------------------------------------------------------------


def test_parse_temporal_datetime_formats() -> None:
    # 4-digit year string and int
    dt_year = parse_temporal_datetime("2024")
    assert dt_year == datetime(2024, 1, 1)

    dt_year_int = parse_temporal_datetime(1996)
    assert dt_year_int == datetime(1996, 1, 1)

    # Standard ISO format with Z
    dt_iso = parse_temporal_datetime("2024-08-01T15:30:00Z")
    assert dt_iso == datetime(2024, 8, 1, 15, 30, 0)

    # TigerGraph standard format
    dt_tg = parse_temporal_datetime("2024-08-01 15:30:00")
    assert dt_tg == datetime(2024, 8, 1, 15, 30, 0)

    # Date only format
    dt_date = parse_temporal_datetime("2024-08-01")
    assert dt_date == datetime(2024, 8, 1, 0, 0, 0)

    # None and empty
    assert parse_temporal_datetime(None) is None
    assert parse_temporal_datetime("") is None
    assert parse_temporal_datetime("invalid-string") is None


def test_temporal_interval_validity_and_overlap() -> None:
    interval = TemporalInterval(
        valid_from=datetime(2020, 1, 1),
        valid_to=datetime(2024, 12, 31),
    )
    # Valid during window
    assert interval.is_valid_at(datetime(2022, 6, 1))
    assert interval.is_valid_at("2020-01-01")
    assert interval.is_valid_at("2024-12-31")

    # Invalid before or after
    assert not interval.is_valid_at(datetime(2019, 12, 31))
    assert not interval.is_valid_at(datetime(2025, 1, 1))

    # Overlaps
    overlapping = TemporalInterval(
        valid_from=datetime(2024, 1, 1),
        valid_to=datetime(2028, 12, 31),
    )
    non_overlapping = TemporalInterval(
        valid_from=datetime(2025, 1, 1),
        valid_to=datetime(2029, 1, 1),
    )
    assert interval.overlaps(overlapping)
    assert not interval.overlaps(non_overlapping)


# ---------------------------------------------------------------------------
# 3. Strategy Step 1: Compare source_authority
# ---------------------------------------------------------------------------


def test_conflict_resolution_by_source_authority() -> None:
    # Fact A: Official IOC report (source_authority = 0.98)
    fact_a = BitemporalFact(
        fact_id="event_100#gold_athlete",
        subject="Men's 100m 2008",
        predicate="gold_athlete",
        value="Usain Bolt",
        source_authority=0.98,
        valid_from=datetime(2008, 8, 16),
        confidence=1.0,
    )
    # Fact B: Third-party unofficial news blog (source_authority = 0.45)
    fact_b = BitemporalFact(
        fact_id="event_100_blog#gold_athlete",
        subject="Men's 100m 2008",
        predicate="gold_athlete",
        value="Richard Thompson",
        source_authority=0.45,
        valid_from=datetime(2008, 8, 16),
        confidence=0.8,
    )

    result = resolve_fact_conflict(fact_a, fact_b)

    # Fact A must win due to higher source_authority
    assert result.status == "resolved"
    assert result.winner is not None
    assert result.winner.fact_id == fact_a.fact_id
    assert result.winner.value == "Usain Bolt"
    assert result.loser is not None
    assert result.loser.fact_id == fact_b.fact_id
    assert result.loser.superseded_by == fact_a.fact_id
    assert result.resolution_strategy == ResolutionStrategy.AUTHORITY.value
    assert result.resolved_by == ResolvedBy.SOURCE_AUTHORITY.value

    # Edge must point from superseded loser to winner
    assert result.conflict_edge is not None
    assert result.conflict_edge.from_event_id == fact_b.fact_id
    assert result.conflict_edge.to_event_id == fact_a.fact_id
    assert result.conflict_edge.resolved_by == ResolvedBy.SOURCE_AUTHORITY.value

    # Version reports must include confidence intervals
    assert len(result.reported_versions) == 2
    assert result.strategy_shift_event is not None
    assert result.strategy_shift_event["strategy_changed"] is True


# ---------------------------------------------------------------------------
# 4. Strategy Step 2: Compare valid_from / valid_to timestamps (Recency)
# ---------------------------------------------------------------------------


def test_conflict_resolution_by_recency() -> None:
    # Both sources have identical authority (0.90)
    # Fact A: Original 2004 result before retesting
    fact_a = BitemporalFact(
        fact_id="event_shotput_2004_orig#gold_athlete",
        subject="Men's Shot Put 2004",
        predicate="gold_athlete",
        value="Yuriy Bilonoh",
        source_authority=0.90,
        valid_from=datetime(2004, 8, 18),
        valid_to=datetime(2012, 12, 5),
    )
    # Fact B: IOC re-analysis decision in 2012 awarding gold to Adam Nelson
    fact_b = BitemporalFact(
        fact_id="event_shotput_2004_retest#gold_athlete",
        subject="Men's Shot Put 2004",
        predicate="gold_athlete",
        value="Adam Nelson",
        source_authority=0.90,
        valid_from=datetime(2013, 3, 5),
        valid_to=None,
    )

    result = resolve_fact_conflict(fact_a, fact_b)

    # Fact B must win because valid_from 2013 > 2004
    assert result.status == "resolved"
    assert result.winner is not None
    assert result.winner.fact_id == fact_b.fact_id
    assert result.winner.value == "Adam Nelson"
    assert result.loser is not None
    assert result.loser.fact_id == fact_a.fact_id
    assert result.loser.superseded_by == fact_b.fact_id
    assert result.resolution_strategy == ResolutionStrategy.RECENCY.value
    assert result.resolved_by == ResolvedBy.VALID_FROM_TIMESTAMP.value

    # Conflict edge points from older fact to newer superseding fact
    assert result.conflict_edge is not None
    assert result.conflict_edge.from_event_id == fact_a.fact_id
    assert result.conflict_edge.to_event_id == fact_b.fact_id
    assert result.conflict_edge.conflict_type == ConflictType.TEMPORAL_UPDATE.value


# ---------------------------------------------------------------------------
# 5. Strategy Step 3: Unresolvable Tie (Dual Report with Confidence Intervals)
# ---------------------------------------------------------------------------


def test_conflict_resolution_unresolvable_tie_reports_both() -> None:
    # Identical authority (0.85) and identical valid_from (2024-07-27)
    fact_a = BitemporalFact(
        fact_id="event_disputed_a#venue",
        subject="Archery Ranking Round 2024",
        predicate="venue",
        value="Les Invalides Arena 1",
        source_authority=0.85,
        valid_from=datetime(2024, 7, 27),
        confidence=0.9,
    )
    fact_b = BitemporalFact(
        fact_id="event_disputed_b#venue",
        subject="Archery Ranking Round 2024",
        predicate="venue",
        value="Esplanade des Invalides Field B",
        source_authority=0.85,
        valid_from=datetime(2024, 7, 27),
        confidence=0.9,
    )

    result = resolve_fact_conflict(fact_a, fact_b)

    assert result.status == "unresolved_tie"
    assert result.winner is None
    assert result.loser is None
    assert result.resolution_strategy == ResolutionStrategy.UNRESOLVED_DUAL_REPORT.value
    assert result.resolved_by == ResolvedBy.UNRESOLVED_BOTH.value

    # Must report both versions with valid confidence intervals
    assert len(result.reported_versions) == 2
    for version in result.reported_versions:
        low, high = version.confidence_interval
        assert 0.0 <= low <= high <= 1.0

    assert fact_a.fact_id in result.confidence_intervals
    assert fact_b.fact_id in result.confidence_intervals
    assert result.conflict_edge is not None
    assert result.conflict_edge.conflict_type == ConflictType.DISPUTED.value


def test_confidence_interval_bounds() -> None:
    low, high = calculate_confidence_interval(authority=0.95, confidence=0.9)
    assert 0.0 <= low <= high <= 1.0

    low_low_auth, high_low_auth = calculate_confidence_interval(authority=0.2, confidence=0.5)
    assert 0.0 <= low_low_auth <= high_low_auth <= 1.0
    # Lower authority should have wider/lower confidence bound
    assert high_low_auth < high


# ---------------------------------------------------------------------------
# 6. Strategy Step 4: Agentic Trace & Strategy Shift Logging
# ---------------------------------------------------------------------------


def test_strategy_shift_logging_and_agent_state_update() -> None:
    fact_a = BitemporalFact(
        fact_id="ev_a#venue",
        subject="Marathon 2024",
        predicate="venue",
        value="Hôtel de Ville",
        source_authority=0.95,
    )
    fact_b = BitemporalFact(
        fact_id="ev_b#venue",
        subject="Marathon 2024",
        predicate="venue",
        value="Stade de France",
        source_authority=0.50,
    )

    result = resolve_fact_conflict(fact_a, fact_b)
    event = create_strategy_shift_event(result)

    # Verify event structure for agentic trace
    assert event["event_type"] == "strategy_shift"
    assert event["strategy_changed"] is True
    assert event["winner_fact_id"] == "ev_a#venue"
    assert "ev_b#venue" in event["superseded_fact_ids"]
    assert event["resolved_by"] == ResolvedBy.SOURCE_AUTHORITY.value

    # Verify applying to AgentState
    state = AgentState(
        query="Where was the 2024 marathon held?",
        qtype="lookup",
    )
    apply_conflict_to_agent_state(state, result)

    assert state.strategy_changed is True
    assert state.strategy_change_rationale is not None
    assert "Resolved by source authority" in state.strategy_change_rationale
    assert any("conflict_resolution:source_authority" in s for s in state.strategy_history)
    assert len(state.tool_history) == 1
    assert state.tool_history[0].tool_name == "bitemporal_conflict_resolution"
    assert state.tool_history[0].llm_tokens == 0


# ---------------------------------------------------------------------------
# 7. Multi-Fact & Collective Conflict Resolution
# ---------------------------------------------------------------------------


def test_resolve_conflicts_for_facts_multi_candidate() -> None:
    # 3 conflicting claims for the same event and predicate
    f1 = BitemporalFact(
        fact_id="claim_1#gold_noc",
        subject="Women's 100m 2024",
        predicate="gold_noc",
        value="LCA",  # Saint Lucia (Julien Alfred) - Official
        source_authority=0.99,
        valid_from=datetime(2024, 8, 3),
    )
    f2 = BitemporalFact(
        fact_id="claim_2#gold_noc",
        subject="Women's 100m 2024",
        predicate="gold_noc",
        value="USA",  # Sha'Carri Richardson was silver
        source_authority=0.60,
        valid_from=datetime(2024, 8, 3),
    )
    f3 = BitemporalFact(
        fact_id="claim_3#gold_noc",
        subject="Women's 100m 2024",
        predicate="gold_noc",
        value="JAM",  # Pre-race prediction
        source_authority=0.40,
        valid_from=datetime(2024, 8, 1),
    )

    results = resolve_conflicts_for_facts([f1, f2, f3])
    assert len(results) == 1
    res = results[0]

    assert res.status == "resolved"
    assert res.winner is not None
    assert res.winner.value == "LCA"
    # Both conflicting claims (f2, f3) must be marked superseded
    assert len(res.superseded_facts) == 2
    assert f2.superseded_by == f1.fact_id
    assert f3.superseded_by == f1.fact_id


def test_no_conflict_when_facts_agree() -> None:
    f1 = BitemporalFact(
        fact_id="claim_1#sport",
        subject="Event 1",
        predicate="sport",
        value="Athletics",
        source_authority=0.9,
    )
    f2 = BitemporalFact(
        fact_id="claim_2#sport",
        subject="Event 1",
        predicate="sport",
        value="Athletics",
        source_authority=0.8,
    )

    results = resolve_conflicts_for_facts([f1, f2])
    assert len(results) == 1
    assert results[0].status == "no_conflict"


# ---------------------------------------------------------------------------
# 8. GraphResolver & GraphClient Integration Tests
# ---------------------------------------------------------------------------


def test_event_dict_to_facts() -> None:
    event_dict = {
        "event_id": "Q12345",
        "name": "Men's 200m 2024",
        "gold_athlete": "Letsile Tebogo",
        "gold_noc": "BOT",
        "venue": "Stade de France",
        "source_authority": 0.95,
        "valid_from": "2024-08-08 20:30:00",
    }
    facts = event_dict_to_facts(event_dict)
    assert len(facts) == 3
    preds = {f.predicate for f in facts}
    assert preds == {"gold_athlete", "gold_noc", "venue"}
    assert all(f.source_authority == 0.95 for f in facts)
    assert all(f.valid_from == datetime(2024, 8, 8, 20, 30, 0) for f in facts)


def test_bitemporal_graph_resolver_apply_to_client() -> None:
    mock_conn = MagicMock()
    mock_client = MagicMock(spec=GraphClient)
    mock_client.conn = mock_conn

    resolver = BitemporalGraphResolver()

    event_a = {
        "event_id": "EV001",
        "name": "Men's High Jump 2020",
        "gold_athlete": "Gianmarco Tamberi",
        "source_authority": 0.99,
        "valid_from": "2021-08-01",
    }
    event_b = {
        "event_id": "EV002",
        "name": "Men's High Jump 2020",
        "gold_athlete": "Mutaz Barshim",  # Shared gold, differing entry
        "source_authority": 0.50,
        "valid_from": "2021-08-01",
    }

    results = resolver.resolve_event_pair(event_a, event_b, predicates=["gold_athlete"])
    assert len(results) == 1
    assert results[0].status == "resolved"

    applied_count = resolver.apply_resolutions_to_client(mock_client, results)
    assert applied_count == 1
    mock_client.upsert_conflict_edge.assert_called_once_with(
        from_event_id="EV002",
        to_event_id="EV001",
        conflict_type="contradiction",
        resolution="superseded_by_authority",
        resolved_by="source_authority",
    )
    mock_conn.upsertVertex.assert_called_once_with(
        "Event",
        "EV002",
        {"superseded_by": "EV001"},
    )


def test_graph_client_bitemporal_methods() -> None:
    mock_conn = MagicMock()
    client = GraphClient(conn=mock_conn)

    # 1. Setup bitemporal schema
    client.setup_bitemporal_schema()
    mock_conn.gsql.assert_called_once_with(_BITEMPORAL_SCHEMA_CHANGE_DDL)

    # 2. Upsert event with bitemporal attributes
    infobox = ParsedInbox(
        event_name="Men's 400m",
        year=2024,
        sport="Athletics",
        gold_athlete="Quincy Hall",
    )
    client.upsert_event(
        doc_id="Q999",
        infobox=infobox,
        title="Men's 400m 2024",
        valid_from="2024-08-07 21:20:00",
        valid_to=None,
        superseded_by=None,
        source_authority=0.98,
    )
    mock_conn.upsertVertex.assert_called_with(
        "Event",
        "Q999",
        {
            "name": "Men's 400m 2024",
            "year": 2024,
            "season": "",
            "sport": "Athletics",
            "venue": "",
            "competitor_count": 0,
            "nation_count": 0,
            "gold_athlete": "Quincy Hall",
            "silver_athlete": "",
            "bronze_athlete": "",
            "gold_noc": "",
            "silver_noc": "",
            "bronze_noc": "",
            "prev_event_id": "",
            "next_event_id": "",
            "filter_mask": 0,
            "superseded_by": "",
            "source_authority": 0.98,
            "valid_from": "2024-08-07 21:20:00",
        },
    )

    # 3. Upsert conflict edge
    client.upsert_conflict_edge(
        from_event_id="EV_OLD",
        to_event_id="EV_NEW",
        conflict_type="temporal_update",
        resolution="superseded_by_timestamp",
        resolved_by="valid_from_timestamp",
    )
    mock_conn.upsertEdge.assert_called_with(
        "Event",
        "EV_OLD",
        "CONFLICTS_WITH",
        "Event",
        "EV_NEW",
        {
            "conflict_type": "temporal_update",
            "resolution": "superseded_by_timestamp",
            "resolved_by": "valid_from_timestamp",
        },
    )
