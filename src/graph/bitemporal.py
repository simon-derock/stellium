# Bitemporal Schema Extension & Conflict Resolution Module for Round 2.
# Implements temporal validity (valid_from, valid_to), provenance (source_authority,
# superseded_by), CONFLICTS_WITH edge semantics, and the 4-step conflict resolution strategy:
# 1. Compare source_authority scores (higher = more authoritative).
# 2. Compare valid_from / valid_to timestamps (newer = supersedes older).
# 3. If unresolvable: report both versions with confidence intervals.
# 4. Log the conflict in the agentic trace as a strategy shift event.
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Datetime Utilities
# ---------------------------------------------------------------------------

_ISO_Z_RE = re.compile(r"Z$")
_TG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_DATE_ONLY_FORMAT = "%Y-%m-%d"


def parse_temporal_datetime(val: str | datetime | int | float | None) -> datetime | None:
    # Parse various temporal formats into a normalized UTC naive datetime.
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is not None:
            return val.astimezone(UTC).replace(tzinfo=None)
        return val
    if isinstance(val, (int, float)):
        # Plausible 4-digit year (e.g. 1896..2100)
        if 1800 <= val <= 2200:
            return datetime(int(val), 1, 1)
        try:
            return datetime.fromtimestamp(float(val), tz=UTC).replace(tzinfo=None)
        except (ValueError, OSError):
            return None

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        # 4-digit year string
        if re.fullmatch(r"\d{4}", s):
            return datetime(int(s), 1, 1)
        # Standard ISO or TigerGraph formats
        clean_s = _ISO_Z_RE.sub("+00:00", s)
        try:
            dt = datetime.fromisoformat(clean_s)
            if dt.tzinfo is not None:
                return dt.astimezone(UTC).replace(tzinfo=None)
            return dt
        except ValueError:
            pass
        for fmt in (_TG_DATETIME_FORMAT, _DATE_ONLY_FORMAT):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue

    return None


# ---------------------------------------------------------------------------
# Enums & Value Objects
# ---------------------------------------------------------------------------


class ConflictType(StrEnum):
    CONTRADICTION = "contradiction"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    TEMPORAL_UPDATE = "temporal_update"


class ResolutionStrategy(StrEnum):
    AUTHORITY = "source_authority"
    RECENCY = "valid_from_timestamp"
    UNRESOLVED_DUAL_REPORT = "unresolved_both"
    NO_CONFLICT = "no_conflict"


class ResolvedBy(StrEnum):
    SOURCE_AUTHORITY = "source_authority"
    VALID_FROM_TIMESTAMP = "valid_from_timestamp"
    UNRESOLVED_BOTH = "unresolved_both"
    NONE = "none"


@dataclass(frozen=True)
class TemporalInterval:
    # Represents the temporal validity window of a fact.

    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def is_valid_at(self, target: datetime | str) -> bool:
        dt = parse_temporal_datetime(target) if isinstance(target, str) else target
        if dt is None:
            return True
        if self.valid_from is not None and dt < self.valid_from:
            return False
        if self.valid_to is not None and dt > self.valid_to:
            return False
        return True

    def overlaps(self, other: TemporalInterval) -> bool:
        if (
            self.valid_from is not None
            and other.valid_to is not None
            and self.valid_from > other.valid_to
        ):
            return False
        if (
            self.valid_to is not None
            and other.valid_from is not None
            and self.valid_to < other.valid_from
        ):
            return False
        return True

    def to_dict(self) -> dict[str, str | None]:
        return {
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
        }


@dataclass
class BitemporalFact:
    # A granular predicate fact carrying temporal validity and source authority.

    fact_id: str
    subject: str
    predicate: str
    value: Any
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_authority: float = 1.0  # Higher = more authoritative
    superseded_by: str | None = None
    source_doc_id: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.valid_from, str):
            self.valid_from = parse_temporal_datetime(self.valid_from)
        if isinstance(self.valid_to, str):
            self.valid_to = parse_temporal_datetime(self.valid_to)

    @property
    def is_superseded(self) -> bool:
        return bool(self.superseded_by)

    @property
    def interval(self) -> TemporalInterval:
        return TemporalInterval(valid_from=self.valid_from, valid_to=self.valid_to)

    def is_valid_at(self, target: datetime | str) -> bool:
        if self.is_superseded:
            return False
        return self.interval.is_valid_at(target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "source_authority": self.source_authority,
            "superseded_by": self.superseded_by,
            "source_doc_id": self.source_doc_id,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ConflictEdge:
    # Represents a CONFLICTS_WITH directed edge between two Event vertices in TigerGraph.

    from_event_id: str
    to_event_id: str
    conflict_type: str = ConflictType.CONTRADICTION.value
    resolution: str = "unresolved"
    resolved_by: str = ResolvedBy.SOURCE_AUTHORITY.value

    def to_dict(self) -> dict[str, str]:
        return {
            "from_event_id": self.from_event_id,
            "to_event_id": self.to_event_id,
            "conflict_type": self.conflict_type,
            "resolution": self.resolution,
            "resolved_by": self.resolved_by,
        }


@dataclass(frozen=True)
class VersionReport:
    # Reported version of an event attribute, used when reporting candidates or ties.

    fact_id: str
    value: Any
    source_authority: float
    valid_from: str | None
    valid_to: str | None
    confidence: float
    confidence_interval: tuple[float, float]
    is_winner: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "value": self.value,
            "source_authority": self.source_authority,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "confidence": self.confidence,
            "confidence_interval": [self.confidence_interval[0], self.confidence_interval[1]],
            "is_winner": self.is_winner,
        }


# ---------------------------------------------------------------------------
# Confidence Interval Calculation
# ---------------------------------------------------------------------------


def calculate_confidence_interval(
    authority: float,
    confidence: float = 1.0,
    sample_size: int = 1,
    z: float = 1.96,
) -> tuple[float, float]:
    # Calculate confidence interval for a fact version based on authority and confidence.
    #
    # Combines point estimate, sample support, and authority uncertainty into a bounded [0.0, 1.0] interval.
    auth = min(1.0, max(0.05, float(authority)))
    conf = min(1.0, max(0.05, float(confidence)))
    p = min(0.99, max(0.05, conf * auth))

    # Margin of error incorporates standard error and source authority uncertainty
    se = math.sqrt((p * (1.0 - p)) / max(1, sample_size))
    auth_uncertainty = (1.0 - auth) * 0.15
    margin = min(0.35, max(0.04, z * se + auth_uncertainty))

    lower = round(max(0.0, p - margin), 3)
    upper = round(min(1.0, p + margin), 3)
    return (lower, upper)


# ---------------------------------------------------------------------------
# Conflict Resolution Result & Strategy Shift
# ---------------------------------------------------------------------------


@dataclass
class ConflictResolutionResult:
    # Structured outcome of bitemporal conflict resolution.

    status: Literal["resolved", "unresolved_tie", "no_conflict"]
    predicate: str
    subject: str
    winner: BitemporalFact | None = None
    loser: BitemporalFact | None = None
    superseded_facts: list[BitemporalFact] = field(default_factory=list)
    resolution_strategy: str = ResolutionStrategy.NO_CONFLICT.value
    resolved_by: str = ResolvedBy.NONE.value
    conflict_edge: ConflictEdge | None = None
    reported_versions: list[VersionReport] = field(default_factory=list)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    strategy_shift_event: dict[str, Any] | None = None
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "predicate": self.predicate,
            "subject": self.subject,
            "winner": self.winner.to_dict() if self.winner else None,
            "loser": self.loser.to_dict() if self.loser else None,
            "superseded_fact_ids": [f.fact_id for f in self.superseded_facts],
            "resolution_strategy": self.resolution_strategy,
            "resolved_by": self.resolved_by,
            "conflict_edge": self.conflict_edge.to_dict() if self.conflict_edge else None,
            "reported_versions": [v.to_dict() for v in self.reported_versions],
            "confidence_intervals": {k: [v[0], v[1]] for k, v in self.confidence_intervals.items()},
            "strategy_shift_event": self.strategy_shift_event,
            "explanation": self.explanation,
        }


def create_strategy_shift_event(result: ConflictResolutionResult) -> dict[str, Any]:
    # Create a strategy shift event for logging in the agentic trace (Step 4).
    winner_id = result.winner.fact_id if result.winner else None
    superseded_ids = [f.fact_id for f in result.superseded_facts]
    if result.loser and result.loser.fact_id not in superseded_ids:
        superseded_ids.append(result.loser.fact_id)

    return {
        "event_type": "strategy_shift",
        "subsystem": "bitemporal_conflict_resolver",
        "strategy_changed": result.status != "no_conflict",
        "strategy_change_rationale": result.explanation,
        "subject": result.subject,
        "predicate": result.predicate,
        "resolution_status": result.status,
        "resolution_strategy": result.resolution_strategy,
        "resolved_by": result.resolved_by,
        "winner_fact_id": winner_id,
        "superseded_fact_ids": superseded_ids,
        "conflict_edge": result.conflict_edge.to_dict() if result.conflict_edge else None,
        "reported_versions": [v.to_dict() for v in result.reported_versions],
        "confidence_intervals": {k: [v[0], v[1]] for k, v in result.confidence_intervals.items()},
    }


def apply_conflict_to_agent_state(state: Any, result: ConflictResolutionResult) -> None:
    # Apply a conflict resolution outcome directly into an AgentState instance.
    if result.status == "no_conflict":
        return
    if hasattr(state, "strategy_changed"):
        state.strategy_changed = True
    if hasattr(state, "strategy_change_rationale"):
        prior = getattr(state, "strategy_change_rationale", None) or ""
        state.strategy_change_rationale = (
            f"{prior}; {result.explanation}" if prior else result.explanation
        )
    if hasattr(state, "strategy_history") and isinstance(state.strategy_history, list):
        state.strategy_history.append(f"conflict_resolution:{result.resolved_by}")
    if hasattr(state, "tool_history") and isinstance(state.tool_history, list):
        from src.models import ToolAuditCall

        call = ToolAuditCall(
            step=len(state.tool_history) + 1,
            tool_name="bitemporal_conflict_resolution",
            input_args={
                "subject": result.subject,
                "predicate": result.predicate,
                "resolution_strategy": result.resolution_strategy,
                "resolved_by": result.resolved_by,
            },
            output_summary=result.explanation[:200],
            llm_tokens=0,
            latency_ms=0.5,
        )
        state.tool_history.append(call)


# ---------------------------------------------------------------------------
# Core Conflict Resolution Engine
# ---------------------------------------------------------------------------


def resolve_fact_conflict(
    fact_a: BitemporalFact,
    fact_b: BitemporalFact,
    authority_tolerance: float = 1e-4,
) -> ConflictResolutionResult:
    # Resolve a conflict between two facts per PLAN_SPEC.md [ORCHESTRA:ROUND2_PREP]:
    #
    # 1. Compare source_authority scores (higher = more authoritative).
    # 2. Compare valid_from / valid_to timestamps (newer = supersedes older).
    # 3. If unresolvable: report both versions with confidence intervals.
    # 4. Log the conflict in the agentic trace as a strategy shift event.
    # Identical values => concordant, no conflict
    if fact_a.value == fact_b.value:
        return ConflictResolutionResult(
            status="no_conflict",
            predicate=fact_a.predicate,
            subject=fact_a.subject,
            resolution_strategy=ResolutionStrategy.NO_CONFLICT.value,
            resolved_by=ResolvedBy.NONE.value,
            explanation="No conflict: facts have identical values.",
        )

    # 1. Compare source_authority scores
    winner: BitemporalFact | None = None
    loser: BitemporalFact | None = None

    auth_diff = fact_a.source_authority - fact_b.source_authority
    if abs(auth_diff) > authority_tolerance:
        if auth_diff > 0:
            winner, loser = fact_a, fact_b
        else:
            winner, loser = fact_b, fact_a

        loser.superseded_by = winner.fact_id
        edge = ConflictEdge(
            from_event_id=loser.fact_id,
            to_event_id=winner.fact_id,
            conflict_type=ConflictType.CONTRADICTION.value,
            resolution="superseded_by_authority",
            resolved_by=ResolvedBy.SOURCE_AUTHORITY.value,
        )
        ci_winner = calculate_confidence_interval(winner.source_authority, winner.confidence)
        ci_loser = calculate_confidence_interval(loser.source_authority, loser.confidence)

        v_winner = VersionReport(
            fact_id=winner.fact_id,
            value=winner.value,
            source_authority=winner.source_authority,
            valid_from=winner.valid_from.isoformat() if winner.valid_from else None,
            valid_to=winner.valid_to.isoformat() if winner.valid_to else None,
            confidence=winner.confidence,
            confidence_interval=ci_winner,
            is_winner=True,
        )
        v_loser = VersionReport(
            fact_id=loser.fact_id,
            value=loser.value,
            source_authority=loser.source_authority,
            valid_from=loser.valid_from.isoformat() if loser.valid_from else None,
            valid_to=loser.valid_to.isoformat() if loser.valid_to else None,
            confidence=loser.confidence,
            confidence_interval=ci_loser,
            is_winner=False,
        )

        explanation = (
            f"Resolved by source authority: Fact '{winner.fact_id}' (authority={winner.source_authority:.3f}) "
            f"supersedes '{loser.fact_id}' (authority={loser.source_authority:.3f}) "
            f"for {winner.subject}.{winner.predicate} ('{winner.value}' vs '{loser.value}')."
        )
        result = ConflictResolutionResult(
            status="resolved",
            predicate=winner.predicate,
            subject=winner.subject,
            winner=winner,
            loser=loser,
            superseded_facts=[loser],
            resolution_strategy=ResolutionStrategy.AUTHORITY.value,
            resolved_by=ResolvedBy.SOURCE_AUTHORITY.value,
            conflict_edge=edge,
            reported_versions=[v_winner, v_loser],
            confidence_intervals={
                winner.fact_id: ci_winner,
                loser.fact_id: ci_loser,
            },
            explanation=explanation,
        )
        result.strategy_shift_event = create_strategy_shift_event(result)
        return result

    # 2. Compare valid_from / valid_to timestamps (newer = supersedes older)
    winner = None
    loser = None

    if fact_a.valid_from is not None or fact_b.valid_from is not None:
        if fact_a.valid_from is not None and fact_b.valid_from is not None:
            if fact_a.valid_from > fact_b.valid_from:
                winner, loser = fact_a, fact_b
            elif fact_b.valid_from > fact_a.valid_from:
                winner, loser = fact_b, fact_a
        elif fact_a.valid_from is not None:
            winner, loser = fact_a, fact_b
        elif fact_b.valid_from is not None:
            winner, loser = fact_b, fact_a

    # If valid_from was equal or neither had valid_from, compare valid_to
    if winner is None and (fact_a.valid_to is not None or fact_b.valid_to is not None):
        if fact_a.valid_to is not None and fact_b.valid_to is not None:
            if fact_a.valid_to > fact_b.valid_to:
                winner, loser = fact_a, fact_b
            elif fact_b.valid_to > fact_a.valid_to:
                winner, loser = fact_b, fact_a
        elif fact_a.valid_to is not None:
            winner, loser = fact_a, fact_b
        elif fact_b.valid_to is not None:
            winner, loser = fact_b, fact_a

    if winner is not None and loser is not None:
        loser.superseded_by = winner.fact_id
        edge = ConflictEdge(
            from_event_id=loser.fact_id,
            to_event_id=winner.fact_id,
            conflict_type=ConflictType.TEMPORAL_UPDATE.value,
            resolution="superseded_by_timestamp",
            resolved_by=ResolvedBy.VALID_FROM_TIMESTAMP.value,
        )
        ci_winner = calculate_confidence_interval(winner.source_authority, winner.confidence)
        ci_loser = calculate_confidence_interval(loser.source_authority, loser.confidence)

        v_winner = VersionReport(
            fact_id=winner.fact_id,
            value=winner.value,
            source_authority=winner.source_authority,
            valid_from=winner.valid_from.isoformat() if winner.valid_from else None,
            valid_to=winner.valid_to.isoformat() if winner.valid_to else None,
            confidence=winner.confidence,
            confidence_interval=ci_winner,
            is_winner=True,
        )
        v_loser = VersionReport(
            fact_id=loser.fact_id,
            value=loser.value,
            source_authority=loser.source_authority,
            valid_from=loser.valid_from.isoformat() if loser.valid_from else None,
            valid_to=loser.valid_to.isoformat() if loser.valid_to else None,
            confidence=loser.confidence,
            confidence_interval=ci_loser,
            is_winner=False,
        )

        winner_ts = winner.valid_from.isoformat() if winner.valid_from else str(winner.valid_to)
        loser_ts = loser.valid_from.isoformat() if loser.valid_from else str(loser.valid_to)
        explanation = (
            f"Resolved by temporal recency: Fact '{winner.fact_id}' (timestamp={winner_ts}) "
            f"supersedes '{loser.fact_id}' (timestamp={loser_ts}) "
            f"for {winner.subject}.{winner.predicate} ('{winner.value}' vs '{loser.value}')."
        )
        result = ConflictResolutionResult(
            status="resolved",
            predicate=winner.predicate,
            subject=winner.subject,
            winner=winner,
            loser=loser,
            superseded_facts=[loser],
            resolution_strategy=ResolutionStrategy.RECENCY.value,
            resolved_by=ResolvedBy.VALID_FROM_TIMESTAMP.value,
            conflict_edge=edge,
            reported_versions=[v_winner, v_loser],
            confidence_intervals={
                winner.fact_id: ci_winner,
                loser.fact_id: ci_loser,
            },
            explanation=explanation,
        )
        result.strategy_shift_event = create_strategy_shift_event(result)
        return result

    # 3. If unresolvable: report both versions with confidence intervals
    ci_a = calculate_confidence_interval(fact_a.source_authority, fact_a.confidence)
    ci_b = calculate_confidence_interval(fact_b.source_authority, fact_b.confidence)

    v_a = VersionReport(
        fact_id=fact_a.fact_id,
        value=fact_a.value,
        source_authority=fact_a.source_authority,
        valid_from=fact_a.valid_from.isoformat() if fact_a.valid_from else None,
        valid_to=fact_a.valid_to.isoformat() if fact_a.valid_to else None,
        confidence=fact_a.confidence,
        confidence_interval=ci_a,
        is_winner=False,
    )
    v_b = VersionReport(
        fact_id=fact_b.fact_id,
        value=fact_b.value,
        source_authority=fact_b.source_authority,
        valid_from=fact_b.valid_from.isoformat() if fact_b.valid_from else None,
        valid_to=fact_b.valid_to.isoformat() if fact_b.valid_to else None,
        confidence=fact_b.confidence,
        confidence_interval=ci_b,
        is_winner=False,
    )

    edge = ConflictEdge(
        from_event_id=fact_a.fact_id,
        to_event_id=fact_b.fact_id,
        conflict_type=ConflictType.DISPUTED.value,
        resolution="unresolved_tie_dual_report",
        resolved_by=ResolvedBy.UNRESOLVED_BOTH.value,
    )
    explanation = (
        f"Unresolvable conflict between '{fact_a.fact_id}' and '{fact_b.fact_id}' on {fact_a.predicate}: "
        f"identical authority ({fact_a.source_authority:.3f}) and timestamp. "
        f"Reporting both versions with confidence intervals: "
        f"'{fact_a.value}' [{ci_a[0]}, {ci_a[1]}] vs '{fact_b.value}' [{ci_b[0]}, {ci_b[1]}]."
    )
    result = ConflictResolutionResult(
        status="unresolved_tie",
        predicate=fact_a.predicate,
        subject=fact_a.subject,
        winner=None,
        loser=None,
        superseded_facts=[],
        resolution_strategy=ResolutionStrategy.UNRESOLVED_DUAL_REPORT.value,
        resolved_by=ResolvedBy.UNRESOLVED_BOTH.value,
        conflict_edge=edge,
        reported_versions=[v_a, v_b],
        confidence_intervals={
            fact_a.fact_id: ci_a,
            fact_b.fact_id: ci_b,
        },
        explanation=explanation,
    )
    result.strategy_shift_event = create_strategy_shift_event(result)
    return result


def resolve_conflicts_for_facts(
    facts: Sequence[BitemporalFact],
    authority_tolerance: float = 1e-4,
) -> list[ConflictResolutionResult]:
    # Resolve conflicts across a collection of facts grouped by (subject, predicate).
    # Group by (subject, predicate)
    grouped: dict[tuple[str, str], list[BitemporalFact]] = {}
    for fact in facts:
        key = (fact.subject, fact.predicate)
        grouped.setdefault(key, []).append(fact)

    results: list[ConflictResolutionResult] = []

    for (subject, predicate), group_facts in grouped.items():
        if len(group_facts) <= 1:
            continue

        # Check unique values
        distinct_values = {f.value for f in group_facts}
        if len(distinct_values) <= 1:
            results.append(
                ConflictResolutionResult(
                    status="no_conflict",
                    predicate=predicate,
                    subject=subject,
                    resolution_strategy=ResolutionStrategy.NO_CONFLICT.value,
                    resolved_by=ResolvedBy.NONE.value,
                    explanation=f"No conflict on {subject}.{predicate}: all {len(group_facts)} facts agree.",
                )
            )
            continue

        # Sort candidates by authority DESC, valid_from DESC, valid_to DESC
        sorted_candidates = sorted(
            group_facts,
            key=lambda f: (
                f.source_authority,
                f.valid_from or datetime.min,
                f.valid_to or datetime.min,
            ),
            reverse=True,
        )

        top_fact = sorted_candidates[0]
        second_fact = sorted_candidates[1]

        # Pairwise resolve top vs second to determine if top wins clearly
        pair_res = resolve_fact_conflict(
            top_fact, second_fact, authority_tolerance=authority_tolerance
        )

        if pair_res.status == "resolved" and pair_res.winner is not None:
            winner = pair_res.winner
            superseded: list[BitemporalFact] = []
            reported_versions: list[VersionReport] = []
            confidence_intervals: dict[str, tuple[float, float]] = {}

            # All other facts with differing values become superseded
            for f in sorted_candidates:
                ci = calculate_confidence_interval(f.source_authority, f.confidence)
                confidence_intervals[f.fact_id] = ci
                is_win = f.fact_id == winner.fact_id
                if not is_win and f.value != winner.value:
                    f.superseded_by = winner.fact_id
                    superseded.append(f)
                reported_versions.append(
                    VersionReport(
                        fact_id=f.fact_id,
                        value=f.value,
                        source_authority=f.source_authority,
                        valid_from=f.valid_from.isoformat() if f.valid_from else None,
                        valid_to=f.valid_to.isoformat() if f.valid_to else None,
                        confidence=f.confidence,
                        confidence_interval=ci,
                        is_winner=is_win,
                    )
                )

            edge = ConflictEdge(
                from_event_id=second_fact.fact_id,
                to_event_id=winner.fact_id,
                conflict_type=pair_res.conflict_edge.conflict_type
                if pair_res.conflict_edge
                else ConflictType.CONTRADICTION.value,
                resolution=pair_res.resolution_strategy,
                resolved_by=pair_res.resolved_by,
            )
            explanation = (
                f"Resolved candidate set of {len(sorted_candidates)} facts for {subject}.{predicate}: "
                f"Winner '{winner.fact_id}' (value='{winner.value}') {pair_res.resolved_by}; "
                f"{len(superseded)} facts superseded."
            )
            res = ConflictResolutionResult(
                status="resolved",
                predicate=predicate,
                subject=subject,
                winner=winner,
                loser=second_fact,
                superseded_facts=superseded,
                resolution_strategy=pair_res.resolution_strategy,
                resolved_by=pair_res.resolved_by,
                conflict_edge=edge,
                reported_versions=reported_versions,
                confidence_intervals=confidence_intervals,
                explanation=explanation,
            )
            res.strategy_shift_event = create_strategy_shift_event(res)
            results.append(res)
        else:
            # Unresolvable tie across top candidates
            results.append(pair_res)

    return results


# ---------------------------------------------------------------------------
# Graph & Event Utilities
# ---------------------------------------------------------------------------

_CORE_PREDICATES = (
    "gold_athlete",
    "silver_athlete",
    "bronze_athlete",
    "gold_noc",
    "silver_noc",
    "bronze_noc",
    "venue",
    "competitor_count",
    "nation_count",
)


def event_dict_to_facts(
    event: dict[str, Any],
    predicates: Sequence[str] | None = None,
) -> list[BitemporalFact]:
    # Extract individual BitemporalFact objects from an Event dictionary.
    event_id = str(event.get("event_id") or event.get("doc_id") or "unknown_event")
    subject = str(event.get("name") or event_id)
    authority = float(event.get("source_authority", 1.0))
    valid_from = parse_temporal_datetime(event.get("valid_from"))
    valid_to = parse_temporal_datetime(event.get("valid_to"))
    superseded_by = event.get("superseded_by") or None
    target_preds = predicates or _CORE_PREDICATES

    facts: list[BitemporalFact] = []
    for pred in target_preds:
        if pred in event and event[pred] not in (None, ""):
            facts.append(
                BitemporalFact(
                    fact_id=f"{event_id}#{pred}",
                    subject=subject,
                    predicate=pred,
                    value=event[pred],
                    valid_from=valid_from,
                    valid_to=valid_to,
                    source_authority=authority,
                    superseded_by=f"{superseded_by}#{pred}" if superseded_by else None,
                    source_doc_id=event.get("doc_id"),
                )
            )
    return facts


class BitemporalGraphResolver:
    # Manages conflict resolution and edge synchronization for graph entities.

    def __init__(self, authority_tolerance: float = 1e-4) -> None:
        self.authority_tolerance = authority_tolerance

    def resolve_event_pair(
        self,
        event_a: dict[str, Any],
        event_b: dict[str, Any],
        predicates: Sequence[str] | None = None,
    ) -> list[ConflictResolutionResult]:
        # Detect and resolve attribute conflicts between two versions of an event.
        facts_a = {f.predicate: f for f in event_dict_to_facts(event_a, predicates)}
        facts_b = {f.predicate: f for f in event_dict_to_facts(event_b, predicates)}

        common_predicates = set(facts_a.keys()) & set(facts_b.keys())
        results: list[ConflictResolutionResult] = []

        for pred in common_predicates:
            res = resolve_fact_conflict(
                facts_a[pred],
                facts_b[pred],
                authority_tolerance=self.authority_tolerance,
            )
            results.append(res)

        return results

    def resolve_events(
        self,
        events: Sequence[dict[str, Any]],
        predicates: Sequence[str] | None = None,
    ) -> list[ConflictResolutionResult]:
        # Resolve conflicts across a collection of event dictionaries.
        all_facts: list[BitemporalFact] = []
        for ev in events:
            all_facts.extend(event_dict_to_facts(ev, predicates))

        return resolve_conflicts_for_facts(all_facts, authority_tolerance=self.authority_tolerance)

    def apply_resolutions_to_client(
        self,
        client: Any,
        results: Sequence[ConflictResolutionResult],
    ) -> int:
        # Apply resolved conflict edges and superseded_by flags to TigerGraph via GraphClient.
        edges_applied = 0
        for r in results:
            if r.conflict_edge is not None and hasattr(client, "upsert_conflict_edge"):
                # Strip predicate suffix from fact_id if present for event-level vertex edge
                from_eid = r.conflict_edge.from_event_id.split("#")[0]
                to_eid = r.conflict_edge.to_event_id.split("#")[0]
                if from_eid != to_eid:
                    client.upsert_conflict_edge(
                        from_event_id=from_eid,
                        to_event_id=to_eid,
                        conflict_type=r.conflict_edge.conflict_type,
                        resolution=r.conflict_edge.resolution,
                        resolved_by=r.conflict_edge.resolved_by,
                    )
                    edges_applied += 1

            # Update superseded_by attribute on loser/superseded vertices if possible
            if hasattr(client, "conn") and hasattr(client.conn, "upsertVertex"):
                for superseded in r.superseded_facts:
                    eid = superseded.fact_id.split("#")[0]
                    sup_by = (
                        superseded.superseded_by.split("#")[0] if superseded.superseded_by else ""
                    )
                    if sup_by:
                        client.conn.upsertVertex(
                            "Event",
                            eid,
                            {"superseded_by": sup_by},
                        )

        return edges_applied
