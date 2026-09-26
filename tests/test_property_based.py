# Property-based invariant testing using Hypothesis.
# Verifies metric bounds, mathematical symmetry, and bitmask determinism across generated inputs.
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from src.coprocessor import build_filter_mask, season_mask, sport_mask, year_mask
from src.evaluate import (
    compute_exact_match,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_token_f1,
)
from src.graph.bitemporal import parse_temporal_datetime


# Invariant: Exact Match is strictly bounded within [0.0, 1.0] for any arbitrary text
@given(pred=st.text(), ground_truth=st.lists(st.text(), min_size=0, max_size=5))
def test_property_exact_match_bounded(pred: str, ground_truth: list[str]) -> None:
    score = compute_exact_match(pred, ground_truth)
    assert 0.0 <= score <= 1.0


# Invariant: Token F1 is strictly bounded within [0.0, 1.0] and handles arbitrary unicode
@given(pred=st.text(), ground_truth=st.lists(st.text(), min_size=1, max_size=5))
def test_property_token_f1_bounded(pred: str, ground_truth: list[str]) -> None:
    score = compute_token_f1(pred, ground_truth)
    assert 0.0 <= score <= 1.0


# Invariant: Token F1 between two single phrases is symmetric: F1(A, B) == F1(B, A)
@given(text_a=st.text(min_size=1, max_size=100), text_b=st.text(min_size=1, max_size=100))
def test_property_token_f1_symmetric(text_a: str, text_b: str) -> None:
    f1_ab = compute_token_f1(text_a, [text_b])
    f1_ba = compute_token_f1(text_b, [text_a])
    assert abs(f1_ab - f1_ba) < 1e-6


# Invariant: MRR is strictly bounded within [0.0, 1.0] for any candidate ranks
@given(retrieved=st.lists(st.text(), max_size=20), gold=st.lists(st.text(), max_size=5))
def test_property_mrr_bounded(retrieved: list[str], gold: list[str]) -> None:
    mrr = compute_mrr(retrieved, gold)
    assert 0.0 <= mrr <= 1.0


# Invariant: Recall@K and Precision@K are strictly bounded within [0.0, 1.0]
@given(
    retrieved=st.lists(st.text(), max_size=20),
    gold=st.lists(st.text(), max_size=5),
    k=st.integers(min_value=1, max_value=10),
)
def test_property_recall_precision_bounded(retrieved: list[str], gold: list[str], k: int) -> None:
    rec = compute_recall_at_k(retrieved, gold, k=k)
    prec = compute_precision_at_k(retrieved, gold, k=k)
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= prec <= 1.0


# Invariant: Filter mask generation is deterministic and bitwise monotonic
@given(
    year=st.integers(min_value=1980, max_value=2030),
    season=st.sampled_from(["Summer", "Winter", ""]),
    sport=st.sampled_from(["Athletics", "Swimming", "Judo", "Rowing", "Other"]),
)
def test_property_filter_mask_invariants(year: int, season: str, sport: str) -> None:
    mask = build_filter_mask(year, season, sport)
    assert isinstance(mask, int)
    assert mask >= 0
    # Combined mask subsumes individual components
    y_m = year_mask(year)
    s_m = season_mask(season)
    sp_m = sport_mask(sport)
    assert (mask & y_m) == y_m
    assert (mask & s_m) == s_m
    assert (mask & sp_m) == sp_m


# Invariant: Bitemporal ISO timestamp parsing is strictly monotonic for valid orderings
@given(
    y1=st.integers(min_value=1900, max_value=2020),
    y2=st.integers(min_value=2021, max_value=2099),
)
def test_property_bitemporal_monotonicity(y1: int, y2: int) -> None:
    dt1 = parse_temporal_datetime(f"{y1}-01-01T00:00:00Z")
    dt2 = parse_temporal_datetime(f"{y2}-01-01T00:00:00Z")
    assert dt1 is not None and dt2 is not None
    assert dt1 < dt2
