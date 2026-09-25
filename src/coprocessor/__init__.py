# High-speed local coprocessor: Roaring Bitmasks + BM25 + RRF + Cross-Encoder reranker.
# Runs entirely in Python memory — no network calls.
# Complements TigerVector HNSW which handles dense semantic search.
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from rank_bm25 import BM25Plus

from src.guardrails import normalize
from src.models import Chunk

# ---------------------------------------------------------------------------
# Filter Mask Bit Layout
# Bit positions for rapid Roaring Bitmask pre-filtering.
# Each document gets a filter_mask uint32 built during ingestion.
# ---------------------------------------------------------------------------

# Year bits: 1988-2026 → 20 possible Olympics → bits 0-19
_YEAR_BASE = 1988
_YEAR_STEP = 2  # Olympics every 4 years but alternating Summer/Winter so 2 year step index
_YEAR_COUNT = 20

# Season bits: bit 20 = Summer, bit 21 = Winter
_BIT_SUMMER = 20
_BIT_WINTER = 21

# Sport bits: 22-31 (top 10 most frequent sports for fast filtering)
# Remaining sports use OR of multiple bits or fall through to text scan.
_SPORT_BITS: dict[str, int] = {
    "athletics": 22,
    "swimming": 23,
    "gymnastics": 24,
    "cycling": 25,
    "rowing": 26,
    "shooting": 27,
    "weightlifting": 28,
    "wrestling": 29,
    "boxing": 30,
    "judo": 31,
}


def build_filter_mask(year: int | None, season: str | None, sport: str | None) -> int:
    # Build a uint32 filter mask for a chunk based on its Olympic event metadata.
    mask = 0
    if year is not None:
        idx = (year - _YEAR_BASE) // _YEAR_STEP
        if 0 <= idx < _YEAR_COUNT:
            mask |= 1 << idx
    if season == "Summer":
        mask |= 1 << _BIT_SUMMER
    elif season == "Winter":
        mask |= 1 << _BIT_WINTER
    if sport:
        sport_lower = sport.lower()
        for key, bit in _SPORT_BITS.items():
            if key in sport_lower:
                mask |= 1 << bit
    return mask


def year_mask(year: int) -> int:
    idx = (year - _YEAR_BASE) // _YEAR_STEP
    if 0 <= idx < _YEAR_COUNT:
        return 1 << idx
    return 0


def season_mask(season: str) -> int:
    if season == "Summer":
        return 1 << _BIT_SUMMER
    elif season == "Winter":
        return 1 << _BIT_WINTER
    return 0


def sport_mask(sport: str) -> int:
    sport_lower = sport.lower()
    mask = 0
    for key, bit in _SPORT_BITS.items():
        if key in sport_lower:
            mask |= 1 << bit
    return mask


# ---------------------------------------------------------------------------
# BM25 Sparse Index
# ---------------------------------------------------------------------------


@dataclass
class BM25Index:
    # Inverted BM25 index over normalized chunk tokens.
    # Sub-millisecond exact token matching for names, codes, numbers.
    _chunks: list[Chunk] = field(default_factory=list)
    _bm25: BM25Plus | None = None

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        # Tokenize normalized raw text (no metadata header to avoid over-weighting).
        tokenized = [normalize(c.raw_text).lower().split() for c in self._chunks]
        self._bm25 = BM25Plus(tokenized)

    def search(self, query: str, top_k: int = 50) -> list[tuple[Chunk, float]]:
        # Returns top_k (chunk, bm25_score) pairs, descending by score.
        if self._bm25 is None or not self._chunks:
            return []
        tokens = normalize(query).lower().split()
        scores = self._bm25.get_scores(tokens)
        # Get indices of top_k non-zero scores
        indexed = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return [(self._chunks[i], float(s)) for i, s in indexed]

    def search_filtered(
        self,
        query: str,
        filter_mask: int,
        top_k: int = 50,
    ) -> list[tuple[Chunk, float]]:
        # Bitmask pre-filter then BM25. O(N) filter is still fast for 2,951 docs.
        if self._bm25 is None or not self._chunks:
            return []
        tokens = normalize(query).lower().split()
        scores = self._bm25.get_scores(tokens)
        if filter_mask == 0:
            indexed = sorted(
                ((i, s) for i, s in enumerate(scores) if s > 0),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
        else:
            indexed = sorted(
                (
                    (i, s)
                    for i, s in enumerate(scores)
                    if s > 0 and (self._chunks[i].filter_mask & filter_mask) != 0
                ),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
        return [(self._chunks[i], float(s)) for i, s in indexed]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

_RRF_K = 60  # Standard RRF constant


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    top_k: int = 30,
) -> list[tuple[str, float]]:
    # Fuses multiple ranked lists into one combined ranking.
    # Each ranked list is [(id, score), ...] in descending score order.
    # Returns [(id, rrf_score), ...] in descending rrf_score order.
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker (lazy-loaded to avoid import cost at startup)
# ---------------------------------------------------------------------------

_reranker: object | None = None
_reranker_loaded = False


def _get_reranker() -> object:
    global _reranker, _reranker_loaded
    if not _reranker_loaded:
        try:
            from sentence_transformers import CrossEncoder

            # Lightweight cross-encoder: fast, accurate enough for top-30→top-2 reranking.
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _reranker = None
        _reranker_loaded = True
    return _reranker


def rerank(
    query: str,
    candidates: list[Chunk],
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    # Cross-encoder reranking: scores each (query, chunk_text) pair.
    # Falls back to input order if reranker unavailable.
    if not candidates:
        return []
    reranker = _get_reranker()
    if reranker is None:
        return [(c, 1.0) for c in candidates[:top_k]]

    from sentence_transformers import CrossEncoder

    assert isinstance(reranker, CrossEncoder)
    pairs = [(query, c.raw_text[:512]) for c in candidates]
    scores: list[float] = reranker.predict(pairs).tolist()
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Unified Coprocessor Interface
# ---------------------------------------------------------------------------


@dataclass
class Coprocessor:
    # Main interface used by all three pipelines.
    bm25: BM25Index = field(default_factory=BM25Index)
    _chunk_map: dict[str, Chunk] = field(default_factory=dict)

    def build(self, chunks: list[Chunk]) -> None:
        # Build BM25 index and chunk lookup map. Called once during startup.
        self.bm25.build(chunks)
        self._chunk_map = {c.chunk_id: c for c in chunks}

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunk_map.get(chunk_id)

    def expand_window(self, chunk_id: str, direction: str = "both") -> list[Chunk]:
        # Fetch predecessor / successor chunks via O(1) pointer lookup.
        chunk = self._chunk_map.get(chunk_id)
        if chunk is None:
            return []
        result: list[Chunk] = []
        if direction in ("prev", "both") and chunk.prev_chunk_id:
            prev = self._chunk_map.get(chunk.prev_chunk_id)
            if prev:
                result.append(prev)
        result.append(chunk)
        if direction in ("next", "both") and chunk.next_chunk_id:
            nxt = self._chunk_map.get(chunk.next_chunk_id)
            if nxt:
                result.append(nxt)
        return result

    def bm25_search(
        self,
        query: str,
        filter_mask: int = 0,
        top_k: int = 30,
    ) -> list[tuple[str, float]]:
        # Returns [(chunk_id, score)] for RRF fusion.
        results = self.bm25.search_filtered(query, filter_mask, top_k)
        return [(c.chunk_id, s) for c, s in results]

    def hybrid_rerank(
        self,
        query: str,
        dense_results: list[tuple[str, float]],  # From TigerGraph vectorSearch
        filter_mask: int = 0,
        final_top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        # Full pipeline: BM25 → RRF fusion with dense results → cross-encoder rerank.
        # dense_results: [(chunk_id, cosine_score)] from TigerGraph
        # BM25 sparse search
        sparse_results = self.bm25_search(query, filter_mask, top_k=30)

        # Normalize dense scores to [0,1] for RRF (already cosine similarity 0-1)
        # RRF only uses rank position, so raw scores are fine
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=30)

        # Resolve chunk_ids to Chunk objects
        candidates: list[Chunk] = []
        for chunk_id, _ in fused:
            chunk = self._chunk_map.get(chunk_id)
            if chunk:
                candidates.append(chunk)

        # Cross-encoder rerank top-30 candidates to top final_top_k
        reranked = rerank(query, candidates, top_k=final_top_k)
        return reranked
