# Local retrieval coprocessor: packed category masks, BM25Plus, RRF, and required reranking.
# Runs entirely in Python memory — no network calls.
# Complements TigerVector HNSW which handles dense semantic search.
from __future__ import annotations

import heapq
import logging
import math
import os
import time
from array import array
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.credentials import (
    available_cohere_keys,
    is_monthly_cap,
    live_key_count,
    park_exhausted_key,
    record_key_call,
    rest_key,
)
from src.guardrails import normalize
from src.models import Chunk

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filter Mask Bit Layout
# Bit positions for compact year/season filtering.
# Each document gets a filter_mask uint32 built during ingestion.
# ---------------------------------------------------------------------------

# Year bits: 1988-2026 → 20 possible Olympics → bits 0-19
_YEAR_BASE = 1988
_YEAR_STEP = 2  # Olympics every 4 years but alternating Summer/Winter so 2 year step index
_YEAR_COUNT = 20

# Season bits: bit 20 = Summer, bit 21 = Winter
_BIT_SUMMER = 20
_BIT_WINTER = 21
_YEAR_BITS = (1 << _YEAR_COUNT) - 1
_SEASON_BITS = (1 << _BIT_SUMMER) | (1 << _BIT_WINTER)


def _tokenize(text: str) -> list[str]:
    # Apply the same Unicode and punctuation normalization to corpus text and queries.
    normalized = normalize(text).lower()
    return "".join(char if char.isalnum() else " " for char in normalized).split()


def _top_positive_scores(
    indexed_scores: Iterable[tuple[int, float]], top_k: int
) -> list[tuple[int, float]]:
    # Select top-k in O(N log K) while preserving source order for tied scores.
    if top_k <= 0:
        return []
    return heapq.nlargest(top_k, indexed_scores, key=lambda item: item[1])


def build_filter_mask(year: int | None, season: str | None) -> int:
    # Build a uint32 year/season mask; sport names remain corpus-derived text, never a fixed list.
    mask = 0
    if year is not None:
        idx = (year - _YEAR_BASE) // _YEAR_STEP
        if 0 <= idx < _YEAR_COUNT:
            mask |= 1 << idx
    if season == "Summer":
        mask |= 1 << _BIT_SUMMER
    elif season == "Winter":
        mask |= 1 << _BIT_WINTER
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


# ---------------------------------------------------------------------------
# BM25 Sparse Index
# ---------------------------------------------------------------------------


@dataclass
class BM25Index:
    # Compact BM25Plus postings avoid retaining millions of repeated Python token strings.
    _chunks: list[Chunk] = field(default_factory=list)
    _postings: dict[str, array[int]] = field(default_factory=dict)
    _doc_lengths: list[int] = field(default_factory=list)
    _avgdl: float = 0.0

    def stats(self) -> dict[str, int]:
        # Size of the in-memory index, for the docs page.
        return {
            "chunks": len(self._chunks),
            "documents": len({chunk.doc_id for chunk in self._chunks}),
            "bm25_terms": len(self._postings),
        }

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        self._postings = {}
        self._doc_lengths = []
        for doc_index, chunk in enumerate(self._chunks):
            # Keep only one vocabulary string and two compact integers per term/document pair.
            frequencies = Counter(_tokenize(chunk.raw_text))
            self._doc_lengths.append(sum(frequencies.values()))
            for token, frequency in frequencies.items():
                posting = self._postings.get(token)
                if posting is None:
                    posting = array("I")
                    self._postings[token] = posting
                posting.extend((doc_index, frequency))
        self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths) if self._chunks else 0.0

    def search(self, query: str, top_k: int = 50) -> list[tuple[Chunk, float]]:
        # Returns top_k (chunk, bm25_score) pairs, descending by score.
        if not self._chunks or top_k <= 0:
            return []
        tokens = _tokenize(query)
        scores = self._score(tokens)
        indexed = _top_positive_scores(
            ((i, float(score)) for i, score in enumerate(scores) if score > 0), top_k
        )
        return [(self._chunks[i], float(s)) for i, s in indexed]

    def search_filtered(
        self,
        query: str,
        filter_mask: int,
        top_k: int = 50,
    ) -> list[tuple[Chunk, float]]:
        # Score all chunks, then apply packed-mask filtering during bounded top-k selection.
        if not self._chunks or top_k <= 0:
            return []
        tokens = _tokenize(query)
        scores = self._score(tokens)
        if filter_mask == 0:
            indexed = _top_positive_scores(
                ((i, float(score)) for i, score in enumerate(scores) if score > 0), top_k
            )
        else:
            year_filter = filter_mask & _YEAR_BITS
            season_filter = filter_mask & _SEASON_BITS
            if not year_filter and not season_filter:
                return []
            indexed = _top_positive_scores(
                (
                    (i, float(score))
                    for i, score in enumerate(scores)
                    if score > 0
                    and (not year_filter or self._chunks[i].filter_mask & year_filter)
                    and (not season_filter or self._chunks[i].filter_mask & season_filter)
                ),
                top_k,
            )
        return [(self._chunks[i], float(s)) for i, s in indexed]

    def _score(self, tokens: list[str]) -> list[float]:
        # Exact BM25Plus scoring: IDF=log((N+1)/df), k1=1.5, b=0.75, delta=1.
        # The shared delta baseline applies to every document; postings add term-frequency gain.
        document_count = len(self._chunks)
        if not tokens or not self._postings:
            return [0.0] * document_count
        k1 = 1.5
        b = 0.75
        delta = 1.0
        scores = [0.0] * document_count
        term_data: list[tuple[float, array[int]]] = []
        baseline = 0.0
        for token in tokens:
            posting = self._postings.get(token)
            if posting is None:
                continue
            inverse_document_frequency = math.log((document_count + 1) / (len(posting) // 2))
            baseline += inverse_document_frequency * delta
            term_data.append((inverse_document_frequency, posting))
        if not term_data:
            return scores
        scores[:] = [baseline] * document_count
        average_length = self._avgdl or 1.0
        for inverse_document_frequency, posting in term_data:
            for offset in range(0, len(posting), 2):
                doc_index = posting[offset]
                frequency = posting[offset + 1]
                length_norm = k1 * (1.0 - b + b * self._doc_lengths[doc_index] / average_length)
                scores[doc_index] += (
                    inverse_document_frequency * frequency * (k1 + 1.0) / (frequency + length_norm)
                )
        return scores


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
# Cohere Rerank: one API call per search, no model weights on this host.
# ---------------------------------------------------------------------------

_RERANK_URL = "https://api.cohere.com/v2/rerank"
_RERANK_MODEL = "rerank-v3.5"
_RERANK_RETRY_ELSEWHERE = frozenset({429, 500, 502, 503, 504})
_reranker: CohereReranker | None = None


@dataclass
class CohereReranker:
    # Scores (query, passage) pairs through Cohere Rerank, using the shared key pool: a spent or
    # rejected key is parked, a throttled one rests while another key takes the call.
    model_name: str = field(
        default_factory=lambda: os.environ.get("COHERE_RERANK_MODEL", _RERANK_MODEL)
    )
    latency_ms: float = 0.0
    timeout_s: float = 30.0
    _turn: int = 0

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        started = time.perf_counter()
        if not pairs:
            self.latency_ms = 0.0
            return []
        documents = [document for _, document in pairs]
        payload = {
            "model": self.model_name,
            "query": pairs[0][0],
            "documents": documents,
            "top_n": len(documents),
            "max_tokens_per_doc": 512,
        }
        results = self._post(payload).get("results", [])
        scores = [0.0] * len(documents)
        for item in results:
            scores[int(item["index"])] = float(item["relevance_score"])
        self.latency_ms = (time.perf_counter() - started) * 1000
        if len(results) != len(documents):
            raise RuntimeError("Cohere Rerank returned a score count that does not match inputs")
        return scores

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        handoffs = 0
        with httpx.Client(timeout=self.timeout_s) as client:
            while True:
                pool = available_cohere_keys()
                if not pool:
                    raise RuntimeError("No Cohere key has calls left for reranking")
                key = pool[self._turn % len(pool)][1]
                self._turn += 1
                can_hand_off = handoffs < live_key_count() and live_key_count() > 1
                try:
                    response = client.post(
                        _RERANK_URL,
                        headers={"Authorization": f"Bearer {key}"},
                        json=payload,
                    )
                except httpx.TransportError as exc:
                    if not can_hand_off:
                        raise RuntimeError("Cohere Rerank is unreachable") from exc
                    rest_key(key, 20.0)
                    handoffs += 1
                    continue
                record_key_call(key, "rerank", response.status_code, response.headers)
                if is_monthly_cap(response.status_code, response.text) or response.status_code in (
                    401,
                    403,
                ):
                    park_exhausted_key(key)
                    continue
                if response.status_code in _RERANK_RETRY_ELSEWHERE and can_hand_off:
                    rest_key(key, 30.0 if response.status_code == 429 else 10.0)
                    handoffs += 1
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"Cohere Rerank failed with HTTP {response.status_code}")
                body: dict[str, Any] = response.json()
                return body


@dataclass
class HybridRetrievalResult:
    # Captures whether the required reranking stage actually executed.
    chunks: list[tuple[Chunk, float]]
    dense_candidate_count: int
    sparse_candidate_count: int
    fused_candidate_count: int
    reranker_model: str
    reranker_executed: bool
    reranker_latency_ms: float


def _get_reranker() -> CohereReranker:
    global _reranker
    if _reranker is None:
        _reranker = CohereReranker()
    return _reranker


def rerank(
    query: str,
    candidates: list[Chunk],
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    # Every nonempty candidate set goes through the reranker.
    if not candidates:
        return []
    reranker = _get_reranker()
    pairs = [(query, c.raw_text) for c in candidates]
    scores = reranker.predict(pairs)
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

    def stats(self) -> dict[str, int]:
        return self.bm25.stats()

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
        return self.hybrid_search(
            query=query,
            dense_results=dense_results,
            filter_mask=filter_mask,
            final_top_k=final_top_k,
        ).chunks

    def hybrid_search(
        self,
        query: str,
        dense_results: list[tuple[str, float]],
        filter_mask: int = 0,
        candidate_k: int = 30,
        final_top_k: int = 5,
        distinct_documents: bool = True,
    ) -> HybridRetrievalResult:
        # Full pipeline: BM25 → RRF fusion with dense results → cross-encoder rerank.
        # dense_results: [(chunk_id, cosine_score)] from TigerGraph
        sparse_results = self.bm25_search(query, filter_mask, top_k=candidate_k)

        # Normalize dense scores to [0,1] for RRF (already cosine similarity 0-1)
        # RRF only uses rank position, so raw scores are fine
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_k=2 * candidate_k)

        # Resolve chunk_ids to Chunk objects
        candidates: list[Chunk] = []
        for chunk_id, _ in fused:
            chunk = self._chunk_map.get(chunk_id)
            if chunk:
                candidates.append(chunk)

        # Rerank the fused candidates to top final_top_k.
        reranker = _get_reranker()
        if not candidates:
            return HybridRetrievalResult(
                chunks=[],
                dense_candidate_count=len(dense_results),
                sparse_candidate_count=len(sparse_results),
                fused_candidate_count=0,
                reranker_model=reranker.model_name,
                reranker_executed=False,
                reranker_latency_ms=0.0,
            )
        executed = True
        try:
            scores = reranker.predict([(query, chunk.raw_text) for chunk in candidates])
        except RuntimeError as exc:
            # Out of rerank calls or unreachable: keep the fused order rather than fail the
            # question, and record that the rerank stage did not run.
            log.warning("rerank skipped, keeping rank-fusion order: %s", exc)
            executed = False
            scores = [1.0 / (rank + 1) for rank in range(len(candidates))]
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        if distinct_documents:
            # One passage per article: near-duplicate chunks of the same event otherwise fill
            # several of the few context slots (public set: +1 answer in context, +2% coverage).
            best_per_doc: dict[str, tuple[Chunk, float]] = {}
            for chunk, score in ranked:
                best_per_doc.setdefault(chunk.doc_id, (chunk, score))
            ranked = list(best_per_doc.values())
        reranked = ranked[:final_top_k]
        return HybridRetrievalResult(
            chunks=reranked,
            dense_candidate_count=len(dense_results),
            sparse_candidate_count=len(sparse_results),
            fused_candidate_count=len(candidates),
            reranker_model=reranker.model_name,
            reranker_executed=executed,
            reranker_latency_ms=reranker.latency_ms if executed else 0.0,
        )
