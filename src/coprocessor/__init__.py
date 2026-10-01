# Local retrieval coprocessor: packed category masks, BM25Plus, RRF, and required reranking.
# Runs entirely in Python memory — no network calls.
# Complements TigerVector HNSW which handles dense semantic search.
from __future__ import annotations

import heapq
import math
import os
import time
from array import array
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from src.guardrails import normalize
from src.models import Chunk

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
# Quantized local cross-encoder reranker (lazy-loaded on first hybrid query).
# ---------------------------------------------------------------------------

_RERANKER_REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
_RERANKER_FILE = "onnx/model_quint8_avx2.onnx"
# Pinned model commit: a moved branch head can never swap the weights under a benchmark.
_RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
_reranker: LocalCrossEncoder | None = None
_reranker_load_error: str | None = None


@dataclass
class LocalCrossEncoder:
    # Int8 MiniLM cross-encoder; ONNX Runtime avoids the much larger PyTorch dependency.
    session: Any
    tokenizer: Any
    model_name: str = _RERANKER_REPO
    latency_ms: float = 0.0

    @classmethod
    def load(cls) -> LocalCrossEncoder:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(
            repo_id=_RERANKER_REPO, filename=_RERANKER_FILE, revision=_RERANKER_REVISION
        )
        tokenizer_path = hf_hub_download(
            repo_id=_RERANKER_REPO, filename="tokenizer.json", revision=_RERANKER_REVISION
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(os.environ.get("RERANKER_CPU_THREADS", "2")))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Avoid ONNX Runtime's extra CPU arena and dynamic-shape buffers on small hosted instances.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=512)
        return cls(session=session, tokenizer=tokenizer)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        import numpy as np

        started = time.perf_counter()
        if not pairs:
            self.latency_ms = 0.0
            return []
        encodings = [self.tokenizer.encode(query, document) for query, document in pairs]
        names = {item.name for item in self.session.get_inputs()}
        batch_size = max(1, min(16, int(os.environ.get("RERANKER_BATCH_SIZE", "1"))))
        normalized_scores: list[float] = []
        for start in range(0, len(encodings), batch_size):
            batch = encodings[start : start + batch_size]
            max_length = max(len(encoding.ids) for encoding in batch)
            input_ids = np.zeros((len(batch), max_length), dtype=np.int64)
            attention_mask = np.zeros_like(input_ids)
            token_type_ids = np.zeros_like(input_ids)
            for row, encoding in enumerate(batch):
                size = len(encoding.ids)
                input_ids[row, :size] = encoding.ids
                attention_mask[row, :size] = encoding.attention_mask
                token_type_ids[row, :size] = encoding.type_ids
            values = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            scores = self.session.run(None, {name: values[name] for name in names})[0]
            normalized_scores.extend(np.asarray(scores).reshape(-1).astype(float).tolist())
        self.latency_ms = (time.perf_counter() - started) * 1000
        if len(normalized_scores) != len(pairs):
            raise RuntimeError("Local reranker returned a score count that does not match inputs")
        return normalized_scores


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


def _get_reranker() -> LocalCrossEncoder:
    global _reranker, _reranker_load_error
    if _reranker is None:
        if _reranker_load_error:
            raise RuntimeError(f"Required local reranker is unavailable: {_reranker_load_error}")
        try:
            _reranker = LocalCrossEncoder.load()
        except Exception as exc:
            _reranker_load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"Required local reranker failed to load: {_reranker_load_error}"
            ) from exc
    return _reranker


def rerank(
    query: str,
    candidates: list[Chunk],
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    # Every nonempty candidate set must pass through the local cross-encoder.
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

        # Cross-encoder rerank the fused candidates to top final_top_k.
        if not candidates:
            return HybridRetrievalResult(
                chunks=[],
                dense_candidate_count=len(dense_results),
                sparse_candidate_count=len(sparse_results),
                fused_candidate_count=0,
                reranker_model=_RERANKER_REPO,
                reranker_executed=False,
                reranker_latency_ms=0.0,
            )
        reranker = _get_reranker()
        pairs = [(query, chunk.raw_text) for chunk in candidates]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[
            :final_top_k
        ]
        return HybridRetrievalResult(
            chunks=reranked,
            dense_candidate_count=len(dense_results),
            sparse_candidate_count=len(sparse_results),
            fused_candidate_count=len(candidates),
            reranker_model=reranker.model_name,
            reranker_executed=True,
            reranker_latency_ms=reranker.latency_ms,
        )
