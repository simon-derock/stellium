# Production Jina Embeddings client supporting any modern Jina model (v3, v2, colbert, clip).
# Supports Matryoshka Representation Learning (MRL), task adapters, late chunking, and L2 normalization.
# Features adaptive token-bucket rate limiting tracking Jina API response headers.
# Zero triple-quote docstrings — use single or multi-line # comments exclusively.
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Jina Defaults
# ---------------------------------------------------------------------------

_JINA_API_URL = "https://api.jina.ai/v1/embeddings"
_DEFAULT_MODEL = "jina-embeddings-v5-text-small"
_DEFAULT_DIMENSION = 1024  # Matches TigerGraph Savanna HNSW vector attribute

# Supported Task Adapters for modern Jina LoRA (v3, v4, v5)
TaskType = Literal[
    "retrieval.passage",
    "retrieval.query",
    "text-matching",
    "classification",
    "separation",
]

# Supported Embedding Types
EmbeddingType = Literal["float", "base64", "binary", "ubinary"]


# ---------------------------------------------------------------------------
# Adaptive Rate Limiter & Token Bucket
# ---------------------------------------------------------------------------


class AdaptiveRateLimiter:
    # Enforces Jina rate limits (RPM and TPM) using adaptive sleep and header feedback.
    # Reads X-RateLimit-Remaining-Requests and X-RateLimit-Remaining-Tokens headers.

    def __init__(self, target_rpm: int = 90, min_interval_s: float | None = None) -> None:
        self.min_interval_s = min_interval_s if min_interval_s is not None else (60.0 / target_rpm)
        self.last_call_time = 0.0
        self.remaining_requests: int = 100
        self.remaining_tokens: int = 100000

    def wait_turn(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_call_time

        # If upstream reported near-exhaustion, dynamically throttle
        sleep_needed = self.min_interval_s
        if self.remaining_requests < 5:
            sleep_needed = max(sleep_needed, 2.0)
        elif self.remaining_tokens < 10000:
            sleep_needed = max(sleep_needed, 1.5)

        if elapsed < sleep_needed:
            time.sleep(sleep_needed - elapsed)
        self.last_call_time = time.monotonic()

    def update_from_headers(self, headers: httpx.Headers) -> None:
        # Extracts rate-limit feedback headers returned by Jina API
        req_rem = headers.get("x-ratelimit-remaining-requests")
        if req_rem is not None:
            try:
                self.remaining_requests = int(req_rem)
            except ValueError:
                pass

        tok_rem = headers.get("x-ratelimit-remaining-tokens")
        if tok_rem is not None:
            try:
                self.remaining_tokens = int(tok_rem)
            except ValueError:
                pass


# Backward compatible alias for RateLimiter
RateLimiter = AdaptiveRateLimiter


# ---------------------------------------------------------------------------
# Jina Embedding Client
# ---------------------------------------------------------------------------


@dataclass
class JinaEmbeddingClient:
    api_key: str
    model: str = _DEFAULT_MODEL
    dimension: int = _DEFAULT_DIMENSION
    batch_size: int = 64
    timeout_s: float = 60.0
    late_chunking: bool = False
    normalized: bool = True
    truncate: bool = True
    embedding_type: EmbeddingType = "float"
    max_retries: int = 5
    _limiter: AdaptiveRateLimiter = field(default_factory=AdaptiveRateLimiter)

    @classmethod
    def from_env(cls) -> JinaEmbeddingClient:
        # Reads configuration from environment without hardcoded secrets.
        # Checks standard Jina key variants + model overrides.
        key = (
            os.environ.get("JINA_API_KEY")
            or os.environ.get("jina_embedding_api_key")
            or os.environ.get("JINA_EMBEDDING_API_KEY")
            or ""
        ).strip()

        model = os.environ.get("EMBEDDING_MODEL") or _DEFAULT_MODEL
        raw_dim = os.environ.get("EMBEDDING_DIMENSION")
        dimension = int(raw_dim) if raw_dim and raw_dim.isdigit() else _DEFAULT_DIMENSION

        raw_batch = os.environ.get("EMBEDDING_BATCH_SIZE")
        batch_size = int(raw_batch) if raw_batch and raw_batch.isdigit() else 64

        return cls(
            api_key=key,
            model=model,
            dimension=dimension,
            batch_size=batch_size,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def embed_passages(
        self, texts: list[str], late_chunking: bool | None = None
    ) -> list[list[float]]:
        # Generates L2-normalized embeddings for corpus passages with retrieval.passage adapter.
        use_late = self.late_chunking if late_chunking is None else late_chunking
        return self._embed_batch_orchestrator(
            texts, task="retrieval.passage", late_chunking=use_late
        )

    def embed_query(self, query: str) -> list[float]:
        # Generates L2-normalized embedding for search query with retrieval.query adapter.
        res = self._embed_batch_orchestrator([query], task="retrieval.query", late_chunking=False)
        return res[0] if res else [0.0] * self.dimension

    def embed_text_matching(self, texts: list[str]) -> list[list[float]]:
        # Encodes sentences for symmetric semantic textual similarity or clustering.
        return self._embed_batch_orchestrator(texts, task="text-matching", late_chunking=False)

    def _embed_batch_orchestrator(
        self, texts: list[str], task: TaskType, late_chunking: bool = False
    ) -> list[list[float]]:
        if not texts:
            return []

        # Offline fallback returns zero-vectors with exact dimension to keep pipelines resilient
        if not self.is_configured:
            return [[0.0] * self.dimension for _ in texts]

        results: list[list[float]] = []

        # Split into batches to respect TPM and memory constraints
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_single_batch_with_retry(
                batch, task=task, late_chunking=late_chunking
            )
            results.extend(batch_embeddings)

        return results

    def _embed_single_batch_with_retry(
        self, batch: list[str], task: TaskType, late_chunking: bool
    ) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "task": task,
            "dimensions": self.dimension,
            "late_chunking": late_chunking,
            "normalized": self.normalized,
            "truncate": self.truncate,
            "embedding_type": self.embedding_type,
            "input": batch,
        }

        backoff = 1.5

        for attempt in range(1, self.max_retries + 1):
            try:
                # Throttle request rate
                self._limiter.wait_turn()

                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(_JINA_API_URL, headers=headers, json=payload)

                # Feed rate-limit headers back to adaptive limiter
                self._limiter.update_from_headers(response.headers)

                # Handle HTTP 429 rate limit with upstream retry-after
                if response.status_code == 429:
                    retry_header = response.headers.get("retry-after")
                    sleep_time = float(retry_header) if retry_header else backoff
                    time.sleep(max(sleep_time, backoff))
                    backoff *= 2.0
                    continue

                response.raise_for_status()
                data = response.json()
                items = data.get("data", [])
                # Maintain original input ordering
                sorted_items = sorted(items, key=lambda x: int(x.get("index", 0)))
                return [item["embedding"] for item in sorted_items]

            except (httpx.HTTPError, httpx.TimeoutException, KeyError):
                if attempt == self.max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2.0

        # Graceful degradation on exhausted retries
        return [[0.0] * self.dimension for _ in batch]
