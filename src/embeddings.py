# Production Jina Embeddings client supporting modern Jina models.
# Supports Matryoshka Representation Learning (MRL), task adapters, and L2 normalization.
# Features adaptive token-bucket rate limiting with HTTP response header feedback.
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Jina Defaults
# ---------------------------------------------------------------------------

_JINA_API_URL = "https://api.jina.ai/v1/embeddings"
GRAPH_EMBEDDING_MODEL = "jina-embeddings-v5-text-small"
GRAPH_EMBEDDING_DIMENSION = 1024
_DEFAULT_MODEL = GRAPH_EMBEDDING_MODEL
_DEFAULT_DIMENSION = GRAPH_EMBEDDING_DIMENSION
_JINA_TOKEN_WINDOW_S = 60.0
_JINA_FREE_TPM = 100_000

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


class EmbeddingRequestError(RuntimeError):
    # Signals that a live embedding request failed instead of returning usable vectors.
    pass


def _retry_after_seconds(value: str | None) -> float | None:
    # Parses both delta-seconds and HTTP-date Retry-After values.
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _rate_reset_seconds(value: str | None) -> float | None:
    # Parses provider reset headers as either delta-seconds or Unix timestamps.
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return _retry_after_seconds(value)
    if parsed >= 1_000_000_000:
        return max(0.0, parsed - time.time())
    return max(0.0, parsed)


def _estimate_input_tokens(texts: list[str]) -> int:
    # A conservative character-based estimate for pacing; provider headers remain authoritative.
    return sum(max(1, (len(text) + 2) // 3) for text in texts)


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
        self.remaining_tokens: int = _JINA_FREE_TPM
        self.token_reset_at: float | None = None

    def wait_turn(self, estimated_tokens: int = 0) -> None:
        now = time.monotonic()
        elapsed = now - self.last_call_time

        # Apply request pacing and wait for the documented token window when the next batch
        # would exceed the provider's remaining token budget.
        sleep_needed = max(0.0, self.min_interval_s - elapsed)
        if self.remaining_requests < 5:
            sleep_needed = max(sleep_needed, 2.0)
        if self.remaining_tokens < estimated_tokens:
            reset_wait = (
                max(0.0, self.token_reset_at - now)
                if self.token_reset_at is not None
                else _JINA_TOKEN_WINDOW_S
            )
            sleep_needed = max(sleep_needed, reset_wait)

        if sleep_needed > 0:
            time.sleep(sleep_needed)
        if self.remaining_tokens < estimated_tokens:
            self.remaining_tokens = _JINA_FREE_TPM
            self.token_reset_at = None
        if estimated_tokens:
            self.remaining_tokens = max(0, self.remaining_tokens - estimated_tokens)
            if self.token_reset_at is None:
                self.token_reset_at = time.monotonic() + _JINA_TOKEN_WINDOW_S
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
                reset_header = headers.get("x-ratelimit-reset-tokens-minute") or headers.get(
                    "x-ratelimit-reset-tokens"
                )
                reset_seconds = _rate_reset_seconds(reset_header)
                self.token_reset_at = time.monotonic() + (
                    reset_seconds if reset_seconds is not None else _JINA_TOKEN_WINDOW_S
                )
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
    max_retries: int = 8
    _limiter: AdaptiveRateLimiter = field(default_factory=AdaptiveRateLimiter)

    @classmethod
    def from_env(cls) -> JinaEmbeddingClient:
        # Reads configuration from environment variables.
        # Checks standard Jina key variants + model overrides.
        key = (
            os.environ.get("JINA_API_KEY")
            or os.environ.get("jina_embedding_api_key")
            or os.environ.get("JINA_EMBEDDING_API_KEY")
            or ""
        ).strip()

        model = (
            os.environ.get("JINA_EMBEDDING_MODEL")
            or os.environ.get("EMBEDDING_MODEL")
            or _DEFAULT_MODEL
        )
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
        return self.embed_queries([query])[0]

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        # Embeds queries in bounded batches using the same adapter as single-query search.
        return self._embed_batch_orchestrator(queries, task="retrieval.query", late_chunking=False)

    def embed_text_matching(self, texts: list[str]) -> list[list[float]]:
        # Encodes sentences for symmetric semantic textual similarity or clustering.
        return self._embed_batch_orchestrator(texts, task="text-matching", late_chunking=False)

    def _embed_batch_orchestrator(
        self, texts: list[str], task: TaskType, late_chunking: bool = False
    ) -> list[list[float]]:
        if not texts:
            return []

        if not self.is_configured:
            raise EmbeddingRequestError(
                "JINA_API_KEY is required to generate corpus-aligned vectors"
            )

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
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Throttle request rate
                estimated_tokens = _estimate_input_tokens(batch)
                self._limiter.wait_turn(estimated_tokens)

                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(_JINA_API_URL, headers=headers, json=payload)

                # Feed rate-limit headers back to adaptive limiter
                self._limiter.update_from_headers(response.headers)

                # Handle HTTP 429 rate limit with upstream retry-after
                if response.status_code == 429:
                    sleep_time = _retry_after_seconds(response.headers.get("retry-after"))
                    if sleep_time is None:
                        reset_header = response.headers.get(
                            "x-ratelimit-reset-tokens-minute"
                        ) or response.headers.get("x-ratelimit-reset-tokens")
                        sleep_time = _rate_reset_seconds(reset_header) or _JINA_TOKEN_WINDOW_S
                    self._limiter.remaining_tokens = 0
                    self._limiter.token_reset_at = time.monotonic() + sleep_time
                    last_error = EmbeddingRequestError(
                        "Jina embedding API rate limit (HTTP 429; "
                        f"retry_after_s={sleep_time:.1f}; "
                        f"remaining_tokens={self._limiter.remaining_tokens})"
                    )
                    if attempt < self.max_retries:
                        time.sleep(max(sleep_time, backoff))
                    backoff *= 2.0
                    continue

                if response.status_code >= 500:
                    last_error = EmbeddingRequestError(
                        f"Jina embedding API returned HTTP {response.status_code}"
                    )
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                    backoff *= 2.0
                    continue

                if response.status_code >= 400:
                    raise EmbeddingRequestError(
                        f"Jina embedding API rejected the request with HTTP {response.status_code}"
                    )

                data = response.json()
                items = data.get("data", [])
                if not isinstance(items, list) or len(items) != len(batch):
                    raise EmbeddingRequestError(
                        "Jina embedding API returned an unexpected number of vectors"
                    )
                ordered: list[list[float] | None] = [None] * len(batch)
                for item in items:
                    index = item.get("index")
                    vector = item.get("embedding")
                    if (
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                        or index >= len(batch)
                        or ordered[index] is not None
                        or not isinstance(vector, list)
                        or len(vector) != self.dimension
                    ):
                        raise EmbeddingRequestError(
                            "Jina embedding API returned an invalid vector record"
                        )
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in vector
                    ):
                        raise EmbeddingRequestError(
                            "Jina embedding API returned a non-finite or non-numeric vector"
                        )
                    if not any(value != 0 for value in vector):
                        raise EmbeddingRequestError("Jina embedding API returned a zero vector")
                    ordered[index] = [float(value) for value in vector]
                if any(vector is None for vector in ordered):
                    raise EmbeddingRequestError("Jina embedding API omitted a vector record")
                return [vector for vector in ordered if vector is not None]

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(backoff)
                backoff *= 2.0

        raise EmbeddingRequestError(
            f"Jina embedding request failed after {self.max_retries} attempts; "
            f"last error: {last_error}"
        ) from last_error
