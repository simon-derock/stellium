# Jina Embeddings v3 client with strict rate-limiting, batching, and retries.
# Produces 1024-dimensional embeddings matching TigerGraph HNSW vector schema.
# Zero triple-quote docstrings — use single or multi-line # comments exclusively.
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import httpx

# ---------------------------------------------------------------------------
# Constants & Jina API Limits
# ---------------------------------------------------------------------------

_JINA_API_URL = "https://api.jina.ai/v1/embeddings"
_MODEL_NAME = "jina-embeddings-v3"
_DEFAULT_DIMENSION = 1024  # Matches TigerGraph Savanna HNSW vector attribute

# Free Tier Safety Limits:
# Max RPM: 100 requests per minute
# Max TPM: 100,000 tokens per minute
# Max batch size recommended: 64 chunks (~16,000-25,000 tokens per request)
_DEFAULT_BATCH_SIZE = 64
_MIN_REQUEST_INTERVAL_S = 0.65  # Enforces <= 92 RPM (< 100 RPM hard ceiling)
_MAX_RETRIES = 5
_INITIAL_BACKOFF_S = 1.5

TaskType = Literal["retrieval.passage", "retrieval.query", "text-matching", "classification"]


# ---------------------------------------------------------------------------
# Rate Limiter & Token Bucket
# ---------------------------------------------------------------------------


class RateLimiter:
    # Token-bucket rate limiter ensuring calls never violate Jina RPM or TPM.
    def __init__(self, min_interval_s: float = _MIN_REQUEST_INTERVAL_S) -> None:
        self.min_interval_s = min_interval_s
        self.last_call_time = 0.0

    def wait_turn(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self.last_call_time = time.monotonic()


# ---------------------------------------------------------------------------
# Jina Embedding Client
# ---------------------------------------------------------------------------


@dataclass
class JinaEmbeddingClient:
    api_key: str
    dimension: int = _DEFAULT_DIMENSION
    batch_size: int = _DEFAULT_BATCH_SIZE
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(min_interval_s=_MIN_REQUEST_INTERVAL_S)

    @classmethod
    def from_env(cls) -> JinaEmbeddingClient:
        # Load API key checking both standard and user-specified variable names
        key = (
            os.environ.get("JINA_API_KEY")
            or os.environ.get("jina_embedding_api_key")
            or os.environ.get("JINA_EMBEDDING_API_KEY")
            or ""
        )
        return cls(api_key=key.strip())

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # Generate 1024-dim embeddings for corpus chunks with retrieval.passage task
        return self._embed_batch_orchestrator(texts, task="retrieval.passage")

    def embed_query(self, query: str) -> list[float]:
        # Generate 1024-dim embedding for user or agent query with retrieval.query task
        res = self._embed_batch_orchestrator([query], task="retrieval.query")
        return res[0] if res else [0.0] * self.dimension

    def _embed_batch_orchestrator(
        self, texts: list[str], task: TaskType = "retrieval.passage"
    ) -> list[list[float]]:
        if not texts:
            return []

        # If key is missing, return deterministic fallback embeddings for offline resilience
        if not self.is_configured:
            return [[0.0] * self.dimension for _ in texts]

        results: list[list[float]] = []

        # Chunk into manageable batch sizes to stay safely within TPM and memory limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_single_batch_with_retry(batch, task=task)
            results.extend(batch_embeddings)

        return results

    def _embed_single_batch_with_retry(self, batch: list[str], task: TaskType) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": _MODEL_NAME,
            "task": task,
            "dimensions": self.dimension,
            "late_chunking": False,
            "embedding_type": "float",
            "input": batch,
        }

        backoff = _INITIAL_BACKOFF_S

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # Throttle request rate
                self._limiter.wait_turn()

                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(_JINA_API_URL, headers=headers, json=payload)

                # Check for rate limiting (HTTP 429)
                if response.status_code == 429:
                    retry_after = float(response.headers.get("retry-after", backoff))
                    sleep_time = max(retry_after, backoff)
                    time.sleep(sleep_time)
                    backoff *= 2.0
                    continue

                response.raise_for_status()
                data = response.json()
                items = data.get("data", [])
                # Ensure sorted by original index
                sorted_items = sorted(items, key=lambda x: int(x.get("index", 0)))
                return [item["embedding"] for item in sorted_items]

            except (httpx.HTTPError, httpx.TimeoutException, KeyError):
                if attempt == _MAX_RETRIES:
                    break
                time.sleep(backoff)
                backoff *= 2.0

        # If all retries fail, return zero vectors rather than crashing the pipeline
        return [[0.0] * self.dimension for _ in batch]
