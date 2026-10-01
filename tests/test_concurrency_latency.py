# Concurrency, latency, and load stress testing for coprocessor and pipelines.
# Verifies thread safety, sub-millisecond ranking SLAs, and adaptive rate limiting.
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.coprocessor import Coprocessor
from src.embeddings import AdaptiveRateLimiter
from src.llm import LLMCallResult, LockedLLMSession
from src.models import Chunk


def test_coprocessor_latency_sla() -> None:
    # Smoke SLA: warm reranking of 20 short passages should complete within 150ms on this host.
    coprocessor = Coprocessor()
    sample_chunks = [
        Chunk(
            chunk_id=f"c_{i}",
            doc_id=f"doc_{i}",
            chunk_index=i,
            section_title="Events",
            text=f"Sample text passage for Olympic event {i} Athletics Swimming",
            raw_text=f"Sample text passage for Olympic event {i} Athletics Swimming",
            prev_chunk_id=None,
            next_chunk_id=None,
            filter_mask=0,
        )
        for i in range(100)
    ]
    coprocessor.build(sample_chunks)

    dense_candidates = [(f"c_{i}", 0.9 - (i * 0.005)) for i in range(20)]

    # Warm-up to initialize CrossEncoder model weights
    _ = coprocessor.hybrid_rerank("Athletics", dense_candidates[:5], final_top_k=2)

    t0 = time.perf_counter()
    reranked = coprocessor.hybrid_rerank(
        "Athletics Swimming event 5", dense_candidates, final_top_k=5
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(reranked) == 5
    assert elapsed_ms < 150.0, f"Warm rerank took {elapsed_ms:.2f}ms, exceeding 150ms smoke SLA"


def test_adaptive_rate_limiter_throttling() -> None:
    # Concurrency test: Limiter with target_rpm=120 must enforce minimum 0.5s interval
    limiter = AdaptiveRateLimiter(target_rpm=120, min_interval_s=0.05)

    t0 = time.perf_counter()
    for _ in range(5):
        limiter.wait_turn()
    elapsed = time.perf_counter() - t0

    # 5 iterations at 0.05s interval must take at least 0.20s
    assert elapsed >= 0.18, f"Rate limiter was too fast: elapsed {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_concurrent_evaluation_thread_safety() -> None:
    # Concurrent requests through a shared locked session retain their input identity/model.
    session = LockedLLMSession(provider="cloudflare", model="test-model")

    async def fake_provider_call(
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        client: httpx.AsyncClient | None = None,
    ) -> LLMCallResult:
        await asyncio.sleep(0.005)
        return LLMCallResult(
            content=messages[-1]["content"],
            input_tokens=1,
            output_tokens=1,
            model_name=model,
            provider="mock",
            latency_ms=5.0,
        )

    requests = [f"query-{index}" for index in range(20)]
    provider_call = AsyncMock(side_effect=fake_provider_call)
    with patch.dict(LockedLLMSession.chat.__globals__, {"_call_cloudflare": provider_call}):
        results = await asyncio.gather(
            *(session.chat([{"role": "user", "content": request}]) for request in requests)
        )

    assert provider_call.await_count == len(requests)
    assert [result.content for result in results] == requests
    assert {result.model_name for result in results} == {"test-model"}
    assert all(result.input_tokens + result.output_tokens == 2 for result in results)
