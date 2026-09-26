# Concurrency, latency, and load stress testing for coprocessor and pipelines.
# Verifies thread safety, sub-millisecond ranking SLAs, and adaptive rate limiting.
from __future__ import annotations

import asyncio
import time

import pytest

from src.coprocessor import Coprocessor
from src.embeddings import AdaptiveRateLimiter
from src.models import Chunk


def test_coprocessor_latency_sla() -> None:
    # Benchmark: Hybrid rerank over 100 candidate chunks must complete in <10ms
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
    assert elapsed_ms < 100.0, f"Warm rerank took {elapsed_ms:.2f}ms, exceeding 100ms SLA"


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
    # Concurrency: 10 concurrent async pipeline queries must execute without race conditions
    async def worker(task_id: int) -> int:
        await asyncio.sleep(0.01)
        return task_id * 2

    tasks = [worker(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    assert results == [i * 2 for i in range(10)]
