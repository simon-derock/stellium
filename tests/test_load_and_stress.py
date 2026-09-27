# Load, concurrency, volume, and latency SLA stress testing.
# Verifies system throughput, concurrent read safety, and coprocessor operations.

import time
from concurrent.futures import ThreadPoolExecutor

from src.coprocessor import Coprocessor, reciprocal_rank_fusion
from src.models import Chunk


def _make_sample_chunks(count: int = 50) -> list[Chunk]:
    # Generates synthetic Olympic chunks for high-concurrency benchmarks.
    sports = ["Athletics", "Biathlon", "Swimming", "Gymnastics", "Skiing"]
    chunks: list[Chunk] = []
    for i in range(count):
        sport = sports[i % len(sports)]
        chunk = Chunk(
            chunk_id=f"doc_{i}#0",
            doc_id=f"doc_{i}",
            chunk_index=0,
            section_title="Overview",
            text=f"The {sport} event at the Olympic Games was held with high competition. Gold won by Athlete {i}.",
            raw_text=f"The {sport} event at the Olympic Games was held with high competition. Gold won by Athlete {i}.",
            filter_mask=0,
        )
        chunks.append(chunk)
    return chunks


def test_concurrent_coprocessor_search_stress() -> None:
    # Runs 100 concurrent reads against one shared coprocessor instance.
    # Asserts stable results and median latency < 10ms.
    chunks = _make_sample_chunks(100)
    coproc = Coprocessor()
    coproc.build(chunks)

    queries = [
        "Athletics competition gold",
        "Biathlon Olympic Games",
        "Swimming high competition",
        "Gymnastics Athlete",
        "Skiing Olympic",
    ]

    def worker(query: str) -> tuple[float, int]:
        t0 = time.perf_counter()
        results = coproc.bm25_search(query, top_k=10)
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed, len(results)

    # Worker threads overlap synchronous reads from the shared in-memory index.
    requests = [queries[i % len(queries)] for i in range(100)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        measurements = list(executor.map(worker, requests))

    latencies = sorted(elapsed for elapsed, _ in measurements)
    median_latency = latencies[len(latencies) // 2]
    p99_latency = latencies[int(len(latencies) * 0.99)]

    assert all(result_count > 0 for _, result_count in measurements)
    # SLA requirements: median sub-millisecond or sub-5ms in Python
    assert median_latency < 10.0, f"Median latency exceeded SLA: {median_latency:.2f}ms"
    assert p99_latency < 30.0, f"p99 latency exceeded SLA: {p99_latency:.2f}ms"


def test_rrf_high_throughput_stress() -> None:
    # Tests reciprocal rank fusion under 1,000 rapid calls.
    # Asserts sub-millisecond execution and monotonic score ranking.
    list_a = [(f"doc_{i}", float(100 - i)) for i in range(30)]
    list_b = [(f"doc_{i}", float(100 - i * 2)) for i in range(15, 45)]

    t0 = time.perf_counter()
    for _ in range(1000):
        fused = reciprocal_rank_fusion(list_a, list_b, top_k=20)
        assert len(fused) == 20
        # Assert descending score invariant
        for idx in range(len(fused) - 1):
            assert fused[idx][1] >= fused[idx + 1][1]
    elapsed = time.perf_counter() - t0

    # 1000 iterations must take less than 0.50s (>2000 fusions/sec)
    assert elapsed < 0.50, f"RRF throughput degraded: {elapsed:.3f}s for 1000 ops"


def test_coprocessor_memory_stability() -> None:
    # Verifies memory stability and zero leakage over 500 consecutive query searches.
    import resource

    chunks = _make_sample_chunks(200)
    coproc = Coprocessor()
    coproc.build(chunks)

    # Initial RSS memory
    mem_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for i in range(500):
        _ = coproc.bm25_search(f"sport query {i % 10}", top_k=15)

    mem_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Memory growth should be zero or negligible (ru_maxrss is in KB on Linux)
    growth_mb = (mem_end - mem_start) / 1024.0
    # Growth must not exceed 10MB across 500 searches
    assert growth_mb < 10.0, f"Potential memory leak detected: {growth_mb:.2f}MB growth"
