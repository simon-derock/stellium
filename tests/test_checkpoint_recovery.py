# Tests for embedding checkpoint disk caching and interruption recovery.
# Verifies atomic append, resume capability, and corrupted line resilience.
from __future__ import annotations

import json
from pathlib import Path

from src.ingest.batch_upsert import (
    load_chunk_embeddings_cache,
    save_chunk_embeddings_batch,
)


def test_checkpoint_cache_save_and_load(tmp_path: Path) -> None:
    cache_file = tmp_path / "test_cache.jsonl"

    sample_pairs = [
        ("chunk_1", [0.1] * 1024),
        ("chunk_2", [-0.2] * 1024),
        ("chunk_3", [0.0] * 1024),
    ]
    save_chunk_embeddings_batch(sample_pairs, cache_file)

    loaded = load_chunk_embeddings_cache(cache_file)
    assert len(loaded) == 3
    assert "chunk_1" in loaded
    assert len(loaded["chunk_1"]) == 1024
    assert loaded["chunk_1"][0] == 0.1
    assert loaded["chunk_2"][0] == -0.2


def test_checkpoint_cache_resume_recovery(tmp_path: Path) -> None:
    cache_file = tmp_path / "resume_cache.jsonl"

    # Step 1: Write first batch (representing chunks embedded before interruption)
    batch_1 = [("c_1", [0.5] * 1024), ("c_2", [0.6] * 1024)]
    save_chunk_embeddings_batch(batch_1, cache_file)

    # Step 2: Resume and load cache
    cache = load_chunk_embeddings_cache(cache_file)
    all_chunk_ids = ["c_1", "c_2", "c_3", "c_4"]

    # Step 3: Only un-cached chunks need to be generated
    missing = [cid for cid in all_chunk_ids if cid not in cache]
    assert missing == ["c_3", "c_4"]

    # Step 4: Write remaining batch
    batch_2 = [(cid, [0.7] * 1024) for cid in missing]
    save_chunk_embeddings_batch(batch_2, cache_file)

    final_cache = load_chunk_embeddings_cache(cache_file)
    assert len(final_cache) == 4
    for cid in all_chunk_ids:
        assert cid in final_cache


def test_checkpoint_cache_corrupted_line_resilience(tmp_path: Path) -> None:
    cache_file = tmp_path / "corrupted_cache.jsonl"

    # Write valid line, malformed line, and another valid line
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": "c_valid_1", "embedding": [0.1] * 1024}) + "\n")
        f.write("CORRUPTED_JSON_DATA_HERE\n")
        f.write(json.dumps({"chunk_id": "c_valid_2", "embedding": [0.2] * 1024}) + "\n")

    # Should gracefully load the 2 valid records without crashing
    cache = load_chunk_embeddings_cache(cache_file)
    assert len(cache) == 2
    assert "c_valid_1" in cache
    assert "c_valid_2" in cache
