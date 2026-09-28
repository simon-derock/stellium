# Tests for embedding checkpoint disk caching and interruption recovery.
# Verifies atomic append, resume capability, and corrupted line resilience.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.embeddings import GRAPH_EMBEDDING_DIMENSION, GRAPH_EMBEDDING_MODEL
from src.ingest.batch_upsert import (
    load_chunk_embeddings_cache,
    save_chunk_embeddings_batch,
)

_EMBEDDING_TASK = "retrieval.passage"


def _texts(chunk_ids: list[str]) -> dict[str, str]:
    return {chunk_id: f"source text for {chunk_id}" for chunk_id in chunk_ids}


def _cache_row(chunk_id: str, embedding: list[float], text: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "embedding": embedding,
        "model": GRAPH_EMBEDDING_MODEL,
        "dimension": GRAPH_EMBEDDING_DIMENSION,
        "task": _EMBEDDING_TASK,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def test_checkpoint_cache_save_and_load(tmp_path: Path) -> None:
    cache_file = tmp_path / "test_cache.jsonl"

    sample_pairs = [
        ("chunk_1", [0.1] * 1024),
        ("chunk_2", [-0.2] * 1024),
        ("chunk_3", [0.3] * 1024),
    ]
    source_texts = _texts([chunk_id for chunk_id, _ in sample_pairs])
    save_chunk_embeddings_batch(sample_pairs, cache_file, source_texts=source_texts)

    loaded = load_chunk_embeddings_cache(cache_file, expected_texts=source_texts)
    assert len(loaded) == 3
    assert "chunk_1" in loaded
    assert len(loaded["chunk_1"]) == 1024
    assert loaded["chunk_1"][0] == 0.1
    assert loaded["chunk_2"][0] == -0.2


def test_checkpoint_cache_resume_recovery(tmp_path: Path) -> None:
    cache_file = tmp_path / "resume_cache.jsonl"

    # Step 1: Write first batch (representing chunks embedded before interruption)
    batch_1 = [("c_1", [0.5] * 1024), ("c_2", [0.6] * 1024)]
    source_texts = _texts(["c_1", "c_2", "c_3", "c_4"])
    save_chunk_embeddings_batch(batch_1, cache_file, source_texts=source_texts)

    # Step 2: Resume and load cache
    cache = load_chunk_embeddings_cache(cache_file, expected_texts=source_texts)
    all_chunk_ids = ["c_1", "c_2", "c_3", "c_4"]

    # Step 3: Only un-cached chunks need to be generated
    missing = [cid for cid in all_chunk_ids if cid not in cache]
    assert missing == ["c_3", "c_4"]

    # Step 4: Write remaining batch
    batch_2 = [(cid, [0.7] * 1024) for cid in missing]
    save_chunk_embeddings_batch(batch_2, cache_file, source_texts=source_texts)

    final_cache = load_chunk_embeddings_cache(cache_file, expected_texts=source_texts)
    assert len(final_cache) == 4
    for cid in all_chunk_ids:
        assert cid in final_cache


def test_checkpoint_cache_corrupted_line_resilience(tmp_path: Path) -> None:
    cache_file = tmp_path / "corrupted_cache.jsonl"

    source_texts = _texts(["c_valid_1", "c_valid_2"])
    save_chunk_embeddings_batch(
        [("c_valid_1", [0.1] * 1024)], cache_file, source_texts=source_texts
    )
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write("CORRUPTED_JSON_DATA_HERE\n")
    save_chunk_embeddings_batch(
        [("c_valid_2", [0.2] * 1024)], cache_file, source_texts=source_texts
    )

    # Should gracefully load the 2 valid records without crashing
    cache = load_chunk_embeddings_cache(cache_file, expected_texts=source_texts)
    assert len(cache) == 2
    assert "c_valid_1" in cache
    assert "c_valid_2" in cache


def test_checkpoint_cache_skips_wrong_dimension_and_non_finite_vectors(
    tmp_path: Path,
) -> None:
    cache_file = tmp_path / "invalid_vectors.jsonl"
    source_texts = _texts(["wrong_dimension", "nan_value", "bool_value", "valid", "zero"])
    rows = [
        _cache_row("wrong_dimension", [0.1] * 3, source_texts["wrong_dimension"]),
        _cache_row("nan_value", [0.1] * 1023 + [float("nan")], source_texts["nan_value"]),
        _cache_row("bool_value", [0.1] * 1023 + [True], source_texts["bool_value"]),
        _cache_row("valid", [0.25] * 1024, source_texts["valid"]),
        _cache_row("zero", [0.0] * 1024, source_texts["zero"]),
    ]
    cache_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    loaded = load_chunk_embeddings_cache(cache_file, expected_texts=source_texts)

    assert list(loaded) == ["valid"]
    assert loaded["valid"] == [0.25] * 1024


def test_checkpoint_cache_rejects_invalid_batch_without_partial_append(
    tmp_path: Path,
) -> None:
    cache_file = tmp_path / "atomic_validation.jsonl"
    cache_file.write_text('{"chunk_id":"existing","embedding":[]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="finite nonzero values"):
        save_chunk_embeddings_batch(
            [("valid", [0.5] * 1024), ("invalid", [float("inf")] * 1024)],
            cache_file,
            source_texts={"valid": "valid text", "invalid": "invalid text"},
        )

    assert cache_file.read_text(encoding="utf-8") == '{"chunk_id":"existing","embedding":[]}\n'


def test_checkpoint_cache_rejects_changed_chunk_text(tmp_path: Path) -> None:
    cache_file = tmp_path / "text_drift.jsonl"
    save_chunk_embeddings_batch(
        [("chunk", [0.5] * 1024)], cache_file, source_texts={"chunk": "original passage"}
    )

    loaded = load_chunk_embeddings_cache(cache_file, expected_texts={"chunk": "changed passage"})

    assert loaded == {}


def test_checkpoint_cache_rejects_unknown_model_provenance(tmp_path: Path) -> None:
    cache_file = tmp_path / "model_drift.jsonl"
    source_texts = {"chunk": "passage"}
    save_chunk_embeddings_batch(
        [("chunk", [0.5] * 1024)], cache_file, source_texts=source_texts, model="other-model"
    )

    loaded = load_chunk_embeddings_cache(cache_file, expected_texts=source_texts)

    assert loaded == {}
