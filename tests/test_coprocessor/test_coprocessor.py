# Unit tests for BM25, bitmask filter, and RRF fusion.
from types import SimpleNamespace
from typing import Any

import pytest
from rank_bm25 import BM25Plus

import src.coprocessor as coprocessor_module
from src.coprocessor import (
    BM25Index,
    Coprocessor,
    LocalCrossEncoder,
    build_filter_mask,
    reciprocal_rank_fusion,
    season_mask,
    year_mask,
)
from src.models import Chunk


def test_bitmask_operations() -> None:
    mask_2012_summer = build_filter_mask(year=2012, season="Summer")
    # Check that individual bitmasks match
    y_mask = year_mask(2012)
    s_mask = season_mask("Summer")
    assert (mask_2012_summer & y_mask) != 0
    assert (mask_2012_summer & s_mask) != 0

    # Winter should not match
    w_mask = season_mask("Winter")
    assert (mask_2012_summer & w_mask) == 0


def test_bm25_search() -> None:
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            chunk_index=0,
            section_title="Canoeing 2012",
            text="Rudolf Dombi and Roland Kokeny won gold in canoe sprint",
            raw_text="Rudolf Dombi and Roland Kokeny won gold in canoe sprint",
        ),
        Chunk(
            chunk_id="c2",
            doc_id="d2",
            chunk_index=0,
            section_title="Marathon 2008",
            text="Samuel Wanjiru won the men marathon at the 2008 Summer Olympics",
            raw_text="Samuel Wanjiru won the men marathon at the 2008 Summer Olympics",
        ),
    ]
    bm25 = BM25Index()
    bm25.build(chunks)

    results = bm25.search("Rudolf Dombi", top_k=5)
    assert len(results) > 0
    assert results[0][0].chunk_id == "c1"

    marathon_results = bm25.search("Wanjiru marathon", top_k=5)
    assert len(marathon_results) > 0
    assert marathon_results[0][0].chunk_id == "c2"


def test_compact_bm25plus_scores_match_reference() -> None:
    chunks = [
        Chunk(
            chunk_id=f"c{index}",
            doc_id=f"d{index}",
            chunk_index=0,
            section_title="Results",
            text=text,
            raw_text=text,
        )
        for index, text in enumerate(
            ["Olympic biathlon gold medal", "Biathlon silver medal", "Swimming gold medal"]
        )
    ]
    index = BM25Index()
    index.build(chunks)
    reference = BM25Plus([coprocessor_module._tokenize(chunk.raw_text) for chunk in chunks])

    actual = index.search("biathlon biathlon unknown", top_k=len(chunks))
    expected = reference.get_scores(coprocessor_module._tokenize("biathlon biathlon unknown"))

    actual_scores = {int(chunk.chunk_id[1:]): score for chunk, score in actual}
    assert actual_scores == pytest.approx(dict(enumerate(expected.tolist())))


def test_bm25_search_normalizes_punctuation_symmetrically() -> None:
    chunks = [
        Chunk(
            chunk_id="distractor#0",
            doc_id="distractor",
            chunk_index=0,
            section_title="Canoeing",
            text="Canoe sprint event",
            raw_text="Canoe sprint event",
        ),
        Chunk(
            chunk_id="target#0",
            doc_id="target",
            chunk_index=0,
            section_title="Sailing",
            text="Sailing RS:X Olympic event",
            raw_text="Sailing RS:X Olympic event",
        ),
    ]
    bm25 = BM25Index()
    bm25.build(chunks)

    results = bm25.search("sailing—RS:X", top_k=2)

    assert results[0][0].doc_id == "target"


def test_bm25_top_k_selection_preserves_stable_ties_and_handles_empty_k() -> None:
    chunks = [
        Chunk(
            chunk_id=f"tie-{index}",
            doc_id=f"tie-{index}",
            chunk_index=0,
            section_title="Same evidence",
            text="Olympic event competitors",
            raw_text="Olympic event competitors",
            filter_mask=build_filter_mask(2012, "Summer"),
        )
        for index in range(3)
    ]
    bm25 = BM25Index()
    bm25.build(chunks)

    assert [chunk.chunk_id for chunk, _ in bm25.search("Olympic", top_k=3)] == [
        "tie-0",
        "tie-1",
        "tie-2",
    ]
    assert [chunk.chunk_id for chunk, _ in bm25.search("Olympic", top_k=1)] == ["tie-0"]
    assert bm25.search("Olympic", top_k=0) == []
    assert (
        bm25.search_filtered("Olympic", build_filter_mask(2012, "Summer"), top_k=1)[0][0].chunk_id
        == "tie-0"
    )
    assert bm25.search_filtered("Olympic", build_filter_mask(2012, "Summer"), top_k=0) == []


def test_bm25_search_filtered_requires_each_requested_facet() -> None:
    shared_text = "Olympic event results and competitors"
    chunks = [
        Chunk(
            chunk_id="2012-summer",
            doc_id="2012-summer",
            chunk_index=0,
            section_title="Event",
            text=shared_text,
            raw_text=shared_text,
            filter_mask=build_filter_mask(2012, "Summer"),
        ),
        Chunk(
            chunk_id="2012-winter",
            doc_id="2012-winter",
            chunk_index=0,
            section_title="Event",
            text=shared_text,
            raw_text=shared_text,
            filter_mask=build_filter_mask(2012, "Winter"),
        ),
        Chunk(
            chunk_id="2008-summer",
            doc_id="2008-summer",
            chunk_index=0,
            section_title="Event",
            text=shared_text,
            raw_text=shared_text,
            filter_mask=build_filter_mask(2008, "Summer"),
        ),
    ]
    bm25 = BM25Index()
    bm25.build(chunks)

    results = bm25.search_filtered(
        "Olympic event",
        build_filter_mask(2012, "Summer"),
        top_k=10,
    )

    assert [chunk.chunk_id for chunk, _ in results] == ["2012-summer"]


def test_bm25_search_filtered_rejects_unknown_nonzero_mask() -> None:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        chunk_index=0,
        section_title="Event",
        text="Olympic event results",
        raw_text="Olympic event results",
    )
    bm25 = BM25Index()
    bm25.build([chunk])

    assert bm25.search_filtered("Olympic event", 1 << 31) == []


def test_rrf_fusion() -> None:
    # Ranked list 1 (dense)
    list1 = [("docA", 0.95), ("docB", 0.85), ("docC", 0.75)]
    # Ranked list 2 (sparse)
    list2 = [("docB", 12.5), ("docA", 8.2), ("docD", 5.1)]

    fused = reciprocal_rank_fusion(list1, list2, top_k=4)
    # docA and docB appear in both lists, so they must rank higher than docC and docD
    top_ids = [doc_id for doc_id, _ in fused[:2]]
    assert "docA" in top_ids
    assert "docB" in top_ids


def test_hybrid_search_runs_reranker_and_records_stage_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        Chunk(
            chunk_id="d1#0",
            doc_id="d1",
            chunk_index=0,
            section_title="Event",
            text="The event had 20 competitors.",
            raw_text="The event had 20 competitors.",
        ),
        Chunk(
            chunk_id="d2#0",
            doc_id="d2",
            chunk_index=0,
            section_title="Event",
            text="The event had 80 competitors.",
            raw_text="The event had 80 competitors.",
        ),
    ]
    coprocessor = Coprocessor()
    coprocessor.build(chunks)
    fake_reranker = SimpleNamespace(
        model_name="local-test-reranker",
        latency_ms=4.0,
        predict=lambda pairs: [0.1, 0.9],
    )
    monkeypatch.setattr(coprocessor_module, "_get_reranker", lambda: fake_reranker)

    result = coprocessor.hybrid_search(
        "Which event had the most competitors?",
        [("d1#0", 0.9), ("d2#0", 0.8)],
        candidate_k=2,
        final_top_k=2,
    )

    assert [chunk.doc_id for chunk, _ in result.chunks] == ["d2", "d1"]
    assert result.reranker_model == "local-test-reranker"
    assert result.reranker_executed
    assert result.reranker_latency_ms == 4.0
    assert result.dense_candidate_count == 2
    assert result.fused_candidate_count == 2


def test_local_reranker_batches_candidates_without_dropping_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="input_ids")]

        def run(self, _outputs: None, inputs: dict[str, Any]) -> list[list[list[float]]]:
            input_ids = inputs["input_ids"]
            self.batch_sizes.append(len(input_ids))
            return [[[float(sum(row))] for row in input_ids]]

    class FakeTokenizer:
        def encode(self, query: str, document: str) -> SimpleNamespace:
            ids = [len(query), len(document)]
            return SimpleNamespace(ids=ids, attention_mask=[1, 1], type_ids=[0, 1])

    monkeypatch.setenv("RERANKER_BATCH_SIZE", "2")
    session = FakeSession()
    reranker = LocalCrossEncoder(session=session, tokenizer=FakeTokenizer())

    scores = reranker.predict(
        [("q", "one"), ("q", "two"), ("q", "three"), ("q", "four"), ("q", "five")]
    )

    assert session.batch_sizes == [2, 2, 1]
    assert scores == [4.0, 4.0, 6.0, 5.0, 5.0]
    assert reranker.latency_ms >= 0
