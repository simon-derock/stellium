# Unit tests for BM25, bitmask filter, and RRF fusion.
from src.coprocessor import (
    BM25Index,
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
