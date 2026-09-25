# Unit tests for BM25, bitmask filter, and RRF fusion.
from src.coprocessor import (
    BM25Index,
    build_filter_mask,
    reciprocal_rank_fusion,
    season_mask,
    sport_mask,
    year_mask,
)
from src.models import Chunk


def test_bitmask_operations() -> None:
    mask_2012_summer_athletics = build_filter_mask(year=2012, season="Summer", sport="Athletics")
    # Check that individual bitmasks match
    y_mask = year_mask(2012)
    s_mask = season_mask("Summer")
    sp_mask = sport_mask("Athletics")

    assert (mask_2012_summer_athletics & y_mask) != 0
    assert (mask_2012_summer_athletics & s_mask) != 0
    assert (mask_2012_summer_athletics & sp_mask) != 0

    # Winter should not match
    w_mask = season_mask("Winter")
    assert (mask_2012_summer_athletics & w_mask) == 0


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
