# Pipeline integration tests for retrieved source text and LLM context contracts.
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.coprocessor import Coprocessor, HybridRetrievalResult
from src.llm import LLMCallResult
from src.models import Chunk
from src.pipelines.graphrag import GraphRAGPipeline
from src.pipelines.rag import RAGPipeline


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="Q123#0",
        doc_id="Q123",
        chunk_index=0,
        section_title="Results",
        text="The 2008 Olympic men's marathon was won by Samuel Wanjiru.",
        raw_text="The 2008 Olympic men's marathon was won by Samuel Wanjiru.",
    )


def _llm_result(content: str) -> LLMCallResult:
    return LLMCallResult(
        content=content,
        input_tokens=20,
        output_tokens=5,
        model_name="test-model",
        provider="test",
        latency_ms=1.0,
    )


def _hybrid_result(chunks: list[tuple[Chunk, float]]) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunks=chunks,
        dense_candidate_count=1,
        sparse_candidate_count=1,
        fused_candidate_count=1,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        reranker_executed=bool(chunks),
        reranker_latency_ms=1.0,
    )


@pytest.mark.asyncio
async def test_rag_passes_retrieved_chunk_text_to_answer_model() -> None:
    coprocessor = Coprocessor()
    coprocessor.build([_chunk()])
    graph = MagicMock()
    graph.vector_search.return_value = [("Q123#0", 0.91)]
    llm = AsyncMock()
    llm.embed.return_value = [[0.0] * 1024]
    llm.chat.return_value = _llm_result("Samuel Wanjiru")

    with patch.object(
        coprocessor, "hybrid_search", return_value=_hybrid_result([(_chunk(), 0.91)])
    ) as hybrid_search:
        result = await RAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "q1", "Who won the 2008 men's Olympic marathon?"
        )

    user_prompt = llm.chat.await_args.args[0][1]["content"]
    assert "Samuel Wanjiru" in user_prompt
    assert result.retrieved_doc_ids == ["Q123"]
    assert result.answer == "Samuel Wanjiru"
    graph.vector_search.assert_called_once_with([0.0] * 1024, top_k=30)
    hybrid_search.assert_called_once()
    assert result.retrieval_metadata["reranker_executed"] is True


@pytest.mark.asyncio
async def test_graphrag_passes_retrieved_chunk_text_to_answer_model() -> None:
    coprocessor = Coprocessor()
    coprocessor.build([_chunk()])
    graph = MagicMock()
    graph.run_generated_gsql.return_value = {
        "rows": [{"event": "Men's marathon", "gold_athlete": "Samuel Wanjiru"}],
        "row_groups": 1,
        "latency_ms": 2.0,
    }
    graph.vector_search.return_value = [("Q123#0", 0.91)]
    llm = AsyncMock()
    llm.chat.side_effect = [
        _llm_result(
            "INTERPRET QUERY () FOR GRAPH OlympicsGraph { Events = {Event.*}; "
            "Matched = SELECT e FROM Events:e WHERE e.year == 2008 AND "
            'lower(e.name) LIKE "%marathon%" LIMIT 20; '
            "PRINT Matched[Matched.name, Matched.gold_athlete]; }"
        ),
        _llm_result("Samuel Wanjiru"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    with patch.object(
        coprocessor, "hybrid_search", return_value=_hybrid_result([(_chunk(), 0.91)])
    ) as hybrid_search:
        result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "q2", "Who won the marathon in 2008?"
        )

    synthesis_prompt = llm.chat.await_args.args[0][1]["content"]
    assert "The 2008 Olympic men's marathon was won by Samuel Wanjiru." in synthesis_prompt
    assert "Reranker score 0.910" in synthesis_prompt
    graph.run_generated_gsql.assert_called_once()
    generated_query = graph.run_generated_gsql.call_args.args[0]
    assert "INTERPRET QUERY () FOR GRAPH OlympicsGraph" in generated_query
    graph.vector_search.assert_called_once_with([0.0] * 1024, top_k=30)
    hybrid_search.assert_called_once_with(
        query="Who won the marathon in 2008?",
        dense_results=[("Q123#0", 0.91)],
        candidate_k=30,
        final_top_k=5,
    )
    assert result.retrieved_doc_ids == ["Q123"]
    assert result.answer == "Samuel Wanjiru"
    assert result.llm_input_tokens == 40
    assert result.llm_output_tokens == 10
    assert result.total_llm_tokens == 50
    assert llm.chat.await_count == 2


@pytest.mark.asyncio
async def test_graphrag_continues_with_passages_when_interpreted_gsql_fails() -> None:
    coprocessor = Coprocessor()
    coprocessor.build([_chunk()])
    graph = MagicMock()
    graph.run_generated_gsql.side_effect = RuntimeError("TigerGraph rejected PRINT syntax")
    graph.vector_search.return_value = [("Q123#0", 0.91)]
    llm = AsyncMock()
    llm.chat.side_effect = [
        _llm_result(
            "INTERPRET QUERY () FOR GRAPH OlympicsGraph { Events = {Event.*}; "
            "Matched = SELECT e FROM Events:e WHERE e.year == 2008 LIMIT 20; "
            "PRINT Matched[Matched.name, Matched.gold_athlete]; }"
        ),
        _llm_result("Samuel Wanjiru"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    with patch.object(
        coprocessor, "hybrid_search", return_value=_hybrid_result([(_chunk(), 0.91)])
    ):
        result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "q-gsql-error", "Who won the 2008 men's Olympic marathon?"
        )

    synthesis_prompt = llm.chat.await_args_list[-1].args[0][1]["content"]
    assert "could not execute" in synthesis_prompt
    assert "Samuel Wanjiru" in synthesis_prompt
    assert result.answer == "Samuel Wanjiru"
    assert result.retrieved_doc_ids == ["Q123"]
    assert llm.chat.await_count == 2


@pytest.mark.asyncio
async def test_graphrag_uses_gsql_for_aggregation_and_counts_all_llm_tokens() -> None:
    coprocessor = Coprocessor()
    coprocessor.build([_chunk()])
    graph = MagicMock()
    graph.run_aggregation.return_value = {
        "count": 5,
        "events": ["Biathlon event A", "Biathlon event B"],
        "gold_doc_ids": ["Q123"],
    }
    graph.vector_search.return_value = []
    llm = AsyncMock()
    llm.chat.side_effect = [
        _llm_result(
            "INTERPRET QUERY () FOR GRAPH OlympicsGraph { Events = {Event.*}; "
            "SumAccum<INT> @@count = 0; Matched = SELECT e FROM Events:e "
            'WHERE e.year == 2018 AND lower(e.sport) == "biathlon" '
            "AND e.competitor_count > 73 ACCUM @@count += 1 LIMIT 2500; "
            "PRINT @@count; }"
        ),
        _llm_result("5"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    graph.run_generated_gsql.return_value = {"rows": [{"@@count": 5}], "row_groups": 1}
    with patch.object(coprocessor, "hybrid_search", return_value=_hybrid_result([])):
        result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "aggregation", "How many 2018 biathlon events had more than 73 competitors?"
        )

    graph.run_generated_gsql.assert_called_once()
    assert "@@count" in llm.chat.await_args_list[-1].args[0][1]["content"]
    assert result.answer == "5"
    assert result.total_llm_tokens == 50


@pytest.mark.asyncio
async def test_graphrag_uses_compiled_superlative_query() -> None:
    coprocessor = Coprocessor()
    coprocessor.build([_chunk()])
    graph = MagicMock()
    graph.run_superlative.return_value = {
        "events": ["Shooting event with most competitors"],
        "competitor_counts": [51],
        "gold_doc_ids": ["Q456"],
    }
    graph.vector_search.return_value = []
    llm = AsyncMock()
    llm.chat.side_effect = [
        _llm_result(
            "INTERPRET QUERY () FOR GRAPH OlympicsGraph { Events = {Event.*}; "
            "Ranked = SELECT e FROM Events:e WHERE e.year == 2016 AND "
            'lower(e.sport) == "shooting" ORDER BY e.competitor_count DESC LIMIT 1; '
            "PRINT Ranked[Ranked.name, Ranked.competitor_count]; }"
        ),
        _llm_result("Shooting event with most competitors"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    graph.run_generated_gsql.return_value = {
        "rows": [{"name": "Shooting event with most competitors", "competitor_count": 51}],
        "row_groups": 1,
    }
    with patch.object(coprocessor, "hybrid_search", return_value=_hybrid_result([])):
        result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "superlative",
            "Which shooting event in the 2016 Summer Olympics had the most competitors?",
        )

    graph.run_generated_gsql.assert_called_once()
    synthesis_prompt = llm.chat.await_args_list[-1].args[0][1]["content"]
    assert "Shooting event with most competitors" in synthesis_prompt
    assert "competitor_count" in synthesis_prompt
    assert "51" in synthesis_prompt
    assert result.answer == "Shooting event with most competitors"


@pytest.mark.asyncio
async def test_graphrag_uses_typed_lookup_for_requested_event_attribute() -> None:
    coprocessor = Coprocessor()
    coprocessor.build([_chunk()])
    graph = MagicMock()
    graph.run_lookup.return_value = {
        "events": ["Judo at the 2016 Summer Olympics – Women's 57 kg"],
        "nation_counts": [23],
        "gold_doc_ids": ["Q789"],
    }
    graph.vector_search.return_value = []
    llm = AsyncMock()
    llm.chat.side_effect = [
        _llm_result(
            "INTERPRET QUERY () FOR GRAPH OlympicsGraph { Events = {Event.*}; "
            "Matched = SELECT e FROM Events:e WHERE e.year == 2016 AND "
            'lower(e.sport) == "judo" AND lower(e.gender) == "women" '
            'AND lower(e.name) LIKE "%57 kg%" LIMIT 10; '
            "PRINT Matched[Matched.name, Matched.nation_count]; }"
        ),
        _llm_result("23"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    graph.run_generated_gsql.return_value = {
        "rows": [{"name": "Judo at the 2016 Summer Olympics – Women's 57 kg", "nation_count": 23}],
        "row_groups": 1,
    }
    with patch.object(coprocessor, "hybrid_search", return_value=_hybrid_result([])):
        result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "lookup", "How many nations competed in Women's 57 kg judo in 2016?"
        )

    synthesis_prompt = llm.chat.await_args_list[-1].args[0][1]["content"]
    assert "nation_count" in synthesis_prompt
    assert "23" in synthesis_prompt
    assert result.answer == "23"
