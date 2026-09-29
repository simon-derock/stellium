# Pipeline integration tests for retrieved source text and LLM context contracts.
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.coprocessor import Coprocessor
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


@pytest.mark.asyncio
async def test_rag_passes_retrieved_chunk_text_to_answer_model() -> None:
    coprocessor = Coprocessor()
    coprocessor.build([_chunk()])
    graph = MagicMock()
    graph.vector_search.return_value = [("Q123#0", 0.91)]
    llm = AsyncMock()
    llm.embed.return_value = [[0.0] * 1024]
    llm.chat.return_value = _llm_result("Samuel Wanjiru")

    result = await RAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
        "q1", "Who won the 2008 men's Olympic marathon?"
    )

    user_prompt = llm.chat.await_args.args[0][1]["content"]
    assert "Samuel Wanjiru" in user_prompt
    assert result.retrieved_doc_ids == ["Q123"]
    assert result.answer == "Samuel Wanjiru"


@pytest.mark.asyncio
async def test_graphrag_passes_retrieved_chunk_text_to_answer_model() -> None:
    coprocessor = Coprocessor()
    coprocessor.build([_chunk()])
    graph = MagicMock()
    graph.run_multihop.return_value = {
        "events": ["Men's marathon"],
        "gold_athletes": ["Samuel Wanjiru"],
        "gold_doc_ids": ["Q123"],
    }
    graph.run_lookup.return_value = {"events": [], "gold_athletes": [], "gold_doc_ids": []}
    graph.vector_search.return_value = [("Q123#0", 0.91)]
    llm = AsyncMock()
    llm.chat.side_effect = [
        _llm_result(
            '{"operation":"multi_hop","year":2008,"sport":"Athletics",'
            '"event_fragment":"marathon","venue":"Olympic Stadium","date":"17 August"}'
        ),
        _llm_result("Samuel Wanjiru"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    with patch.object(
        coprocessor, "hybrid_rerank", return_value=[(_chunk(), 0.91)]
    ) as hybrid_rerank:
        result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "q2", "Who won the marathon in 2008?"
        )

    synthesis_prompt = llm.chat.await_args.args[0][1]["content"]
    assert "The 2008 Olympic men's marathon was won by Samuel Wanjiru." in synthesis_prompt
    assert "Hybrid rank score 0.910" in synthesis_prompt
    graph.run_multihop.assert_called_once_with(
        venue_fragment="Olympic Stadium", date_fragment="17 August", year=2008
    )
    graph.vector_search.assert_called_once_with([0.0] * 1024, top_k=30)
    hybrid_rerank.assert_called_once_with(
        query="Who won the marathon in 2008?",
        dense_results=[("Q123#0", 0.91)],
        final_top_k=5,
    )
    assert "Q123" in result.retrieved_doc_ids
    assert result.answer == "Samuel Wanjiru"
    assert result.llm_input_tokens == 40
    assert result.llm_output_tokens == 10
    assert result.total_llm_tokens == 50


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
            '{"operation":"aggregation","year":2018,"sport":"Biathlon",'
            '"min_competitors":74,"max_competitors":0}'
        ),
        _llm_result("5"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    with patch.object(coprocessor, "hybrid_rerank", return_value=[]):
        result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "aggregation", "How many 2018 biathlon events had more than 73 competitors?"
        )

    graph.run_aggregation.assert_called_once_with(
        sport="Biathlon", year=2018, min_competitors=74, max_competitors=0
    )
    assert "Exact matching event count: 5" in llm.chat.await_args_list[-1].args[0][1]["content"]
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
            '{"operation":"superlative","year":2016,"sport":"Shooting",'
            '"season":"Summer","order":"desc","limit":1}'
        ),
        _llm_result("Shooting event with most competitors"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    with patch.object(coprocessor, "hybrid_rerank", return_value=[]):
        result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "superlative",
            "Which shooting event in the 2016 Summer Olympics had the most competitors?",
        )

    graph.run_superlative.assert_called_once_with(
        sport="Shooting", year=2016, season="Summer", order="desc", limit=1
    )
    synthesis_prompt = llm.chat.await_args_list[-1].args[0][1]["content"]
    assert "Shooting event with most competitors" in synthesis_prompt
    assert "Competitors: 51" in synthesis_prompt
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
            '{"operation":"lookup","year":2016,"sport":"Judo","gender":"Women",'
            '"event_fragment":"57 kg","attribute":"nation_count"}'
        ),
        _llm_result("23"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    with patch.object(coprocessor, "hybrid_rerank", return_value=[]):
        result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
            "lookup", "How many nations competed in Women's 57 kg judo in 2016?"
        )

    graph.run_lookup.assert_called_once_with(
        event_fragment="57 kg", year=2016, sport="Judo", gender="Women"
    )
    synthesis_prompt = llm.chat.await_args_list[-1].args[0][1]["content"]
    assert "Requested nation_count: 23" in synthesis_prompt
    assert result.answer == "23"
