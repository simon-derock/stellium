# Pipeline integration tests for retrieved source text and LLM context contracts.
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.coprocessor import Coprocessor, HybridRetrievalResult
from src.llm import LLMCallResult
from src.models import Chunk
from src.pipelines.graphrag import GraphRAGPipeline
from src.pipelines.rag import RAGPipeline
from tests.graph_fixtures import EVENTS, seeded_graph


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


def _graphrag(*replies: LLMCallResult) -> tuple[GraphRAGPipeline, AsyncMock]:
    graph, catalog, coprocessor = seeded_graph()
    llm = AsyncMock()
    llm.embed.return_value = [[0.0] * 1024]
    llm.chat.side_effect = list(replies)
    return GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor, catalog=catalog), llm


def _plan(**fields: object) -> LLMCallResult:
    return _llm_result(json.dumps(fields))


@pytest.mark.asyncio
async def test_graphrag_runs_one_typed_lookup_between_two_llm_calls() -> None:
    pipeline, llm = _graphrag(
        _plan(
            operation="event_attribute", event="women's single sculls", sport="Rowing", year=2012
        ),
        _llm_result("Eve Wren"),
    )

    result = await pipeline.run("g1", "Who won the 2012 women's single sculls?")

    assert result.answer == "Eve Wren"
    assert llm.chat.await_count == 2
    assert result.total_llm_tokens == 50
    assert result.retrieved_doc_ids == ["R12W"]
    answer_prompt = llm.chat.await_args_list[1].args[0][1]["content"]
    # The graph value and the cited article's opening text both reach the answer model.
    assert '"answer": "Eve Wren"' in answer_prompt
    assert "[R12W]" in answer_prompt
    assert result.retrieval_metadata["graph_status"] == "conclusive"
    llm.embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_graphrag_counts_with_code_bounds_and_validates_the_answer() -> None:
    pipeline, _ = _graphrag(
        _plan(
            operation="count_events",
            sport="rowing",
            year=2008,
            comparison="more_than",
            threshold=30,
        ),
        _llm_result("There were two such events."),
    )

    result = await pipeline.run("g2", "How many 2008 rowing events had more than 30 competitors?")

    # The prose answer is not a supported value, so the verified graph count replaces it.
    assert result.answer == "2"
    assert result.retrieval_metadata["answer_source"] == "graph_value"


@pytest.mark.asyncio
async def test_graphrag_reports_ties_from_the_ranking() -> None:
    names = [EVENTS[2]["name"], EVENTS[3]["name"]]
    pipeline, llm = _graphrag(
        _plan(operation="rank_events", sport="rowing", year=2008, order="desc"),
        _llm_result("; ".join(names)),
    )

    result = await pipeline.run("g3", "Which 2008 rowing event had the most competitors?")

    assert result.answer == "; ".join(names)
    assert result.retrieval_metadata["graph_status"] == "ambiguous"
    assert '"candidates"' in llm.chat.await_args_list[1].args[0][1]["content"]


@pytest.mark.asyncio
async def test_graphrag_falls_back_to_hybrid_passages_without_a_graph_operation() -> None:
    pipeline, llm = _graphrag(_plan(operation="none"), _llm_result("Not found in corpus"))
    chunk = _chunk()
    assert pipeline.coprocessor is not None
    with (
        patch.object(
            pipeline.coprocessor, "hybrid_search", return_value=_hybrid_result([(chunk, 0.9)])
        ) as hybrid_search,
        patch.object(pipeline.graph, "vector_search", return_value=[]),
    ):
        result = await pipeline.run("g4", "Why was the marathon notable?")

    hybrid_search.assert_called_once()
    llm.embed.assert_awaited_once()
    assert result.retrieved_doc_ids == ["Q123"]
    assert result.retrieval_metadata["reranker_executed"] is True
    assert "Samuel Wanjiru" in llm.chat.await_args_list[1].args[0][1]["content"]


@pytest.mark.asyncio
async def test_graphrag_tolerates_unparseable_extraction() -> None:
    pipeline, _ = _graphrag(
        _llm_result("I think this is about rowing."), _llm_result("Not found in corpus")
    )
    assert pipeline.coprocessor is not None
    with (
        patch.object(pipeline.coprocessor, "hybrid_search", return_value=_hybrid_result([])),
        patch.object(pipeline.graph, "vector_search", return_value=[]),
    ):
        result = await pipeline.run("g5", "Tell me about rowing.")

    assert result.answer == "Not found in corpus"
    assert result.retrieval_metadata["graph_operation"] == "none"
