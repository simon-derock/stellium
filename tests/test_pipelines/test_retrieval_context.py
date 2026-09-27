# Pipeline integration tests for retrieved source text and LLM context contracts.
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
        _llm_result('{"year": 2008, "sport": "Athletics", "venue": "Olympic Stadium"}'),
        _llm_result("Samuel Wanjiru"),
    ]
    llm.embed.return_value = [[0.0] * 1024]

    result = await GraphRAGPipeline(graph=graph, llm=llm, coprocessor=coprocessor).run(
        "q2", "Who won the marathon in 2008?"
    )

    synthesis_prompt = llm.chat.await_args.args[0][1]["content"]
    assert "The 2008 Olympic men's marathon was won by Samuel Wanjiru." in synthesis_prompt
    assert "Q123" in result.retrieved_doc_ids
    assert result.answer == "Samuel Wanjiru"
