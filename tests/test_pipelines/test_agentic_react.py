# Unit tests for the Pure ReAct Agent Harness and trajectory execution.
# Verifies Thought-Action-Observation loop, tool dispatch, strategy adaptation, and trace logging.
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.coprocessor import Coprocessor
from src.graph.mock import create_mock_graph_client
from src.llm import LLMCallResult, LockedLLMSession
from src.pipelines.agentic import AgenticPipeline, parse_react_response


def test_parse_react_response_action_and_input() -> None:
    raw_output = """Thought: I need to check the event aggregates for Biathlon in 2018.
Action: gsql_aggregate
Action Input: {"sport": "Biathlon", "target_year": 2018, "min_competitors": 74}"""

    parsed = parse_react_response(raw_output)
    assert parsed.thought == "I need to check the event aggregates for Biathlon in 2018."
    assert parsed.action == "gsql_aggregate"
    assert parsed.action_input == {
        "sport": "Biathlon",
        "target_year": 2018,
        "min_competitors": 74,
    }
    assert not parsed.is_terminal


def test_parse_react_response_final_answer() -> None:
    raw_output = """Thought: The graph returned conclusive evidence.
Final Answer: Chen Ding"""

    parsed = parse_react_response(raw_output)
    assert parsed.thought == "The graph returned conclusive evidence."
    assert parsed.final_answer == "Chen Ding"
    assert parsed.is_terminal


def test_parse_react_response_finish_tool() -> None:
    raw_output = """Thought: Done.
Action: finish
Action Input: {"answer": "Usain Bolt", "confidence": 0.99, "citations": ["Q123"]}"""

    parsed = parse_react_response(raw_output)
    assert parsed.action == "finish"
    assert parsed.final_answer == "Usain Bolt"
    assert parsed.is_terminal


@pytest.mark.asyncio
async def test_agentic_react_full_trajectory() -> None:
    graph = create_mock_graph_client()
    coprocessor = Coprocessor()
    session = LockedLLMSession(provider="cloudflare", model="mock-model")

    pipeline = AgenticPipeline(graph=graph, coprocessor=coprocessor, llm=session)

    # Mock response demonstrating ReAct Thought and Action dispatch
    mock_res = LLMCallResult(
        content="""Thought: The user asks for biathlon events with >73 competitors in 2018. I will invoke gsql_aggregate.
Action: gsql_aggregate
Action Input: {"sport": "Biathlon", "target_year": 2018, "min_competitors": 74, "max_competitors": 0}""",
        input_tokens=120,
        output_tokens=45,
        model_name="mock-model",
        provider="mock",
        latency_ms=15.0,
    )

    with patch.object(LockedLLMSession, "chat", return_value=mock_res):
        result = await pipeline.run(
            qid="test-pub-1",
            question="According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
        )

        assert result.qid == "test-pub-1"
        assert result.pipeline == "agentic"
        assert result.answer == "5"
        assert result.total_llm_tokens > 0

        # Verify trace structure matching hackathon rubric
        trace = result.agentic_trace
        assert trace is not None
        assert trace["step_count"] >= 1
        assert "gsql_aggregate" in trace["tools_called"][0]["tool_name"]
        assert trace["stopping_reason"] != ""
        assert trace["confidence_score"] >= 0.90


@pytest.mark.asyncio
async def test_agentic_react_strategy_adaptation() -> None:
    graph = create_mock_graph_client()
    coprocessor = Coprocessor()
    session = LockedLLMSession(provider="cloudflare", model="mock-model")

    pipeline = AgenticPipeline(graph=graph, coprocessor=coprocessor, llm=session)

    # Step 1: LLM selects gsql_lookup which returns empty
    step1_res = LLMCallResult(
        content="""Thought: First try structured lookup.
Action: gsql_lookup
Action Input: {"event_name_fragment": "nonexistent_event_xyz", "target_year": 2020}""",
        input_tokens=100,
        output_tokens=30,
        model_name="mock-model",
        provider="mock",
        latency_ms=10.0,
    )

    # Step 2: LLM adapts strategy to vector_search and finishes
    step2_res = LLMCallResult(
        content="""Thought: Lookup was empty. Adapting strategy to vector search on text chunks.
Action: finish
Action Input: {"answer": "Adapted Answer", "confidence": 0.85, "citations": ["doc-1"]}""",
        input_tokens=150,
        output_tokens=35,
        model_name="mock-model",
        provider="mock",
        latency_ms=10.0,
    )

    with patch.object(LockedLLMSession, "chat", side_effect=[step1_res, step2_res]):
        result = await pipeline.run(qid="test-adapt", question="Unknown question?")
        assert result.answer == "Adapted Answer"
        assert result.agentic_trace is not None
        assert result.agentic_trace["step_count"] >= 2
