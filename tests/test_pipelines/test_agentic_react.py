# Unit tests for the Pure ReAct Agent Harness and trajectory execution.
# Verifies Thought-Action-Observation loop, tool dispatch, strategy adaptation, and trace logging.
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.coprocessor import Coprocessor
from src.graph.mock import create_mock_graph_client
from src.llm import LLMCallResult, LockedLLMSession
from src.pipelines.agentic import _REACT_SYSTEM_PROMPT, AgenticPipeline, parse_react_response


def test_react_prompt_does_not_contain_public_benchmark_questions() -> None:
    dataset = (
        Path(__file__).resolve().parents[2] / "hackathon-resources/questions/eval_public.jsonl"
    )
    prompt = _REACT_SYSTEM_PROMPT.casefold()
    assert "default document retriever" in prompt
    assert "prefer `hybrid_search`" in prompt

    for line in dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            question = json.loads(line)["question"].casefold()
            assert question not in prompt


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
    mock_final = LLMCallResult(
        content="Thought: The aggregate matches the requested threshold.\nFinal Answer: 5",
        input_tokens=80,
        output_tokens=10,
        model_name="mock-model",
        provider="mock",
        latency_ms=8.0,
    )
    with patch.object(LockedLLMSession, "chat", side_effect=[mock_res, mock_final]) as chat:
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
        assert trace["step_count"] == len(trace["llm_calls"]) + len(trace["tools_called"])
        assert trace["agents_invoked"] == ["ReActOrchestrator"]
        assert "gsql_aggregate" in trace["tools_called"][0]["tool_name"]
        assert trace["tools_called"][0]["llm_tokens"] == 0
        assert (
            sum(call["prompt_tokens"] + call["completion_tokens"] for call in trace["llm_calls"])
            == result.total_llm_tokens
        )
        assert trace["stopping_reason"] != ""
        assert trace["confidence_score"] >= 0.90
        assert chat.await_count == 2


@pytest.mark.asyncio
async def test_agentic_retrieved_document_order_is_stable_for_rank_metrics() -> None:
    graph = create_mock_graph_client()
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
    )
    responses = [
        LLMCallResult(
            content="Thought: Count matching events.\nAction: gsql_aggregate\nAction Input: "
            '{"sport":"Biathlon","target_year":2018,"min_competitors":74}',
            input_tokens=80,
            output_tokens=30,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content="Thought: The exact count is supported by the graph.\nFinal Answer: 2",
            input_tokens=60,
            output_tokens=8,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
    ]

    with (
        patch.object(
            graph,
            "run_aggregation",
            return_value={
                "count": 2,
                "events": ["Event A", "Event B"],
                "gold_doc_ids": ["Q2", "Q1", "Q2"],
            },
        ),
        patch.object(LockedLLMSession, "chat", side_effect=responses),
    ):
        result = await pipeline.run(qid="ordered-citations", question="Count events?")

    assert result.retrieved_doc_ids == ["Q2", "Q1"]
    assert result.agentic_trace is not None
    assert result.agentic_trace["citations"] == ["Q2", "Q1"]


@pytest.mark.asyncio
async def test_agentic_does_not_replace_bounded_aggregate_with_sample_count() -> None:
    graph = create_mock_graph_client()
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
    )
    responses = [
        LLMCallResult(
            content='Thought: Count matching events.\nAction: gsql_aggregate\nAction Input: {"sport":"Biathlon","target_year":2018,"min_competitors":74,"max_competitors":0}',
            input_tokens=90,
            output_tokens=35,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content='Thought: Inspect the first event.\nAction: gsql_lookup\nAction Input: {"event_name_fragment":"Mixed relay","target_year":2018,"sport":"Biathlon","attribute":"competitor_count"}',
            input_tokens=100,
            output_tokens=40,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
    ]

    with (
        patch.object(
            graph,
            "run_aggregation",
            return_value={"count": 5, "events": ["Mixed relay"], "gold_doc_ids": ["Q1"]},
        ),
        patch.object(graph, "run_lookup") as lookup,
        patch.object(LockedLLMSession, "chat", side_effect=responses),
    ):
        result = await pipeline.run(
            qid="aggregate-count-not-sample-attribute",
            question="How many biathlon events in 2018 had at least 74 competitors?",
        )

    assert result.answer == "5"
    assert result.agentic_trace is not None
    assert "cannot replace it" in result.agentic_trace["stopping_reason"]
    lookup.assert_not_called()


@pytest.mark.asyncio
async def test_agentic_fallback_synthesis_uses_tool_observations_and_value_only_contract() -> None:
    graph = create_mock_graph_client()
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
        max_iterations=1,
    )
    responses = [
        LLMCallResult(
            content='Thought: Look up the named event.\nAction: gsql_lookup\nAction Input: {"event_name_fragment":"synthetic event","target_year":2020}',
            input_tokens=80,
            output_tokens=30,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content="Synthetic Champion",
            input_tokens=100,
            output_tokens=5,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
    ]
    lookup_result = {
        "events": ["Synthetic event"],
        "competitor_counts": [],
        "nation_counts": [],
        "gold_athletes": ["Synthetic Champion"],
        "venues": [],
        "gold_doc_ids": ["Q_SYNTHETIC"],
    }

    with (
        patch.object(graph, "run_lookup", return_value=lookup_result),
        patch.object(LockedLLMSession, "chat", side_effect=responses) as chat,
    ):
        result = await pipeline.run(
            qid="fallback-evidence-contract",
            question="Who won the synthetic event?",
        )

    assert result.answer == "Synthetic Champion"
    synthesis_messages = chat.await_args_list[-1].args[0]
    synthesis_contract = synthesis_messages[0]["content"].casefold()
    assert "only facts in the supplied tool observations" in synthesis_contract
    assert "return only the requested answer value" in synthesis_contract
    assert "Synthetic Champion" in synthesis_messages[1]["content"]


@pytest.mark.asyncio
async def test_agentic_fallback_rejects_thought_only_synthesis_output() -> None:
    graph = create_mock_graph_client()
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
        max_iterations=1,
    )
    responses = [
        LLMCallResult(
            content="Thought: Check the named event.\nAction: gsql_lookup\nAction Input: "
            '{"event_name_fragment":"synthetic event","target_year":2020}',
            input_tokens=80,
            output_tokens=30,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content="Thought: The evidence is ambiguous, so I should inspect another event.",
            input_tokens=90,
            output_tokens=25,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
    ]

    with (
        patch.object(graph, "run_lookup", return_value={}),
        patch.object(LockedLLMSession, "chat", side_effect=responses),
    ):
        result = await pipeline.run(qid="thought-only-synthesis", question="Who won?")

    assert result.answer == "Not found in corpus"
    assert result.agentic_trace is not None
    assert result.agentic_trace["invalid_response_count"] == 1
    assert "Rejected nonterminal output" in result.agentic_trace["stopping_reason"]


@pytest.mark.asyncio
async def test_agentic_rejects_unstructured_thought_instead_of_returning_it() -> None:
    graph = create_mock_graph_client()
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
        max_iterations=2,
    )
    responses = [
        LLMCallResult(
            content="Thought: I think the answer is an unsupported guess.",
            input_tokens=40,
            output_tokens=12,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content='Thought: Verify through the graph.\nAction: gsql_lookup\nAction Input: {"event_name_fragment":"winner","attribute":"gold_athlete"}',
            input_tokens=50,
            output_tokens=25,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content="Not found in corpus",
            input_tokens=60,
            output_tokens=5,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
    ]

    with (
        patch.object(graph, "run_lookup", return_value={}),
        patch.object(LockedLLMSession, "chat", side_effect=responses) as chat,
    ):
        result = await pipeline.run(qid="malformed-react-output", question="Who won?")

    assert result.answer == "Not found in corpus"
    assert result.agentic_trace is not None
    assert result.agentic_trace["invalid_response_count"] == 1
    assert chat.await_count == 3
    retry_messages = chat.await_args_list[1].args[0]
    assert any("protocol error" in message["content"].lower() for message in retry_messages)


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


@pytest.mark.asyncio
async def test_agentic_reuses_observation_for_repeated_identical_tool_action() -> None:
    graph = create_mock_graph_client()
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
    )
    first_action = LLMCallResult(
        content="Thought: Search structured event data.\nAction: gsql_lookup\nAction Input: "
        '{"event_name_fragment":"synthetic event","target_year":2020,"sport":"Example",'
        '"attribute":"nation_count"}',
        input_tokens=80,
        output_tokens=30,
        model_name="mock-model",
        provider="mock",
        latency_ms=5.0,
    )
    repeated_action = LLMCallResult(
        content="Thought: Repeat the same lookup.\nAction: gsql_lookup\nAction Input: "
        '{"attribute":"nation_count","sport":"Example","target_year":2020,'
        '"event_name_fragment":"synthetic event"}',
        input_tokens=80,
        output_tokens=30,
        model_name="mock-model",
        provider="mock",
        latency_ms=5.0,
    )
    finish = LLMCallResult(
        content="Thought: Finish after reviewing the result.\nAction: finish\nAction Input: "
        '{"answer":"Not found in corpus","confidence":0.0,"citations":[]}',
        input_tokens=70,
        output_tokens=20,
        model_name="mock-model",
        provider="mock",
        latency_ms=5.0,
    )
    empty_lookup: dict[str, list[str]] = {
        "events": [],
        "competitor_counts": [],
        "nation_counts": [],
        "gold_athletes": [],
        "venues": [],
        "gold_doc_ids": [],
    }

    with (
        patch.object(graph, "run_lookup", return_value=empty_lookup) as lookup,
        patch.object(LockedLLMSession, "chat", side_effect=[first_action, repeated_action, finish]),
    ):
        result = await pipeline.run("repeated-action", "Find an attribute for a synthetic event.")

    assert result.agentic_trace is not None
    calls = result.agentic_trace["tools_called"]
    lookup.assert_called_once()
    assert [call["tool_name"] for call in calls] == ["gsql_lookup", "gsql_lookup"]
    assert result.answer == "Not found in corpus"
    assert "reused its cached observation" in calls[1]["output_summary"]
    assert calls[1]["latency_ms"] == 0.0


@pytest.mark.asyncio
async def test_agentic_lookup_returns_requested_nation_count_from_unique_event() -> None:
    graph = create_mock_graph_client()
    lookup_result = {
        "events": ["Judo at the 2016 Summer Olympics – Women's 57 kg"],
        "competitor_counts": [36],
        "nation_counts": [23],
        "gold_athletes": ["Kayla Harrison"],
        "venues": ["Carioca Arena 2"],
        "gold_doc_ids": ["Q123"],
    }
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
    )
    response = LLMCallResult(
        content='Thought: Retrieve the requested nation count.\nAction: gsql_lookup\nAction Input: {"event_name_fragment":"57 kg","target_year":2016,"sport":"Judo","gender":"Women","attribute":"nation_count"}',
        input_tokens=90,
        output_tokens=35,
        model_name="mock-model",
        provider="mock",
        latency_ms=5.0,
    )

    with (
        patch.object(graph, "run_lookup", return_value=lookup_result) as lookup,
        patch.object(LockedLLMSession, "chat", return_value=response) as chat,
    ):
        result = await pipeline.run(
            qid="lookup-nations",
            question="How many nations competed in Women's 57 kg judo?",
        )

    assert result.answer == "23"
    assert result.agentic_trace is not None
    assert "requested nation_count" in result.agentic_trace["stopping_reason"]
    lookup.assert_called_once_with(event_fragment="57 kg", year=2016, sport="Judo", gender="Women")
    chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_agentic_corrects_superlative_arguments_sent_to_lookup_tool() -> None:
    graph = create_mock_graph_client()
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
    )
    response = LLMCallResult(
        content=(
            "Thought: Rank the shooting events by competitors.\n"
            "Action: gsql_lookup\n"
            'Action Input: {"sport":"shooting","target_year":2016,"season":"Summer",'
            '"order_by":"desc","result_limit":1}'
        ),
        input_tokens=90,
        output_tokens=35,
        model_name="mock-model",
        provider="mock",
        latency_ms=5.0,
    )
    expected_event = "Shooting at the 2016 Summer Olympics – Women's 10 metre air rifle"
    ranked_event = {
        "events": [expected_event],
        "competitor_counts": [51],
        "gold_doc_ids": ["Q25396733"],
    }

    with (
        patch.object(graph, "run_superlative", return_value=ranked_event) as superlative,
        patch.object(LockedLLMSession, "chat", return_value=response) as chat,
    ):
        result = await pipeline.run(
            qid="superlative-action-schema",
            question="Which shooting event had the highest number of competitors in 2016?",
        )

    superlative.assert_called_once_with(
        sport="shooting", year=2016, season="Summer", order="desc", limit=1
    )
    chat.assert_awaited_once()
    assert result.answer == expected_event
    assert result.agentic_trace is not None
    assert result.agentic_trace["tools_called"][0]["tool_name"] == "gsql_superlative"
    assert result.agentic_trace["action_corrections"] == [
        {
            "requested_tool": "gsql_lookup",
            "executed_tool": "gsql_superlative",
            "reason": "Arguments matched the superlative tool schema.",
        }
    ]


@pytest.mark.asyncio
async def test_agentic_retries_when_aggregate_action_contains_event_attribute_request() -> None:
    graph = create_mock_graph_client()
    lookup_result = {
        "events": ["Biathlon at the 2018 Winter Olympics – Women's pursuit"],
        "competitor_counts": [58],
        "nation_counts": [24],
        "gold_athletes": ["Laura Dahlmeier"],
        "venues": ["Alpensia"],
        "gold_doc_ids": ["Q47155365"],
    }
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
    )
    responses = [
        LLMCallResult(
            content='Thought: Get the nations for this event.\nAction: gsql_aggregate\nAction Input: {"sport":"Biathlon","target_year":2018,"event_name_fragment":"Women\'s pursuit","attribute":"nation_count"}',
            input_tokens=90,
            output_tokens=35,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content='Thought: Aggregation cannot return an event attribute; use lookup.\nAction: gsql_lookup\nAction Input: {"event_name_fragment":"Women\'s pursuit","target_year":2018,"sport":"Biathlon","gender":"Women","attribute":"nation_count"}',
            input_tokens=100,
            output_tokens=40,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
    ]

    with (
        patch.object(graph, "run_aggregation") as aggregate,
        patch.object(graph, "run_lookup", return_value=lookup_result) as lookup,
        patch.object(LockedLLMSession, "chat", side_effect=responses) as chat,
    ):
        result = await pipeline.run(
            qid="lookup-aggregate-misroute",
            question="How many nations competed in the Women's pursuit at the 2018 Winter Olympics?",
        )

    assert result.answer == "24"
    aggregate.assert_not_called()
    lookup.assert_called_once_with(
        event_fragment="Women's pursuit", year=2018, sport="Biathlon", gender="Women"
    )
    assert chat.await_count == 2


@pytest.mark.asyncio
async def test_agentic_does_not_return_aggregate_count_for_medalist_question() -> None:
    graph = create_mock_graph_client()
    lookup_result = {
        "events": ["Athletics at the 2008 Summer Olympics – Women's pole vault"],
        "competitor_counts": [38],
        "nation_counts": [24],
        "gold_athletes": ["Yelena Isinbayeva"],
        "venues": ["Beijing National Stadium"],
        "gold_doc_ids": ["Q123"],
    }
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
    )
    responses = [
        LLMCallResult(
            content='Thought: I will query the 2008 event records.\nAction: gsql_aggregate\nAction Input: {"target_year":2008,"sport":"","min_competitors":0,"max_competitors":0}',
            input_tokens=90,
            output_tokens=35,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content='Thought: The count does not answer who won. Query the medalist for the event.\nAction: gsql_lookup\nAction Input: {"event_name_fragment":"Women\'s pole vault","target_year":2008,"sport":"Athletics","gender":"Women","attribute":"gold_athlete"}',
            input_tokens=110,
            output_tokens=40,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
    ]

    with (
        patch.object(graph, "run_aggregation", return_value={"count": 209, "events": []}),
        patch.object(graph, "run_lookup", return_value=lookup_result) as lookup,
        patch.object(LockedLLMSession, "chat", side_effect=responses) as chat,
    ):
        result = await pipeline.run(
            qid="wrong-aggregate-action",
            question="Who won the gold medal at the Beijing National Stadium on 16 August 2008?",
        )

    assert result.answer == "Yelena Isinbayeva"
    assert result.answer != "209"
    lookup.assert_called_once()
    assert chat.await_count == 2


@pytest.mark.asyncio
async def test_agentic_temporal_does_not_select_first_of_ambiguous_events() -> None:
    graph = create_mock_graph_client()
    temporal_result = {
        "prev_events": ["Men's 200 metre freestyle", "Women's 200 metre freestyle"],
        "gold_athletes": ["Michael Phelps", "Allison Schmitt"],
        "gold_doc_ids": ["Q1", "Q2"],
    }
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
    )
    responses = [
        LLMCallResult(
            content='Thought: Search for the preceding edition.\nAction: gsql_temporal\nAction Input: {"sport":"Swimming","gender":"Women","event_name_fragment":"200 metre freestyle","current_year":2016}',
            input_tokens=90,
            output_tokens=35,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content="Thought: Refine the gender qualifier to match the question.\nFinal Answer: Allison Schmitt",
            input_tokens=110,
            output_tokens=20,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
    ]

    with (
        patch.object(graph, "run_temporal", return_value=temporal_result) as temporal,
        patch.object(LockedLLMSession, "chat", side_effect=responses) as chat,
    ):
        result = await pipeline.run(
            qid="temporal-ambiguous",
            question="Who won the women's 200 metre freestyle immediately before 2016?",
        )

    assert result.answer == "Allison Schmitt"
    temporal.assert_called_once_with(
        sport="Swimming",
        event_name_fragment="200 metre freestyle",
        current_year=2016,
        gender="Women",
    )
    assert chat.await_count == 2


@pytest.mark.asyncio
async def test_agentic_multihop_does_not_select_first_of_ambiguous_events() -> None:
    graph = create_mock_graph_client()
    multihop_result = {
        "events": [
            "Women's skeet",
            "Men's 10 metre running target",
            "Candidate 3",
            "Candidate 4",
            "Candidate 5",
            "Candidate 6",
        ],
        "gold_athletes": [
            "Zemfira Meftahatdinova",
            "Yang Ling",
            "Athlete 3",
            "Athlete 4",
            "Athlete 5",
            "Athlete 6",
        ],
        "gold_doc_ids": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
    }
    pipeline = AgenticPipeline(
        graph=graph,
        coprocessor=Coprocessor(),
        llm=LockedLLMSession(provider="cloudflare", model="mock-model"),
    )
    responses = [
        LLMCallResult(
            content='Thought: Search venue and event date.\nAction: gsql_multihop\nAction Input: {"venue_name_fragment":"Sydney International Shooting Centre","target_date_fragment":"21 September 2000","target_year":2000}',
            input_tokens=90,
            output_tokens=35,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
        LLMCallResult(
            content="Thought: Match the event to the exact date qualifier.\nFinal Answer: Yang Ling",
            input_tokens=110,
            output_tokens=20,
            model_name="mock-model",
            provider="mock",
            latency_ms=5.0,
        ),
    ]

    with (
        patch.object(graph, "run_multihop", return_value=multihop_result) as multihop,
        patch.object(LockedLLMSession, "chat", side_effect=responses) as chat,
    ):
        result = await pipeline.run(
            qid="multihop-ambiguous",
            question="Who won the event at Sydney International Shooting Centre on 21 September 2000?",
        )

    assert result.answer == "Yang Ling"
    multihop.assert_called_once_with(
        venue_fragment="Sydney International Shooting Centre",
        date_fragment="21 September 2000",
        year=2000,
    )
    assert chat.await_count == 2
    observation_messages = chat.await_args_list[1].args[0]
    assert "Candidate 6" in observation_messages[-1]["content"]
    assert "Athlete 6" in observation_messages[-1]["content"]


def test_parse_react_response_direct_json_schema() -> None:
    # Verifies production parser on direct JSON tool-call output
    raw_output = """{
        "thought": "Querying TigerGraph for preceding event winners",
        "action": "gsql_temporal",
        "action_input": {"sport": "Athletics", "current_year": 2016, "event_name_fragment": "20 kilometres walk"}
    }"""
    parsed = parse_react_response(raw_output)
    assert parsed.thought == "Querying TigerGraph for preceding event winners"
    assert parsed.action == "gsql_temporal"
    assert parsed.action_input["sport"] == "Athletics"
    assert parsed.action_input["current_year"] == 2016
    assert not parsed.is_terminal


def test_parse_react_response_markdown_fenced_and_nested() -> None:
    # Verifies production parser with markdown code fences and nested JSON objects
    raw_output = """Thought: Searching hybrid index with complex nested filter.
Action: hybrid_search
Action Input: ```json
{
    "query": "flag bearer 2006",
    "filter": {"season": "Winter", "year": 2006}
}
```"""
    parsed = parse_react_response(raw_output)
    assert parsed.thought == "Searching hybrid index with complex nested filter."
    assert parsed.action == "hybrid_search"
    assert parsed.action_input["filter"]["season"] == "Winter"


def test_parse_react_response_markdown_bold_headers() -> None:
    # Verifies production parser on markdown bold headers
    raw_output = """**Thought:** The evidence across graph edges confirms Chen Ding won the 2012 gold.
**Final Answer:** Chen Ding"""
    parsed = parse_react_response(raw_output)
    assert "Chen Ding" in parsed.thought
    assert parsed.final_answer == "Chen Ding"
    assert parsed.is_terminal
