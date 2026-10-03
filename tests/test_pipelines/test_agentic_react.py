# Agent harness: planning loop, tool dispatch, stopping rules, and the judge-facing trace.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.graph.mock import MockTigerGraphConnection
from src.llm import LLMCallResult, LockedLLMSession
from src.models import Chunk, PipelineResult
from src.pipelines.agentic import _REACT_SYSTEM_PROMPT, AgenticPipeline, parse_react_response
from tests.graph_fixtures import EVENTS, seeded_graph


def _trace(result: PipelineResult) -> dict[str, Any]:
    assert result.agentic_trace is not None
    return result.agentic_trace


def _reply(content: str, tokens: int = 50) -> LLMCallResult:
    return LLMCallResult(
        content=content,
        input_tokens=tokens,
        output_tokens=10,
        model_name="mock-model",
        provider="mock",
        latency_ms=1.0,
    )


def _action(tool: str, arguments: dict[str, Any]) -> LLMCallResult:
    return _reply(f"Thought: next step.\nAction: {tool}\nAction Input: {json.dumps(arguments)}")


def _pipeline(*replies: LLMCallResult) -> tuple[AgenticPipeline, AsyncMock]:
    graph, catalog, coprocessor = seeded_graph()
    session = LockedLLMSession(provider="cloudflare", model="mock-model")
    chat = AsyncMock(side_effect=list(replies))
    session.chat = chat  # type: ignore[method-assign]
    pipeline = AgenticPipeline(graph=graph, coprocessor=coprocessor, llm=session, catalog=catalog)
    return pipeline, chat


def test_react_prompt_does_not_contain_public_benchmark_questions() -> None:
    dataset = (
        Path(__file__).resolve().parents[2] / "hackathon-resources/questions/eval_public.jsonl"
    )
    prompt = _REACT_SYSTEM_PROMPT.casefold()
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert json.loads(line)["question"].casefold() not in prompt


@pytest.mark.asyncio
async def test_conclusive_graph_value_stops_after_one_planning_call() -> None:
    pipeline, chat = _pipeline(
        _action(
            "event_attribute", {"event": "women's single sculls", "sport": "Rowing", "year": 2012}
        )
    )

    result = await pipeline.run("q1", "Who won the 2012 women's single sculls?")

    trace = _trace(result)
    assert result.answer == "Eve Wren"
    assert chat.await_count == 1
    assert result.total_llm_tokens == 60
    assert trace["tools_called"][0]["llm_tokens"] == 0
    assert trace["agents_invoked"] == [
        "OrchestratorAgent",
        "EntityLinkingAgent",
        "GraphTraversalAgent",
        "EvidenceEvaluationAgent",
    ]
    assert result.retrieved_doc_ids == ["R12W"]
    assert "stated in the cited source article" in trace["stopping_reason"]
    assert trace["strategy_changed"] is False


@pytest.mark.asyncio
async def test_planner_refines_an_ambiguous_link_and_records_the_strategy_change() -> None:
    pipeline, chat = _pipeline(
        _action("event_attribute", {"event": "single sculls", "sport": "Rowing", "year": 2004}),
        _action(
            "event_attribute", {"event": "men's single sculls", "sport": "Rowing", "year": 2004}
        ),
    )

    result = await pipeline.run("q2", "Who won the 2004 men's single sculls?")

    assert result.answer == "Ada Stone"
    assert chat.await_count == 2
    second_prompt = chat.await_args_list[1].args[0]
    assert "candidates" in second_prompt[-1]["content"]


@pytest.mark.asyncio
async def test_unresolvable_candidates_are_reported_rather_than_guessed() -> None:
    venue_call = _action("event_at_venue_date", {"venue": "Lake A", "date": "15 to 21 August"})
    pipeline, chat = _pipeline(
        venue_call, _reply("Thought: Both fit.\nFinal Answer: Ada Stone; Bea Moss")
    )

    result = await pipeline.run("q3", "Who won the event held at Lake A on 15 to 21 August?")

    assert result.answer == "Ada Stone; Bea Moss"
    assert chat.await_count == 2


@pytest.mark.asyncio
async def test_iteration_limit_with_pending_ambiguity_reports_all_candidates() -> None:
    venue_call = _action("event_at_venue_date", {"venue": "Lake A", "date": "15 to 21 August"})
    pipeline, _ = _pipeline(*[venue_call] * 5)

    result = await pipeline.run("q4", "Who won the event held at Lake A on 15 to 21 August?")

    assert result.answer == "Ada Stone; Bea Moss"
    assert "reporting every candidate" in _trace(result)["stopping_reason"]
    calls = _trace(result)["tools_called"]
    assert "Identical call already made" in calls[1]["output_summary"]


@pytest.mark.asyncio
async def test_a_day_with_nothing_is_not_moved_to_another_year() -> None:
    moved = _action(
        "event_at_venue_date", {"venue": "Lake A", "date": "15 to 21 August 2004", "year": 2004}
    )
    pipeline, _ = _pipeline(
        moved, _reply("Thought: nothing on that day.\nFinal Answer: Not found in corpus")
    )

    result = await pipeline.run("q-day", "Who won the event held at Lake A on 15 August 2000?")

    assert result.answer == "Not found in corpus"
    assert "exactly as the question states" in _trace(result)["tools_called"][0]["output_summary"]


@pytest.mark.asyncio
async def test_a_list_of_the_wrong_kind_is_never_reported_as_a_tie() -> None:
    listing = _action("find_events", {"sport": "rowing"})
    pipeline, _ = _pipeline(
        listing,
        _action("find_events", {"sport": "rowing", "year": 2004}),
        _action("find_events", {"sport": "rowing", "season": "Summer"}),
        _action("find_events", {"sport": "rowing", "gender": "Men"}),
        _action("find_events", {"sport": "rowing", "year": 2008}),
        _reply("Not found in corpus"),
    )

    result = await pipeline.run("q-who", "Who won the gold medal in rowing on a day nobody raced?")

    assert result.answer == "Not found in corpus"
    assert "reporting every candidate" not in _trace(result)["stopping_reason"]


@pytest.mark.asyncio
async def test_a_listing_total_grounds_a_count_answer() -> None:
    pipeline, chat = _pipeline(
        _action("find_events", {"sport": "rowing", "year": 2004}),
        _reply('Thought: the listing has the total.\nAction: finish\nAction Input: {"answer": 2}'),
    )

    result = await pipeline.run("q-total", "How many rowing events were held at the 2004 Games?")

    assert result.answer == "2"
    assert chat.await_count == 2


@pytest.mark.asyncio
async def test_answer_from_memory_is_rejected_until_grounded() -> None:
    pipeline, chat = _pipeline(
        _reply("Thought: I recall it.\nFinal Answer: Zed Unknown"),
        _action("rank_events", {"sport": "rowing", "year": 2004}),
    )

    result = await pipeline.run(
        "q5", "Which rowing event at the 2004 Games had the most competitors?"
    )

    assert result.answer == EVENTS[0]["name"]
    assert _trace(result)["invalid_response_count"] == 1
    assert chat.await_count == 2


@pytest.mark.asyncio
async def test_small_talk_stops_after_two_tool_free_answers() -> None:
    pipeline, chat = _pipeline(
        _reply("Thought: a greeting.\nFinal Answer: Hello!"),
        _reply("Thought: still a greeting.\nFinal Answer: Hello!"),
    )

    result = await pipeline.run("q-hi", "hi")

    assert result.answer == "Not found in corpus"
    assert chat.await_count == 2
    assert _trace(result)["stopping_reason"] == "The question asks for nothing a tool can look up"


@pytest.mark.asyncio
async def test_failed_lookup_then_passage_search_records_strategy_change() -> None:
    pipeline, _ = _pipeline(
        _action("count_events", {"sport": "curling", "year": 2008, "threshold": 1}),
        _action("hybrid_search", {"query": "curling 2008"}),
        _reply("Thought: Nothing relevant.\nFinal Answer: Not found in corpus"),
    )
    hybrid = AsyncMock()
    with patch.object(pipeline.coprocessor, "hybrid_search") as search:
        search.return_value.chunks = []
        search.return_value.dense_candidate_count = 0
        search.return_value.sparse_candidate_count = 0
        search.return_value.fused_candidate_count = 0
        search.return_value.reranker_model = "mock"
        search.return_value.reranker_executed = False
        search.return_value.reranker_latency_ms = 0.0
        result = await pipeline.run("q6", "How many curling events in 2008?")

    trace = _trace(result)
    assert hybrid.await_count == 0
    assert result.answer == "Not found in corpus"
    assert trace["strategy_changed"] is True
    assert "count_events was error; switched to hybrid_search" in trace["strategy_change_rationale"]
    assert "DocumentRetrievalAgent" in trace["agents_invoked"]


@pytest.mark.asyncio
async def test_an_event_found_on_the_way_is_a_hop_when_the_question_asks_who() -> None:
    pipeline, chat = _pipeline(
        _action("rank_events", {"sport": "rowing", "year": 2004, "order": "desc"}),
        _action(
            "event_attribute",
            {
                "event": "Rowing at the 2004 Summer Olympics – Men's single sculls",
                "attribute": "gold_athlete",
            },
        ),
    )

    result = await pipeline.run(
        "q11", "Who won gold in the rowing event with the most competitors at the 2004 Games?"
    )

    trace = _trace(result)
    assert result.answer == "Ada Stone"
    assert chat.await_count == 2
    assert [call["tool_name"] for call in trace["tools_called"]] == [
        "rank_events",
        "event_attribute",
    ]
    hop = chat.await_args_list[1].args[0][-1]["content"]
    assert "intermediate result; the question asks for a person" in hop
    assert trace["stopping_reason"].startswith("event_attribute returned one verified graph value")


@pytest.mark.asyncio
async def test_a_tie_on_the_way_is_answered_for_every_tied_event() -> None:
    # 2008 men's and women's single sculls both had 33 competitors.
    pipeline, chat = _pipeline(
        _action("rank_events", {"sport": "rowing", "year": 2008, "order": "desc"}),
        _action(
            "event_attribute",
            {
                "event": "Rowing at the 2008 Summer Olympics – Men's single sculls",
                "attribute": "gold_athlete",
            },
        ),
        _action(
            "event_attribute",
            {
                "event": "Rowing at the 2008 Summer Olympics – Women's single sculls",
                "attribute": "gold_athlete",
            },
        ),
    )

    result = await pipeline.run(
        "q13", "Who won gold in the rowing event with the most competitors at the 2008 Games?"
    )

    trace = _trace(result)
    assert result.answer == "Cal Reed; Dee Lake"
    assert chat.await_count == 3
    assert "Women's single sculls" in chat.await_args_list[2].args[0][-1]["content"]
    assert trace["stopping_reason"] == "every tied event returned a verified graph value"


@pytest.mark.asyncio
async def test_ranking_question_still_stops_on_the_event() -> None:
    pipeline, chat = _pipeline(
        _action("rank_events", {"sport": "rowing", "year": 2004, "order": "desc"})
    )

    result = await pipeline.run(
        "q12", "Which rowing event at the 2004 Summer Olympics had the most competitors?"
    )

    assert result.answer == "Rowing at the 2004 Summer Olympics – Men's single sculls"
    assert chat.await_count == 1


@pytest.mark.asyncio
async def test_dense_only_search_is_not_a_separate_tool() -> None:
    # Hybrid search already fuses the TigerGraph vector hits, so a dense-only call is refused.
    assert "vector_search" not in _REACT_SYSTEM_PROMPT
    pipeline, chat = _pipeline(
        _action("vector_search", {"query": "rowing 2012"}),
        _action(
            "event_attribute", {"event": "women's single sculls", "sport": "Rowing", "year": 2012}
        ),
    )
    with patch.object(pipeline.graph, "vector_search") as dense:
        result = await pipeline.run("q10", "Who won the 2012 women's single sculls?")

    trace = _trace(result)
    assert result.answer == "Eve Wren"
    assert chat.await_count == 2
    assert dense.call_count == 0
    assert "unknown tool" in trace["tools_called"][0]["output_summary"]
    assert "SimilaritySearchAgent" not in trace["agents_invoked"]


@pytest.mark.asyncio
async def test_unavailable_graph_stops_instead_of_guessing() -> None:
    pipeline, chat = _pipeline(_action("count_events", {"sport": "rowing", "threshold": 1}))
    with patch.object(
        pipeline.graph, "run_aggregation", side_effect=RuntimeError("Starting workspace")
    ):
        result = await pipeline.run("q7", "How many rowing events?")

    assert result.answer == "Not found in corpus"
    assert "unavailable" in _trace(result)["stopping_reason"]
    assert chat.await_count == 1


@pytest.mark.asyncio
async def test_generated_gsql_runs_only_guarded_queries() -> None:
    safe_query = (
        "INTERPRET QUERY () FOR GRAPH OlympicsGraph { Events = {Event.*}; "
        "Matched = SELECT e FROM Events:e WHERE e.year == 2008 LIMIT 10; "
        "PRINT Matched[Matched.name]; }"
    )
    pipeline, _ = _pipeline(
        _action("gsql_query", {"query": safe_query.replace("Event.*", "ChatMessage.*")}),
        _reply("Thought: Rejected.\nFinal Answer: Not found in corpus"),
    )

    result = await pipeline.run("q8", "List 2008 events")

    assert "error" in _trace(result)["tools_called"][0]["output_summary"].lower()
    assert isinstance(pipeline.graph.conn, MockTigerGraphConnection)
    assert pipeline.graph.conn.gsql_history == []
    assert "QueryGenerationAgent" in _trace(result)["agents_invoked"]


@pytest.mark.asyncio
async def test_initial_passages_are_an_opt_in_ablation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STELLIUM_AGENT_INITIAL_HYBRID", "1")
    pipeline, chat = _pipeline(
        _action("count_events", {"sport": "rowing", "year": 2004, "threshold": 1})
    )
    chunk = Chunk(
        chunk_id="R04M#0", doc_id="R04M", chunk_index=0, section_title="t", text="x", raw_text="x"
    )
    with patch.object(pipeline.coprocessor, "hybrid_search") as search:
        search.return_value.chunks = [(chunk, 0.9)]
        search.return_value.dense_candidate_count = 1
        search.return_value.sparse_candidate_count = 1
        search.return_value.fused_candidate_count = 1
        search.return_value.reranker_model = "mock"
        search.return_value.reranker_executed = True
        search.return_value.reranker_latency_ms = 1.0
        result = await pipeline.run("q9", "How many rowing events in 2004?")

    assert result.answer == "2"
    assert _trace(result)["tools_called"][0]["tool_name"] == "hybrid_search"
    assert "Initial passages" in chat.await_args_list[0].args[0][-1]["content"]


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


@pytest.mark.asyncio
async def test_wrong_tool_arguments_trigger_a_replan_not_a_silent_answer() -> None:
    # "How many nations competed in <event>" is an attribute lookup; count_events must refuse it.
    pipeline, chat = _pipeline(
        _action(
            "count_events", {"sport": "Rowing", "year": 2012, "event": "women's single sculls"}
        ),
        _action(
            "event_attribute",
            {
                "event": "women's single sculls",
                "sport": "Rowing",
                "year": 2012,
                "attribute": "nation_count",
            },
        ),
    )

    result = await pipeline.run(
        "q10", "How many nations competed in the 2012 women's single sculls?"
    )

    trace = _trace(result)
    assert result.answer == "14"
    assert chat.await_count == 2
    assert "count_events does not take event" in trace["tools_called"][0]["output_summary"]
    assert trace["strategy_changed"] is True


def test_a_passage_may_ground_a_year_that_is_asked_for_but_never_a_count() -> None:
    from src.pipelines.agentic import _answer_parts_grounded, _asks_for_year

    passage = "Title: Hostel (2005 film)\n  released: 2005-09-17\n  runtime: 94 minutes"
    assert _asks_for_year("In which year was Hostel released?")
    assert _answer_parts_grounded(
        "2005", set(), [passage], _asks_for_year("In which year was Hostel released?")
    )
    # A count stays graph-only even when the number happens to appear in a passage.
    assert not _asks_for_year("How many events were held in the year 2008?")
    assert not _answer_parts_grounded(
        "94", set(), [passage], _asks_for_year("How many minutes is Hostel?")
    )
    # A year must appear whole in the passage, not inside a longer number.
    assert not _answer_parts_grounded("2005", set(), ["Catalogue 120051."], True)
