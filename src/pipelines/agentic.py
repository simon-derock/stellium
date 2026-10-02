# Pipeline 3: Agentic GraphRAG.
# An LLM orchestrator plans the investigation one step at a time: it picks a specialist tool,
# reads the observation, and decides what is still missing. Graph tools link entities and read
# exact values from TigerGraph at zero LLM cost; passage search and guarded generated GSQL cover
# what the typed tools cannot. The harness stops as soon as a tool returns a verified value.
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.coprocessor import Coprocessor
from src.graph import GraphClient
from src.graph.mock import MockTigerGraphConnection
from src.guardrails import normalize, sanitize_output
from src.linking import EventCatalog
from src.llm import LockedLLMSession
from src.models import AgentState, EvidenceItem, PipelineResult, ToolAuditCall
from src.pipelines.toolkit import (
    EVENT_ATTRIBUTES,
    TOOL_PARAMETERS,
    GraphToolkit,
    ToolOutcome,
    asked_kind,
    catalog_for,
    unsupported_arguments,
)

logger = logging.getLogger(__name__)

ORCHESTRATOR = "OrchestratorAgent"
_PASSAGE_CHARS = 700
_PASSAGES_SHOWN = 4
# A handful of same-kind candidates is a real tie worth reporting; more is an unfinished search.
_MAX_REPORTED_CANDIDATES = 6
_YEAR = re.compile(r"\b(?:18|19|20)\d\d\b")

# Which specialist owns each non-graph tool, for the trace.
_TEXT_TOOL_AGENTS = {
    "hybrid_search": "DocumentRetrievalAgent",
    "gsql_query": "QueryGenerationAgent",
}

# Prompt text only; it describes GSQL syntax to the model and never builds a query.
_REACT_SYSTEM_PROMPT = f"""You answer questions over a TigerGraph graph of Olympic events and the Wikipedia articles behind it. Each Event has a canonical title ("<Sport> at the <year> <Season> Olympics – <event>") and the attributes {", ".join(EVENT_ATTRIBUTES)}.
Work one step at a time from the question, the evidence so far, and what is still missing. Graph tools cost no LLM tokens and return source-checked values: prefer them, and search text only for facts outside those attributes.
Tools (Action Input is one JSON object; omit unknown fields):
- count_events {{sport, year, season, gender, comparison, threshold}}: how many events meet a competitor-count condition; comparison is more_than, at_least, fewer_than, at_most or exactly. Not for one event's own nation_count or competitor_count.
- rank_events {{sport, year, season, gender, order}}: the event with the most ("desc") or fewest ("asc") competitors; reports ties.
- event_attribute {{event, attribute, sport, year, season, gender}}: one attribute of one event.
- previous_edition {{event, year, attribute, sport, season, gender}}: the same event at the Games immediately before `year`. Pass the year the question names; never work out the earlier year yourself.
- event_at_venue_date {{venue, date, year, attribute}}: the event held at a venue on a date. Copy the venue and date exactly as the question writes them.
- find_events {{event, sport, year, season, gender}}: matching canonical events, to explore or disambiguate.
- hybrid_search {{query}}: reranked passages from TigerGraph vector search fused with BM25.
- gsql_query {{query}}: one read-only `INTERPRET QUERY () FOR GRAPH OlympicsGraph {{ ... }}` when no tool above fits: a single SELECT with LIMIT, then PRINT; double-quoted strings; lower(...) to compare strings.
- finish {{answer, citations}}: the final value, copied from an observation.
Rules:
- Use names, dates and numbers exactly as the question writes them. Never answer from memory.
- Finish with only what the question asks for: the name(s), a bare number, a venue, or the full canonical event title. An event found on the way is a step; look up the attribute the question wants.
- Several candidates: narrow them with a qualifier the question contains; if none separates them, finish with all of them joined by "; ".
- A tool errors or finds nothing: change the tool or its arguments; never repeat a call.
- The evidence never establishes it: finish with "Not found in corpus".
Reply with exactly:
Thought: <one short sentence>
Action: <tool name>
Action Input: <JSON object>"""  # nosec B608


# ---------------------------------------------------------------------------
# ReAct Step Parsing
# ---------------------------------------------------------------------------


@dataclass
class ReActParsedStep:
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    is_terminal: bool = False


def _reportable_tie(outcome: ToolOutcome, asked: str | None) -> bool:
    # Events for a "who" question, or two hundred titles, are a search still in progress, not a
    # tie; only a short list of the asked kind is worth reporting when the budget runs out.
    kind = outcome.answer_kind
    return (
        outcome.ambiguous
        and len(outcome.candidates) <= _MAX_REPORTED_CANDIDATES
        and (asked is None or kind is None or kind == asked)
    )


def _rewrites_the_day(question: str, arguments: dict[str, Any]) -> bool:
    # A venue-and-day lookup answers only for the day the question names. Moving it to another
    # year when the asked day held nothing turns "no such event" into someone else's medal.
    asked = set(_YEAR.findall(question))
    given = set(_YEAR.findall(f"{arguments.get('date', '')} {arguments.get('year', '')}"))
    return bool(asked) and bool(given - asked)


def _answer_is_grounded(answer: str, verified_values: set[str], passages: list[str]) -> bool:
    # Refuse a terminal answer unless it is an exact graph value or exact passage span.
    candidate = answer.strip()
    if normalize(candidate).casefold() == "not found in corpus":
        return True
    if candidate in verified_values:
        return True
    if candidate.isdecimal():
        return False
    normalized_answer = normalize(candidate).casefold()
    return bool(normalized_answer) and any(
        normalized_answer in normalize(passage).casefold() for passage in passages
    )


def _sanitize_trace_value(value: Any) -> Any:
    # Prevent secrets in prompts, observations, or provider errors from entering trace files.
    if isinstance(value, str):
        return sanitize_output(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_trace_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_trace_value(item) for item in value]
    return value


def _backend_unavailable(error: str) -> bool:
    # TigerGraph Savanna can return a startup HTML page while a workspace is resuming.
    message = error.casefold()
    return any(
        marker in message
        for marker in (
            "starting workspace",
            "connection timeout",
            "timed out",
            "http status 502",
            "http status 503",
            "http status 504",
            "bad gateway",
            "gateway timeout",
        )
    )


def _extract_balanced_json(text: str) -> dict[str, Any]:
    # Multi-pass balanced brace JSON extractor.
    # Handles markdown fences, nested objects, single-quote keys, and trailing commas.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start_idx = cleaned.find("{")
    if start_idx == -1:
        return {}

    depth = 0
    in_string = False
    escape = False
    end_idx = -1

    for i in range(start_idx, len(cleaned)):
        char = cleaned[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

    json_str = cleaned[start_idx:end_idx] if end_idx != -1 else cleaned[start_idx:]

    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    try:
        normalized = json_str.replace("'", '"')
        normalized = (
            normalized.replace("True", "true").replace("False", "false").replace("None", "null")
        )
        repaired_lines = []
        for line in normalized.splitlines():
            s = line.rstrip()
            if s.endswith(","):
                repaired_lines.append(s[:-1])
            else:
                repaired_lines.append(s)
        data = json.loads("\n".join(repaired_lines))
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


def parse_react_response(text: str) -> ReActParsedStep:
    # Production-grade multi-grammar parser for ReAct reasoning streams.
    # Supports native JSON tool-calling schema and structured text ReAct blocks.
    step = ReActParsedStep()
    raw = text.strip()
    if not raw:
        return step

    # Grammar 1: Direct JSON tool-call schema
    if raw.startswith("{") or raw.startswith("```json"):
        json_obj = _extract_balanced_json(raw)
        if json_obj:
            if "thought" in json_obj:
                step.thought = str(json_obj["thought"]).strip()
            if "final_answer" in json_obj:
                step.final_answer = sanitize_output(str(json_obj["final_answer"]).strip())
                step.is_terminal = True
                return step
            if "action" in json_obj:
                step.action = str(json_obj["action"]).strip().lower()
                step.action_input = json_obj.get("action_input") or {}
                if step.action == "finish":
                    step.is_terminal = True
                    step.final_answer = sanitize_output(
                        str(step.action_input.get("answer", "")).strip()
                    )
                return step

    # Grammar 2: Token-boundary stream lexer for text ReAct blocks
    lines = raw.splitlines()
    current_section: str | None = None
    thought_lines: list[str] = []
    final_answer_lines: list[str] = []
    action_input_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("final answer:") or lower.startswith("**final answer:**"):
            current_section = "final_answer"
            content = stripped.split(":", 1)[1].strip(" *")
            if content:
                final_answer_lines.append(content)
            continue
        elif lower.startswith("thought:") or lower.startswith("**thought:**"):
            current_section = "thought"
            content = stripped.split(":", 1)[1].strip(" *")
            if content:
                thought_lines.append(content)
            continue
        elif lower.startswith("action:") or lower.startswith("**action:**"):
            current_section = "action"
            act_name = stripped.split(":", 1)[1].strip(" *`'\"")
            step.action = act_name.lower()
            continue
        elif lower.startswith("action input:") or lower.startswith("**action input:**"):
            current_section = "action_input"
            content = stripped.split(":", 1)[1].strip()
            if content:
                action_input_lines.append(content)
            continue
        elif lower.startswith("observation:"):
            break

        if current_section == "thought":
            thought_lines.append(stripped)
        elif current_section == "final_answer":
            final_answer_lines.append(stripped)
        elif current_section == "action_input":
            action_input_lines.append(stripped)

    step.thought = " ".join(thought_lines).strip()

    if final_answer_lines:
        ans_text = " ".join(final_answer_lines).strip()
        step.final_answer = sanitize_output(ans_text)
        step.is_terminal = True
        return step

    if action_input_lines:
        input_raw = "\n".join(action_input_lines).strip()
        step.action_input = _extract_balanced_json(input_raw)

    if step.action == "finish":
        step.is_terminal = True
        ans = step.action_input.get("answer")
        if ans:
            step.final_answer = sanitize_output(str(ans).strip())

    return step


# ---------------------------------------------------------------------------
# Agent Pipeline
# ---------------------------------------------------------------------------


def _answer_parts_grounded(answer: str, verified: set[str], passages: list[str]) -> bool:
    # A multi-candidate answer ("A; B") is grounded only if every part is.
    parts = [part.strip() for part in answer.split(";") if part.strip()]
    return bool(parts) and all(_answer_is_grounded(part, verified, passages) for part in parts)


@dataclass
class _RunLog:
    # Everything the trace needs, collected while the loop runs.
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    conversation: list[dict[str, Any]] = field(default_factory=list)
    retrieval_stages: list[dict[str, Any]] = field(default_factory=list)
    agents: list[str] = field(default_factory=lambda: [ORCHESTRATOR])
    citations: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    invalid_responses: int = 0

    def agent(self, *names: str) -> None:
        for name in names:
            if name not in self.agents:
                self.agents.append(name)

    def cite(self, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            if doc_id and doc_id not in self.citations:
                self.citations.append(doc_id)


@dataclass
class AgenticPipeline:
    graph: GraphClient
    coprocessor: Coprocessor
    llm: LockedLLMSession
    max_iterations: int = 5
    catalog: EventCatalog | None = None

    async def _embed_query(self, query: str) -> list[float]:
        # Offline graph mocks use a stable vector and never call external embedding providers.
        if isinstance(self.graph.conn, MockTigerGraphConnection):
            return [0.0] * 1024
        return (await self.llm.embed([query]))[0]

    async def _passages(
        self, query: str, log: _RunLog, step: int
    ) -> tuple[dict[str, Any], list[EvidenceItem]]:
        # TigerGraph vector hits are one input to the fusion, so a dense-only tool would be a subset.
        query_vector = await self._embed_query(query)
        dense = self.graph.vector_search(query_vector, top_k=30)
        hybrid = self.coprocessor.hybrid_search(
            query=query, dense_results=dense, candidate_k=30, final_top_k=_PASSAGES_SHOWN
        )
        log.retrieval_stages.append(
            {
                "step": step,
                "dense_candidates": hybrid.dense_candidate_count,
                "bm25_candidates": hybrid.sparse_candidate_count,
                "rrf_candidates": hybrid.fused_candidate_count,
                "reranker": hybrid.reranker_model,
                "reranker_executed": hybrid.reranker_executed,
                "reranker_latency_ms": hybrid.reranker_latency_ms,
            }
        )
        evidence = [
            EvidenceItem(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text[:_PASSAGE_CHARS],
                relevance_score=score,
                source="hybrid_search",
            )
            for chunk, score in hybrid.chunks
        ]
        observation = {
            "passages": [
                {"doc_id": item.doc_id, "chunk_id": item.chunk_id, "text": item.text}
                for item in evidence
            ]
        }
        return observation, evidence

    def _generated_gsql(self, query: str) -> tuple[dict[str, Any], list[str], list[str]]:
        # Returns the observation, cited documents, and canonical values the query surfaced.
        if not query:
            return {"error": "gsql_query requires a complete query"}, [], []
        result = self.graph.run_generated_gsql(query)
        rows = result.get("rows", [])
        citations: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.casefold() in {"wikidata_qid", "doc_id", "v_id"} and isinstance(
                        item, str
                    ):
                        citations.append(item)
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(rows)
        values = [
            str(value)
            for candidate in result.get("event_candidates", [])
            if isinstance(candidate, dict)
            for value in candidate.values()
            if value is not None
        ]
        observation = {
            "rows": json.dumps(rows, ensure_ascii=False, default=str)[:4000],
            "event_candidates": result.get("event_candidates", []),
        }
        return observation, list(dict.fromkeys(citations)), values

    async def _execute(
        self,
        toolkit: GraphToolkit,
        tool: str,
        arguments: dict[str, Any],
        log: _RunLog,
        step: int,
        evidence: list[EvidenceItem],
        question: str = "",
    ) -> tuple[dict[str, Any], ToolOutcome | None, list[str]]:
        # Runs one tool. Returns its observation, the typed outcome for graph tools, and any
        # values it verified (used to ground a later finish).
        if tool == "event_at_venue_date" and _rewrites_the_day(question, arguments):
            refusal = "Use the venue and date exactly as the question states them; another year is a different question."
            return {"error": refusal}, None, []
        if tool in TOOL_PARAMETERS:
            rejected = unsupported_arguments(tool, arguments)
            if rejected:
                accepted = ", ".join(TOOL_PARAMETERS[tool])
                return (
                    {
                        "error": (
                            f"{tool} does not take {', '.join(rejected)}; it accepts {accepted}. "
                            "Choose the tool whose inputs match the question."
                        )
                    },
                    None,
                    [],
                )
            try:
                outcome = toolkit.invoke(tool, arguments)
            except TypeError as exc:
                return {"error": f"invalid arguments for {tool}: {exc}"}, None, []
            log.agent(*outcome.agents)
            log.cite(outcome.citations)
            evidence.extend(outcome.evidence)
            observation = dict(outcome.observation)
            if outcome.candidates and outcome.answer is None:
                observation["candidates"] = outcome.candidates
            values = [outcome.answer] if outcome.answer else list(outcome.candidates)
            # A listing's size is counted by the graph, as verified as count_events' own answer.
            if isinstance(outcome.observation.get("total"), int):
                values.append(str(outcome.observation["total"]))
            return observation, outcome, values
        if tool == "hybrid_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return {"error": f"{tool} requires a query"}, None, []
            observation, found = await self._passages(query, log, step)
            log.agent(_TEXT_TOOL_AGENTS[tool])
            evidence.extend(found)
            return observation, None, []
        if tool == "gsql_query":
            log.agent(_TEXT_TOOL_AGENTS[tool])
            observation, citations, values = self._generated_gsql(str(arguments.get("query", "")))
            log.cite(citations)
            evidence.extend(
                EvidenceItem(doc_id=doc_id, text="", source="gsql") for doc_id in citations
            )
            return observation, None, values
        return {"error": f"unknown tool {tool!r}"}, None, []

    async def _chat(self, messages: list[dict[str, str]], log: _RunLog, operation: str) -> str:
        started = time.perf_counter()
        # Send a snapshot: the loop keeps appending to its own transcript after the call.
        result = await self.llm.chat(list(messages), max_tokens=256, temperature=0.0)
        log.input_tokens += result.input_tokens
        log.output_tokens += result.output_tokens
        log.llm_calls.append(
            {
                "step": len(log.llm_calls) + 1,
                "operation": operation,
                "model_name": result.model_name,
                "provider": result.provider,
                "token_source": result.token_source,
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                # Wall time includes any client-side rate-limit pacing; provider time does not.
                "latency_ms": (time.perf_counter() - started) * 1000,
                "provider_latency_ms": result.latency_ms,
            }
        )
        return result.content

    async def run(self, qid: str, question: str) -> PipelineResult:
        started = time.perf_counter()
        state = AgentState(query=question, qtype="agentic_react", model_name=self.llm.model)
        log = _RunLog()
        toolkit = GraphToolkit(
            self.graph, self.catalog or catalog_for(self.graph), self.coprocessor
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        verified: set[str] = set()
        asked = asked_kind(question)
        # Events tied in a ranking the question passes through, each with its value once read.
        tied: dict[str, str | None] = {}
        pending_candidates: list[str] = []
        observations: list[str] = []
        cache: dict[str, dict[str, Any]] = {}
        previous: tuple[str, str] | None = None
        step = 0
        refused_finishes = 0

        if os.environ.get("STELLIUM_AGENT_INITIAL_HYBRID", "").casefold() in {"1", "true", "yes"}:
            # Optional ablation: hand the planner passages before its first decision.
            observation, found = await self._passages(question, log, 0)
            state.evidence.extend(found)
            log.agent(_TEXT_TOOL_AGENTS["hybrid_search"])
            state.tool_history.append(
                ToolAuditCall(
                    step=0,
                    tool_name="hybrid_search",
                    input_args={"query": question, "initial": True},
                    output_summary=json.dumps(observation, ensure_ascii=False)[:1500],
                )
            )
            messages.append(
                {"role": "user", "content": f"Initial passages: {json.dumps(observation)}"}
            )

        for _ in range(self.max_iterations):
            content = await self._chat(messages, log, "orchestrator")
            parsed = parse_react_response(content)
            messages.append({"role": "assistant", "content": content.strip()})
            log.conversation.append(
                {
                    "event": "plan",
                    "thought": parsed.thought,
                    "action": parsed.action,
                    "action_input": _sanitize_trace_value(parsed.action_input),
                    "final_answer": parsed.final_answer,
                }
            )
            passages = [item.text for item in state.evidence if item.text]

            if parsed.is_terminal and parsed.final_answer:
                if _answer_parts_grounded(parsed.final_answer, verified, passages):
                    state.final_answer = parsed.final_answer
                    state.confidence_score = 0.9
                    state.stopping_reason = (
                        "Orchestrator finished with an answer grounded in tool observations"
                    )
                    if parsed.action == "finish":
                        requested = parsed.action_input.get("citations", [])
                        if isinstance(requested, list):
                            log.cite([str(doc) for doc in requested if str(doc) in log.citations])
                    break
                log.invalid_responses += 1
                refused_finishes += 1
                if refused_finishes >= 2 and not state.tool_history:
                    # Twice it answered without consulting anything (a greeting, chit-chat): no
                    # tool applies, so there is nothing to ground and nothing more to spend.
                    state.final_answer = "Not found in corpus"
                    state.stopping_reason = "The question asks for nothing a tool can look up"
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That answer is not a value from any observation. Use a tool to "
                            "gather support, or answer with a value copied from an observation."
                        ),
                    }
                )
                continue

            if not parsed.action:
                log.invalid_responses += 1
                messages.append(
                    {
                        "role": "user",
                        "content": "Reply with Thought, Action, and Action Input (or finish).",
                    }
                )
                continue

            tool, arguments = parsed.action, parsed.action_input
            step += 1
            cache_key = f"{tool}:{json.dumps(arguments, sort_keys=True)}"
            tool_started = time.perf_counter()
            outcome: ToolOutcome | None = None
            if cache_key in cache:
                observation = {
                    "notice": "Identical call already made; its observation is unchanged.",
                    "previous_observation": cache[cache_key],
                }
                values: list[str] = []
            else:
                try:
                    observation, outcome, values = await self._execute(
                        toolkit, tool, arguments, log, step, state.evidence, question
                    )
                except Exception as exc:
                    observation, values = {"error": f"{tool} failed: {exc}"}, []
                cache[cache_key] = observation
            tool_latency = (time.perf_counter() - tool_started) * 1000
            verified.update(values)
            # A verified value of the wrong kind is a hop, not the answer: the event with the most
            # competitors when the question asks who won it.
            value_kind = outcome.answer_kind if outcome is not None and outcome.conclusive else None
            intermediate = asked is not None and value_kind is not None and value_kind != asked
            if intermediate:
                observation = {
                    **observation,
                    "note": f"This {value_kind} is an intermediate result; the question asks for a {asked}.",
                }
            if outcome is not None and outcome.ambiguous and "tie" in outcome.observation:
                tied = dict.fromkeys(outcome.candidates)
            # A value for one tied event is part of the answer; the rest still need theirs.
            tie_open = False
            if tied and outcome is not None and outcome.answer is not None and not intermediate:
                event_name = str(outcome.observation.get("event", ""))
                if event_name in tied:
                    tied[event_name] = outcome.answer
                    missing = [name for name, value in tied.items() if value is None]
                    tie_open = bool(missing)
                    if tie_open:
                        observation = {
                            **observation,
                            "note": "Tied events still without this value: " + "; ".join(missing),
                        }
            state.strategy_history.append(tool)
            state.tool_history.append(
                ToolAuditCall(
                    step=step,
                    tool_name=tool,
                    input_args=arguments,
                    output_summary=sanitize_output(
                        json.dumps(observation, ensure_ascii=False, default=str)[:1500]
                    ),
                    latency_ms=tool_latency,
                )
            )
            log.conversation.append(
                {
                    "event": "observation",
                    "step": step,
                    "tool": tool,
                    "latency_ms": tool_latency,
                    "observation": _sanitize_trace_value(observation),
                }
            )

            error = str(observation.get("error", ""))
            if _backend_unavailable(error):
                state.final_answer = "Not found in corpus"
                state.stopping_reason = "TigerGraph was unavailable; stopped instead of guessing"
                break

            status = (
                "conclusive"
                if outcome is not None and outcome.conclusive
                else "ambiguous"
                if outcome is not None and outcome.ambiguous
                else "error"
                if error
                else observation.get("status", "observed")
            )
            if (
                previous
                and previous[1] in {"error", "no_match", "ambiguous"}
                and tool != previous[0]
            ):
                state.strategy_changed = True
                state.strategy_change_rationale = (
                    f"{previous[0]} was {previous[1]}; switched to {tool} for the missing evidence"
                )
            previous = (tool, str(status))

            if (
                outcome is not None
                and outcome.answer is not None
                and not intermediate
                and not tie_open
            ):
                # Evidence-sufficiency stop: a verified value of the asked kind, or one per tied event.
                checked = any(item.text for item in outcome.evidence)
                state.confidence_score = 0.98 if checked else 0.9
                if tied and all(tied.values()):
                    state.final_answer = "; ".join(dict.fromkeys(str(v) for v in tied.values()))
                    state.stopping_reason = "every tied event returned a verified graph value"
                else:
                    state.final_answer = outcome.answer
                    state.stopping_reason = f"{tool} returned one verified graph value" + (
                        " stated in the cited source article" if checked else ""
                    )
                break
            if outcome is not None and _reportable_tie(outcome, asked):
                pending_candidates = list(outcome.candidates)

            observation_text = json.dumps(observation, ensure_ascii=False, default=str)
            observations.append(observation_text)
            messages.append({"role": "user", "content": f"Observation: {observation_text}"})

        answered_ties = [str(value) for value in tied.values() if value]
        if not state.final_answer and answered_ties:
            state.final_answer = "; ".join(dict.fromkeys(answered_ties))
            state.confidence_score = 0.5
            state.stopping_reason = "Iteration budget used before every tied event was answered"
        if not state.final_answer and pending_candidates:
            state.final_answer = "; ".join(pending_candidates)
            state.confidence_score = 0.3
            state.stopping_reason = (
                "Evidence could not separate several matching events; reporting every candidate"
            )
        if not state.final_answer:
            await self._synthesize(question, observations, state, verified, log)

        return self._result(qid, question, state, log, started)

    async def _synthesize(
        self,
        question: str,
        observations: list[str],
        state: AgentState,
        verified: set[str],
        log: _RunLog,
    ) -> None:
        # Iteration budget exhausted without a finish: one last value-only answer from evidence.
        passages = [item.text for item in state.evidence if item.text]
        context = "\n---\n".join(observations[-4:] + passages[:_PASSAGES_SHOWN])
        content = await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied observations and passages. Return only "
                        "the value: a full canonical event title, a number, or a name. If they do "
                        'not establish it, return "Not found in corpus".'
                    ),
                },
                {"role": "user", "content": f"{context}\n\nQuestion: {question}\nAnswer:"},
            ],
            log,
            "answer_synthesis",
        )
        parsed = parse_react_response(content)
        answer = (parsed.final_answer or content).strip()
        if not parsed.action and _answer_parts_grounded(answer, verified, passages):
            state.final_answer = answer
            state.confidence_score = 0.7
            state.stopping_reason = "Iteration budget used; synthesized a grounded answer"
        else:
            log.invalid_responses += 1
            state.final_answer = "Not found in corpus"
            state.stopping_reason = "Iteration budget used; no grounded answer was available"

    def _result(
        self, qid: str, question: str, state: AgentState, log: _RunLog, started: float
    ) -> PipelineResult:
        state.step_count = len(log.llm_calls) + len(state.tool_history)
        elapsed_ms = (time.perf_counter() - started) * 1000
        log.cite([item.doc_id for item in state.evidence if item.doc_id])
        total_tokens = log.input_tokens + log.output_tokens
        trace: dict[str, Any] = {
            "step_count": state.step_count,
            "retrieval_methods": state.strategy_history,
            "agents_invoked": log.agents,
            "specialized_tools_used": list(dict.fromkeys(state.strategy_history)),
            "llm_calls": log.llm_calls,
            "tools_called": [call.model_dump() for call in state.tool_history],
            "chunks_retrieved": len({item.chunk_id for item in state.evidence if item.chunk_id}),
            "citations": log.citations,
            "strategy_changed": state.strategy_changed,
            "strategy_change_rationale": state.strategy_change_rationale,
            "stopping_reason": state.stopping_reason,
            "final_answer": state.final_answer,
            "confidence_score": state.confidence_score,
            "total_tokens": total_tokens,
            "total_latency_ms": elapsed_ms,
            "invalid_response_count": log.invalid_responses,
            "retrieval_stages": log.retrieval_stages,
            "conversation_trace": log.conversation,
        }
        trace = _sanitize_trace_value(trace)
        trace_path = os.environ.get("STELLIUM_AGENT_TRACE_PATH", "").strip()
        if trace_path:
            from pathlib import Path

            trace_file = Path(trace_path)
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            with trace_file.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"qid": qid, "question": question, **trace}) + "\n")
        return PipelineResult(
            qid=qid,
            pipeline="agentic",
            question=question,
            answer=sanitize_output(state.final_answer or "Not found in corpus"),
            llm_input_tokens=log.input_tokens,
            llm_output_tokens=log.output_tokens,
            total_llm_tokens=total_tokens,
            context_tokens=sum(len(item.text) // 4 for item in state.evidence if item.text),
            latency_ms=elapsed_ms,
            retrieved_doc_ids=log.citations,
            retrieval_metadata={
                "retrieval_stages": log.retrieval_stages,
                "reranker_executions": sum(
                    bool(stage.get("reranker_executed")) for stage in log.retrieval_stages
                ),
            },
            agentic_trace=trace,
            model_name=self.llm.model,
            provider=self.llm.provider,
        )
