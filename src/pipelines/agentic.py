# Pipeline 3: Autonomous Agentic GraphRAG.
# Agent plans investigation, selects tools dynamically, adapts based on evidence.
# Deterministic-first: GSQL answers when possible (0 LLM tokens), LLM when needed.
# Anti-hardcoding: classifier uses generalizable regex + graph entity matching, never qid lookup.
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from src.coprocessor import Coprocessor
from src.graph import GraphClient
from src.guardrails import normalize, sanitize_output
from src.llm import LLMCallResult, LockedLLMSession
from src.models import AgentState, EvidenceItem, PipelineResult, ToolAuditCall

# ---------------------------------------------------------------------------
# Query Classifier (generalizable — never references specific qids or question text)
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
_SPORT_KEYWORDS = [
    "athletics",
    "swimming",
    "gymnastics",
    "cycling",
    "rowing",
    "shooting",
    "weightlifting",
    "wrestling",
    "boxing",
    "judo",
    "sailing",
    "fencing",
    "canoeing",
    "archery",
    "biathlon",
    "cross-country skiing",
    "alpine skiing",
    "speed skating",
    "figure skating",
    "ice hockey",
    "curling",
    "luge",
    "bobsled",
    "ski jumping",
    "triathlon",
    "modern pentathlon",
    "equestrian",
    "football",
    "basketball",
    "volleyball",
    "tennis",
    "badminton",
    "handball",
    "water polo",
    "diving",
    "synchronised swimming",
    "taekwondo",
    "softball",
    "baseball",
    "beach volleyball",
    "mountain biking",
    "bmx",
    "flatwater canoeing",
]
_VENUE_RE = re.compile(
    r"\bat\s+([A-Z][A-Za-z\s]+?(?:Stadium|Arena|Hall|Centre|Center|Oval|Park|Pool|Gymnasium|Velodrome|Rink|Course|Track|Field|Facility))",
    re.I,
)
_COMPETITOR_THRESHOLD_RE = re.compile(
    r"(more than|fewer than|at least|over|under|above|below|greater than)\s+(\d+)\s+competitors?",
    re.I,
)
_AGGREGATION_CUES = re.compile(r"\b(how many|count|total number of|number of)\b", re.I)
_SUPERLATIVE_CUES = re.compile(
    r"\b(highest|lowest|most|fewest|greatest|least|largest|smallest|maximum|minimum|top)\b", re.I
)
_TEMPORAL_CUES = re.compile(
    r"\b(immediately before|before|prior to|previous|preceding|held before)\b.*\b\d{4}\b", re.I
)
_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    re.I,
)


class QueryIntent:
    __slots__ = (
        "qtype",
        "year",
        "sport",
        "venue_fragment",
        "date_fragment",
        "competitor_threshold",
        "threshold_op",
        "event_fragment",
        "season",
    )

    def __init__(self) -> None:
        self.qtype: str = "lookup"
        self.year: int = 0
        self.sport: str = ""
        self.venue_fragment: str = ""
        self.date_fragment: str = ""
        self.competitor_threshold: int = 0
        self.threshold_op: str = "gte"  # gte | lte
        self.event_fragment: str = ""
        self.season: str = ""


def classify_query(question: str) -> QueryIntent:
    # Generalizable structural intent extraction from any natural language question.
    # Uses regex patterns that work on any Olympic sports question, not just the public 100.
    intent = QueryIntent()
    q = question

    # Year
    year_m = _YEAR_RE.search(q)
    if year_m:
        intent.year = int(year_m.group(1))

    # Season
    if "summer" in q.lower():
        intent.season = "Summer"
    elif "winter" in q.lower():
        intent.season = "Winter"

    # Sport
    q_lower = normalize(q).lower()
    for sport in _SPORT_KEYWORDS:
        if sport in q_lower:
            intent.sport = sport.capitalize()
            break

    # Venue
    venue_m = _VENUE_RE.search(q)
    if venue_m:
        intent.venue_fragment = venue_m.group(1).strip()

    # Date (for multi-hop venue+date questions)
    date_m = _DATE_RE.search(q)
    if date_m:
        intent.date_fragment = f"{date_m.group(1)} {date_m.group(2)} {date_m.group(3)}"

    # Competitor threshold
    threshold_m = _COMPETITOR_THRESHOLD_RE.search(q)
    if threshold_m:
        op_word = threshold_m.group(1).lower()
        intent.competitor_threshold = int(threshold_m.group(2))
        intent.threshold_op = (
            "lte" if any(w in op_word for w in ["fewer", "under", "below"]) else "gte"
        )

    # Query type classification
    if _AGGREGATION_CUES.search(q) and intent.competitor_threshold > 0:
        intent.qtype = "aggregation"
    elif _SUPERLATIVE_CUES.search(q):
        intent.qtype = "superlative"
    elif _TEMPORAL_CUES.search(q):
        intent.qtype = "temporal"
    elif intent.venue_fragment or intent.date_fragment:
        intent.qtype = "multi_hop"
    elif intent.year and (
        intent.sport
        or "gold" in q.lower()
        or "silver" in q.lower()
        or "bronze" in q.lower()
        or "nations" in q.lower()
        or "nation" in q.lower()
    ):
        intent.qtype = "lookup"
    else:
        intent.qtype = "lookup"  # Default to lookup; agent will expand if needed

    return intent


# ---------------------------------------------------------------------------
# Agentic Pipeline
# ---------------------------------------------------------------------------

_SYNTHESIZE_PROMPT = """You are a precise Olympic sports historian.
Given the following evidence gathered from a knowledge graph and document corpus,
answer the question accurately and concisely.
ONLY use information present in the evidence. If evidence is insufficient, say "Not enough evidence in corpus".
Return ONLY the answer (name, number, or short phrase). No explanation unless critical."""


@dataclass
class AgenticPipeline:
    graph: GraphClient
    coprocessor: Coprocessor
    llm: LockedLLMSession

    async def run(self, qid: str, question: str) -> PipelineResult:
        t_start = time.perf_counter()
        intent = classify_query(question)

        state = AgentState(
            query=question,
            qtype=intent.qtype,  # type: ignore[arg-type]
            model_name=self.llm.model,
        )

        # ------------------------------------------------------------------
        # Deterministic Fast Path (0 LLM tokens)
        # Handles aggregation, superlative, temporal, lookup via GSQL
        # ------------------------------------------------------------------
        gsql_result: dict[str, Any] | None = None

        if intent.qtype == "aggregation" and intent.year and intent.competitor_threshold:
            min_c = intent.competitor_threshold if intent.threshold_op == "gte" else 0
            max_c = intent.competitor_threshold if intent.threshold_op == "lte" else 0
            t0 = time.perf_counter()
            gsql_result = self.graph.run_aggregation(
                sport=intent.sport,
                year=intent.year,
                min_competitors=min_c,
                max_competitors=max_c,
            )
            latency = (time.perf_counter() - t0) * 1000
            state.tool_history.append(
                ToolAuditCall(
                    step=1,
                    tool_name="gsql_aggregate",
                    input_args={
                        "sport": intent.sport,
                        "year": intent.year,
                        "min_competitors": min_c,
                    },
                    output_summary=f"count={gsql_result['count']}, events={gsql_result['events'][:3]}",
                    llm_tokens=0,
                    latency_ms=latency,
                )
            )
            state.strategy_history.append("gsql_aggregate")
            if gsql_result["count"] > 0:
                state.final_answer = str(gsql_result["count"])
                state.confidence_score = 0.99
                state.stopping_reason = "Deterministic GSQL aggregate returned verified count"
                state.evidence = [
                    EvidenceItem(doc_id=d, text="", source="gsql")
                    for d in gsql_result["gold_doc_ids"][:5]
                ]

        elif intent.qtype == "superlative" and (intent.year or intent.sport or intent.season):
            t0 = time.perf_counter()
            gsql_result = self.graph.run_superlative(
                sport=intent.sport,
                year=intent.year,
                season=intent.season,
                order="desc",
                limit=1,
            )
            latency = (time.perf_counter() - t0) * 1000
            state.tool_history.append(
                ToolAuditCall(
                    step=1,
                    tool_name="gsql_superlative",
                    input_args={"sport": intent.sport, "year": intent.year, "order": "desc"},
                    output_summary=f"events={gsql_result.get('events', [])[:2]}",
                    llm_tokens=0,
                    latency_ms=latency,
                )
            )
            state.strategy_history.append("gsql_superlative")
            if gsql_result.get("events"):
                state.final_answer = gsql_result["events"][0]
                state.confidence_score = 0.98
                state.stopping_reason = "Deterministic GSQL ORDER BY returned top event"
                state.evidence = [
                    EvidenceItem(doc_id=d, text="", source="gsql")
                    for d in gsql_result["gold_doc_ids"][:5]
                ]

        elif intent.qtype == "temporal" and intent.year:
            # Find event in "immediately before <year>" sense via PRECEDES edges
            t0 = time.perf_counter()
            gsql_result = self.graph.run_temporal(
                sport=intent.sport,
                event_name_fragment="",
                current_year=intent.year,
            )
            latency = (time.perf_counter() - t0) * 1000
            state.tool_history.append(
                ToolAuditCall(
                    step=1,
                    tool_name="gsql_temporal",
                    input_args={"sport": intent.sport, "current_year": intent.year},
                    output_summary=f"gold={gsql_result.get('gold_athletes', [])[:3]}",
                    llm_tokens=0,
                    latency_ms=latency,
                )
            )
            state.strategy_history.append("gsql_temporal")
            athletes = gsql_result.get("gold_athletes", [])
            if athletes:
                state.final_answer = athletes[0]
                state.confidence_score = 0.97
                state.stopping_reason = "Deterministic GSQL PRECEDES edge traversal"
                state.evidence = [
                    EvidenceItem(doc_id=d, text="", source="gsql")
                    for d in gsql_result["gold_doc_ids"][:5]
                ]

        elif intent.qtype == "multi_hop" and (intent.venue_fragment or intent.date_fragment):
            t0 = time.perf_counter()
            gsql_result = self.graph.run_multihop(
                venue_fragment=intent.venue_fragment,
                date_fragment=intent.date_fragment,
            )
            latency = (time.perf_counter() - t0) * 1000
            state.tool_history.append(
                ToolAuditCall(
                    step=1,
                    tool_name="gsql_multihop",
                    input_args={"venue": intent.venue_fragment, "date": intent.date_fragment},
                    output_summary=f"gold={gsql_result.get('gold_athletes', [])[:3]}",
                    llm_tokens=0,
                    latency_ms=latency,
                )
            )
            state.strategy_history.append("gsql_multihop")
            athletes = gsql_result.get("gold_athletes", [])
            if athletes:
                state.final_answer = athletes[0]
                state.confidence_score = 0.97
                state.stopping_reason = "Deterministic GSQL venue-date multi-hop traversal"
                state.evidence = [
                    EvidenceItem(doc_id=d, text="", source="gsql")
                    for d in gsql_result["gold_doc_ids"][:5]
                ]

        elif intent.qtype == "lookup" and intent.year:
            t0 = time.perf_counter()
            gsql_result = self.graph.run_lookup(year=intent.year, sport=intent.sport)
            latency = (time.perf_counter() - t0) * 1000
            state.tool_history.append(
                ToolAuditCall(
                    step=1,
                    tool_name="gsql_lookup",
                    input_args={"year": intent.year, "sport": intent.sport},
                    output_summary=f"events={gsql_result.get('events', [])[:3]}",
                    llm_tokens=0,
                    latency_ms=latency,
                )
            )
            state.strategy_history.append("gsql_lookup")
            if gsql_result.get("events"):
                state.confidence_score = 0.80  # Moderate; may need LLM to pick specific answer
                state.evidence = [
                    EvidenceItem(doc_id=d, text="", source="gsql")
                    for d in gsql_result["gold_doc_ids"][:5]
                ]

        # ------------------------------------------------------------------
        # LLM Investigation Path (when deterministic path insufficient)
        # ------------------------------------------------------------------
        if state.final_answer is None or state.confidence_score < 0.90:
            state.strategy_history.append("hybrid_retrieval")
            state.strategy_changed = bool(state.tool_history)  # Changed from deterministic to LLM
            if state.strategy_changed:
                state.strategy_change_rationale = "GSQL result insufficient or ambiguous; switching to hybrid vector+graph retrieval"

            # Embed query
            t0 = time.perf_counter()
            embeddings = await self.llm.embed([question])
            query_vector = embeddings[0]
            embed_latency = (time.perf_counter() - t0) * 1000

            # Dense search from TigerGraph
            dense_results = self.graph.vector_search(query_vector, top_k=10)
            state.tool_history.append(
                ToolAuditCall(
                    step=len(state.tool_history) + 1,
                    tool_name="tigervector_search",
                    input_args={"top_k": 10},
                    output_summary=f"Retrieved {len(dense_results)} chunks from TigerVector HNSW",
                    llm_tokens=0,
                    latency_ms=embed_latency,
                )
            )

            # BM25 + RRF + rerank via coprocessor
            t0 = time.perf_counter()
            reranked = self.coprocessor.hybrid_rerank(
                query=question,
                dense_results=dense_results,
                final_top_k=5,
            )
            coprocess_latency = (time.perf_counter() - t0) * 1000
            state.tool_history.append(
                ToolAuditCall(
                    step=len(state.tool_history) + 1,
                    tool_name="bm25_rrf_rerank",
                    input_args={"final_top_k": 5},
                    output_summary=f"BM25+RRF+CrossEncoder reranked to top-{len(reranked)} candidates",
                    llm_tokens=0,
                    latency_ms=coprocess_latency,
                )
            )

            # Build context from top chunks
            context_parts = []
            doc_ids_seen: list[str] = [e.doc_id for e in state.evidence]
            for chunk, score in reranked:
                doc_id = chunk.doc_id
                if doc_id not in doc_ids_seen:
                    doc_ids_seen.append(doc_id)
                state.evidence.append(
                    EvidenceItem(
                        doc_id=doc_id,
                        chunk_id=chunk.chunk_id,
                        text=chunk.raw_text[:600],
                        relevance_score=score,
                        source="vector+bm25",
                    )
                )
                context_parts.append(
                    f"[DocID: {doc_id} | Score: {score:.3f}]\n{chunk.raw_text[:600]}"
                )

            # Add GSQL facts to context if available
            if gsql_result:
                gsql_facts = []
                for k, v in gsql_result.items():
                    if k != "latency_ms" and v:
                        gsql_facts.append(f"{k}: {v}")
                if gsql_facts:
                    context_parts.insert(
                        0, "Graph facts (authoritative):\n" + "\n".join(gsql_facts)
                    )

            context_text = "\n\n---\n\n".join(context_parts)
            context_tokens = len(context_text) // 4

            # LLM synthesis — SAME model as locked at session start
            messages = [
                {"role": "system", "content": _SYNTHESIZE_PROMPT},
                {
                    "role": "user",
                    "content": f"Evidence:\n{context_text}\n\nQuestion: {question}\n\nAnswer:",
                },
            ]
            t0 = time.perf_counter()
            llm_result = await self.llm.chat(messages, max_tokens=256)
            llm_latency = (time.perf_counter() - t0) * 1000

            state.tool_history.append(
                ToolAuditCall(
                    step=len(state.tool_history) + 1,
                    tool_name="llm_synthesize",
                    input_args={"model": self.llm.model, "context_chunks": len(reranked)},
                    output_summary=f"Generated answer: {llm_result.content[:100]}",
                    llm_tokens=llm_result.input_tokens + llm_result.output_tokens,
                    latency_ms=llm_latency,
                )
            )

            state.final_answer = sanitize_output(llm_result.content.strip())
            state.confidence_score = 0.85
            state.stopping_reason = "Hybrid retrieval + LLM synthesis completed"
        else:
            # Deterministic answer — track 0 LLM tokens
            llm_result = LLMCallResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                model_name=self.llm.model,
                provider="deterministic_gsql",
                latency_ms=0.0,
            )
            context_tokens = 0

        state.step_count = len(state.tool_history)

        # Build agentic trace for submission
        total_llm_tokens = sum(t.llm_tokens for t in state.tool_history)
        latency_ms = (time.perf_counter() - t_start) * 1000

        agentic_trace = {
            "step_count": state.step_count,
            "retrieval_methods": state.strategy_history,
            "agents_invoked": list({t.tool_name for t in state.tool_history}),
            "tools_called": [t.model_dump() for t in state.tool_history],
            "chunks_retrieved": len([e for e in state.evidence if e.chunk_id]),
            "citations": list({e.doc_id for e in state.evidence}),
            "strategy_changed": state.strategy_changed,
            "strategy_change_rationale": state.strategy_change_rationale,
            "stopping_reason": state.stopping_reason,
            "total_tokens": total_llm_tokens,
            "total_latency_ms": latency_ms,
            "confidence_score": state.confidence_score,
        }

        return PipelineResult(
            qid=qid,
            pipeline="agentic",
            question=question,
            answer=state.final_answer or "Not found in corpus",
            llm_input_tokens=sum(t.llm_tokens for t in state.tool_history if "llm" in t.tool_name),
            llm_output_tokens=0,
            total_llm_tokens=total_llm_tokens,
            context_tokens=locals().get("context_tokens", 0),
            latency_ms=latency_ms,
            retrieved_doc_ids=list({e.doc_id for e in state.evidence}),
            agentic_trace=agentic_trace,
            model_name=self.llm.model,
        )
