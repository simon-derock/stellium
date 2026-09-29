# Pipeline 2: Hybrid GraphRAG (fixed pipeline, non-agentic baseline).
# Fixed sequence: entity extraction -> graph expansion -> dense/sparse fusion -> reranking -> synthesis.
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from src.coprocessor import Coprocessor
from src.graph import GraphClient
from src.guardrails import sanitize_output
from src.llm import LockedLLMSession
from src.models import PipelineResult

logger = logging.getLogger(__name__)

_ENTITY_LINKING_SYSTEM_PROMPT = """You are an entity extraction engine for Olympic sports questions.
Translate the question into exactly one fixed graph operation and copy its constraints.
Return ONLY a valid JSON object matching this schema:
{"operation":"aggregation|superlative|temporal|multi_hop|lookup|none","year":2012,"sport":"Athletics","season":"Summer","gender":"Women","event_fragment":"100 metres","venue":"National Stadium","date":"16 August","attribute":"gold_athlete|nation_count|competitor_count|venue","min_competitors":0,"max_competitors":0,"order":"desc","limit":1}
Use "aggregation" for counts of matching events, "lookup" for an attribute of one named event, "superlative" for events with the most/fewest competitors, "temporal" for a preceding edition, and "multi_hop" for an event identified by venue/date. For strict comparisons, convert "more than N" to min_competitors N+1 and "fewer than N" or "less than N" to max_competitors N-1. Copy entities and date wording from the question; do not invent values or use a fixed sport/entity list. Use empty strings and zero for unknown or unconstrained fields. Do not include markdown or commentary."""

_GRAPHRAG_SYSTEM_PROMPT = """You are a precise sports historian with access to Olympic event data.
Answer the question using ONLY the provided graph context (entities, relationships, and passages).
If the answer is not in the context, respond with "Not found in corpus".
Be concise: give the direct answer, not a full sentence when a name or number suffices."""

_GRAPHRAG_USER_TEMPLATE = """Graph context:
{graph_context}

Question: {question}

Answer:"""


@dataclass
class ExtractedEntities:
    operation: str = "none"
    year: int = 0
    sport: str = ""
    season: str = ""
    gender: str = ""
    event_fragment: str = ""
    venue: str = ""
    date: str = ""
    attribute: str = "gold_athlete"
    min_competitors: int = 0
    max_competitors: int = 0
    order: str = "desc"
    limit: int = 1
    input_tokens: int = 0
    output_tokens: int = 0


def _as_int(value: Any, default: int = 0) -> int:
    # Reject booleans and values that cannot be converted to an integer.
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


async def _extract_entities_via_llm(llm: LockedLLMSession, question: str) -> ExtractedEntities:
    # Uses one bounded LLM call to produce a typed plan for the fixed GraphRAG sequence.
    messages = [
        {"role": "system", "content": _ENTITY_LINKING_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    res = None
    try:
        res = await llm.chat(messages, max_tokens=128, temperature=0.0)
        cleaned = res.content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data: dict[str, Any] = json.loads(cleaned)

        operations = {"aggregation", "superlative", "temporal", "multi_hop", "lookup", "none"}
        operation = str(data.get("operation", "none")).strip().lower()
        order = str(data.get("order", "desc")).strip().lower()
        attribute = str(data.get("attribute", "gold_athlete")).strip().lower()
        if attribute not in {"gold_athlete", "nation_count", "competitor_count", "venue"}:
            attribute = "gold_athlete"
        return ExtractedEntities(
            operation=operation if operation in operations else "none",
            year=_as_int(data.get("year")),
            sport=str(data.get("sport") or "").strip(),
            season=str(data.get("season") or "").strip(),
            gender=str(data.get("gender") or "").strip(),
            event_fragment=str(data.get("event_fragment") or "").strip(),
            venue=str(data.get("venue") or "").strip(),
            date=str(data.get("date") or "").strip(),
            attribute=attribute,
            min_competitors=max(0, _as_int(data.get("min_competitors"))),
            max_competitors=max(0, _as_int(data.get("max_competitors"))),
            order=order if order in {"asc", "desc"} else "desc",
            limit=min(10, max(1, _as_int(data.get("limit"), 1))),
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
        )
    except Exception:
        # Graceful fallback when running in offline mode or on unparseable JSON
        return ExtractedEntities(
            input_tokens=res.input_tokens if res else 0,
            output_tokens=res.output_tokens if res else 0,
        )


@dataclass
class GraphRAGPipeline:
    graph: GraphClient
    llm: LockedLLMSession
    coprocessor: Coprocessor | None = None

    async def run(self, qid: str, question: str) -> PipelineResult:
        t_start = time.perf_counter()

        # Step 1: Entity linking — dynamic LLM entity extraction
        entities = await _extract_entities_via_llm(self.llm, question)

        # Step 2: Execute one graph operation from the bounded extraction plan.
        graph_facts: list[str] = []
        doc_ids: list[str] = []

        if entities.operation == "aggregation":
            result = self.graph.run_aggregation(
                sport=entities.sport,
                year=entities.year,
                min_competitors=entities.min_competitors,
                max_competitors=entities.max_competitors,
            )
            graph_facts.append(
                f"Exact matching event count: {result.get('count', 0)}. "
                f"Matching events: {', '.join(result.get('events', []))}"
            )
            doc_ids.extend(result.get("gold_doc_ids", []))
        elif entities.operation == "superlative":
            result = self.graph.run_superlative(
                sport=entities.sport,
                year=entities.year,
                season=entities.season,
                order=entities.order,
                limit=entities.limit,
            )
            for event, count in zip(result.get("events", []), result.get("competitor_counts", [])):
                graph_facts.append(f"Ranked event: {event} | Competitors: {count}")
            doc_ids.extend(result.get("gold_doc_ids", []))
        elif entities.operation == "temporal":
            result = self.graph.run_temporal(
                sport=entities.sport,
                event_name_fragment=entities.event_fragment,
                current_year=entities.year,
                gender=entities.gender,
            )
            for event, athlete in zip(
                result.get("prev_events", []), result.get("gold_athletes", [])
            ):
                graph_facts.append(f"Preceding-edition event: {event} | Gold: {athlete}")
            doc_ids.extend(result.get("gold_doc_ids", []))
        elif entities.operation == "multi_hop":
            result = self.graph.run_multihop(
                venue_fragment=entities.venue,
                date_fragment=entities.date,
                year=entities.year,
            )
            for event, athlete in zip(result.get("events", []), result.get("gold_athletes", [])):
                graph_facts.append(f"Event: {event} | Gold: {athlete}")
            doc_ids.extend(result.get("gold_doc_ids", []))
        elif entities.operation == "lookup":
            result = self.graph.run_lookup(
                event_fragment=entities.event_fragment,
                year=entities.year,
                sport=entities.sport,
                gender=entities.gender,
            )
            values = {
                "gold_athlete": result.get("gold_athletes", []),
                "nation_count": result.get("nation_counts", []),
                "competitor_count": result.get("competitor_counts", []),
                "venue": result.get("venues", []),
            }.get(entities.attribute, [])
            for index, event in enumerate(result.get("events", [])):
                value = values[index] if index < len(values) else ""
                graph_facts.append(f"Event: {event} | Requested {entities.attribute}: {value}")
            doc_ids.extend(result.get("gold_doc_ids", []))

        # Step 3: Hybrid retrieval is the default passage retriever for GraphRAG.
        embeddings = await self.llm.embed([question])
        dense_results = self.graph.vector_search(embeddings[0], top_k=30)
        if self.coprocessor:
            passage_results = self.coprocessor.hybrid_rerank(
                query=question,
                dense_results=dense_results,
                final_top_k=5,
            )
        else:
            passage_results = []

        if passage_results:
            for chunk, score in passage_results:
                doc_id = chunk.doc_id
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                graph_facts.append(
                    f"[Hybrid rank score {score:.3f} | chunk: {chunk.chunk_id} | doc: {doc_id}]\n"
                    f"{chunk.text}"
                )
        else:
            # Keep GraphRAG operational when the local coprocessor has no indexed candidates.
            for chunk_id, score in dense_results[:5]:
                doc_id = chunk_id.rsplit("#", 1)[0]
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                fallback_chunk = self.coprocessor.get_chunk(chunk_id) if self.coprocessor else None
                if fallback_chunk:
                    graph_facts.append(
                        f"[Dense fallback score {score:.3f} | doc: {doc_id}]\n{fallback_chunk.text}"
                    )
                else:
                    graph_facts.append(f"[Dense fallback score {score:.3f} | doc: {doc_id}]")

        # Step 4: Single-turn LLM synthesis over assembled graph context
        context_text = "\n".join(graph_facts) if graph_facts else "No relevant graph data found."
        context_tokens = len(context_text) // 4

        messages = [
            {"role": "system", "content": _GRAPHRAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _GRAPHRAG_USER_TEMPLATE.format(
                    graph_context=context_text, question=question
                ),
            },
        ]
        llm_result = await self.llm.chat(messages, max_tokens=256)
        answer = sanitize_output(llm_result.content.strip())

        latency_ms = (time.perf_counter() - t_start) * 1000
        return PipelineResult(
            qid=qid,
            pipeline="graphrag",
            question=question,
            answer=answer,
            llm_input_tokens=entities.input_tokens + llm_result.input_tokens,
            llm_output_tokens=entities.output_tokens + llm_result.output_tokens,
            total_llm_tokens=(
                entities.input_tokens
                + entities.output_tokens
                + llm_result.input_tokens
                + llm_result.output_tokens
            ),
            context_tokens=context_tokens,
            latency_ms=latency_ms,
            # Preserve first-seen retrieval order for deterministic rank metrics.
            retrieved_doc_ids=list(dict.fromkeys(doc_ids)),
            model_name=llm_result.model_name,
            provider=llm_result.provider,
        )
