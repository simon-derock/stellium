# Pipeline 2: Hybrid GraphRAG (fixed pipeline, non-agentic baseline).
# Fixed sequence: entity extraction -> 1-2 hop graph expansion -> dense vector fusion -> LLM synthesis.
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
Extract the target year, sport, and venue mentioned in the user question.
Return ONLY a valid JSON object matching this schema:
{"year": 2012, "sport": "Athletics", "venue": "National Stadium"}
Use null for any field not explicitly mentioned or inferred from the question. Do not include markdown formatting or commentary."""

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
    year: int = 0
    sport: str = ""
    venue: str = ""


async def _extract_entities_via_llm(llm: LockedLLMSession, question: str) -> ExtractedEntities:
    # Uses a single-turn LLM call to extract structured entities from query text.
    messages = [
        {"role": "system", "content": _ENTITY_LINKING_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    try:
        res = await llm.chat(messages, max_tokens=128, temperature=0.0)
        cleaned = res.content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data: dict[str, Any] = json.loads(cleaned)

        raw_year = data.get("year")
        year = int(raw_year) if raw_year is not None and str(raw_year).isdigit() else 0
        sport = str(data.get("sport") or "").strip()
        venue = str(data.get("venue") or "").strip()
        return ExtractedEntities(year=year, sport=sport, venue=venue)
    except Exception:
        # Graceful fallback when running in offline mode or on unparseable JSON
        return ExtractedEntities()


@dataclass
class GraphRAGPipeline:
    graph: GraphClient
    llm: LockedLLMSession
    coprocessor: Coprocessor | None = None

    async def run(self, qid: str, question: str) -> PipelineResult:
        t_start = time.perf_counter()

        # Step 1: Entity linking — dynamic LLM entity extraction
        entities = await _extract_entities_via_llm(self.llm, question)

        # Step 2: Fixed 1-2 hop graph traversal based on extracted entities
        graph_facts: list[str] = []
        doc_ids: list[str] = []

        if entities.venue:
            result = self.graph.run_multihop(venue_fragment=entities.venue)
            for event, athlete in zip(result.get("events", []), result.get("gold_athletes", [])):
                graph_facts.append(f"Event: {event} | Gold: {athlete}")
            doc_ids.extend(result.get("gold_doc_ids", []))

        if entities.year and entities.sport:
            result = self.graph.run_lookup(year=entities.year, sport=entities.sport)
            for event, gold, nations in zip(
                result.get("events", []),
                result.get("gold_athletes", []),
                result.get("nation_counts", []),
            ):
                graph_facts.append(f"Event: {event} | Gold: {gold} | Nations: {nations}")
            doc_ids.extend(result.get("gold_doc_ids", []))
        elif entities.year:
            result = self.graph.run_lookup(year=entities.year)
            for event, gold in zip(
                result.get("events", [])[:10], result.get("gold_athletes", [])[:10]
            ):
                graph_facts.append(f"Event: {event} | Gold: {gold}")
            doc_ids.extend(result.get("gold_doc_ids", [])[:10])

        # Step 3: Fixed dense vector search for supporting passages (fixed top-3)
        embeddings = await self.llm.embed([question])
        dense_results = self.graph.vector_search(embeddings[0], top_k=3)
        for chunk_id, score in dense_results:
            doc_id = chunk_id.rsplit("#", 1)[0]
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
            chunk = self.coprocessor.get_chunk(chunk_id) if self.coprocessor else None
            if chunk:
                graph_facts.append(
                    f"[Semantic match score {score:.3f} | doc: {doc_id}]\n{chunk.text}"
                )
            else:
                graph_facts.append(f"[Semantic match score {score:.3f} | doc: {doc_id}]")

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
            llm_input_tokens=llm_result.input_tokens,
            llm_output_tokens=llm_result.output_tokens,
            total_llm_tokens=llm_result.input_tokens + llm_result.output_tokens,
            context_tokens=context_tokens,
            latency_ms=latency_ms,
            retrieved_doc_ids=list(set(doc_ids)),
            model_name=llm_result.model_name,
            provider=llm_result.provider,
        )
