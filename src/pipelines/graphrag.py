# Pipeline 2: Hybrid GraphRAG (fixed pipeline, NON-agentic).
# Fixed sequence: entity link → 1-2 hop graph expand → fuse → LLM answer.
# No backtracking, no strategy adaptation, no dynamic tool selection.
# Weakness exposed: entity linking failure cascades, no aggregation, fixed depth.
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from src.graph import GraphClient
from src.guardrails import sanitize_output
from src.llm import LockedLLMSession
from src.models import PipelineResult

_GRAPHRAG_SYSTEM_PROMPT = """You are a precise sports historian with access to Olympic event data.
Answer the question using ONLY the provided graph context (entities, relationships, and passages).
If the answer is not in the context, respond with "Not found in corpus".
Be concise."""

_GRAPHRAG_USER_TEMPLATE = """Graph context:
{graph_context}

Question: {question}

Answer:"""

# Simple year extractor for entity linking
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
    "skiing",
    "skating",
    "curling",
    "hockey",
    "football",
    "basketball",
    "volleyball",
    "tennis",
    "badminton",
]


def _extract_year(query: str) -> int:
    m = _YEAR_RE.search(query)
    return int(m.group(1)) if m else 0


def _extract_sport(query: str) -> str:
    q_lower = query.lower()
    for sport in _SPORT_KEYWORDS:
        if sport in q_lower:
            return sport.capitalize()
    return ""


def _extract_venue(query: str) -> str:
    # Look for "at [Proper Noun] [Venue]" patterns
    m = re.search(
        r"\bat\s+([A-Z][A-Za-z\s]+(?:Stadium|Arena|Hall|Centre|Center|Oval|Park|Pool|Gymnasium|Velodrome|Rink))",
        query,
    )
    return m.group(1).strip() if m else ""


@dataclass
class GraphRAGPipeline:
    graph: GraphClient
    llm: LockedLLMSession

    async def run(self, qid: str, question: str) -> PipelineResult:
        t_start = time.perf_counter()

        # Step 1: Entity linking — extract structured parameters from query text
        year = _extract_year(question)
        sport = _extract_sport(question)
        venue = _extract_venue(question)

        # Step 2: Graph traversal — fixed 1-2 hop expansion
        graph_facts: list[str] = []
        doc_ids: list[str] = []

        if venue:
            # Multi-hop: venue → events at that venue
            result = self.graph.run_multihop(venue_fragment=venue)
            for event, athlete in zip(result["events"], result["gold_athletes"]):
                graph_facts.append(f"Event: {event} | Gold: {athlete}")
            doc_ids.extend(result["gold_doc_ids"])

        if year and sport:
            # Lookup: events by year+sport
            result = self.graph.run_lookup(year=year, sport=sport)
            for event, gold, nations in zip(
                result["events"], result["gold_athletes"], result["nation_counts"]
            ):
                graph_facts.append(f"Event: {event} | Gold: {gold} | Nations: {nations}")
            doc_ids.extend(result["gold_doc_ids"])
        elif year:
            result = self.graph.run_lookup(year=year)
            for event, gold in zip(result["events"][:10], result["gold_athletes"][:10]):
                graph_facts.append(f"Event: {event} | Gold: {gold}")
            doc_ids.extend(result["gold_doc_ids"][:10])

        # Step 3: Vector search for supporting text (fixed top-3)
        embeddings = await self.llm.embed([question])
        dense_results = self.graph.vector_search(embeddings[0], top_k=3)
        for chunk_id, score in dense_results:
            doc_id = chunk_id.rsplit("#", 1)[0]
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
            graph_facts.append(f"[Semantic match score {score:.3f} | doc: {doc_id}]")

        # Step 4: Single-turn LLM synthesis
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
        )
