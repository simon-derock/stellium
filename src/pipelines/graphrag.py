# Pipeline 2: Hybrid GraphRAG (fixed pipeline, non-agentic baseline).
# Fixed sequence: entity extraction -> graph expansion -> dense/sparse fusion -> reranking -> synthesis.
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from src.coprocessor import Coprocessor
from src.graph import GraphClient
from src.guardrails import sanitize_output, validate_generated_gsql
from src.llm import LockedLLMSession
from src.models import PipelineResult

logger = logging.getLogger(__name__)

_GSQL_GENERATION_SYSTEM_PROMPT = """You write one read-only GSQL query for TigerGraph graph OlympicsGraph, using the schema and examples below.

The caller executes your exact query with TigerGraph's interpreted-query API. Return ONLY a complete query in this form:
INTERPRET QUERY () FOR GRAPH OlympicsGraph { ... }

Schema available to this query:
- Event: event_id, name, year, season, sport, gender, venue, competitor_count, nation_count, gold_athlete, silver_athlete, bronze_athlete, gold_noc, silver_noc, bronze_noc, prev_event_id, next_event_id, filter_mask, valid_from, valid_to, superseded_by, source_authority.
- Venue: venue_id, name. Edge HELD_AT has start_date and end_date.
- Document: doc_id, title, url, wikidata_qid, wikipedia_pageid, approx_tokens, filter_mask.
- Chunk: chunk_id, doc_id, chunk_index, section_title, text, raw_text, prev_chunk_id, next_chunk_id, filter_mask.
- Edges: HAS_CHUNK (Document→Chunk), DOCUMENTED_IN (Event→Document), HELD_AT (Event→Venue), PRECEDES and SUCCEEDS (Event→Event).

Rules:
- Use only those vertex types, edge types, and attributes. Session and ChatMessage are forbidden.
- Use the exact question semantics; preserve every year, season, sport, gender, event, date, venue, threshold, and requested attribute.
- Return the smallest useful result. Every SELECT must have LIMIT 2500 or less; use a smaller limit whenever appropriate. The corpus has 2,210 Event vertices.
- Include PRINT for the result. One SELECT only. No comments, markdown, DDL, DML, file access, loops, or other commands.
- Use the interpreted-query JSON API v2 PRINT grammar: `PRINT Matched[Matched.name, Matched.year];`. Do not use legacy `PRINT Matched.name;` syntax or alias a global accumulator in PRINT.
- Compare string attributes case-insensitively: use `lower(e.sport) == lower("Fencing")`, not `e.sport == "fencing"`. Apply this form to season, gender, and event-name equality; use `LIKE` with `%fragment%` for partial names.
- For an exact count, declare `SumAccum<INT> @@match_count = 0;`, increment it in the bounded SELECT with `ACCUM @@match_count += 1`, then print `@@match_count` directly. The 2,210 Event vertices fit within `LIMIT 2500` for corpus-wide counts.
- For a superlative, order by `competitor_count` and return both `name` and `competitor_count`. If the question asks which event, the final answer is the complete event name; use the count only to rank events.
- For strict thresholds, translate “more than N” to > N and “fewer than N” to < N.
- Event has no `date` attribute. For venue/date questions, traverse `HELD_AT` in the single SELECT and filter the linked `Venue.name` plus `HELD_AT.start_date` (or `end_date`); do not invent `Event.date`.
- The guarded subset accepts `LIKE`, not `CONTAINS`. Use `%fragment%` patterns for substring matches. Use double-quoted strings when a value contains an apostrophe. Do not add comments or use a second SELECT.
- Match the executor's supported GSQL subset exactly: the only function call allowed is `lower(...)`; do not call `max`, `min`, `sum`, `avg`, `count`, `contains`, `regex`, conversion functions, or user-defined functions. Do not use `GROUP BY`, nested SELECTs, subqueries, or SQL/Cypher syntax. For counts, use only the `SumAccum<INT>` pattern shown above.
- Before returning, check that the query has exactly one SELECT, every SELECT has an explicit LIMIT, every referenced type/edge/attribute is in the schema above, and the final statement is PRINT. Return the query only, with no explanation or markdown fence.
- Do not guess values or answer from general knowledge.

Example — named event attribute:
INTERPRET QUERY () FOR GRAPH OlympicsGraph {
  Events = {Event.*};
  Matched = SELECT e FROM Events:e
    WHERE e.year == 2008 AND lower(e.name) LIKE "%marathon%"
    LIMIT 20;
  PRINT Matched[Matched.name, Matched.gold_athlete];
}

Example — highest competitor count:
INTERPRET QUERY () FOR GRAPH OlympicsGraph {
  Events = {Event.*};
  Ranked = SELECT e FROM Events:e
    WHERE e.year == 2016 AND lower(e.sport) == lower("Shooting")
    ORDER BY e.competitor_count DESC LIMIT 1;
  PRINT Ranked[Ranked.name, Ranked.competitor_count];
}

Example — event by venue and date (date belongs to HELD_AT, not Event):
INTERPRET QUERY () FOR GRAPH OlympicsGraph {
  Events = {Event.*};
  Matched = SELECT e FROM Events:e -(HELD_AT:h)- Venue:v
    WHERE e.year == 2012
      AND lower(v.name) LIKE "%london velopark%"
      AND lower(h.start_date) LIKE "%3 august%"
    LIMIT 20;
  PRINT Matched[Matched.name, Matched.gold_athlete];
}

Example — preceding event through the directed PRECEDES edge:
INTERPRET QUERY () FOR GRAPH OlympicsGraph {
  Events = {Event.*};
  Prior = SELECT prior FROM Events:current -(PRECEDES:p)-> Event:prior
    WHERE current.year == 2016
      AND lower(current.sport) == "athletics"
      AND lower(current.name) LIKE "%women%10,000%metres%"
    LIMIT 20;
  PRINT Prior[Prior.name, Prior.year, Prior.gold_athlete];
}

Example — exact count with a bounded SELECT:
INTERPRET QUERY () FOR GRAPH OlympicsGraph {
  SumAccum<INT> @@match_count = 0;
  Events = {Event.*};
  Matched = SELECT e FROM Events:e
    WHERE e.year == 2018 AND lower(e.sport) == lower("Biathlon")
      AND e.competitor_count > 73
    ACCUM @@match_count += 1 LIMIT 2500;
  PRINT @@match_count;
}

For accumulator output, print the global accumulator directly without `AS`. For a vertex set,
use JSON API v2 projection syntax such as `PRINT Matched[Matched.name, Matched.year];`. Always use double-quoted string literals; the deployed interpreted-query endpoint rejects single-quoted values. Apostrophes inside a value belong inside the double quotes, e.g. `"%men's epee%"`. Do not use escapes or include query delimiters in values."""

_GRAPHRAG_SYSTEM_PROMPT = """You are a precise sports historian with access to Olympic event data.
Answer the question using ONLY the provided graph context (entities, relationships, and passages).
If the answer is not in the context, respond with "Not found in corpus".
Be concise, but preserve the complete canonical entity name exactly as it appears in the graph or source.
For an Olympic event, include the sport and Olympic edition when they are part of the event name;
do not shorten the answer to only the event suffix. Do not add competitor counts unless asked.
For a question asking for a number, return only the number."""

_GRAPHRAG_USER_TEMPLATE = """Graph context:
{graph_context}

Question: {question}

Answer:"""


async def _generate_gsql_query(llm: LockedLLMSession, question: str) -> tuple[str, int, int]:
    # Produces a complete query, then rejects it unless it passes the read-only GSQL allowlist.
    response = await llm.chat(
        [
            {"role": "system", "content": _GSQL_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Write a GSQL query for this question:\n{question}"},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    query = response.content.strip()
    if query.startswith("```"):
        query = query.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return query, response.input_tokens, response.output_tokens


@dataclass
class GraphRAGPipeline:
    graph: GraphClient
    llm: LockedLLMSession
    coprocessor: Coprocessor | None = None

    async def run(self, qid: str, question: str) -> PipelineResult:
        t_start = time.perf_counter()
        if self.coprocessor is None:
            raise RuntimeError("GraphRAG requires the in-memory BM25 coprocessor")

        # Step 1: Generate GSQL from the question and execute only after guardrail validation.
        generated_query, query_input_tokens, query_output_tokens = await _generate_gsql_query(
            self.llm, question
        )
        graph_facts: list[str] = []
        doc_ids: list[str] = []
        safe_query, rejection_reason = validate_generated_gsql(generated_query)
        if safe_query:
            try:
                graph_result = self.graph.run_generated_gsql(generated_query)
                event_candidates = graph_result.get("event_candidates", [])
                if event_candidates:
                    graph_facts.append(
                        "Canonical Event rows returned by GSQL, in query order: "
                        + json.dumps(event_candidates, ensure_ascii=False, default=str)
                    )
                graph_facts.append(
                    "Generated read-only GSQL result: "
                    + json.dumps(graph_result.get("rows", []), ensure_ascii=False, default=str)[
                        :16_000
                    ]
                )
            except Exception as exc:
                # Keep an interpreted-query failure from aborting the fixed GraphRAG pipeline.
                logger.warning("Generated GraphRAG GSQL failed at runtime: %s", type(exc).__name__)
                graph_facts.append("Generated graph query could not execute; use passage evidence.")
        else:
            logger.warning("Generated GraphRAG GSQL was rejected: %s", rejection_reason)
            graph_facts.append(f"Graph query rejected by guardrail: {rejection_reason}")

        # Step 3: Hybrid retrieval is the default passage retriever for GraphRAG.
        embeddings = await self.llm.embed([question])
        dense_results = self.graph.vector_search(embeddings[0], top_k=30)
        retrieval = self.coprocessor.hybrid_search(
            query=question,
            dense_results=dense_results,
            candidate_k=30,
            final_top_k=5,
        )

        if retrieval.chunks:
            for chunk, score in retrieval.chunks:
                doc_id = chunk.doc_id
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                graph_facts.append(
                    f"[Reranker score {score:.3f} | chunk: {chunk.chunk_id} | doc: {doc_id}]\n"
                    f"{chunk.text}"
                )
        # Empty candidate sets stay empty; do not claim a rerank occurred or bypass the stage.

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
            llm_input_tokens=query_input_tokens + llm_result.input_tokens,
            llm_output_tokens=query_output_tokens + llm_result.output_tokens,
            total_llm_tokens=(
                query_input_tokens
                + query_output_tokens
                + llm_result.input_tokens
                + llm_result.output_tokens
            ),
            context_tokens=context_tokens,
            latency_ms=latency_ms,
            # Preserve first-seen retrieval order for deterministic rank metrics.
            retrieved_doc_ids=list(dict.fromkeys(doc_ids)),
            retrieval_metadata={
                "dense_candidates": retrieval.dense_candidate_count,
                "bm25_candidates": retrieval.sparse_candidate_count,
                "rrf_candidates": retrieval.fused_candidate_count,
                "reranker": retrieval.reranker_model,
                "reranker_executed": retrieval.reranker_executed,
                "reranker_latency_ms": retrieval.reranker_latency_ms,
            },
            model_name=llm_result.model_name,
            provider=llm_result.provider,
        )
