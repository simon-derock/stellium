# Pipeline 2: GraphRAG, a fixed and acyclic baseline.
# Exactly two LLM calls: one turns the question into a structured graph lookup, the other writes
# the answer. In between, one typed graph operation runs (entity linking, then TigerGraph), and
# the documents it cites supply the supporting text. Nothing loops, retries, or changes strategy;
# that adaptivity is what the agentic pipeline is measured against.
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from src.coprocessor import Coprocessor
from src.graph import GraphClient
from src.guardrails import sanitize_output
from src.linking import EventCatalog
from src.llm import LockedLLMSession
from src.models import PipelineResult
from src.pipelines.agentic import _answer_parts_grounded, _extract_balanced_json
from src.pipelines.toolkit import (
    EVENT_ATTRIBUTES,
    TOOL_PARAMETERS,
    GraphToolkit,
    ToolOutcome,
    catalog_for,
)

logger = logging.getLogger(__name__)

_SUPPORTING_DOCS = 3
_SUPPORT_CHARS = 700

_EXTRACTION_PROMPT = f"""Map the question to one structured lookup over a graph of Olympic events. Return only a JSON object, using just the keys that apply:
{{"operation": "", "sport": "", "year": 0, "season": "", "gender": "", "event": "", "venue": "", "date": "", "attribute": "", "comparison": "", "threshold": 0, "order": ""}}

operation is one of:
- count_events: how many events meet a competitor-count condition (comparison: more_than, at_least, fewer_than, at_most, exactly; threshold: the number)
- rank_events: which event had the most (order "desc") or fewest (order "asc") competitors
- event_attribute: an attribute of one named event
- previous_edition: an attribute of the same event at the Games immediately before `year` (use the year named in the question)
- event_at_venue_date: an attribute of the event held at a venue on a date
- none: anything else

attribute is one of: {", ".join(EVENT_ATTRIBUTES)}.
Copy names, dates, and numbers exactly as the question writes them; omit every key the question does not state."""

_ANSWER_PROMPT = """Answer the question using only the graph result and passages provided.
If the graph result contains a value or answer, return that value exactly; if it lists several candidates that the passages cannot separate, return all of them joined by "; ".
Return only the answer: the full canonical event title for "which event", a bare number for counts, the name(s) for people.
If nothing provided establishes the answer, return "Not found in corpus"."""


@dataclass
class GraphRAGPipeline:
    graph: GraphClient
    llm: LockedLLMSession
    coprocessor: Coprocessor | None = None
    catalog: EventCatalog | None = None

    def _supporting_passages(self, doc_ids: list[str]) -> list[tuple[str, str]]:
        # Graph-to-document retrieval: the opening section of each article the graph cited.
        passages: list[tuple[str, str]] = []
        if self.coprocessor is None:
            return passages
        for doc_id in doc_ids[:_SUPPORTING_DOCS]:
            chunk = self.coprocessor.get_chunk(f"{doc_id}#0")
            if chunk is not None:
                passages.append((doc_id, chunk.text[:_SUPPORT_CHARS]))
        return passages

    async def _hybrid_passages(self, question: str) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        # Fallback when no graph operation applies: the same hybrid retrieval RAG uses.
        if self.coprocessor is None:
            return [], {}
        embeddings = await self.llm.embed([question])
        dense = self.graph.vector_search(embeddings[0], top_k=30)
        hybrid = self.coprocessor.hybrid_search(
            query=question, dense_results=dense, candidate_k=30, final_top_k=5
        )
        passages = [(chunk.doc_id, chunk.text[:_SUPPORT_CHARS]) for chunk, _ in hybrid.chunks]
        metadata = {
            "dense_candidates": hybrid.dense_candidate_count,
            "bm25_candidates": hybrid.sparse_candidate_count,
            "rrf_candidates": hybrid.fused_candidate_count,
            "reranker": hybrid.reranker_model,
            "reranker_executed": hybrid.reranker_executed,
            "reranker_latency_ms": hybrid.reranker_latency_ms,
        }
        return passages, metadata

    async def run(self, qid: str, question: str) -> PipelineResult:
        started = time.perf_counter()
        toolkit = GraphToolkit(
            self.graph, self.catalog or catalog_for(self.graph), self.coprocessor
        )

        # Call 1: structured extraction of the graph lookup.
        extraction = await self.llm.chat(
            [
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=160,
            temperature=0.0,
        )
        plan = _extract_balanced_json(extraction.content)
        operation = str(plan.get("operation", "none")).strip()

        # One typed graph operation, chosen by the extracted operation name.
        outcome: ToolOutcome | None = None
        if operation in TOOL_PARAMETERS and operation != "find_events":
            try:
                outcome = toolkit.invoke(operation, plan)
            except (TypeError, ValueError) as exc:
                logger.warning("GraphRAG graph operation %s failed: %s", operation, exc)

        doc_ids = list(outcome.citations) if outcome else []
        retrieval: dict[str, Any] = {}
        if outcome is not None and (outcome.answer is not None or outcome.candidates):
            passages = self._supporting_passages(doc_ids)
        else:
            passages, retrieval = await self._hybrid_passages(question)
            doc_ids.extend(doc_id for doc_id, _ in passages if doc_id not in doc_ids)

        graph_result: dict[str, Any] = (
            {"operation": operation, **outcome.observation}
            if outcome is not None
            else {"operation": operation, "status": "no graph operation applied"}
        )
        if outcome is not None and outcome.answer is not None:
            graph_result["answer"] = outcome.answer
        elif outcome is not None and outcome.candidates:
            graph_result["candidates"] = outcome.candidates
        context = "Graph result: " + json.dumps(graph_result, ensure_ascii=False, default=str)
        context += "".join(f"\n\n[{doc_id}]\n{text}" for doc_id, text in passages)

        # Call 2: answer synthesis over graph facts and their source passages.
        synthesis = await self.llm.chat(
            [
                {"role": "system", "content": _ANSWER_PROMPT},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}\nAnswer:"},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        answer = sanitize_output(synthesis.content.strip())
        answer_source = "synthesis"
        # Answer validation: a synthesized value that neither the graph nor a passage supports is
        # replaced by the graph's verified value, so formatting slips cannot corrupt a fact.
        verified = {outcome.answer} if outcome is not None and outcome.answer else set()
        if verified and not _answer_parts_grounded(answer, verified, [t for _, t in passages]):
            answer = next(iter(verified))
            answer_source = "graph_value"

        input_tokens = extraction.input_tokens + synthesis.input_tokens
        output_tokens = extraction.output_tokens + synthesis.output_tokens
        return PipelineResult(
            qid=qid,
            pipeline="graphrag",
            question=question,
            answer=answer,
            llm_input_tokens=input_tokens,
            llm_output_tokens=output_tokens,
            total_llm_tokens=input_tokens + output_tokens,
            context_tokens=len(context) // 4,
            latency_ms=(time.perf_counter() - started) * 1000,
            retrieved_doc_ids=list(dict.fromkeys(doc_ids)),
            retrieval_metadata={
                "graph_operation": operation,
                "graph_plan": plan,
                "graph_tool_latency_ms": outcome.latency_ms if outcome else 0.0,
                "graph_status": (
                    "conclusive"
                    if outcome is not None and outcome.answer is not None
                    else "ambiguous"
                    if outcome is not None and outcome.ambiguous
                    else "unresolved"
                ),
                "answer_source": answer_source,
                "token_sources": sorted({extraction.token_source, synthesis.token_source}),
                **retrieval,
            },
            model_name=synthesis.model_name,
            provider=synthesis.provider,
        )
