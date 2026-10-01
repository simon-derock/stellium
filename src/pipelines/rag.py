# Pipeline 1: Hybrid RAG control. Dense + BM25 → RRF → local reranker → answer.
from __future__ import annotations

import time
from dataclasses import dataclass

from src.coprocessor import Coprocessor
from src.graph import GraphClient
from src.guardrails import sanitize_output
from src.llm import LockedLLMSession
from src.models import PipelineResult

_RAG_SYSTEM_PROMPT = """Answer the question using only the context passages provided.
Return only the answer, with no sentence around it: the full canonical event title for "which event", a bare number for counts, the name(s) for people.
If the passages do not establish the answer, return "Not found in corpus"."""

_RAG_USER_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGPipeline:
    graph: GraphClient
    llm: LockedLLMSession
    coprocessor: Coprocessor | None = None

    async def run(self, qid: str, question: str) -> PipelineResult:
        t_start = time.perf_counter()

        if self.coprocessor is None:
            raise RuntimeError("Hybrid RAG requires the in-memory BM25 coprocessor")

        # Step 1: Dense and sparse candidates → RRF → mandatory local reranking.
        embeddings = await self.llm.embed([question])
        query_vector = embeddings[0]
        dense_results = self.graph.vector_search(query_vector, top_k=30)
        retrieval = self.coprocessor.hybrid_search(
            query=question,
            dense_results=dense_results,
            candidate_k=30,
            final_top_k=5,
        )

        # Step 2: Resolve chunk texts and collect doc IDs
        doc_ids: list[str] = []
        context_parts: list[str] = []
        context_tokens = 0

        for chunk, score in retrieval.chunks:
            doc_id = chunk.doc_id
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
            context_parts.append(
                f"[Reranker score: {score:.3f} | Chunk: {chunk.chunk_id} | Doc: {doc_id}]\n"
                f"{chunk.text}"
            )
            context_tokens += len(chunk.text) // 4

        context_text = "\n\n".join(context_parts)

        # Step 3: Single-turn LLM call with retrieved context
        messages = [
            {"role": "system", "content": _RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _RAG_USER_TEMPLATE.format(context=context_text, question=question),
            },
        ]
        llm_result = await self.llm.chat(messages, max_tokens=256)
        answer = sanitize_output(llm_result.content.strip())

        latency_ms = (time.perf_counter() - t_start) * 1000
        return PipelineResult(
            qid=qid,
            pipeline="rag",
            question=question,
            answer=answer,
            llm_input_tokens=llm_result.input_tokens,
            llm_output_tokens=llm_result.output_tokens,
            total_llm_tokens=llm_result.input_tokens + llm_result.output_tokens,
            context_tokens=context_tokens,
            latency_ms=latency_ms,
            retrieved_doc_ids=doc_ids,
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
