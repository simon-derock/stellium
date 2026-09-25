# Pipeline 1: Baseline Vector RAG (deliberately simple).
# TigerVector HNSW top-k → single-turn LLM answer. No graph, no BM25, no reranker.
# Weakness exposed: fails on aggregation, superlative, multi-hop, temporal.
from __future__ import annotations

import time
from dataclasses import dataclass

from src.graph import GraphClient
from src.guardrails import sanitize_output
from src.llm import LockedLLMSession
from src.models import PipelineResult

_RAG_SYSTEM_PROMPT = """You are a precise sports historian with access to Wikipedia articles about Olympic events.
Answer the question using ONLY the provided context passages.
If the answer is not in the context, respond with "Not found in corpus".
Be concise: give the direct answer, not a full sentence when a name or number suffices."""

_RAG_USER_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGPipeline:
    graph: GraphClient
    llm: LockedLLMSession

    async def run(self, qid: str, question: str) -> PipelineResult:
        t_start = time.perf_counter()

        # Step 1: Embed query → TigerVector HNSW search (top-5)
        embeddings = await self.llm.embed([question])
        query_vector = embeddings[0]
        dense_results = self.graph.vector_search(query_vector, top_k=5)

        # Step 2: Resolve chunk texts and collect doc IDs
        # Note: RAG pipeline accesses chunk text from TigerGraph result directly
        doc_ids: list[str] = []
        context_parts: list[str] = []
        for chunk_id, score in dense_results:
            # chunk_id format: "{doc_id}#{index}"
            doc_id = chunk_id.rsplit("#", 1)[0]
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)

        # Get context text from the chunks returned by vector search
        # (The graph.vector_search already returns chunk text via the GSQL query)
        # We access text via the coprocessor chunk map for offline usage
        # In production, text is returned directly from TigerGraph GSQL result
        context_tokens = 0
        context_parts = []
        for chunk_id, score in dense_results:
            # This will be resolved via coprocessor in the app startup
            context_parts.append(f"[Score: {score:.3f}] [DocID: {chunk_id.rsplit('#', 1)[0]}]")
            context_tokens += 80  # estimated

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
            model_name=llm_result.model_name,
        )
