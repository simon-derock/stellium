# Shared Pydantic domain models used across all modules.
# Single source of truth for all data contracts.
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Corpus / Ingestion Models
# ---------------------------------------------------------------------------


class CorpusDoc(BaseModel):
    # One row from corpus.jsonl
    doc_id: str
    wikidata_qid: str
    wikipedia_pageid: int
    title: str
    url: str
    text: str
    approx_tokens: int


class ParsedInbox(BaseModel):
    # Structured fields extracted from [Infobox Olympic event] header
    event_name: str | None = None
    year: int | None = None
    season: str | None = None  # "Summer" | "Winter"
    sport: str | None = None
    gender: str | None = None
    venue: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    competitor_count: int | None = None
    nation_count: int | None = None
    gold_athlete: str | None = None
    silver_athlete: str | None = None
    bronze_athlete: str | None = None
    gold_noc: str | None = None
    silver_noc: str | None = None
    bronze_noc: str | None = None
    prev_year: int | None = None
    next_year: int | None = None


class Chunk(BaseModel):
    # One chunk extracted from a document, ready for embedding + graph upsert
    chunk_id: str  # f"{doc_id}#{chunk_index}"
    doc_id: str
    chunk_index: int
    section_title: str
    text: str  # Full text including injected metadata header
    raw_text: str  # Text without injected header (for BM25)
    prev_chunk_id: str | None = None  # uint16 index pointer as string
    next_chunk_id: str | None = None
    infobox: ParsedInbox | None = None  # Only on chunk_index==0
    filter_mask: int = 0  # Roaring bitmask flags for year/season/sport


# ---------------------------------------------------------------------------
# Evaluation / Query Models
# ---------------------------------------------------------------------------


class EvalQuestion(BaseModel):
    # One row from eval_public.jsonl or eval_hidden.jsonl
    qid: str
    question: str
    qtype: Literal["aggregation", "lookup", "multi_hop", "superlative", "temporal"]
    answer: list[str] | None = None  # None for hidden questions
    gold_doc_ids: list[str] | None = None  # None for hidden questions


class EvidenceItem(BaseModel):
    doc_id: str
    chunk_id: str | None = None
    text: str
    relevance_score: float = 1.0
    source: str = "vector"  # "vector" | "bm25" | "gsql" | "graph_traverse"


class ToolAuditCall(BaseModel):
    step: int
    tool_name: str
    input_args: dict[str, Any]
    output_summary: str
    llm_tokens: int = 0  # 0 for deterministic steps
    latency_ms: float = 0.0


class AgentState(BaseModel):
    query: str
    qtype: Literal["aggregation", "lookup", "multi_hop", "superlative", "temporal"]
    model_name: str = ""  # Locked model for this run
    sub_questions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    traversed_vertices: list[str] = Field(default_factory=list)
    tool_history: list[ToolAuditCall] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = 5
    strategy_history: list[str] = Field(default_factory=list)
    strategy_changed: bool = False
    strategy_change_rationale: str | None = None
    stopping_reason: str = "initialized"
    confidence_score: float = 0.0
    final_answer: str | None = None


# ---------------------------------------------------------------------------
# Pipeline Result Models
# ---------------------------------------------------------------------------


class PipelineResult(BaseModel):
    # Output of one pipeline execution for one question
    qid: str
    pipeline: Literal["rag", "graphrag", "agentic"]
    question: str
    answer: str
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    total_llm_tokens: int = 0
    context_tokens: int = 0
    latency_ms: float = 0.0
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    # Agentic only — empty for rag/graphrag
    agentic_trace: dict[str, Any] | None = None
    model_name: str = ""


class CompareResult(BaseModel):
    # Side-by-side results for all 3 pipelines on one question
    qid: str
    question: str
    qtype: str
    rag: PipelineResult
    graphrag: PipelineResult
    agentic: PipelineResult


# ---------------------------------------------------------------------------
# Dashboard / Snapshot DTOs (Lunarbit GraphSurface compatible)
# ---------------------------------------------------------------------------


class GraphNodeDTO(BaseModel):
    id: str
    type: str
    layer: str
    label: str
    weight: float = 1.0
    source_count: int = 1
    confidence: float = 1.0
    privacy_state: str = "public"
    scope: str = "olympics"


class GraphEdgeDTO(BaseModel):
    id: str
    source: str
    target: str
    relationship_type: str
    confidence: float = 1.0
    provenance_label: str = ""


class MetricDTO(BaseModel):
    label: str
    value: str
    unit: str
    scope: str = "pipeline"


class SnapshotDTO(BaseModel):
    metrics: list[MetricDTO] = Field(default_factory=list)
    graph_nodes: list[GraphNodeDTO] = Field(default_factory=list)
    graph_edges: list[GraphEdgeDTO] = Field(default_factory=list)
    findings: list[dict[str, str]] = Field(default_factory=list)
    disclosure: str = "Stellium Agentic GraphRAG — TigerGraph Hackathon 2026"
