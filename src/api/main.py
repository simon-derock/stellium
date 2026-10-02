# FastAPI application exposing comparative endpoints and Lunarbit Snapshot DTOs.
# Endpoints:
# - POST /api/v1/query/rag
# - POST /api/v1/query/graphrag
# - POST /api/v1/query/agentic
# - POST /api/v1/query/compare (all 3 side-by-side)
# - POST /api/v1/evaluate/batch (batch evaluation for judges)
# - GET  /api/v1/graph/snapshot (Lunarbit GraphSurface compatible payload)
# - GET  /api/v1/sessions/{session_id}/history (persistent session hydration)
from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pyTigerGraph.common.exception import TigerGraphException

from src.coprocessor import Coprocessor
from src.graph import GraphClient, connect, create_mock_graph_client, wait_for_graph_ready
from src.guardrails import check_query
from src.ingest import load_all_chunks
from src.llm import make_session
from src.models import (
    CompareResult,
    EvalQuestion,
    PipelineResult,
    SnapshotDTO,
)
from src.pipelines.agentic import AgenticPipeline
from src.pipelines.graphrag import GraphRAGPipeline
from src.pipelines.rag import RAGPipeline

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

_coprocessor: Coprocessor = Coprocessor()
_graph: GraphClient | None = None


def _get_mock_graph() -> GraphClient:
    return create_mock_graph_client()


def _load_graph_environment() -> None:
    # Load local credentials before checking TG_HOST; connect() loads them too late for routing.
    try:
        import dotenv

        dotenv.load_dotenv()
    except ImportError:
        pass


def get_graph() -> GraphClient:
    global _graph
    if _graph is not None:
        return _graph
    _load_graph_environment()
    use_mock = os.environ.get("TG_USE_MOCK", "").casefold() in {"1", "true", "yes"}
    if use_mock:
        return _get_mock_graph()
    if not os.environ.get("TG_HOST", "").strip():
        raise RuntimeError(
            "TG_HOST is not configured; refusing to answer from the mock graph. "
            "Set TG_USE_MOCK=1 only for explicit local testing."
        )
    try:
        wait_for_graph_ready(os.environ["TG_HOST"], timeout_s=120.0)
        _graph = GraphClient(conn=connect())
    except Exception as exc:
        raise RuntimeError(
            "TigerGraph client initialization failed; refusing to substitute mock graph data."
        ) from exc
    return _graph


def _get_request_graph() -> GraphClient:
    try:
        return get_graph()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_coprocessor() -> Coprocessor:
    global _coprocessor
    return _coprocessor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _coprocessor
    # Startup: load corpus into memory coprocessor
    corpus_path = "hackathon-resources/corpus/corpus.jsonl"
    if os.path.exists(corpus_path):
        chunks = load_all_chunks(corpus_path)
        _coprocessor.build(chunks)

    yield


app = FastAPI(
    title="Stellium: Agentic GraphRAG API",
    description="TigerGraph Hackathon 2026 — 3-Way Comparative Benchmark & Live Visualization Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static Files & Dashboard Mount
# ---------------------------------------------------------------------------

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    index_file = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Dashboard UI not found")


_PROVIDERS = {"cohere", "cloudflare", "gemini", "mistral", "offline"}


def _default_provider() -> str:
    # The demo answers with the same model the benchmark was measured on unless configured.
    return os.environ.get("STELLIUM_LLM_PROVIDER", "cohere")


class QueryRequest(BaseModel):
    query: str
    qid: str = "custom-001"
    provider: str = Field(default_factory=_default_provider)


class BatchEvalRequest(BaseModel):
    questions: list[EvalQuestion]
    pipeline: str = "agentic"  # "rag" | "graphrag" | "agentic" | "all"
    provider: str = Field(default_factory=_default_provider)


# Bursts queue behind a fixed number of in-flight questions instead of overrunning provider
# rate limits; a request that cannot start in time gets a clean 503 with Retry-After.
_query_slots = asyncio.Semaphore(int(os.environ.get("STELLIUM_MAX_CONCURRENT_QUERIES", "4")))
_QUEUE_TIMEOUT_S = float(os.environ.get("STELLIUM_QUEUE_TIMEOUT_S", "60"))


@asynccontextmanager
async def _admitted(provider: str) -> AsyncIterator[None]:
    if provider not in _PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider {provider!r}")
    try:
        await asyncio.wait_for(_query_slots.acquire(), timeout=_QUEUE_TIMEOUT_S)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="All query slots are busy; retry shortly.",
            headers={"Retry-After": "10"},
        ) from exc
    try:
        yield
    except RuntimeError as exc:
        # Provider quota or outage after bounded retries: report it, never a fabricated answer.
        raise HTTPException(
            status_code=503,
            detail=f"Language model unavailable: {exc}",
            headers={"Retry-After": "30"},
        ) from exc
    finally:
        _query_slots.release()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check(deep: bool = False) -> dict[str, Any]:
    # deep=true also runs a real graph read, which keeps an idle workspace from suspending.
    status: dict[str, Any] = {"status": "ok", "system": "stellium", "version": "0.1.0"}
    if deep:
        started = time.perf_counter()
        venues = _get_request_graph().ping()
        status["graph"] = {
            "venues": venues,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    return status


@app.exception_handler(TigerGraphException)
@app.exception_handler(requests.RequestException)
async def graph_unavailable(_request: Request, _exc: Exception) -> JSONResponse:
    # A suspended or restarting graph is a temporary outage, not a server bug.
    return JSONResponse(
        status_code=503,
        content={"detail": "The graph database is waking up or unreachable; retry shortly."},
        headers={"Retry-After": "15"},
    )


@app.post("/api/v1/query/rag", response_model=PipelineResult)
async def query_rag(req: QueryRequest) -> PipelineResult:
    safe, reason = check_query(req.query, is_batch=False)
    if not safe:
        raise HTTPException(status_code=400, detail=reason)

    graph = _get_request_graph()
    async with _admitted(req.provider), make_session(req.provider) as session:
        pipe = RAGPipeline(graph=graph, llm=session, coprocessor=get_coprocessor())
        return await pipe.run(req.qid, req.query)


@app.post("/api/v1/query/graphrag", response_model=PipelineResult)
async def query_graphrag(req: QueryRequest) -> PipelineResult:
    safe, reason = check_query(req.query, is_batch=False)
    if not safe:
        raise HTTPException(status_code=400, detail=reason)

    graph = _get_request_graph()
    async with _admitted(req.provider), make_session(req.provider) as session:
        pipe = GraphRAGPipeline(graph=graph, llm=session, coprocessor=get_coprocessor())
        return await pipe.run(req.qid, req.query)


@app.post("/api/v1/query/agentic", response_model=PipelineResult)
async def query_agentic(req: QueryRequest) -> PipelineResult:
    safe, reason = check_query(req.query, is_batch=False)
    if not safe:
        raise HTTPException(status_code=400, detail=reason)

    graph = _get_request_graph()
    coproc = get_coprocessor()
    async with _admitted(req.provider), make_session(req.provider) as session:
        pipe = AgenticPipeline(graph=graph, coprocessor=coproc, llm=session)
        return await pipe.run(req.qid, req.query)


@app.post("/api/v1/query/compare", response_model=CompareResult)
async def query_compare(req: QueryRequest) -> CompareResult:
    # Executes all 3 pipelines side-by-side for comparative judging
    safe, reason = check_query(req.query, is_batch=False)
    if not safe:
        raise HTTPException(status_code=400, detail=reason)

    graph = _get_request_graph()
    coproc = get_coprocessor()
    async with (
        _admitted(req.provider),
        make_session(req.provider) as rag_llm,
        make_session(req.provider) as graphrag_llm,
        make_session(req.provider) as agentic_llm,
    ):
        # Same model in every pipeline; separate sessions let the three run side by side.
        rag_res, graphrag_res, agentic_res = await asyncio.gather(
            RAGPipeline(graph=graph, llm=rag_llm, coprocessor=coproc).run(req.qid, req.query),
            GraphRAGPipeline(graph=graph, llm=graphrag_llm, coprocessor=coproc).run(
                req.qid, req.query
            ),
            AgenticPipeline(graph=graph, coprocessor=coproc, llm=agentic_llm).run(
                req.qid, req.query
            ),
        )

    return CompareResult(
        qid=req.qid,
        question=req.query,
        qtype=agentic_res.agentic_trace.get("qtype", "general")
        if agentic_res.agentic_trace
        else "general",
        rag=rag_res,
        graphrag=graphrag_res,
        agentic=agentic_res,
    )


@app.post("/api/v1/evaluate/batch")
async def evaluate_batch(req: BatchEvalRequest) -> list[dict[str, Any]]:
    # Batch evaluation: bypasses input guardrails (trusted eval questions)
    graph = _get_request_graph()
    coproc = get_coprocessor()
    session = make_session(req.provider)
    results: list[dict[str, Any]] = []

    async with session:
        rag_pipe = RAGPipeline(graph=graph, llm=session, coprocessor=coproc)
        graphrag_pipe = GraphRAGPipeline(graph=graph, llm=session, coprocessor=coproc)
        agentic_pipe = AgenticPipeline(graph=graph, coprocessor=coproc, llm=session)

        for q in req.questions:
            entry: dict[str, Any] = {"qid": q.qid, "question": q.question, "qtype": q.qtype}
            if req.pipeline in ("rag", "all"):
                r = await rag_pipe.run(q.qid, q.question)
                entry["rag_answer"] = r.answer
                entry["rag_tokens"] = r.total_llm_tokens
            if req.pipeline in ("graphrag", "all"):
                gr = await graphrag_pipe.run(q.qid, q.question)
                entry["graphrag_answer"] = gr.answer
                entry["graphrag_tokens"] = gr.total_llm_tokens
            if req.pipeline in ("agentic", "all"):
                ag = await agentic_pipe.run(q.qid, q.question)
                entry["agentic_answer"] = ag.answer
                entry["agentic_tokens"] = ag.total_llm_tokens
                entry["agentic_trace"] = ag.agentic_trace

            results.append(entry)

    return results


_REPO_ROOT = Path(__file__).resolve().parents[2]
_METRICS_PATH = _REPO_ROOT / "docs" / "metrics" / "public.json"
_PUBLIC_QUESTIONS = _REPO_ROOT / "hackathon-resources" / "questions" / "eval_public.jsonl"


@app.get("/api/v1/metrics")
async def benchmark_metrics() -> dict[str, Any]:
    # The committed benchmark summary that the dashboard renders; never recomputed per request.
    if not _METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No benchmark metrics have been published")
    metrics: dict[str, Any] = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
    return metrics


@app.get("/api/v1/presets")
async def preset_questions() -> list[dict[str, str]]:
    # Public questions for the demo picker, with gold answers stripped.
    if not _PUBLIC_QUESTIONS.exists():
        return []
    presets = []
    for line in _PUBLIC_QUESTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            question = json.loads(line)
            presets.append(
                {key: str(question[key]) for key in ("qid", "qtype", "question") if key in question}
            )
    return presets


@app.get("/api/v1/graph/snapshot", response_model=SnapshotDTO)
async def get_graph_snapshot() -> SnapshotDTO:
    # Do not present example graph data or unmeasured metrics as live results.
    return SnapshotDTO(
        disclosure="Query-specific graph snapshot and measured benchmark metrics are not available yet.",
    )


@app.get("/api/v1/sessions/{session_id}/history")
async def get_session_history(session_id: str) -> dict[str, Any]:
    # Returns chat history and investigation memories for this session
    return {
        "session_id": session_id,
        "created_at": time.time(),
        "messages": [],
        "memories": [],
    }
