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
import hmac
import json
import logging
import os
import time
from collections import Counter, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pyTigerGraph.common.exception import TigerGraphException

from src.api.graph_view import project
from src.api.limits import QuestionBudget, client_id
from src.coprocessor import Coprocessor
from src.credentials import key_report, live_key_count
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
from src.pipelines.intent import classify_intent
from src.pipelines.rag import RAGPipeline
from src.pipelines.toolkit import catalog_for

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

# Investigation views stay readable: a pipeline cites a handful of events, never dozens.
_MAX_FOCUS = 20
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


logger = logging.getLogger(__name__)

# Savanna suspends a workspace after the idle interval set in its console, and only real query
# traffic counts as activity. While the API runs it reads the graph on this beat, so nobody's
# first question waits for a resume. 0 turns the beat off.
_KEEPALIVE_S = float(os.environ.get("STELLIUM_GRAPH_KEEPALIVE_S", "300"))


# What the operator status page reports: when the process started, the last keep-alive beats,
# and the most recent live questions with their timings.
_STARTED_AT = time.time()
_beat: dict[str, Any] = {"last_ok_at": None, "last_error": None, "last_error_at": None}
_recent_questions: deque[dict[str, Any]] = deque(maxlen=25)


async def _keep_graph_awake(interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            venues = await asyncio.to_thread(get_graph().ping)
            _beat["last_ok_at"] = time.time()
            logger.info("graph keep-alive: %s venues", venues)
        except Exception as exc:
            # One missed beat is not an outage; the next one tries again.
            _beat.update(last_error=str(exc)[:200], last_error_at=time.time())
            logger.warning("graph keep-alive failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _coprocessor
    # Startup: load corpus into memory coprocessor
    corpus_path = "hackathon-resources/corpus/corpus.jsonl"
    if os.path.exists(corpus_path):
        chunks = load_all_chunks(corpus_path)
        _coprocessor.build(chunks)

    beat = asyncio.create_task(_keep_graph_awake(_KEEPALIVE_S)) if _KEEPALIVE_S > 0 else None
    yield
    if beat:
        beat.cancel()


app = FastAPI(
    title="Stellium: Agentic GraphRAG API",
    description="TigerGraph Hackathon 2026 — 3-Way Comparative Benchmark & Live Visualization Backend",
    version="0.1.0",
    lifespan=lifespan,
)

# The deployed console reaches the API through its own origin's proxy, so browsers need CORS
# only for local development; any other site calling from a browser is refused.
_CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "STELLIUM_CORS_ORIGINS", "http://localhost:5173,http://localhost:4173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# A question is at most a few kilobytes; anything far larger is refused before it is parsed.
_MAX_BODY_BYTES = 64 * 1024


@app.middleware("http")
async def guard_requests(request: Request, call_next: Any) -> Any:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > _MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


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


def _public_providers() -> set[str]:
    # Visitors use the benchmarked model; other providers open only when listed explicitly.
    listed = os.environ.get("STELLIUM_ALLOWED_PROVIDERS", "")
    names = {name.strip() for name in listed.split(",") if name.strip()} or {_default_provider()}
    return names & _PROVIDERS


class QueryRequest(BaseModel):
    query: str
    qid: str = Field(default="custom-001", max_length=100)
    provider: str = Field(default_factory=_default_provider)


class BatchEvalRequest(BaseModel):
    questions: list[EvalQuestion] = Field(max_length=200)
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
    except (RuntimeError, httpx.HTTPStatusError) as exc:
        # Provider quota or outage after bounded retries: report it, never a fabricated answer.
        # The cause stays in the server log; visitors get no account or key details.
        logger.warning("language model unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The language model is unavailable right now; retry shortly.",
            headers={"Retry-After": "30"},
        ) from exc
    finally:
        _query_slots.release()


_budget = QuestionBudget()


def _screen(req: QueryRequest, request: Request) -> None:
    # Cheap refusals first, so a refused question never spends the daily allowance.
    if req.provider not in _public_providers():
        raise HTTPException(status_code=400, detail=f"Unknown provider {req.provider!r}")
    safe, reason = check_query(req.query, is_batch=False)
    if not safe:
        raise HTTPException(status_code=400, detail=reason)
    _budget.spend(client_id(request))


def _require_admin(request: Request) -> None:
    # Batch evaluation spends the model allowance freely, so it stays hidden without a token.
    token = os.environ.get("STELLIUM_ADMIN_TOKEN", "")
    if not token:
        raise HTTPException(status_code=404, detail="Not Found")
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(status_code=401, detail="A valid admin token is required")


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


def _memory_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def _graph_probe() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        venues = _get_request_graph().ping()
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}
    return {
        "ok": True,
        "venues": venues,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.get("/api/v1/ops/status", include_in_schema=False)
async def operator_status(request: Request) -> dict[str, Any]:
    # The owner's private page: allowances, key health, graph reachability and recent questions.
    # Cohere and Savanna publish no balance API, so key counts are this process's own calls.
    _require_admin(request)
    keys = key_report()
    return {
        "service": {
            "started_at": _STARTED_AT,
            "uptime_s": round(time.time() - _STARTED_AT),
            "commit": os.environ.get("RENDER_GIT_COMMIT", "")[:7] or None,
            "memory_mb": _memory_mb(),
            "provider": _default_provider(),
            "max_concurrent_questions": int(os.environ.get("STELLIUM_MAX_CONCURRENT_QUERIES", "4")),
        },
        "graph": await asyncio.to_thread(_graph_probe),
        "keepalive": {"interval_s": _KEEPALIVE_S, **_beat},
        "questions": _budget.snapshot(),
        "keys": keys,
        "recent": list(_recent_questions),
    }


@app.get("/api/v1/ops/alerts", include_in_schema=False)
async def operator_alerts(request: Request) -> JSONResponse:
    # For an external monitor: 503 names what needs attention, so the monitor's failure email is
    # the alert. Kept apart from the keep-alive so a daily-cap alert never pauses the keep-alive.
    _require_admin(request)
    issues = []
    if not (await asyncio.to_thread(_graph_probe))["ok"]:
        issues.append("graph unreachable")
    if _default_provider() == "cohere" and live_key_count() == 0:
        issues.append("no model key has calls left this month")
    allowance = _budget.snapshot()
    if allowance["daily_cap"] and allowance["spent_today"] >= allowance["daily_cap"]:
        issues.append("daily question cap reached")
    return JSONResponse(
        status_code=503 if issues else 200, content={"ok": not issues, "issues": issues}
    )


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
async def query_rag(req: QueryRequest, request: Request) -> PipelineResult:
    _screen(req, request)

    graph = _get_request_graph()
    async with _admitted(req.provider), make_session(req.provider) as session:
        pipe = RAGPipeline(graph=graph, llm=session, coprocessor=get_coprocessor())
        return await pipe.run(req.qid, req.query)


@app.post("/api/v1/query/graphrag", response_model=PipelineResult)
async def query_graphrag(req: QueryRequest, request: Request) -> PipelineResult:
    _screen(req, request)

    graph = _get_request_graph()
    async with _admitted(req.provider), make_session(req.provider) as session:
        pipe = GraphRAGPipeline(graph=graph, llm=session, coprocessor=get_coprocessor())
        return await pipe.run(req.qid, req.query)


@app.post("/api/v1/query/agentic", response_model=PipelineResult)
async def query_agentic(req: QueryRequest, request: Request) -> PipelineResult:
    _screen(req, request)

    graph = _get_request_graph()
    coproc = get_coprocessor()
    async with _admitted(req.provider), make_session(req.provider) as session:
        pipe = AgenticPipeline(graph=graph, coprocessor=coproc, llm=session)
        return await pipe.run(req.qid, req.query)


def _note_question(
    query: str,
    intent: str,
    started: float,
    tokens: int,
    results: dict[str, PipelineResult] | None = None,
) -> None:
    _recent_questions.appendleft(
        {
            "at": time.time(),
            "question": query[:120],
            "intent": intent,
            "total_ms": round((time.perf_counter() - started) * 1000),
            "llm_tokens": tokens,
            "pipelines": {
                name: {"answer": r.answer[:80], "latency_ms": round(r.latency_ms)}
                for name, r in (results or {}).items()
            },
        }
    )


_LANE_NAMES = ("rag", "graphrag", "agentic")


def _lane(name: str, outcome: PipelineResult | BaseException) -> PipelineResult | None:
    if isinstance(outcome, asyncio.CancelledError):
        raise outcome
    if isinstance(outcome, BaseException):
        logger.warning("%s pipeline failed in compare: %r", name, outcome)
        return None
    return outcome


@app.post("/api/v1/query/compare", response_model=CompareResult)
async def query_compare(req: QueryRequest, request: Request) -> CompareResult:
    # Executes all 3 pipelines side-by-side for comparative judging
    _screen(req, request)

    started = time.perf_counter()
    async with _admitted(req.provider):
        intent = await classify_intent(req.query, req.provider)
    if intent.label == "chat":
        _note_question(req.query, "chat", started, intent.tokens)
        return CompareResult(
            qid=req.qid,
            question=req.query,
            qtype="chat",
            intent="chat",
            intent_tokens=intent.tokens,
        )

    graph = _get_request_graph()
    coproc = get_coprocessor()
    async with (
        _admitted(req.provider),
        make_session(req.provider) as rag_llm,
        make_session(req.provider) as graphrag_llm,
        make_session(req.provider) as agentic_llm,
    ):
        # Same model in every pipeline; separate sessions let the three run side by side. One
        # pipeline's failure blanks its own lane only; the other answers still go out.
        outcomes = await asyncio.gather(
            RAGPipeline(graph=graph, llm=rag_llm, coprocessor=coproc).run(req.qid, req.query),
            GraphRAGPipeline(graph=graph, llm=graphrag_llm, coprocessor=coproc).run(
                req.qid, req.query
            ),
            AgenticPipeline(graph=graph, coprocessor=coproc, llm=agentic_llm).run(
                req.qid, req.query
            ),
            return_exceptions=True,
        )
        lanes = [_lane(name, outcome) for name, outcome in zip(_LANE_NAMES, outcomes, strict=True)]
        if not any(lanes):
            # Nothing to show: the first failure decides the status (graph waking, model down).
            raise next(o for o in outcomes if isinstance(o, BaseException))
    rag_res, graphrag_res, agentic_res = lanes

    _note_question(
        req.query,
        "ask",
        started,
        intent.tokens + sum(r.total_llm_tokens for r in lanes if r),
        {name: r for name, r in zip(_LANE_NAMES, lanes, strict=True) if r},
    )
    return CompareResult(
        qid=req.qid,
        question=req.query,
        qtype=agentic_res.agentic_trace.get("qtype", "general")
        if agentic_res and agentic_res.agentic_trace
        else "general",
        intent_tokens=intent.tokens,
        rag=rag_res,
        graphrag=graphrag_res,
        agentic=agentic_res,
    )


@app.post("/api/v1/evaluate/batch")
async def evaluate_batch(req: BatchEvalRequest, request: Request) -> list[dict[str, Any]]:
    # Batch evaluation: bypasses input guardrails (trusted eval questions, admin token only)
    _require_admin(request)
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
_METRICS_DIR = _REPO_ROOT / "docs" / "metrics"
_METRICS_PATH = _METRICS_DIR / "public.json"
# Published benchmark documents the docs page may read, by name.
_METRIC_SETS = (
    "public",
    "paraphrase",
    "compositional",
    "unanswerable",
    "offtemplate",
    "hidden_oracle",
)
_PUBLIC_QUESTIONS = _REPO_ROOT / "hackathon-resources" / "questions" / "eval_public.jsonl"


@app.get("/api/v1/metrics")
async def benchmark_metrics() -> dict[str, Any]:
    # The committed benchmark summary that the dashboard renders; never recomputed per request.
    if not _METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No benchmark metrics have been published")
    metrics: dict[str, Any] = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
    return metrics


@app.get("/api/v1/metrics/index")
async def published_metric_sets() -> list[str]:
    # Which sets have results yet, so the console only asks for what exists.
    return [name for name in _METRIC_SETS if (_METRICS_DIR / f"{name}.json").exists()]


@app.get("/api/v1/metrics/{name}")
async def named_metrics(name: str) -> dict[str, Any]:
    if name not in _METRIC_SETS or not (_METRICS_DIR / f"{name}.json").exists():
        raise HTTPException(status_code=404, detail=f"No published metrics named {name!r}")
    document: dict[str, Any] = json.loads((_METRICS_DIR / f"{name}.json").read_text("utf-8"))
    return document


@app.get("/api/v1/graph/stats")
async def graph_stats() -> dict[str, Any]:
    # Live sizes for the docs page, read from the loaded catalog and index, not hard-coded.
    catalog = catalog_for(_get_request_graph())
    series = Counter((r.sport, r.season, r.label) for r in catalog.records if r.year)
    return {
        "events": len(catalog.records),
        "games": len({(r.year, r.season) for r in catalog.records if r.year}),
        "sports": len(catalog.sports),
        "venues": len({r.venue for r in catalog.records if r.venue}),
        # Events with an earlier edition in the corpus: the PRECEDES hops the agent can take.
        "previous_edition_links": sum(count - 1 for count in series.values()),
        **get_coprocessor().stats(),
    }


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
async def get_graph_snapshot(
    view: Literal["constellation", "venues", "lineage", "investigation"] = "constellation",
    focus: str = "",
    sport: str = "",
) -> SnapshotDTO:
    # The canvas payload, projected from the graph-loaded catalog. `focus` is a comma-separated
    # list of event ids, normally a pipeline's citations, for the investigation view.
    catalog = catalog_for(_get_request_graph())
    event_ids = [part.strip() for part in focus.split(",") if part.strip()][:_MAX_FOCUS]
    return project(catalog, view, focus=event_ids, sport=sport or None)


@app.get("/api/v1/sessions/{session_id}/history")
async def get_session_history(session_id: str) -> dict[str, Any]:
    # Returns chat history and investigation memories for this session
    return {
        "session_id": session_id,
        "created_at": time.time(),
        "messages": [],
        "memories": [],
    }
