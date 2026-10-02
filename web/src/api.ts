// Typed client for the STELLIUM API. In development Vite proxies /api; in production
// VITE_API_BASE points at the backend host.
import type { Snapshot, ViewId } from "./graph/graph";

export const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export type PipelineId = "rag" | "graphrag" | "agentic";

export interface TraceEvent {
  event: "plan" | "observation";
  thought?: string;
  action?: string;
  action_input?: Record<string, unknown>;
  final_answer?: string | null;
  step?: number;
  tool?: string;
  latency_ms?: number;
  observation?: Record<string, unknown>;
}

export interface AgentTrace {
  step_count: number;
  agents_invoked: string[];
  llm_calls: { prompt_tokens: number; completion_tokens: number; latency_ms: number }[];
  tools_called: { step: number; tool_name: string; latency_ms: number }[];
  citations: string[];
  strategy_changed: boolean;
  strategy_change_rationale: string | null;
  stopping_reason: string;
  confidence_score: number;
  conversation_trace: TraceEvent[];
}

export interface PipelineResult {
  qid: string;
  pipeline: PipelineId;
  question: string;
  answer: string;
  llm_input_tokens: number;
  llm_output_tokens: number;
  total_llm_tokens: number;
  latency_ms: number;
  retrieved_doc_ids: string[];
  retrieval_metadata: Record<string, unknown>;
  agentic_trace: AgentTrace | null;
}

export interface CompareResult {
  qid: string;
  question: string;
  rag: PipelineResult;
  graphrag: PipelineResult;
  agentic: PipelineResult;
}

export interface Preset {
  qid: string;
  qtype: string;
  question: string;
}

export interface Overall {
  questions: number;
  scored: number;
  exact_match: number;
  exact_match_strict: number;
  token_f1: number;
  tokens_mean: number;
  latency_mean_s: number;
  latency_p50_s: number;
  latency_p95_s: number;
  cost_per_100_usd: number;
  citations_mean: number;
}

export interface PipelineMetrics {
  label: string;
  overall: Overall;
  by_type: Record<string, { questions: number; exact_match: number; tokens_mean: number }>;
}

export interface Metrics {
  pipelines: Record<PipelineId, PipelineMetrics>;
  agent: {
    questions: number;
    steps_mean: number;
    llm_calls_mean: number;
    single_call_answers: number;
    strategy_changes: number;
    tools: Record<string, number>;
    specialists: Record<string, number>;
  };
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON error bodies keep the status text.
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export function fetchSnapshot(view: ViewId, focus: string[] = []): Promise<Snapshot> {
  const params = new URLSearchParams({ view });
  if (focus.length) params.set("focus", focus.join(","));
  return request<Snapshot>(`/api/v1/graph/snapshot?${params}`);
}

export function compare(query: string): Promise<CompareResult> {
  return request<CompareResult>("/api/v1/query/compare", {
    method: "POST",
    body: JSON.stringify({ query, qid: "live" }),
  });
}

export const fetchPresets = () => request<Preset[]>("/api/v1/presets");
export const fetchMetrics = () => request<Metrics>("/api/v1/metrics");
export const fetchHealth = (deep = false) =>
  request<{ status: string; graph?: unknown }>(`/health${deep ? "?deep=true" : ""}`);
