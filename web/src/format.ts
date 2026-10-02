// Pure presentation helpers, kept apart from components so they can be unit tested.
import type { CompareResult, PipelineId, TraceEvent } from "./api";

export const PIPELINES: { id: PipelineId; name: string; color: string }[] = [
  { id: "rag", name: "RAG", color: "var(--rag)" },
  { id: "graphrag", name: "GraphRAG", color: "var(--graphrag)" },
  { id: "agentic", name: "Agentic GraphRAG", color: "var(--agentic)" },
];

export function tokens(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

export function seconds(ms: number): string {
  return ms >= 10_000 ? `${(ms / 1000).toFixed(0)} s` : `${(ms / 1000).toFixed(1)} s`;
}

export function percent(part: number, whole: number): string {
  return whole ? `${Math.round((100 * part) / whole)}%` : "n/a";
}

// Every event a result cited, in order of first appearance, for the investigation view.
export function citedEvents(result: CompareResult | null): string[] {
  if (!result) return [];
  const ids = [result.agentic, result.graphrag, result.rag].flatMap((r) => r.retrieved_doc_ids);
  return [...new Set(ids)];
}

export interface TraceStep {
  index: number;
  thought: string;
  action: string;
  input: string;
  observation: string;
  note: string | null;
  sourceLine: string | null;
  latencyMs: number | null;
}

// Pair each planning event with the observation that followed it.
export function traceSteps(events: TraceEvent[]): TraceStep[] {
  const steps: TraceStep[] = [];
  for (const event of events) {
    if (event.event === "plan") {
      steps.push({
        index: steps.length + 1,
        thought: event.thought ?? "",
        action: event.final_answer ? "finish" : (event.action ?? ""),
        input: event.final_answer ?? compactJson(event.action_input ?? {}),
        observation: "",
        note: null,
        sourceLine: null,
        latencyMs: null,
      });
      continue;
    }
    const step = steps[steps.length - 1];
    if (!step || !event.observation) continue;
    const { note, source_check: sourceCheck, linking: _linking, ...rest } = event.observation;
    step.observation = summarize(rest);
    step.note = typeof note === "string" ? note : null;
    const firstSource = Array.isArray(sourceCheck) ? sourceCheck[0] : null;
    step.sourceLine =
      firstSource && typeof firstSource === "object" && "line" in firstSource
        ? String((firstSource as { line: unknown }).line)
        : null;
    step.latencyMs = event.latency_ms ?? null;
  }
  return steps;
}

function compactJson(value: Record<string, unknown>): string {
  return Object.entries(value)
    .map(([key, item]) => `${key}: ${typeof item === "string" ? item : JSON.stringify(item)}`)
    .join(" · ");
}

// One readable line per observation: the value when there is one, else the first few fields.
export function summarize(observation: Record<string, unknown>, limit = 180): string {
  if ("value" in observation && observation.value != null) return `value → ${String(observation.value)}`;
  if ("error" in observation) return `error → ${String(observation.error)}`;
  if (Array.isArray(observation.top) && observation.top.length) {
    const top = observation.top[0] as { event?: string; competitor_count?: number };
    return `top → ${top.event ?? "?"} (${top.competitor_count ?? "?"})`;
  }
  const text = compactJson(observation);
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}
