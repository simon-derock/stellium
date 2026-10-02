// Pure presentation helpers, kept apart from components so they can be unit tested.
import type { CompareResult, PipelineId, TraceEvent } from "./api";

// Best first: the agent leads every table, card, chart and answer list; RAG, the baseline, closes.
export const PIPELINES: { id: PipelineId; name: string; color: string }[] = [
  { id: "agentic", name: "Agentic GraphRAG", color: "var(--agentic)" },
  { id: "graphrag", name: "GraphRAG", color: "var(--graphrag)" },
  { id: "rag", name: "RAG", color: "var(--rag)" },
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
export function isNotFound(answer: string): boolean {
  return /^not found in corpus\.?$/i.test(answer.trim());
}

export function citedEvents(result: CompareResult | null): string[] {
  if (!result) return [];
  // A pipeline that found nothing retrieved near misses, not evidence; leave those out.
  const ids = [result.agentic, result.graphrag, result.rag]
    .filter((r): r is NonNullable<typeof r> => r !== null && !isNotFound(r.answer))
    .flatMap((r) => r.retrieved_doc_ids);
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

// ---------------------------------------------------------------------------
// Theme legibility: the canvas themes were designed for the graph, not for UI chrome. Work out
// whether a theme is light or dark, and lift an accent that is too faint to read as text.
// ---------------------------------------------------------------------------

function channels(hex: string): [number, number, number] | null {
  const value = hex.trim().replace("#", "");
  const full = value.length === 3 ? value.split("").map((c) => c + c).join("") : value;
  if (!/^[0-9a-f]{6}$/i.test(full)) return null;
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255) as [number, number, number];
}

export function luminance(hex: string): number | null {
  const rgb = channels(hex);
  if (!rgb) return null;
  const [r, g, b] = rgb.map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a: string, b: string): number {
  const la = luminance(a);
  const lb = luminance(b);
  if (la === null || lb === null) return 21;
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

export function chromeFor(vars: Record<string, string>): { scheme: "light" | "dark"; overrides: Record<string, string> } {
  const background = vars["--background"] ?? "#08090a";
  const accent = vars["--accent"] ?? "#cba36a";
  const scheme = (luminance(background) ?? 0) > 0.4 ? "light" : "dark";
  const overrides: Record<string, string> = {};
  // Below 3:1 an accent cannot carry a label; pull it towards the theme's ink until it can.
  if (contrast(accent, background) < 3) {
    overrides["--accent"] = `color-mix(in oklab, ${accent} 55%, var(--foreground))`;
  }
  return { scheme, overrides };
}
