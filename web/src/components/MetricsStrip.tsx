// The headline numbers on the first screen, read from the published benchmark documents.
import type { Metrics, OracleSummary } from "../api";
import { tokens } from "../format";

export interface BenchData {
  public: Metrics | null;
  paraphrase: Metrics | null;
  compositional: Metrics | null;
  unanswerable: Metrics | null;
  offtemplate: Metrics | null;
  oracle: OracleSummary | null;
}

function score(metrics: Metrics | null, pipeline: "rag" | "graphrag" | "agentic") {
  const overall = metrics?.pipelines[pipeline]?.overall;
  return overall ? { value: overall.exact_match, of: overall.scored } : null;
}

export function MetricsStrip({ data, onOpen }: { data: BenchData; onOpen: () => void }) {
  const agent = score(data.public, "agentic");
  const graph = score(data.public, "graphrag");
  const rag = score(data.public, "rag");
  const hidden = data.oracle?.hidden.pipelines.agentic;
  const comp = score(data.compositional, "agentic");
  const para = score(data.paraphrase, "agentic");
  const overall = data.public?.pipelines.agentic?.overall;
  type Cell = { label: string; value: string; unit?: string; color?: string };
  const candidates: (Cell | null | undefined | false)[] = [
    agent && { label: "Agentic · public", value: `${agent.value}`, unit: `/ ${agent.of}`, color: "var(--agentic)" },
    graph && { label: "GraphRAG · public", value: `${graph.value}`, unit: `/ ${graph.of}`, color: "var(--graphrag)" },
    rag && { label: "RAG · public", value: `${rag.value}`, unit: `/ ${rag.of}`, color: "var(--rag)" },
    hidden && { label: "Hidden 50 · oracle", value: `${hidden.agree}`, unit: `/ ${hidden.scored}` },
    comp && { label: "Two-step", value: `${comp.value}`, unit: `/ ${comp.of}` },
    para && { label: "Paraphrased", value: `${para.value}`, unit: `/ ${para.of}` },
    overall && { label: "Agent tokens / q", value: tokens(overall.tokens_mean) },
    data.public && { label: "LLM calls / q", value: data.public.agent.llm_calls_mean.toFixed(2) },
  ];
  const cells = candidates.filter((cell): cell is Cell => Boolean(cell));

  if (!cells.length) return null;
  return (
    <button className="strip pointer-events-auto text-left" onClick={onOpen} title="Open the full benchmark">
      {cells.map((cell) => (
        <div key={cell.label} className="strip-cell">
          <div className="strip-label" style={cell.color ? { color: cell.color } : undefined}>
            {cell.label}
          </div>
          <div className="strip-value">
            {cell.value}
            {cell.unit && <small>{cell.unit}</small>}
          </div>
        </div>
      ))}
    </button>
  );
}
