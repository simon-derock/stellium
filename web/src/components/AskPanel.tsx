// Question in, three pipelines out: the same question answered by RAG, GraphRAG and the agent,
// side by side with their LLM tokens, latency and citations.
import { useMemo, useState } from "react";
import type { CompareResult, Preset } from "../api";
import { PIPELINES, seconds, tokens } from "../format";

interface Props {
  initialQuestion: string;
  presets: Preset[];
  result: CompareResult | null;
  busy: boolean;
  error: string | null;
  onAsk: (question: string) => void;
  onFocus: (eventIds: string[]) => void;
}

const QTYPE_NAMES: Record<string, string> = {
  aggregation: "Count",
  superlative: "Rank",
  lookup: "Lookup",
  temporal: "Previous Games",
  multi_hop: "Venue and date",
};

export function AskPanel({ initialQuestion, presets, result, busy, error, onAsk, onFocus }: Props) {
  const [question, setQuestion] = useState(initialQuestion);
  const [qtype, setQtype] = useState<string>("multi_hop");
  const byType = useMemo(() => {
    const groups = new Map<string, Preset[]>();
    for (const preset of presets) groups.set(preset.qtype, [...(groups.get(preset.qtype) ?? []), preset]);
    return groups;
  }, [presets]);
  const samples = (byType.get(qtype) ?? []).slice(0, 4);
  const submit = () => question.trim() && !busy && onAsk(question.trim());

  return (
    <section className="plate pointer-events-auto flex max-h-full w-full flex-col overflow-hidden">
      <div className="px-4 pt-3.5 pb-3">
        <div className="tag">Ask the Olympic graph</div>
        <textarea
          className="ask-input mt-2"
          rows={3}
          value={question}
          placeholder="Who won gold in the event held at Richmond Olympic Oval on 14 February 2010?"
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) submit();
          }}
        />
        <div className="mt-2 flex items-center justify-between gap-3">
          <div className="flex flex-wrap gap-1">
            {[...byType.keys()].map((type) => (
              <button
                key={type}
                className="chip"
                style={type === qtype ? { color: "var(--foreground)", borderColor: "var(--accent)" } : undefined}
                onClick={() => setQtype(type)}
              >
                {QTYPE_NAMES[type] ?? type}
              </button>
            ))}
          </div>
          <button className="ask-submit" disabled={busy || !question.trim()} onClick={submit}>
            {busy ? "Asking…" : "Ask all three"}
          </button>
        </div>
        {samples.length > 0 && (
          <ul className="mt-2.5 grid gap-1">
            {samples.map((preset) => (
              <li key={preset.qid}>
                <button
                  className="w-full truncate text-left font-mono text-[10px] text-muted-foreground transition-colors hover:text-foreground"
                  title={preset.question}
                  onClick={() => setQuestion(preset.question)}
                >
                  {preset.question}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {error && (
        <div className="lane" style={{ ["--lane" as string]: "var(--destructive)" }}>
          <div className="tag">Unavailable</div>
          <div className="lane-answer text-[15px]">{error}</div>
        </div>
      )}

      <div className="no-scrollbar overflow-y-auto">
        {PIPELINES.map((pipeline) => {
          const run = result?.[pipeline.id];
          const calls = run?.agentic_trace?.llm_calls.length;
          return (
            <article key={pipeline.id} className="lane" style={{ ["--lane" as string]: pipeline.color }}>
              <div className="flex items-baseline justify-between">
                <span className="tag" style={{ color: pipeline.color }}>
                  {pipeline.name}
                </span>
                {run && <span className="tag">{seconds(run.latency_ms)}</span>}
              </div>
              <div className="lane-answer" style={{ opacity: run ? 1 : 0.35 }}>
                {busy ? "…" : (run?.answer ?? "Waiting for a question")}
              </div>
              {run && (
                <>
                  <div className="lane-meta">
                    <span>
                      <b>{tokens(run.total_llm_tokens)}</b> LLM tokens
                    </span>
                    {calls !== undefined && (
                      <span>
                        <b>{calls}</b> LLM {calls === 1 ? "call" : "calls"}
                      </span>
                    )}
                    {pipeline.id === "graphrag" && typeof run.retrieval_metadata.graph_operation === "string" && (
                      <span>
                        graph op <b>{run.retrieval_metadata.graph_operation}</b>
                      </span>
                    )}
                  </div>
                  {run.retrieved_doc_ids.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {run.retrieved_doc_ids.slice(0, 5).map((doc) => (
                        <button key={doc} className="chip" onClick={() => onFocus([doc])} title="Show in the graph">
                          {doc}
                        </button>
                      ))}
                    </div>
                  )}
                </>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
