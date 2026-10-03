// Question in, three pipelines out: the same question answered by RAG, GraphRAG and the agent,
// side by side with their LLM tokens, latency and citations.
import { useMemo, useState } from "react";
import type { CompareResult, Preset } from "../api";
import { PIPELINES, isNotFound, seconds, tokens } from "../format";

// The agent's answer leads; the two baselines follow for comparison.
const LANES = PIPELINES;

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
  multi_hop: "Venue & date",
  temporal: "Previous Games",
  aggregation: "Count",
  superlative: "Rank",
  lookup: "Lookup",
};

export function AskPanel({ initialQuestion, presets, result, busy, error, onAsk, onFocus }: Props) {
  const [question, setQuestion] = useState(initialQuestion);
  const [qtype, setQtype] = useState<string>("multi_hop");
  const byType = useMemo(() => {
    const groups = new Map<string, Preset[]>();
    for (const preset of presets) groups.set(preset.qtype, [...(groups.get(preset.qtype) ?? []), preset]);
    return groups;
  }, [presets]);
  const types = Object.keys(QTYPE_NAMES).filter((type) => byType.has(type));
  const samples = (byType.get(qtype) ?? []).slice(0, 3);
  const submit = () => question.trim() && !busy && onAsk(question.trim());
  // Once answers are on screen the examples fold away to give them the room.
  const [examplesOpen, setExamplesOpen] = useState(false);
  const chat = result?.intent === "chat" && !busy;
  const answering = Boolean(result || busy) && !chat;
  // One example of each kind of question the graph answers, for the welcome card.
  const starters = ["multi_hop", "temporal", "superlative"].flatMap((type) => byType.get(type)?.slice(0, 1) ?? []);
  // The welcome card carries its own examples, so the list stays folded behind it.
  const showExamples = (!answering && !chat) || examplesOpen;

  return (
    <section className="plate pointer-events-auto flex max-h-full w-full flex-col overflow-hidden">
      <div className="px-4 pt-4 pb-4">
        <div className="mb-2.5 flex items-baseline justify-between px-0.5">
          <label htmlFor="ask" className="text-[13.5px] font-semibold">
            Ask the Olympic graph
          </label>
          <span className="kbd">Ctrl ↵ to send</span>
        </div>
        <div className="ask-field">
          <textarea
            id="ask"
            className="ask-input"
            rows={3}
            value={question}
            placeholder="Type a question, or pick an example below…"
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) submit();
            }}
          />
          <div className="flex items-center justify-between gap-2 px-3 pt-1 pb-2.5">
            <span className="kbd">One question, three pipelines</span>
            <button className="ask-submit" disabled={busy || !question.trim()} onClick={submit}>
              {busy ? "Asking…" : "Ask"}
              <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
                <path d="M3 8h9.5M8.5 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
        </div>

        {types.length > 0 && (answering || chat) && (
          <button className="examples-toggle" aria-expanded={showExamples} onClick={() => setExamplesOpen((open) => !open)}>
            {showExamples ? "Hide examples" : "Examples"}
          </button>
        )}
        {types.length > 0 && showExamples && (
          <div className="mt-3.5">
            <div className="segmented no-scrollbar max-w-full overflow-x-auto">
              {types.map((type) => (
                <button key={type} aria-pressed={type === qtype} onClick={() => setQtype(type)}>
                  {QTYPE_NAMES[type]}
                </button>
              ))}
            </div>
            <ul className="mt-2 grid grid-cols-[minmax(0,1fr)] px-0.5">
              {samples.map((preset) => (
                <li key={preset.qid}>
                  <button
                    className="sample"
                    title={preset.question}
                    onClick={() => {
                      setQuestion(preset.question);
                      setExamplesOpen(false);
                    }}
                  >
                    {preset.question}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {error && (
        <div className="lane" style={{ ["--lane" as string]: "var(--destructive)" }}>
          <span className="lane-name">Unavailable</span>
          <div className="lane-answer text-[14px] text-[color:var(--soft)]">{error}</div>
        </div>
      )}

      {chat && (
        <div className="welcome">
          <div className="welcome-title">Hello. Ask me about the Olympics.</div>
          <p className="welcome-body">
            Every answer comes from a TigerGraph graph of Olympic events and the Wikipedia articles behind them,
            worked out three ways side by side. Try one of these:
          </p>
          <ul className="grid gap-1.5">
            {starters.map((preset) => (
              <li key={preset.qid}>
                <button
                  className="welcome-sample"
                  title={preset.question}
                  onClick={() => {
                    setQuestion(preset.question);
                    onAsk(preset.question);
                  }}
                >
                  <span className="welcome-sample-kind">{QTYPE_NAMES[preset.qtype]}</span>
                  <span className="truncate">{preset.question}</span>
                </button>
              </li>
            ))}
          </ul>
          <div className="welcome-meta">
            Intent check · {tokens(result?.intent_tokens ?? 0)} LLM tokens · no pipeline needed
          </div>
        </div>
      )}

      {answering && (
        <div className="no-scrollbar overflow-y-auto">
          {LANES.map((pipeline) => {
            const run = result?.[pipeline.id];
            const calls = run?.agentic_trace?.llm_calls.length;
            return (
              <article key={pipeline.id} className="lane" style={{ ["--lane" as string]: pipeline.color }}>
                <div className="flex items-baseline justify-between">
                  <span className="lane-name">{pipeline.name}</span>
                  {run && <span className="kbd">{seconds(run.latency_ms)}</span>}
                </div>
                <div className="lane-answer" style={{ opacity: run ? 1 : 0.4 }}>
                  {busy ? "Thinking…" : run ? run.answer : result ? "No answer this time. Ask again in a moment." : null}
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
                      {pipeline.id === "graphrag" &&
                        typeof run.retrieval_metadata.graph_operation === "string" &&
                        run.retrieval_metadata.graph_operation !== "none" && (
                        <span>
                          via <b>{run.retrieval_metadata.graph_operation}</b>
                        </span>
                      )}
                    </div>
                    {run.retrieved_doc_ids.length > 0 && !isNotFound(run.answer) && (
                      <div className="mt-2.5 flex flex-wrap gap-1.5">
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
      )}
    </section>
  );
}
