// The agent's reasoning as it happened: each thought, the tool it chose, what the graph returned,
// and the source line that confirmed the value. Graph steps cost zero LLM tokens.
import type { AgentTrace } from "../api";
import { traceSteps } from "../format";

export function TracePanel({ trace }: { trace: AgentTrace | null }) {
  if (!trace) return null;
  const steps = traceSteps(trace.conversation_trace ?? []);
  const llmTokens = trace.llm_calls.reduce((sum, call) => sum + call.prompt_tokens + call.completion_tokens, 0);

  return (
    <section className="plate pointer-events-auto flex max-h-full w-full flex-col overflow-hidden">
      <header className="flex items-baseline justify-between px-4 pt-4 pb-3">
        <span className="tag" style={{ color: "var(--agentic)" }}>
          Agent trace
        </span>
        <span className="tag">
          {trace.llm_calls.length} LLM · {trace.tools_called.length} graph · {llmTokens.toLocaleString("en-US")} tok
        </span>
      </header>
      <ol className="no-scrollbar overflow-y-auto">
        {steps.map((step) => (
          <li key={step.index} className="trace-step">
            <span className="trace-index">{String(step.index).padStart(2, "0")}</span>
            <div>
              <div className="trace-thought">{step.thought}</div>
              <div className="trace-call">
                {step.action} <span>{step.input}</span>
              </div>
              {step.observation && (
                <div className="trace-obs">
                  {step.observation}
                  {step.latencyMs !== null && ` · ${Math.round(step.latencyMs)} ms · 0 tok`}
                </div>
              )}
              {step.sourceLine && <div className="trace-obs">source line → “{step.sourceLine}”</div>}
              {step.note && <div className="trace-note">{step.note}</div>}
            </div>
          </li>
        ))}
      </ol>
      <footer className="border-t border-[color:var(--hair)] px-4 py-3">
        <div className="tag">Specialists</div>
        <div className="mt-1 text-[12px] leading-relaxed text-[color:var(--soft)]">{trace.agents_invoked.join(" · ")}</div>
        <div className="tag mt-2">Stopped because</div>
        <div className="mt-1 text-[12.5px] text-foreground">{trace.stopping_reason}</div>
        {trace.strategy_changed && trace.strategy_change_rationale && (
          <div className="mt-1 text-[12px] text-[color:var(--soft)]">{trace.strategy_change_rationale}</div>
        )}
      </footer>
    </section>
  );
}
