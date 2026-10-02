// The measured benchmark, read from the committed metrics document the API serves.
import type { Metrics, PipelineId } from "../api";
import { PIPELINES, percent, seconds, tokens } from "../format";

const QTYPE_NAMES: Record<string, string> = {
  aggregation: "Aggregation · count events over a threshold",
  superlative: "Superlative · event with the most competitors",
  lookup: "Lookup · one attribute of one event",
  temporal: "Temporal · the previous Games",
  multi_hop: "Multi-hop · venue and date to medallist",
};

export function BenchmarkSheet({ metrics }: { metrics: Metrics | null }) {
  if (!metrics) {
    return (
      <section className="sheet pointer-events-auto p-6">
        <div className="tag">Loading the measured benchmark…</div>
      </section>
    );
  }
  const ids = PIPELINES.filter((p) => metrics.pipelines[p.id]);
  const qtypes = Object.keys(metrics.pipelines.agentic?.by_type ?? {});
  const agent = metrics.agent;
  const toolTotal = Object.values(agent.tools).reduce((a, b) => a + b, 0);

  return (
    <section className="sheet no-scrollbar pointer-events-auto max-h-full overflow-y-auto">
      <header className="flex flex-wrap items-end justify-between gap-3 border-b border-border px-6 pt-5 pb-4">
        <div>
          <div className="tag">Benchmark</div>
          <h2 className="mt-1.5 font-[family-name:var(--font-display)] text-[30px] leading-none">
            100 public questions, three pipelines, one model
          </h2>
        </div>
        <p className="max-w-[26rem] font-mono text-[10px] leading-relaxed text-muted-foreground">
          Exact match after normalising case, punctuation and diacritics. Tokens are LLM input plus
          output as the provider bills them; graph queries and retrieval count zero.
        </p>
      </header>

      <div className="grid grid-cols-1 border-b border-border md:grid-cols-3">
        {ids.map((pipeline) => {
          const o = metrics.pipelines[pipeline.id as PipelineId].overall;
          return (
            <div key={pipeline.id} className="border-border px-6 py-5 md:border-r md:last:border-r-0">
              <div className="tag" style={{ color: pipeline.color }}>
                {pipeline.name}
              </div>
              <div className="figure mt-3">
                {o.exact_match.toFixed(0)}
                <small>/ {o.scored}</small>
              </div>
              <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 font-mono text-[10px]">
                <dt className="text-muted-foreground">Strict match</dt>
                <dd className="text-right">{percent(o.exact_match_strict, o.scored)}</dd>
                <dt className="text-muted-foreground">LLM tokens / q</dt>
                <dd className="text-right">{tokens(o.tokens_mean)}</dd>
                <dt className="text-muted-foreground">Latency p50 / p95</dt>
                <dd className="text-right">
                  {seconds(o.latency_p50_s * 1000)} / {seconds(o.latency_p95_s * 1000)}
                </dd>
                <dt className="text-muted-foreground">Cost / 100 q</dt>
                <dd className="text-right">${o.cost_per_100_usd.toFixed(2)}</dd>
              </dl>
            </div>
          );
        })}
      </div>

      <div className="px-6 py-5">
        <table className="ledger">
          <thead>
            <tr>
              <th>Question type</th>
              {ids.map((p) => (
                <th key={p.id} className="num" style={{ color: p.color }}>
                  {p.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {qtypes.map((qtype) => (
              <tr key={qtype}>
                <td>{QTYPE_NAMES[qtype] ?? qtype}</td>
                {ids.map((p) => {
                  const group = metrics.pipelines[p.id as PipelineId].by_type[qtype];
                  if (!group) return <td key={p.id} className="num">n/a</td>;
                  return (
                    <td key={p.id} className="num" style={{ ["--lane" as string]: p.color }}>
                      <div>
                        {group.exact_match.toFixed(0)}/{group.questions}
                        <span className="text-muted-foreground"> · {tokens(group.tokens_mean)} tok</span>
                      </div>
                      <div className="bar mt-1.5 ml-auto w-28">
                        <i style={{ width: `${(100 * group.exact_match) / group.questions}%` }} />
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 gap-6 border-t border-border px-6 py-5 md:grid-cols-[1fr_1.4fr]">
        <div>
          <div className="tag" style={{ color: "var(--agentic)" }}>
            How the agent worked
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 font-mono text-[10.5px]">
            <dt className="text-muted-foreground">LLM calls / question</dt>
            <dd className="text-right">{agent.llm_calls_mean.toFixed(2)}</dd>
            <dt className="text-muted-foreground">Steps / question</dt>
            <dd className="text-right">{agent.steps_mean.toFixed(2)}</dd>
            <dt className="text-muted-foreground">Answered in one LLM call</dt>
            <dd className="text-right">
              {agent.single_call_answers}/{agent.questions}
            </dd>
            <dt className="text-muted-foreground">Changed strategy</dt>
            <dd className="text-right">
              {agent.strategy_changes}/{agent.questions}
            </dd>
          </dl>
        </div>
        <div>
          <div className="tag">Tools it chose</div>
          <ul className="mt-3 grid gap-2">
            {Object.entries(agent.tools)
              .sort((a, b) => b[1] - a[1])
              .map(([tool, count]) => (
                <li key={tool} className="grid grid-cols-[9rem_1fr_2rem] items-center gap-3 font-mono text-[10px]">
                  <span>{tool}</span>
                  <span className="bar" style={{ ["--lane" as string]: "var(--agentic)" }}>
                    <i style={{ width: `${(100 * count) / Math.max(1, toolTotal)}%` }} />
                  </span>
                  <span className="text-right text-muted-foreground">{count}</span>
                </li>
              ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
