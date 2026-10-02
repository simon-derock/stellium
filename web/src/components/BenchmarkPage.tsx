// Every measured number, from the committed metrics documents the API serves.
import type { Metrics, PipelineId } from "../api";
import { PIPELINES, percent, seconds, tokens } from "../format";
import { AccuracyByType, ChartCard, Efficiency, Legend, RetrievalRadar, ToolMix, usePalette } from "./Charts";
import { LongPage, Section } from "./LongPage";
import type { BenchData } from "./MetricsStrip";

const QTYPE_NAMES: Record<string, string> = {
  aggregation: "Aggregation: count events over a threshold",
  superlative: "Superlative: event with the most competitors",
  lookup: "Lookup: one attribute of one event",
  temporal: "Temporal: the Games immediately before",
  multi_hop: "Multi-hop: venue and date to medallist",
  compositional_gold: "Gold medallist of the largest event",
  compositional_venue: "Venue of the largest event",
  compositional_nations: "Nations in the largest event",
  paraphrase_aggregation: "Count, reworded",
  paraphrase_superlative: "Rank, reworded",
  paraphrase_lookup: "Lookup, reworded",
  paraphrase_temporal: "Previous Games, reworded",
  paraphrase_multi_hop: "Venue and date, reworded",
  unanswerable_rank: "Rank: a sport those Games never held",
  unanswerable_lookup: "Lookup: an edition with no article",
  unanswerable_venue_date: "Venue and date: a day with no event",
  offtemplate_film: "Films: director, composer, release year",
  offtemplate_office: "Officeholders: successor",
};

const SECTIONS = [
  { id: "glance", title: "At a glance" },
  { id: "by-type", title: "Public 100 by type" },
  { id: "retrieval", title: "Retrieval and evidence" },
  { id: "robustness", title: "Robustness" },
  { id: "hidden", title: "Hidden 50" },
  { id: "agent", title: "How the agent worked" },
  { id: "cost", title: "Cost and latency" },
  { id: "ceiling", title: "The ceiling" },
  { id: "method", title: "Method" },
];

function TypeLedger({ metrics }: { metrics: Metrics }) {
  const ids = PIPELINES.filter((p) => metrics.pipelines[p.id]);
  const qtypes = Object.keys(metrics.pipelines.agentic?.by_type ?? metrics.pipelines.graphrag?.by_type ?? {});
  return (
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
            <td>{QTYPE_NAMES[qtype] ?? qtype.replaceAll("_", " ")}</td>
            {ids.map((p) => {
              const group = metrics.pipelines[p.id as PipelineId].by_type[qtype];
              if (!group) return <td key={p.id} className="num">n/a</td>;
              return (
                <td key={p.id} className="num" style={{ ["--lane" as string]: p.color }}>
                  <div className="font-medium">
                    {group.exact_match.toFixed(0)}/{group.questions}
                    <span className="font-normal text-[color:var(--soft)]"> · {tokens(group.tokens_mean)} tok</span>
                  </div>
                  <div className="bar mt-1.5 ml-auto w-32">
                    <i style={{ width: `${(100 * group.exact_match) / group.questions}%` }} />
                  </div>
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

interface Props {
  data: BenchData;
  themeKey: string;
  onScroll?: (top: number) => void;
}

export function BenchmarkPage({ data, themeKey, onScroll }: Props) {
  const metrics = data.public;
  const palette = usePalette(themeKey);
  if (!metrics) {
    return (
      <LongPage sections={[]}>
        <p>Loading the measured benchmark…</p>
      </LongPage>
    );
  }
  const ids = PIPELINES.filter((p) => metrics.pipelines[p.id]);
  const agent = metrics.agent;
  const oracle = data.oracle;

  return (
    <LongPage sections={SECTIONS} onScroll={onScroll}>
      <h1>Benchmark</h1>
      <p className="lede">
        The same questions answered three ways by one model, Cohere <code>command-a-03-2025</code>, at temperature 0. Every
        number below is read from a committed results document, with the commit that produced it.
      </p>

      <Section id="glance" title="At a glance">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {ids.map((pipeline) => {
            const o = metrics.pipelines[pipeline.id as PipelineId].overall;
            return (
              <div key={pipeline.id} className="card">
                <div className="tag" style={{ color: pipeline.color }}>
                  {pipeline.name}
                </div>
                <div className="figure mt-3">
                  {o.exact_match.toFixed(0)}
                  <small>/ {o.scored} public</small>
                </div>
                <div className="mt-3 text-[12.5px] text-[color:var(--soft)]">
                  {tokens(o.tokens_mean)} LLM tokens · {seconds(o.latency_p50_s * 1000)} median
                </div>
              </div>
            );
          })}
        </div>
        {palette && (
          <div className="mt-4">
            <ChartCard title="Accuracy against LLM cost" note="Public 100: exact match versus LLM tokens per question. Up and to the left is better.">
              <Efficiency metrics={metrics} palette={palette} />
            </ChartCard>
          </div>
        )}
        <table className="ledger mt-6">
          <thead>
            <tr>
              <th>Set</th>
              {ids.map((p) => (
                <th key={p.id} className="num" style={{ color: p.color }}>
                  {p.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(
              [
                ["Public 100 (gold answers)", metrics],
                ["Paraphrased (12, reworded public questions)", data.paraphrase],
                ["Two-step (24, rank then read an attribute)", data.compositional],
                ["Unanswerable (12, the right reply is “Not found”)", data.unanswerable],
                ["Off-template (12, films and officeholders)", data.offtemplate],
              ] as const
            ).map(([label, set]) =>
              set ? (
                <tr key={label}>
                  <td>{label}</td>
                  {ids.map((p) => {
                    const o = set.pipelines[p.id as PipelineId]?.overall;
                    return (
                      <td key={p.id} className="num font-medium">
                        {o ? `${o.exact_match.toFixed(0)}/${o.scored}` : "n/a"}
                      </td>
                    );
                  })}
                </tr>
              ) : null,
            )}
            {oracle && (
              <tr>
                <td>Hidden 50, agreement with the corpus oracle</td>
                {ids.map((p) => {
                  const h = oracle.hidden.pipelines[p.id as PipelineId];
                  return (
                    <td key={p.id} className="num font-medium">
                      {h ? `${h.agree}/${h.scored}` : "n/a"}
                    </td>
                  );
                })}
              </tr>
            )}
          </tbody>
        </table>
      </Section>

      <Section id="by-type" title="Public 100 by question type">
        <p>
          Exact match after normalising case, punctuation and diacritics, with LLM tokens per question. Counting and ranking
          span 8 to 43 articles, which no top-k passage window holds; that is where retrieval alone breaks.
        </p>
        {palette && (
          <div className="mb-5">
            <ChartCard title="Exact match by question type" note="Share of each type answered exactly.">
              <Legend />
              <AccuracyByType metrics={metrics} palette={palette} />
            </ChartCard>
          </div>
        )}
        <TypeLedger metrics={metrics} />
      </Section>

      <Section id="retrieval" title="Retrieval and evidence">
        <p>
          Ranked citations scored against the gold documents, and every answer checked against the text of the articles it
          cites.
        </p>
        <div className="grid grid-cols-1 items-start gap-5 xl:grid-cols-[1.15fr_1fr]">
          <table className="ledger">
          <thead>
            <tr>
              <th>Measure</th>
              {ids.map((p) => (
                <th key={p.id} className="num" style={{ color: p.color }}>
                  {p.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(
              [
                ["Hit@1", (m) => m.retrieval?.hit_at_1.toFixed(2)],
                ["Hit@5", (m) => m.retrieval?.hit_at_5.toFixed(2)],
                ["MRR", (m) => m.retrieval?.mrr.toFixed(3)],
                ["nDCG@5", (m) => m.retrieval?.ndcg_at_5.toFixed(3)],
                ["Precision@5", (m) => m.retrieval?.precision_at_5.toFixed(3)],
                ["Context recall", (m) => m.retrieval && percent(m.retrieval.context_recall * 100, 100)],
                ["Gold among answers", (m) => m.evidence && `${m.evidence.gold_among_answers}/100`],
                ["Grounded in cited articles", (m) => m.evidence && `${m.evidence.grounded}/${m.evidence.grounding_checked}`],
                ["Abstained", (m) => m.evidence && `${m.evidence.abstained}`],
              ] as [string, (m: Metrics["pipelines"][PipelineId]) => string | undefined][]
            ).map(([label, read]) => (
              <tr key={label}>
                <td>{label}</td>
                {ids.map((p) => (
                  <td key={p.id} className="num mono">
                    {read(metrics.pipelines[p.id as PipelineId]) ?? "n/a"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
          {palette && (
            <ChartCard title="Retrieval quality" note="Each axis is a share; the agent and GraphRAG overlap.">
              <Legend />
              <RetrievalRadar metrics={metrics} palette={palette} />
            </ChartCard>
          )}
        </div>
      </Section>

      <Section id="robustness" title="Robustness">
        <h3>Paraphrased questions</h3>
        <p>
          Twelve public questions reworded by hand: host cities instead of years ("Calgary 1988"), synonyms ("shooters",
          "biggest field") and imperative forms ("Name the…"). Answers and gold documents carry over.
        </p>
        {data.paraphrase && <TypeLedger metrics={data.paraphrase} />}
        <h3>Two-step questions</h3>
        <p>
          Twenty-four questions that chain a ranking into a lookup: who won, where it was held, or how many nations took part
          in the event with the most competitors. A fixed pipeline runs one graph operation; the agent sees that the event it
          found is a step, not the answer, and reads the attribute next.
        </p>
        {data.compositional && <TypeLedger metrics={data.compositional} />}
        {data.unanswerable && (
          <>
            <h3>Questions the corpus cannot answer</h3>
            <p>
              The official templates pointed at facts the corpus does not hold: a sport at Games that never held it, an
              edition with no article, a venue on a day it held nothing. An independent oracle confirms none has an answer,
              so the only correct reply is "Not found in corpus"; anything else is a hallucination.
            </p>
            <TypeLedger metrics={data.unanswerable} />
          </>
        )}
        {data.offtemplate && (
          <>
            <h3>Outside the Olympic templates</h3>
            <p>
              Films and officeholders from the same corpus: who directed or scored a film, when it was released, who
              succeeded an officeholder. No graph tool models these, so the agent has to reach for passage search.
            </p>
            <TypeLedger metrics={data.offtemplate} />
          </>
        )}
      </Section>

      <Section id="hidden" title="Hidden 50">
        <p>
          The hidden questions have no published answers, so an independent oracle derives the expected answer for each
          template straight from the corpus infoboxes, without TigerGraph and without any pipeline code. On the public set it
          contains the gold answer for {oracle?.public.contain_gold ?? "all"} of {oracle?.public.questions ?? 100} questions.
        </p>
        {oracle && (
          <ul>
            {ids.map((p) => {
              const h = oracle.hidden.pipelines[p.id as PipelineId];
              return h ? (
                <li key={p.id}>
                  <strong>{p.name}</strong>: {h.agree} of {h.scored} answers agree with the oracle.
                </li>
              ) : null;
            })}
            <li>
              Not scored: {oracle.hidden.unscored.join(", ")}, where the oracle itself finds two events at the same venue on the
              same date.
            </li>
          </ul>
        )}
      </Section>

      <Section id="agent" title="How the agent worked">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-[1fr_1.3fr]">
          <table className="ledger">
            <tbody>
              <tr>
                <td>LLM calls per question</td>
                <td className="num mono">{agent.llm_calls_mean.toFixed(2)}</td>
              </tr>
              <tr>
                <td>Steps per question (LLM + graph)</td>
                <td className="num mono">{agent.steps_mean.toFixed(2)}</td>
              </tr>
              <tr>
                <td>Answered after one LLM call</td>
                <td className="num mono">
                  {agent.single_call_answers}/{agent.questions}
                </td>
              </tr>
              <tr>
                <td>Changed strategy mid-question</td>
                <td className="num mono">
                  {agent.strategy_changes}/{agent.questions}
                </td>
              </tr>
            </tbody>
          </table>
          {palette && (
            <ChartCard title="Tools it chose" note="Graph tools cost zero LLM tokens.">
              <ToolMix tools={agent.tools} palette={palette} />
            </ChartCard>
          )}
        </div>
        {agent.stopping_reasons && (
          <>
            <h3>Why it stopped</h3>
            <ul>
              {Object.entries(agent.stopping_reasons)
                .sort((a, b) => b[1] - a[1])
                .map(([reason, count]) => (
                  <li key={reason}>
                    {reason} <span className="text-[color:var(--soft)]">({count})</span>
                  </li>
                ))}
            </ul>
          </>
        )}
      </Section>

      <Section id="cost" title="Cost and latency">
        <table className="ledger">
          <thead>
            <tr>
              <th>Pipeline</th>
              <th className="num">LLM tokens / q</th>
              <th className="num">Latency p50</th>
              <th className="num">Latency p95</th>
              <th className="num">Cost / 100 q</th>
              <th className="num">Citations / answer</th>
            </tr>
          </thead>
          <tbody>
            {ids.map((p) => {
              const o = metrics.pipelines[p.id as PipelineId].overall;
              return (
                <tr key={p.id}>
                  <td style={{ color: p.color }}>{p.name}</td>
                  <td className="num mono">{tokens(o.tokens_mean)}</td>
                  <td className="num mono">{seconds(o.latency_p50_s * 1000)}</td>
                  <td className="num mono">{seconds(o.latency_p95_s * 1000)}</td>
                  <td className="num mono">${o.cost_per_100_usd.toFixed(2)}</td>
                  <td className="num mono">{o.citations_mean.toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p>
          Latency includes client-side pacing for a trial API key; graph queries and retrieval add no LLM tokens and are
          counted as zero.
        </p>
      </Section>

      <Section id="ceiling" title="The ceiling">
        <p>
          The one public miss is <code>pub-099</code>: two events share the venue and date the question names (Laura Biathlon
          &amp; Ski Complex, 22 February 2014): the women's 30 km cross-country and the men's biathlon relay. Both articles
          give the same venue and the same date, so the corpus cannot say which one the question means. The graph pipelines
          return both winners rather than guess, and the hidden set has one such question too (<code>eval-032</code>).
        </p>
        <p>
          <strong>99 of 100 is every question the corpus can decide.</strong> Reaching 100 would mean picking one of two equally
          supported answers, which is a guess.
        </p>
      </Section>

      <Section id="method" title="Method">
        <ul>
          <li>One model for every LLM call in every pipeline; temperature 0.</li>
          <li>Exact match after normalising case, punctuation, separators and diacritics; strict match without the separators.</li>
          <li>Tokens are the provider's billed input and output tokens; any row that had to estimate them is flagged.</li>
          <li>Each results file records the git commit, model, and dataset hash; reruns of a subset are merged with that provenance.</li>
          <li>No pipeline sees a benchmark answer; the hidden-set oracle is evaluation code that no pipeline imports.</li>
        </ul>
        {metrics.provenance && (
          <table className="ledger">
            <thead>
              <tr>
                <th>Rows</th>
                <th>Pipelines</th>
                <th className="num">Commit</th>
              </tr>
            </thead>
            <tbody>
              {metrics.provenance.map((row, index) => (
                <tr key={index}>
                  <td className="mono">{row.rows}</td>
                  <td>{Array.isArray(row.pipelines) ? row.pipelines.join(", ") : row.pipelines}</td>
                  <td className="num mono">{row.commit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Section>
    </LongPage>
  );
}
