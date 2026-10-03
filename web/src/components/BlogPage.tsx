import { useEffect, useRef } from "react";
import { Wordmark } from "./Wordmark";

// The write-up at /blog: what was built, what the numbers say, and when an agent is worth it.
// Figures here are the published results; the Benchmark page reads the same documents live.
const SITE = "https://stellium.philipsimonderock.com";
const REPO = "https://github.com/simon-derock/stellium";

const VERDICT: [string, string, string, string, string][] = [
  ["One attribute of one event", "19/19 · 855", "19/19 · 941", "19/19 · 2,513", "Agent, 9% fewer tokens than GraphRAG"],
  ["Winner at the previous Games", "22/22 · 1,341", "22/22 · 1,023", "22/22 · 2,547", "GraphRAG: one lookup suffices"],
  ["Venue and date to the winner", "27/28 · 786", "27/28 · 919", "24/28 · 2,522", "Agent, 14% fewer tokens"],
  ["Count events over a threshold", "21/21 · 747", "21/21 · 1,418", "4/21 · 2,580", "Agent, 47% fewer tokens"],
  ["Event with the most competitors", "10/10 · 733", "10/10 · 968", "2/10 · 2,572", "Agent, 24% fewer tokens"],
  ["Any of the above, reworded", "12/12 · 983", "12/12 · 1,082", "5/12 · 2,411", "Agent, 9% fewer tokens"],
  ["Two steps: rank, then read an attribute", "24/24 · 1,737", "19/24 · 1,170", "18/24 · 2,657", "Agent, decisive"],
  ["Outside the templates", "12/12 · 2,334", "11/12 · 1,374", "12/12 · 2,120", "RAG: retrieval is enough"],
  ["Nothing in the corpus answers it", "12/12 · 6,034", "12/12 · 1,730", "11/12 · 2,623", "GraphRAG: declines for less"],
];

function Figure({ src, alt, caption }: { src: string; alt: string; caption: string }) {
  return (
    <figure className="my-8">
      <img src={src} alt={alt} loading="lazy" className="w-full rounded-2xl border border-[color:var(--hair)]" />
      <figcaption className="mt-3 text-[13px] text-[color:var(--faint)]">{caption}</figcaption>
    </figure>
  );
}

export function BlogPage() {
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    document.title = "When does a question need an agent? · STELLIUM";
    // Sections rise in as they arrive, as on the other long pages.
    const scroller = root.current;
    if (!scroller || !("IntersectionObserver" in window)) return;
    const sections = Array.from(scroller.querySelectorAll<HTMLElement>(".reveal"));
    const reveal = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          (entry.target as HTMLElement).dataset.shown = "true";
          reveal.unobserve(entry.target);
        }
      },
      { root: scroller, rootMargin: "0px 0px -6% 0px", threshold: 0.02 },
    );
    sections.forEach((section) => reveal.observe(section));
    scroller.dataset.reveal = "on";
    return () => reveal.disconnect();
  }, []);

  return (
    <div ref={root} className="h-full overflow-y-auto bg-[color:var(--background)] text-[color:var(--foreground)]">
      <header className="mx-auto flex max-w-[46rem] items-center justify-between px-5 pt-6 md:pt-8">
        <a href="/" aria-label="STELLIUM home" className="no-underline">
          <Wordmark />
        </a>
        <nav className="flex gap-1">
          <a className="nav-cta" href="/">
            Live demo
          </a>
          <a className="nav-cta" href={REPO} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </nav>
      </header>

      <article className="prose mx-auto max-w-[46rem] px-5 pt-12 pb-24 md:pt-16">
        <div className="text-[12px] font-semibold tracking-[0.16em] text-[color:var(--accent)] uppercase">
          TigerGraph Agentic GraphRAG Hackathon 2026
        </div>
        <h1 className="mt-4">When does a question need an agent?</h1>
        <p className="lede">
          I built three question-answering pipelines over the same TigerGraph data, gave them the same model, and measured
          where an agent earns its tokens and where it is overkill.
        </p>
        <p className="mt-5 text-[13.5px] text-[color:var(--faint)]">Philip Simon Derock · 3 October 2026 · 7 min read</p>

        <Figure
          src="/blog/demo.gif"
          alt="The STELLIUM console answering a count question: the agent and GraphRAG answer 5, RAG answers 2, and the graph moves to the cited events"
          caption="The live console: one question, three pipelines, the agent's trace, and the events it cited."
        />

        <section className="reveal">
          <h2>The question</h2>
          <p>
            Agents are often treated as an upgrade: add planning, add tools, get better answers. But every step costs tokens
            and time, and some questions are answered just as well without any of it. The hackathon asked the useful version
            of the question: which questions need an agent, and which don't?
          </p>
          <p>
            To answer it I built STELLIUM, a console that sends each question to three pipelines at once and shows their
            answers side by side, over a TigerGraph graph of 2,210 Olympic events and the 2,951 Wikipedia articles they come
            from.
          </p>
        </section>

        <section className="reveal">
          <h2>Three pipelines, one model</h2>
          <p>
            All three use Cohere <code>command-a-03-2025</code> at temperature 0, so the only difference is how much of the
            work the graph does.
          </p>
          <ul>
            <li>
              <strong>RAG</strong> searches passages two ways, TigerGraph's native vector index and keyword search, fuses the
              lists, reranks them with Cohere Rerank, and answers from the top five articles in one model call.
            </li>
            <li>
              <strong>GraphRAG</strong> turns the question into one typed graph operation (a count, a ranking, a lookup),
              runs it on TigerGraph, and phrases the answer in a second call.
            </li>
            <li>
              <strong>The agent</strong> plans one step at a time over seven specialists: entity linking, graph traversal,
              aggregation, evidence checking, passage retrieval and more. It stops as soon as it holds a value the graph
              verified.
            </li>
          </ul>
          <p>
            RAG here is deliberately strong. Hybrid search with a reranker reaches 71 of 100 on the public questions, and on
            simple lookups it is perfect. Comparing against a weak baseline would make any agent look good.
          </p>
        </section>

        <section className="reveal">
          <h2>Where retrieval breaks</h2>
          <p>
            Ask "How many biathlon events at the 2018 Winter Olympics had more than 73 competitors?" and RAG answers two.
            The answer is five. The events are spread over several articles, and five passages cannot hold all of them, so
            retrieval undercounts. Counting and ranking are where plain retrieval fails: 4 of 21 counts and 2 of 10 rankings
            are right. Both graph pipelines get all of them, because the graph holds every event at once.
          </p>
          <Figure
            src="/blog/console.webp"
            alt="Three answers side by side with the agent's trace: one LLM call, the count_events tool, five events"
            caption="The agent's trace: one model call, one graph query, and the reason it stopped."
          />
        </section>

        <section className="reveal">
          <h2>When is the agent worth it?</h2>
          <p>
            For each type of question, the right choice is the cheapest pipeline among those with the top accuracy. Cells
            show correct answers and LLM tokens per question.
          </p>
          <div className="ledger-wrap">
            <table className="ledger">
              <thead>
                <tr>
                  <th>Question type</th>
                  <th className="num">Agent</th>
                  <th className="num">GraphRAG</th>
                  <th className="num">RAG</th>
                  <th>Right choice</th>
                </tr>
              </thead>
              <tbody>
                {VERDICT.map(([shape, agent, graph, rag, choice]) => (
                  <tr key={shape}>
                    <td>{shape}</td>
                    <td className="num whitespace-nowrap">{agent}</td>
                    <td className="num whitespace-nowrap">{graph}</td>
                    <td className="num whitespace-nowrap">{rag}</td>
                    <td>{choice}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            <strong>The agent is the right choice for six of nine types, decisive for one, and overkill for three.</strong>{" "}
            It is decisive when a question chains two steps, such as "how many nations competed in the rowing event with the
            most competitors at Rio 2016?". The agent finds the largest event, notes in its trace that this is an
            intermediate result, and then reads the nation count. It gets all 24 such questions; GraphRAG's single fixed
            operation gets 19.
          </p>
          <p>
            It is overkill when one fixed lookup already answers the question, when nothing in the corpus does (GraphRAG
            declines for under a third of the tokens), and outside the templates, where reranked retrieval alone is right.
          </p>
          <Figure
            src="/blog/verdict.webp"
            alt="The Benchmark page's verdict table, computed from the published results"
            caption="The same verdict on the live Benchmark page, computed from the committed results files."
          />
        </section>

        <section className="reveal">
          <h2>Why the agent costs less, not more</h2>
          <p>
            The agent averages 908 tokens per answer, against 1,056 for GraphRAG and 2,543 for RAG, and 84 of 100 public
            questions take a single model call. Two choices make that possible. Its tools are typed graph operations, so one
            call to TigerGraph replaces several passages of context. And it stops on the first verified value of the kind the
            question asks for, instead of collecting more evidence it does not need.
          </p>
        </section>

        <section className="reveal">
          <h2>Results</h2>
          <div className="ledger-wrap">
            <table className="ledger">
              <thead>
                <tr>
                  <th>Measure</th>
                  <th className="num">Agent</th>
                  <th className="num">GraphRAG</th>
                  <th className="num">RAG</th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["Public 100, exact match", "99", "99", "71"],
                    ["LLM tokens per answer", "908", "1,056", "2,543"],
                    ["Median latency", "1.9 s", "6.5 s", "1.8 s"],
                    ["Hidden 50, agreement with a corpus oracle", "49/49", "49/49", "28/49"],
                    ["Two-step questions", "24/24", "19/24", "18/24"],
                    ["Reworded questions", "12/12", "12/12", "5/12"],
                    ["Unanswerable, correctly declined", "12/12", "12/12", "11/12"],
                    ["Outside the templates", "12/12", "11/12", "12/12"],
                  ] as const
                ).map(([label, a, g, r]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td className="num">{a}</td>
                    <td className="num">{g}</td>
                    <td className="num">{r}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            The one public miss cannot be decided from the corpus: two events share the venue and date the question names,
            and both articles say so. The graph pipelines return both winners rather than guess. The same happens once in the
            hidden set, which is why the oracle scores 49 questions, not 50. The oracle derives answers from the corpus on its
            own and is not the organisers' ground truth.
          </p>
        </section>

        <section className="reveal">
          <h2>Checking the work</h2>
          <p>
            Every number on the Benchmark page is read from a committed results file that records the commit, model and
            dataset hash behind it. When I replaced the local reranker with Cohere Rerank, I re-ran every result that touches
            reranking and documented why the rest cannot change. The project has 246 commits since 25 September, 351 tests,
            and a CI gate with strict typing, linting, security scanning and a dependency audit on every push. The live
            console has per-visitor limits, a strict content security policy, and a private status page.
          </p>
        </section>

        <section className="reveal">
          <h2>What's next</h2>
          <p>
            Real knowledge changes and contradicts itself. The graph schema already has edges for conflicting facts and for
            which source supersedes which. The next step is wiring them into the agent's evidence check, so it can say not
            only what the answer is, but which source it trusts and why.
          </p>
          <div className="mt-8 flex flex-wrap gap-2">
            <a className="nav-cta" href="/">
              Try the live demo
            </a>
            <a className="nav-cta" href="/?tab=benchmark">
              Benchmark
            </a>
            <a className="nav-cta" href={REPO} target="_blank" rel="noreferrer">
              Source on GitHub
            </a>
          </div>
        </section>
      </article>

      <footer className="mx-auto max-w-[46rem] px-5 pb-10 text-[12.5px] text-[color:var(--faint)]">
        <a className="signature" href="https://philipsimonderock.com" target="_blank" rel="noreferrer">
          <span>
            Built by <b>Philip Simon Derock</b>
          </span>
          <span className="signature-dot" aria-hidden="true" />
          <span>{SITE.replace("https://", "")}</span>
        </a>
      </footer>
    </div>
  );
}
