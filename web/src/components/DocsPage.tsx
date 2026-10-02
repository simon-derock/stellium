// How STELLIUM is built: the three pipelines, the TigerGraph schema and its access paths, retrieval,
// the agent, and every graph view and drawing style the canvas offers.
import type { GraphStats } from "../api";
import { THEMES, VIEWS, VIZ_PROFILES } from "../graph/graph";
import { LongPage, Section } from "./LongPage";

const SECTIONS = [
  { id: "overview", title: "Overview" },
  { id: "architecture", title: "Architecture" },
  { id: "schema", title: "Graph schema" },
  { id: "access", title: "Access paths and cost" },
  { id: "rag", title: "Retrieval (RAG)" },
  { id: "graphrag", title: "GraphRAG" },
  { id: "agent", title: "The agent" },
  { id: "views", title: "Views and formations" },
  { id: "api", title: "API" },
];

const VERTICES: [string, string, string, string][] = [
  ["Event", "event_id (Wikidata QID)", "name, year, season, sport, gender, venue, competitor_count, nation_count, gold/silver/bronze athlete and NOC, prev_event_id, next_event_id, filter_mask", "One Olympic event: the unit every question is about."],
  ["Document", "doc_id", "title, url, wikidata_qid, wikipedia_pageid, approx_tokens, filter_mask", "The Wikipedia article an event is documented in."],
  ["Chunk", "chunk_id (doc_id#index)", "doc_id, chunk_index, section_title, text, raw_text, prev_chunk_id, next_chunk_id, filter_mask, embedding (1024-d HNSW)", "A passage: what retrieval ranks and what answers are checked against."],
  ["Venue", "venue_id", "name", "Where events were held."],
  ["Session / ChatMessage", "session_id / msg_id", "created_at, role, content, pipeline, token_count", "Conversation history for the live demo."],
];

const EDGES: [string, string, string][] = [
  ["DOCUMENTED_IN", "Event → Document", "The article that states an event's facts."],
  ["HAS_CHUNK", "Document → Chunk", "An article's passages, in order."],
  ["HELD_AT", "Event → Venue", "Carries start_date and end_date on the edge itself."],
  ["PRECEDES", "Event → Event", "To the same event at the previous Games, with time_diff."],
  ["SUCCEEDS", "Event → Event", "To the same event at the next Games, with time_diff."],
  ["CONFLICTS_WITH", "Event → Event", "Conflict type, resolution and resolver, for disagreeing sources."],
  ["HAS_MESSAGE", "Session → ChatMessage", "Ordered by msg_order."],
];

const TOOLS: [string, string, string][] = [
  ["count_events", "AggregationAgent", "How many events of a sport at one Games meet a competitor-count condition."],
  ["rank_events", "AggregationAgent", "The event with the most or fewest competitors; reports ties and breaks one only when exactly one article restates its count in prose."],
  ["event_attribute", "GraphTraversalAgent", "One attribute of one linked event, read back from TigerGraph."],
  ["previous_edition", "GraphTraversalAgent", "Links the event, then follows PRECEDES one hop with an installed query."],
  ["event_at_venue_date", "GraphTraversalAgent", "The event held at a venue on a date: the venue, and the start date on its HELD_AT edge."],
  ["find_events", "EntityLinkingAgent", "Canonical events that fit some constraints, to explore or disambiguate."],
  ["hybrid_search", "DocumentRetrievalAgent", "Reranked passages for facts outside the structured attributes."],
  ["gsql_query", "QueryGenerationAgent", "One guarded, read-only, LLM-written INTERPRET QUERY when no typed tool fits."],
];

function Stat({ label, value }: { label: string; value?: number }) {
  return (
    <div className="card">
      <div className="tag">{label}</div>
      <div className="stat-value mt-2">
        {value === undefined ? "…" : value.toLocaleString("en-US")}
      </div>
    </div>
  );
}

export function DocsPage({ stats, onScroll }: { stats: GraphStats | null; onScroll?: (top: number) => void }) {
  return (
    <LongPage sections={SECTIONS} onScroll={onScroll}>
      <h1>How STELLIUM works</h1>
      <p className="lede">
        STELLIUM answers questions about Olympic events from a corpus of Wikipedia articles three ways, side by side: plain
        retrieval, a fixed GraphRAG pipeline, and an agent that plans over typed TigerGraph tools. Same model, same corpus,
        same questions; the difference is how much of the work the graph does.
      </p>

      <Section id="overview" title="Overview">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Events" value={stats?.events} />
          <Stat label="Games" value={stats?.games} />
          <Stat label="Sports" value={stats?.sports} />
          <Stat label="Venues" value={stats?.venues} />
          <Stat label="Articles indexed" value={stats?.documents} />
          <Stat label="Passages" value={stats?.chunks} />
          <Stat label="BM25 terms" value={stats?.bm25_terms} />
          <Stat label="Previous-edition links" value={stats?.previous_edition_links} />
        </div>
        <p className="mt-4">
          Counts are live: read from the catalog loaded out of TigerGraph and from the in-memory passage index the API is
          serving right now.
        </p>
      </Section>

      <Section id="architecture" title="Architecture">
        <pre>
          <code>{`question
  │
  ├─ RAG ─────────── hybrid retrieval ─────────────────────────────▶ 1 LLM call ─▶ answer
  │                  TigerGraph HNSW (top 30) + BM25 → RRF → rerank → 5 articles
  │
  ├─ GraphRAG ────── 1 LLM call: question → typed plan (JSON)
  │                  → 1 graph operation on TigerGraph (0 tokens)
  │                  → 1 LLM call over the graph result + its source passages ─▶ answer
  │
  └─ Agentic ─────── ReAct loop, at most 5 planning calls
                     each step: Thought → one tool (graph tools cost 0 tokens) → observation
                     stops on a verified value of the asked kind ─▶ answer + trace`}</code>
        </pre>
        <ul>
          <li>
            <strong>Answers come from the graph, not the model.</strong> Graph tools return attribute values read back from
            TigerGraph and checked against a line of the cited article; the LLM chooses tools and phrases nothing new.
          </li>
          <li>
            <strong>One model everywhere.</strong> Cohere <code>command-a-03-2025</code> for every LLM call; Cohere{" "}
            <code>embed-v4.0</code> (1024 dimensions) for passage vectors stored in TigerGraph.
          </li>
        </ul>
      </Section>

      <Section id="schema" title="Graph schema">
        <p>
          One graph, <code>OlympicsGraph</code>, on TigerGraph Savanna. Vertices use their primary id as an attribute, so every
          lookup by id is a direct seek.
        </p>
        <h3>Vertices</h3>
        <table className="ledger">
          <thead>
            <tr>
              <th>Vertex</th>
              <th>Primary id</th>
              <th>Attributes</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {VERTICES.map(([name, id, attrs, role]) => (
              <tr key={name}>
                <td className="mono">{name}</td>
                <td className="mono">{id}</td>
                <td className="text-[12.5px] text-[color:var(--soft)]">{attrs}</td>
                <td className="text-[12.5px]">{role}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>Edges (directed)</h3>
        <table className="ledger">
          <thead>
            <tr>
              <th>Edge</th>
              <th>From → to</th>
              <th>Meaning</th>
            </tr>
          </thead>
          <tbody>
            {EDGES.map(([name, ends, meaning]) => (
              <tr key={name}>
                <td className="mono">{name}</td>
                <td className="mono">{ends}</td>
                <td>{meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>Two doubly linked lists</h3>
        <ul>
          <li>
            <strong>Editions of an event.</strong> Ingestion groups articles into series by sport and normalised event name,
            sorts each series by year, and writes <code>prev_event_id</code> and <code>next_event_id</code> on every Event, plus{" "}
            <code>PRECEDES</code> and <code>SUCCEEDS</code> edges carrying the gap in years. "Who won the men's 20 km walk at
            the Games before 2016?" is one hop from the 2016 event, whatever year that edition was held.
          </li>
          <li>
            <strong>Passages of an article.</strong> Chunks are ids <code>doc_id#0</code>, <code>doc_id#1</code>… with{" "}
            <code>prev_chunk_id</code> and <code>next_chunk_id</code>, so a passage's neighbours are a constant-time hop, both
            in the graph and in the API's in-memory chunk map.
          </li>
        </ul>
        <h3>Prepared for evolving facts</h3>
        <p>
          Events also carry <code>valid_from</code>, <code>valid_to</code>, <code>superseded_by</code> and{" "}
          <code>source_authority</code>, and <code>CONFLICTS_WITH</code> records two sources that disagree and how the
          disagreement was resolved.
        </p>
        <pre>
          <code>{`CREATE VERTEX Event (PRIMARY_ID event_id STRING, name STRING, year INT, season STRING,
  sport STRING, gender STRING, venue STRING, competitor_count INT, nation_count INT,
  gold_athlete STRING, ..., prev_event_id STRING, next_event_id STRING, filter_mask UINT, ...)
  WITH PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE DIRECTED EDGE HELD_AT  (FROM Event, TO Venue, start_date STRING, end_date STRING)
CREATE DIRECTED EDGE PRECEDES (FROM Event, TO Event, time_diff INT)
CREATE DIRECTED EDGE SUCCEEDS (FROM Event, TO Event, time_diff INT)
ALTER VERTEX Chunk ADD VECTOR ATTRIBUTE embedding (DIMENSION = 1024, METRIC = "COSINE", INDEXTYPE = "HNSW")`}</code>
        </pre>
      </Section>

      <Section id="access" title="Access paths and cost">
        <table className="ledger">
          <thead>
            <tr>
              <th>Path</th>
              <th>How</th>
              <th className="num">Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Event by id or exact title</td>
              <td>Hash maps built once from the graph at startup (by id, by spacing-insensitive title)</td>
              <td className="num mono">O(1)</td>
            </tr>
            <tr>
              <td>Attribute values</td>
              <td>One REST read of the linked vertices by primary id, so every value comes from TigerGraph</td>
              <td className="num mono">O(k) for k events</td>
            </tr>
            <tr>
              <td>Previous edition</td>
              <td>Installed query seeded at the linked event, one PRECEDES hop</td>
              <td className="num mono">O(1) hop</td>
            </tr>
            <tr>
              <td>Neighbouring passage</td>
              <td>prev_chunk_id / next_chunk_id pointer</td>
              <td className="num mono">O(1)</td>
            </tr>
            <tr>
              <td>Year and season filter</td>
              <td>
                <code>filter_mask</code>: one bit per Games year from 1988 (bits 0 to 19) plus Summer (bit 20) and Winter (bit 21);
                a filter is one bitwise AND
              </td>
              <td className="num mono">O(1) per passage</td>
            </tr>
            <tr>
              <td>Dense passage search</td>
              <td>TigerGraph native HNSW over Chunk.embedding, cosine</td>
              <td className="num mono">≈ O(log n)</td>
            </tr>
            <tr>
              <td>Keyword passage search</td>
              <td>BM25Plus over an inverted index of compact integer postings; top-k by heap</td>
              <td className="num mono">O(postings + n log k)</td>
            </tr>
          </tbody>
        </table>
        <p>
          Entity linking runs in memory against the graph-loaded catalog: exact title first, then order-insensitive token
          overlap within the sport, Games and gender the question names. Numbers must match exactly, so "200 metre" never links
          to "4 × 200 metre relay".
        </p>
      </Section>

      <Section id="rag" title="Retrieval (RAG)">
        <ul>
          <li>
            <strong>Chunking.</strong> Paragraph-grouped passages of about 400 tokens, with the last paragraph repeated in the next
            passage. Passage 0 carries a metadata header built from the infobox; every passage carries its article title, so an
            embedding never loses its document.
          </li>
          <li>
            <strong>Two retrievers.</strong> TigerGraph HNSW vector search returns 30 candidates; BM25Plus over the same passages
            returns its own ranking.
          </li>
          <li>
            <strong>Fusion and reranking.</strong> Reciprocal rank fusion merges the two lists, a local int8 MiniLM cross-encoder
            reranks them, and the top passages from five distinct articles become the context.
          </li>
          <li>
            <strong>Answer.</strong> One LLM call returns only the value, or "Not found in corpus".
          </li>
        </ul>
        <p>
          This is the baseline. It wins when the answer sits in one passage, and loses when a question counts or ranks across
          dozens of articles: no top-5 window can hold 43 infoboxes.
        </p>
      </Section>

      <Section id="graphrag" title="GraphRAG">
        <ol className="list-decimal pl-5">
          <li>One LLM call maps the question to a typed plan: an operation plus only the fields the question states.</li>
          <li>The plan runs as one typed graph operation: zero LLM tokens, values read from TigerGraph.</li>
          <li>
            A second LLM call answers from the graph result and its source passages; if the reply is not supported by the graph
            value or a passage, the verified graph value replaces it.
          </li>
        </ol>
      </Section>

      <Section id="agent" title="The agent">
        <p>
          A ReAct orchestrator plans one step at a time (Thought, Action, Action Input) and gets at most five planning steps. Each tool
          belongs to a named specialist, and every step lands in the trace with its latency and token count.
        </p>
        <table className="ledger">
          <thead>
            <tr>
              <th>Tool</th>
              <th>Specialist</th>
              <th>What it does</th>
            </tr>
          </thead>
          <tbody>
            {TOOLS.map(([tool, agent, what]) => (
              <tr key={tool}>
                <td className="mono">{tool}</td>
                <td className="mono">{agent}</td>
                <td>{what}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>Stopping and guard rails</h3>
        <ul>
          <li>Stops as soon as a tool returns one verified value of the kind the question asks for (a person, a number, a venue, an event).</li>
          <li>A value of the wrong kind is a step: the event with the most competitors when the question asks who won it.</li>
          <li>A ranking tie is followed for every tied event, and the answer lists each value.</li>
          <li>An answer that is not an exact graph value or a passage span is refused, and the agent must gather support.</li>
          <li>Identical repeated calls return a notice instead of running again; a failed call prompts a change of strategy.</li>
          <li>While a TigerGraph workspace resumes, reads wait for it and retry once instead of failing.</li>
        </ul>
      </Section>

      <Section id="views" title="Views and formations">
        <h3>Views</h3>
        <table className="ledger">
          <tbody>
            {VIEWS.map((view) => (
              <tr key={view.id}>
                <td className="w-44 font-medium">{view.name}</td>
                <td className="text-[color:var(--soft)]">{view.hint}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          Every edge the canvas draws names its source: an edge in the graph (HELD_AT, PRECEDES) or the Event attribute a
          grouping is read from (year and season for Games, sport for Sport).
        </p>
        <h3>Styles ({VIZ_PROFILES.length})</h3>
        <table className="ledger">
          <thead>
            <tr>
              <th>Style</th>
              <th>Bodies</th>
              <th>Links</th>
              <th>Formation</th>
              <th>Character</th>
            </tr>
          </thead>
          <tbody>
            {VIZ_PROFILES.map((viz) => (
              <tr key={viz.id}>
                <td className="font-medium">{viz.name}</td>
                <td className="mono">{viz.nodeMark}</td>
                <td className="mono">{viz.edgeMark}</td>
                <td className="mono">{viz.formation}</td>
                <td className="text-[color:var(--soft)]">{viz.hint}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>Themes ({THEMES.length})</h3>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
          {THEMES.map((theme) => (
            <div key={theme.id} className="card flex items-center gap-3 !py-2.5">
              <span className="flex gap-[2px]">
                {[theme.palette.paper, ...Object.values(theme.palette.layers)].map((color, index) => (
                  <span key={index} className="h-5 w-[5px] rounded-[1px]" style={{ background: color }} />
                ))}
              </span>
              <span className="min-w-0">
                <span className="block text-[13px] font-medium">{theme.name}</span>
                <span className="block truncate text-[11.5px] text-[color:var(--soft)]">{theme.hint}</span>
              </span>
            </div>
          ))}
        </div>
      </Section>

      <Section id="api" title="API">
        <table className="ledger">
          <tbody>
            {(
              [
                ["POST /api/v1/query/compare", "All three pipelines on one question, side by side"],
                ["POST /api/v1/query/{rag|graphrag|agentic}", "One pipeline"],
                ["GET /api/v1/graph/snapshot?view=&focus=", "The canvas payload for a view, optionally around cited events"],
                ["GET /api/v1/graph/stats", "Live catalog and index sizes"],
                ["GET /api/v1/metrics/{set}", "Published benchmark documents: public, paraphrase, compositional, hidden_oracle"],
                ["GET /health?deep=true", "A real graph read, which also wakes a suspended workspace"],
              ] as const
            ).map(([route, what]) => (
              <tr key={route}>
                <td className="mono w-[22rem]">{route}</td>
                <td>{what}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
    </LongPage>
  );
}
