// STELLIUM console: the Olympic graph full bleed, a question plate on the left, the agent's trace
// on the right, the headline numbers along the bottom, and the full benchmark and docs one tab away.
import { flushSync } from "react-dom";
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import {
  ApiError,
  compare,
  fetchHealth,
  untilAwake,
  fetchMetricSet,
  fetchPublishedSets,
  fetchMetrics,
  fetchOracle,
  fetchPresets,
  fetchSnapshot,
  fetchStats,
  type CompareResult,
  type GraphStats,
  type Preset,
} from "./api";
import { AskPanel } from "./components/AskPanel";
import { DocsPage } from "./components/DocsPage";
import { Menu } from "./components/Menu";
import { Pill } from "./components/Pill";
import { MetricsStrip, type BenchData } from "./components/MetricsStrip";
import { TracePanel } from "./components/TracePanel";
import { Wordmark } from "./components/Wordmark";
import { chromeFor, citedEvents } from "./format";
import { GraphSurface } from "./graph/GraphSurface";
import {
  THEMES,
  VIEWS,
  VIZ_PROFILES,
  type GraphNode,
  type Snapshot,
  type ViewId,
} from "./graph/graph";

// Charts load with the benchmark page, so the first screen stays light.
const BenchmarkPage = lazy(() => import("./components/BenchmarkPage").then((m) => ({ default: m.BenchmarkPage })));

type Mode = "ask" | "benchmark" | "docs";
const MODES: { id: Mode; name: string }[] = [
  { id: "ask", name: "Ask" },
  { id: "benchmark", name: "Benchmark" },
  { id: "docs", name: "Docs" },
];
type Health = "checking" | "ok" | "waking" | "down";

const EMPTY: Snapshot = { graph_nodes: [], graph_edges: [], disclosure: "" };
const REPO_URL = "https://github.com/simon-derock/stellium";

// Shareable links: ?q= asks a question on load, ?tab=benchmark or ?tab=docs opens that page,
// ?style= and ?theme= pick the drawing. Without them every visit opens on Nova over Gilt.
const LINK = new URLSearchParams(window.location.search);
const LINKED_QUESTION = LINK.get("q")?.trim().slice(0, 500) ?? "";

// Ask panel: left-5 + w-[26rem]; trace panel: right-5 + w-[24rem] (both from lg up).
const PANEL_LEFT_PX = 20 + 416;
const PANEL_RIGHT_PX = 20 + 384;

function useWide() {
  const query = "(min-width: 1024px)";
  const [wide, setWide] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const list = window.matchMedia(query);
    const update = () => setWide(list.matches);
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, []);
  return wide;
}

export function App() {
  const [mode, setModeNow] = useState<Mode>(() => MODES.find((m) => m.id === LINK.get("tab"))?.id ?? "ask");
  // Switching pages cross-dissolves through a View Transition where the browser has one, so the
  // shared frame holds still and only the content changes; elsewhere the switch is instant.
  const setMode = useCallback((next: Mode) => {
    const doc = document as Document & { startViewTransition?: (update: () => void) => unknown };
    if (!doc.startViewTransition || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setModeNow(next);
      return;
    }
    doc.startViewTransition(() => flushSync(() => setModeNow(next)));
  }, []);
  const [vizId, setVizId] = useState(() => VIZ_PROFILES.find((v) => v.id === LINK.get("style"))?.id ?? "nova");
  // As in Lunarbit, every style carries its own theme; picking a theme overrides it for the visit.
  const [themeId, setThemeId] = useState(
    () =>
      THEMES.find((t) => t.id === LINK.get("theme"))?.id ??
      VIZ_PROFILES.find((v) => v.id === vizId)?.preferredTheme ??
      "gilt",
  );
  const [viewId, setViewId] = useState<ViewId>("venues");
  const [focus, setFocus] = useState<string[]>([]);
  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [bench, setBench] = useState<BenchData>({
    public: null,
    paraphrase: null,
    compositional: null,
    unanswerable: null,
    offtemplate: null,
    oracle: null,
  });
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health>("checking");

  const theme = THEMES.find((t) => t.id === themeId) ?? THEMES[0]!;
  // On wide screens the ask panel (left) and the agent trace (right) sit over the canvas; the
  // graph frames itself between them.
  const wide = useWide();
  const tracing = Boolean(result?.agentic?.agentic_trace);
  // On narrower screens the panel sits below the graph: measure the band between the header and
  // the panel's top edge, live, so the whole graph frames into the space actually left for it.
  const headerRef = useRef<HTMLElement>(null);
  const askRef = useRef<HTMLDivElement>(null);
  const [band, setBand] = useState<{ top: number; bottom: number } | null>(null);
  useLayoutEffect(() => {
    const header = headerRef.current;
    const panel = askRef.current?.firstElementChild as HTMLElement | null | undefined;
    if (wide || mode !== "ask" || !header || !panel) {
      setBand(null);
      return;
    }
    const measure = () => {
      const top = Math.round(header.getBoundingClientRect().bottom + 14);
      const bottom = Math.round(window.innerHeight - panel.getBoundingClientRect().top + 14);
      setBand((prev) => (prev && Math.abs(prev.top - top) < 4 && Math.abs(prev.bottom - bottom) < 4 ? prev : { top, bottom }));
    };
    measure();
    const watch = new ResizeObserver(measure);
    watch.observe(header);
    watch.observe(panel);
    window.addEventListener("resize", measure);
    return () => {
      watch.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [wide, mode]);
  const clear = useMemo(
    () =>
      mode === "ask" && wide
        ? { left: PANEL_LEFT_PX, right: tracing ? PANEL_RIGHT_PX : 0 }
        : { left: 0, right: 0, ...band },
    [mode, wide, tracing, band],
  );
  const viz = VIZ_PROFILES.find((v) => v.id === vizId) ?? VIZ_PROFILES[0]!;
  const chrome = useMemo(() => chromeFor(theme.vars), [theme]);



  // Fetch the benchmark page's chart bundle while idle, so its first opening is instant.
  useEffect(() => {
    const warm = () => void import("./components/BenchmarkPage");
    const idle = (window as Window & { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback;
    if (idle) idle(warm);
    else window.setTimeout(warm, 1500);
  }, []);

  // A deep health check reads the graph, so it also wakes a suspended workspace.
  useEffect(() => {
    const slow = window.setTimeout(() => setHealth((h) => (h === "checking" ? "waking" : h)), 2500);
    // Every startup read waits out a host that is still waking, so a cold start fills in late
    // instead of leaving sections empty.
    untilAwake(() => fetchHealth(true))
      .then(() => setHealth("ok"))
      .catch(() => setHealth("down"))
      .finally(() => window.clearTimeout(slow));
    untilAwake(fetchPresets).then(setPresets).catch(() => setPresets([]));
    // Each published document arrives on its own; a missing one only hides its numbers.
    untilAwake(fetchMetrics).then((d) => setBench((b) => ({ ...b, public: d }))).catch(() => undefined);
    // Ask only for the sets that have been published, so nothing 404s in the console.
    untilAwake(fetchPublishedSets)
      .then((published) => {
        for (const name of ["paraphrase", "compositional", "unanswerable", "offtemplate"] as const) {
          if (!published.includes(name)) continue;
          fetchMetricSet(name).then((d) => setBench((b) => ({ ...b, [name]: d }))).catch(() => undefined);
        }
        if (published.includes("hidden_oracle")) {
          fetchOracle().then((d) => setBench((b) => ({ ...b, oracle: d }))).catch(() => undefined);
        }
      })
      .catch(() => undefined);
    untilAwake(fetchStats).then(setStats).catch(() => setStats(null));
    return () => window.clearTimeout(slow);
  }, []);

  // A suspended TigerGraph workspace answers 503 while it resumes: keep the page calm, say so,
  // and retry every few seconds until the graph is back.
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let live = true;
    let timer = 0;
    fetchSnapshot(viewId, viewId === "investigation" ? focus : [])
      .then((next) => {
        if (!live) return;
        setSnapshot(next);
        setHealth("ok");
        if (retry) fetchStats().then(setStats).catch(() => undefined);
      })
      .catch(() => {
        if (!live) return;
        setHealth("waking");
        setSnapshot((s) => ({ ...s, disclosure: "Waking the graph; this takes a few seconds after a quiet spell." }));
        if (retry < 40) timer = window.setTimeout(() => setRetry((n) => n + 1), 4000);
        else setHealth("down");
      });
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [viewId, focus, retry]);

  const ask = useCallback(async (question: string) => {
    setBusy(true);
    setError(null);
    try {
      const next = await compare(question);
      setResult(next);
      // Open the evidence only when there is some; "Not found" keeps the current view.
      const cited = citedEvents(next);
      if (cited.length) {
        setFocus(cited);
        setViewId("investigation");
      }
      setHealth("ok");
    } catch (exc) {
      setError(
        exc instanceof ApiError && exc.status === 503
          ? "The graph or the model is busy. Try again in a few seconds."
          : exc instanceof Error
            ? exc.message
            : "Something went wrong.",
      );
    } finally {
      setBusy(false);
    }
  }, []);

  // Once per page load, even when StrictMode mounts effects twice in development.
  const linkedAsked = useRef(false);
  useEffect(() => {
    if (!LINKED_QUESTION || linkedAsked.current) return;
    linkedAsked.current = true;
    void ask(LINKED_QUESTION);
  }, [ask]);

  const focusOn = useCallback((eventIds: string[]) => {
    setFocus(eventIds);
    setViewId("investigation");
  }, []);

  const onSelect = useCallback(
    (node: GraphNode | null) => {
      setSelected(node);
      // Clicking an event outside the investigation view opens its neighbourhood.
      if (node?.type === "Event" && viewId !== "investigation") focusOn([node.id]);
    },
    [viewId, focusOn],
  );


  return (
    <main
      style={{ ...theme.vars, ...chrome.overrides } as CSSProperties}
      data-scheme={chrome.scheme}
      data-mode={mode}
      className={`app-shell theme-${themeId} relative h-dvh w-full overflow-hidden bg-background text-foreground`}
    >
      <div className="parallax">
        <GraphSurface
          nodes={snapshot.graph_nodes}
          edges={snapshot.graph_edges}
          palette={theme.palette}
          viz={viz}
          selectedId={selected?.id ?? null}
          onSelect={onSelect}
          onLinkSelect={() => undefined}
          clear={clear}
        />
      </div>
      <div className="hdr pointer-events-none absolute inset-0" />
      <div className="grain pointer-events-none absolute inset-0" />

      <header ref={headerRef} className="pointer-events-none absolute inset-x-0 top-0 z-20 flex flex-wrap items-center justify-between gap-3 px-4 pt-4 md:px-5 md:pt-5">
        <div className="pointer-events-auto py-2 pl-1">
          <Wordmark />
        </div>
        <div className="pointer-events-auto w-full md:w-auto">
          <Pill label="Navigation and graph display" activeKey={mode}>
            {MODES.map((m) => (
              <button
                key={m.id}
                className="nav-item"
                data-glide={m.id}
                aria-current={mode === m.id ? "page" : undefined}
                onClick={() => setMode(m.id)}
              >
                {m.name}
              </button>
            ))}
            <a className="nav-cta" href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Source on GitHub">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 .7a11.3 11.3 0 0 0-3.57 22c.57.1.78-.25.78-.55v-2.1c-3.18.69-3.85-1.34-3.85-1.34-.52-1.32-1.27-1.67-1.27-1.67-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.24 3.34.95.1-.74.4-1.24.73-1.53-2.54-.29-5.21-1.27-5.21-5.66 0-1.25.45-2.27 1.18-3.07-.12-.29-.51-1.45.11-3.03 0 0 .96-.31 3.13 1.17A10.9 10.9 0 0 1 12 6c.97 0 1.95.13 2.86.38 2.17-1.48 3.13-1.17 3.13-1.17.62 1.58.23 2.74.11 3.03.73.8 1.18 1.82 1.18 3.07 0 4.4-2.68 5.36-5.23 5.64.41.36.78 1.07.78 2.16v3.2c0 .3.2.66.79.55A11.3 11.3 0 0 0 12 .7Z" />
              </svg>
              GitHub
            </a>
            <span className="pill-divider" aria-hidden="true" />
            <Menu
              tag="View"
              value={viewId}
              onChange={(id) => setViewId(id as ViewId)}
              align="end"
              width="19rem"
              wheelExplore
              options={VIEWS.map((v) => ({ id: v.id, name: v.name, hint: v.hint }))}
            />
            <Menu
              tag="Style"
              value={vizId}
              onChange={(id) => {
                setVizId(id);
                const next = VIZ_PROFILES.find((v) => v.id === id);
                if (next && THEMES.some((t) => t.id === next.preferredTheme)) setThemeId(next.preferredTheme);
              }}
              align="end"
              width="19rem"
              keepOpenOnSelect
              wheelExplore
              options={VIZ_PROFILES.map((v) => ({
                id: v.id,
                name: v.name,
                hint: `${v.hint} · ${THEMES.find((t) => t.id === v.preferredTheme)?.name ?? "own"} theme`,
              }))}
            />
            <Menu
              tag="Theme"
              value={themeId}
              onChange={setThemeId}
              align="end"
              width="21rem"
              keepOpenOnSelect
              wheelExplore
              options={THEMES.map((t) => ({ id: t.id, name: t.name, hint: t.hint, swatches: Object.values(t.palette.layers) }))}
            />
          </Pill>
        </div>
      </header>

      {mode === "ask" ? (
        <>
          <div ref={askRef} className="pointer-events-none absolute inset-x-4 top-[44dvh] bottom-[4.5rem] z-10 flex items-end md:top-[10.5rem] md:bottom-[8.5rem] lg:top-[10rem] xl:top-[6.75rem] lg:right-auto lg:left-5 lg:w-[26rem] lg:items-start">
            <AskPanel
              initialQuestion={LINKED_QUESTION}
              presets={presets}
              result={result}
              busy={busy}
              error={error}
              onAsk={ask}
              onFocus={focusOn}
            />
          </div>
          {result?.agentic?.agentic_trace && (
            <div className="pointer-events-none absolute top-[10rem] xl:top-[6.75rem] right-5 bottom-[8.5rem] z-10 hidden w-[24rem] items-start lg:flex">
              <TracePanel trace={result.agentic.agentic_trace} />
            </div>
          )}
          <div className="pointer-events-none absolute inset-x-0 bottom-[3.75rem] z-10 hidden justify-center px-5 md:flex">
            <MetricsStrip data={bench} onOpen={() => setMode("benchmark")} />
          </div>
        </>
      ) : (
        <div className="pointer-events-none absolute inset-x-4 top-[13.5rem] bottom-[3.75rem] z-10 mx-auto flex max-w-[78rem] md:top-[10rem] xl:top-[6.75rem]">
          {mode === "benchmark" ? (
            <Suspense fallback={null}>
              <BenchmarkPage data={bench} themeKey={themeId} />
            </Suspense>
          ) : (
            <DocsPage stats={stats} />
          )}
        </div>
      )}

      <footer className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex items-end justify-center gap-3 px-4 pb-3 md:justify-between md:px-5">
        {/* The canvas explains itself on hover; a selected node is the only caption worth space. */}
        <div className="pointer-events-auto hidden min-w-0 truncate pb-2 text-[12px] text-[color:var(--faint)] md:block">
          {selected ? `${selected.type} · ${selected.label}` : ""}
        </div>
        <div className="pointer-events-auto flex items-center gap-3">
          {/* Status only speaks up when something is off: waking or unreachable. */}
          {(health === "waking" || health === "down") && (
            <span className="inline-flex items-center gap-2">
              <span className="status-dot" data-state={health} />
              <span className="tag">{health === "waking" ? "Waking the graph" : "Graph unreachable"}</span>
            </span>
          )}
          <a className="signature" href="https://philipsimonderock.com" target="_blank" rel="noreferrer">
            {/* One line on a phone: the shorter wording keeps the pill clear of the page above it. */}
            <span>
              <span className="hidden sm:inline">Built by </span>
              <b>Philip Simon Derock</b>
            </span>
            <span className="signature-dot" aria-hidden="true" />
            <span>
              <span className="hidden sm:inline">TigerGraph </span>GraphRAG Hackathon 2026
            </span>
          </a>
        </div>
      </footer>
    </main>
  );
}
