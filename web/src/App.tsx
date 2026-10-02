// STELLIUM console: the Olympic graph full bleed, a question plate on the left, the agent's trace
// on the right, the headline numbers along the bottom, and the full benchmark and docs one tab away.
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  ApiError,
  compare,
  fetchHealth,
  fetchMetricSet,
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
import { citedEvents } from "./format";
import { GraphSurface } from "./graph/GraphSurface";
import {
  LAYER_NAMES,
  THEMES,
  VIEWS,
  VIZ_PROFILES,
  type GraphNode,
  type LayerId,
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

function remembered(key: string, fallback: string, allowed: string[]): string {
  try {
    const value = localStorage.getItem(`stellium.v2.${key}`);
    return value && allowed.includes(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

function remember(key: string, value: string) {
  try {
    localStorage.setItem(`stellium.v2.${key}`, value);
  } catch {
    // Private windows and blocked storage just forget the choice.
  }
}

// Shareable links: ?q= asks a question on load, ?tab=benchmark or ?tab=docs opens that page.
const LINK = new URLSearchParams(window.location.search);
const LINKED_QUESTION = LINK.get("q")?.trim().slice(0, 500) ?? "";

export function App() {
  const [mode, setMode] = useState<Mode>(() => MODES.find((m) => m.id === LINK.get("tab"))?.id ?? "ask");
  const [vizId, setVizId] = useState(() => remembered("style", "nova", VIZ_PROFILES.map((v) => v.id)));
  // As in Lunarbit, every style carries its own theme; picking a theme overrides it for the visit.
  const [themeId, setThemeId] = useState(
    () => VIZ_PROFILES.find((v) => v.id === vizId)?.preferredTheme ?? "gilt",
  );
  const [viewId, setViewId] = useState<ViewId>("constellation");
  const [focus, setFocus] = useState<string[]>([]);
  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [bench, setBench] = useState<BenchData>({ public: null, paraphrase: null, compositional: null, oracle: null });
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health>("checking");

  const theme = THEMES.find((t) => t.id === themeId) ?? THEMES[0]!;
  const viz = VIZ_PROFILES.find((v) => v.id === vizId) ?? VIZ_PROFILES[0]!;

  useEffect(() => remember("style", vizId), [vizId]);

  // Parallax: the canvas drifts at a fraction of the page scroll, eased towards its target on
  // every frame so the motion stays smooth whatever the scroll device sends.
  const canvasLayer = useRef<HTMLDivElement>(null);
  const drift = useRef({ target: 0, now: 0, frame: 0 });
  const onPageScroll = useCallback((top: number) => {
    const d = drift.current;
    d.target = -Math.min(top * 0.12, 140);
    if (d.frame) return;
    const step = () => {
      d.now += (d.target - d.now) * 0.12;
      canvasLayer.current?.style.setProperty("--parallax", `${d.now.toFixed(2)}px`);
      d.frame = Math.abs(d.target - d.now) > 0.1 ? requestAnimationFrame(step) : 0;
    };
    d.frame = requestAnimationFrame(step);
  }, []);
  useEffect(() => {
    if (mode === "ask") onPageScroll(0);
  }, [mode, onPageScroll]);

  // A deep health check reads the graph, so it also wakes a suspended workspace.
  useEffect(() => {
    const slow = window.setTimeout(() => setHealth((h) => (h === "checking" ? "waking" : h)), 2500);
    fetchHealth(true)
      .then(() => setHealth("ok"))
      .catch(() => setHealth("down"))
      .finally(() => window.clearTimeout(slow));
    fetchPresets().then(setPresets).catch(() => setPresets([]));
    // Each published document arrives on its own; a missing one only hides its numbers.
    fetchMetrics().then((d) => setBench((b) => ({ ...b, public: d }))).catch(() => undefined);
    fetchMetricSet("paraphrase").then((d) => setBench((b) => ({ ...b, paraphrase: d }))).catch(() => undefined);
    fetchMetricSet("compositional").then((d) => setBench((b) => ({ ...b, compositional: d }))).catch(() => undefined);
    fetchOracle().then((d) => setBench((b) => ({ ...b, oracle: d }))).catch(() => undefined);
    fetchStats().then(setStats).catch(() => setStats(null));
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
      setFocus(citedEvents(next));
      setViewId("investigation");
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

  const legend = useMemo(() => {
    const present = new Set(snapshot.graph_nodes.map((n) => n.layer));
    return (Object.keys(LAYER_NAMES) as LayerId[]).filter((layer) => present.has(layer));
  }, [snapshot]);

  return (
    <main
      style={theme.vars as CSSProperties}
      className={`app-shell theme-${themeId} relative h-dvh w-full overflow-hidden bg-background text-foreground`}
    >
      <div ref={canvasLayer} className="parallax">
        <GraphSurface
          nodes={snapshot.graph_nodes}
          edges={snapshot.graph_edges}
          palette={theme.palette}
          viz={viz}
          selectedId={selected?.id ?? null}
          onSelect={onSelect}
          onLinkSelect={() => undefined}
        />
      </div>
      <div className="hdr pointer-events-none absolute inset-0" />
      <div className="grain pointer-events-none absolute inset-0" />

      <header className="pointer-events-none absolute inset-x-0 top-0 z-20 flex flex-wrap items-center justify-between gap-3 px-4 pt-4 md:px-5 md:pt-5">
        <div className="pointer-events-auto py-2 pl-1">
          <Wordmark />
        </div>
        <div className="pointer-events-auto flex w-full flex-col gap-2 md:w-auto md:flex-row md:items-center">
          <Pill label="Pages" activeKey={mode}>
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
          </Pill>
          <Pill label="Graph display">
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
          <div className="pointer-events-none absolute inset-x-4 bottom-28 z-10 flex max-h-[52vh] items-end lg:top-[10rem] xl:top-[6.75rem] lg:right-auto lg:bottom-28 lg:left-5 lg:max-h-none lg:w-[26rem] lg:items-start">
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
          {result?.agentic.agentic_trace && (
            <div className="pointer-events-none absolute top-[10rem] xl:top-[6.75rem] right-5 bottom-28 z-10 hidden w-[24rem] items-start lg:flex">
              <TracePanel trace={result.agentic.agentic_trace} />
            </div>
          )}
          <div className="pointer-events-none absolute inset-x-0 bottom-12 z-10 hidden justify-center px-5 md:flex">
            <MetricsStrip data={bench} onOpen={() => setMode("benchmark")} />
          </div>
        </>
      ) : (
        <div className="pointer-events-none absolute inset-x-4 top-[13.5rem] bottom-12 z-10 mx-auto flex max-w-[78rem] md:top-[10rem] xl:top-[6.75rem]">
          {mode === "benchmark" ? (
            <Suspense fallback={null}>
              <BenchmarkPage data={bench} themeKey={themeId} onScroll={onPageScroll} />
            </Suspense>
          ) : (
            <DocsPage stats={stats} onScroll={onPageScroll} />
          )}
        </div>
      )}

      <footer className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex flex-wrap items-end justify-between gap-2 px-5 pb-4">
        <div className="pointer-events-auto hidden flex-wrap items-center gap-x-4 gap-y-1 md:flex">
          {legend.map((layer) => (
            <span key={layer} className="tag inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: theme.palette.layers[layer] }} />
              {LAYER_NAMES[layer]}
            </span>
          ))}
          <span className="text-[12px] text-[color:var(--soft)]">
            {selected ? `${selected.type} · ${selected.label}` : snapshot.disclosure}
          </span>
        </div>
        <div className="pointer-events-auto flex items-center gap-2">
          <span className="status-dot" data-state={health} />
          <span className="tag">
            {health === "ok"
              ? "TigerGraph live"
              : health === "waking"
                ? "Waking the graph"
                : health === "down"
                  ? "Graph unreachable"
                  : "Checking the graph"}
          </span>
        </div>
      </footer>
    </main>
  );
}
