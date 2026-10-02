// STELLIUM console: the Olympic graph full bleed, a question plate on the left, the agent's trace
// on the right, and the measured benchmark one tab away.
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  ApiError,
  compare,
  fetchHealth,
  fetchMetrics,
  fetchPresets,
  fetchSnapshot,
  type CompareResult,
  type Metrics,
  type Preset,
} from "./api";
import { AskPanel } from "./components/AskPanel";
import { BenchmarkSheet } from "./components/BenchmarkSheet";
import { Menu } from "./components/Menu";
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

type Mode = "ask" | "benchmark";
type Health = "checking" | "ok" | "waking" | "down";

const EMPTY: Snapshot = { graph_nodes: [], graph_edges: [], disclosure: "" };
const REPO_URL = "https://github.com/simon-derock/stellium";

function remembered(key: string, fallback: string, allowed: string[]): string {
  try {
    const value = localStorage.getItem(`stellium.${key}`);
    return value && allowed.includes(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

function remember(key: string, value: string) {
  try {
    localStorage.setItem(`stellium.${key}`, value);
  } catch {
    // Private windows and blocked storage just forget the choice.
  }
}

// Shareable links: ?q= asks a question on load, ?tab=benchmark opens the benchmark.
const LINK = new URLSearchParams(window.location.search);
const LINKED_QUESTION = LINK.get("q")?.trim().slice(0, 500) ?? "";

export function App() {
  const [mode, setMode] = useState<Mode>(LINK.get("tab") === "benchmark" ? "benchmark" : "ask");
  const [themeId, setThemeId] = useState(() => remembered("theme", "obsidian-observatory", THEMES.map((t) => t.id)));
  const [vizId, setVizId] = useState(() => remembered("style", "orrery", VIZ_PROFILES.map((v) => v.id)));
  const [viewId, setViewId] = useState<ViewId>("constellation");
  const [focus, setFocus] = useState<string[]>([]);
  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health>("checking");

  const theme = THEMES.find((t) => t.id === themeId) ?? THEMES[0]!;
  const viz = VIZ_PROFILES.find((v) => v.id === vizId) ?? VIZ_PROFILES[0]!;

  useEffect(() => remember("theme", themeId), [themeId]);
  useEffect(() => remember("style", vizId), [vizId]);

  // A deep health check reads the graph, so it also wakes a suspended workspace.
  useEffect(() => {
    const slow = window.setTimeout(() => setHealth((h) => (h === "checking" ? "waking" : h)), 2500);
    fetchHealth(true)
      .then(() => setHealth("ok"))
      .catch(() => setHealth("down"))
      .finally(() => window.clearTimeout(slow));
    fetchPresets().then(setPresets).catch(() => setPresets([]));
    fetchMetrics().then(setMetrics).catch(() => setMetrics(null));
    return () => window.clearTimeout(slow);
  }, []);

  useEffect(() => {
    let live = true;
    fetchSnapshot(viewId, viewId === "investigation" ? focus : [])
      .then((next) => live && setSnapshot(next))
      .catch(() => live && setSnapshot({ ...EMPTY, disclosure: "The graph is not reachable right now." }));
    return () => {
      live = false;
    };
  }, [viewId, focus]);

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
      <GraphSurface
        nodes={snapshot.graph_nodes}
        edges={snapshot.graph_edges}
        palette={theme.palette}
        viz={viz}
        selectedId={selected?.id ?? null}
        onSelect={onSelect}
        onLinkSelect={() => undefined}
      />
      <div className="hdr pointer-events-none absolute inset-0" />
      <div className="grain pointer-events-none absolute inset-0" />

      <header className="pointer-events-none absolute inset-x-0 top-0 z-10 flex flex-wrap items-start justify-between gap-3 p-4">
        <div className="pointer-events-auto px-1 pt-1">
          <Wordmark />
        </div>
        <div className="header-controls pointer-events-auto relative flex flex-wrap items-center justify-end gap-x-2 gap-y-1">
          <Menu
            tag="view"
            value={viewId}
            onChange={(id) => setViewId(id as ViewId)}
            align="end"
            width="17rem"
            wheelExplore
            options={VIEWS.map((v) => ({ id: v.id, name: v.name, hint: v.hint }))}
          />
          <Menu
            tag="style"
            value={vizId}
            onChange={(id) => {
              setVizId(id);
              const next = VIZ_PROFILES.find((v) => v.id === id);
              if (next && THEMES.some((t) => t.id === next.preferredTheme)) setThemeId(next.preferredTheme);
            }}
            align="end"
            width="17rem"
            keepOpenOnSelect
            wheelExplore
            options={VIZ_PROFILES.map((v) => ({ id: v.id, name: v.name, hint: v.hint }))}
          />
          <Menu
            tag="theme"
            value={themeId}
            onChange={setThemeId}
            align="end"
            width="17rem"
            keepOpenOnSelect
            wheelExplore
            options={THEMES.map((t) => ({ id: t.id, name: t.name, hint: t.hint, swatches: Object.values(t.palette.layers) }))}
          />
          <nav className="ml-3 flex items-center gap-1 border-l border-border pl-3">
            <button className="tab" aria-current={mode === "ask" ? "page" : undefined} onClick={() => setMode("ask")}>
              Ask
            </button>
            <button
              className="tab"
              aria-current={mode === "benchmark" ? "page" : undefined}
              onClick={() => setMode("benchmark")}
            >
              Benchmark
            </button>
            <a className="tab inline-flex items-center" href={REPO_URL} target="_blank" rel="noreferrer">
              Code
            </a>
          </nav>
        </div>
      </header>

      {mode === "ask" ? (
        <>
          <div className="pointer-events-none absolute inset-x-4 bottom-14 z-10 flex max-h-[58vh] items-end lg:top-[6.5rem] lg:right-auto lg:bottom-14 lg:max-h-none lg:w-[25rem] lg:items-start">
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
            <div className="pointer-events-none absolute top-[6.5rem] right-4 bottom-14 z-10 hidden w-[23rem] items-start lg:flex">
              <TracePanel trace={result.agentic.agentic_trace} />
            </div>
          )}
        </>
      ) : (
        <div className="pointer-events-none absolute inset-x-4 top-[6.5rem] bottom-14 z-10 mx-auto flex max-w-[66rem] items-start">
          <BenchmarkSheet metrics={metrics} />
        </div>
      )}

      <footer className="pointer-events-none absolute inset-x-0 bottom-0 z-10 flex flex-wrap items-end justify-between gap-2 p-4">
        <div className="pointer-events-auto flex flex-wrap items-center gap-x-4 gap-y-1">
          {legend.map((layer) => (
            <span key={layer} className="tag inline-flex items-center gap-1.5">
              <span className="h-2 w-[3px]" style={{ background: theme.palette.layers[layer] }} />
              {LAYER_NAMES[layer]}
            </span>
          ))}
          <span className="font-mono text-[9.5px] text-muted-foreground">
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
