// Benchmark charts in the page's own palette: the three pipelines are the theme's tonal ramp,
// grids are hairlines, and tooltips match the menus. Colours are resolved from the live theme so
// SVG fills follow a theme change.
import { useLayoutEffect, useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { Metrics, PipelineId } from "../api";
import { PIPELINES, tokens } from "../format";

export interface Palette {
  rag: string;
  graphrag: string;
  agentic: string;
  text: string;
  faint: string;
  grid: string;
}

// Turn any CSS colour (var(), color-mix(), oklab) into plain rgba by painting one pixel, so every
// SVG renderer accepts it.
function resolve(probe: HTMLElement, value: string, ctx: CanvasRenderingContext2D): string {
  probe.style.color = value;
  ctx.clearRect(0, 0, 1, 1);
  ctx.fillStyle = getComputedStyle(probe).color;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
  return `rgba(${r}, ${g}, ${b}, ${((a ?? 255) / 255).toFixed(3)})`;
}

export function usePalette(themeKey: string): Palette | null {
  const [palette, setPalette] = useState<Palette | null>(null);
  useLayoutEffect(() => {
    const host = document.querySelector("main") ?? document.body;
    const probe = document.createElement("span");
    host.appendChild(probe);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 1;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (ctx) {
      setPalette({
        rag: resolve(probe, "var(--rag)", ctx),
        graphrag: resolve(probe, "var(--graphrag)", ctx),
        agentic: resolve(probe, "var(--agentic)", ctx),
        text: resolve(probe, "var(--soft)", ctx),
        faint: resolve(probe, "var(--faint)", ctx),
        grid: resolve(probe, "var(--hair)", ctx),
      });
    }
    probe.remove();
  }, [themeKey]);
  return palette;
}

function Tip({ active, payload, label, unit }: { active?: boolean; payload?: { name?: string; value?: number; color?: string; payload?: Record<string, unknown> }[]; label?: string; unit?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      {label && <div className="mb-1.5 font-semibold text-[color:var(--foreground)]">{label}</div>}
      {payload.map((item) => (
        <div key={item.name} className="flex items-center justify-between gap-5 text-[color:var(--soft)]">
          <span className="inline-flex items-center gap-2">
            <i className="inline-block h-2 w-2 rounded-full" style={{ background: item.color }} />
            {item.name}
          </span>
          <span className="font-semibold text-[color:var(--foreground)] tabular-nums">
            {typeof item.value === "number" ? `${Math.round(item.value * 10) / 10}${unit ?? ""}` : item.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function ChartCard({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return (
    <figure className="chart-card m-0">
      <figcaption className="mb-3">
        <div className="chart-title">{title}</div>
        {note && <div className="chart-note">{note}</div>}
      </figcaption>
      {children}
    </figure>
  );
}

export function Legend() {
  return (
    <div className="legend mb-3">
      {PIPELINES.map((p) => (
        <span key={p.id} style={{ ["--lane" as string]: p.color }}>
          <i />
          {p.name}
        </span>
      ))}
    </div>
  );
}

const axis = (palette: Palette) => ({
  stroke: palette.grid,
  tick: { fill: palette.faint, fontSize: 12 },
  tickLine: false,
  axisLine: false,
});

const QTYPE_SHORT: Record<string, string> = {
  aggregation: "Count",
  superlative: "Rank",
  lookup: "Lookup",
  temporal: "Previous Games",
  multi_hop: "Venue & date",
};

export function AccuracyByType({ metrics, palette }: { metrics: Metrics; palette: Palette }) {
  const qtypes = Object.keys(metrics.pipelines.agentic?.by_type ?? {});
  const data = qtypes.map((qtype) => {
    const row: Record<string, string | number> = { type: QTYPE_SHORT[qtype] ?? qtype };
    for (const p of PIPELINES) {
      const g = metrics.pipelines[p.id]?.by_type[qtype];
      row[p.name] = g ? (100 * g.exact_match) / g.questions : 0;
    }
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} barGap={3} barCategoryGap="26%" margin={{ top: 8, right: 4, left: -18, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke={palette.grid} />
        <XAxis dataKey="type" {...axis(palette)} />
        <YAxis domain={[0, 100]} ticks={[0, 50, 100]} {...axis(palette)} />
        <Tooltip cursor={{ fill: palette.grid }} content={<Tip unit="%" />} />
        {PIPELINES.map((p) => (
          <Bar
            key={p.id}
            dataKey={p.name}
            fill={palette[p.id]}
            radius={[5, 5, 1, 1]}
            isAnimationActive={false}
            animationEasing="ease-out"
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function RetrievalRadar({ metrics, palette }: { metrics: Metrics; palette: Palette }) {
  const measures: [string, (id: PipelineId) => number][] = [
    ["Hit@1", (id) => metrics.pipelines[id]?.retrieval?.hit_at_1 ?? 0],
    ["Hit@5", (id) => metrics.pipelines[id]?.retrieval?.hit_at_5 ?? 0],
    ["MRR", (id) => metrics.pipelines[id]?.retrieval?.mrr ?? 0],
    ["nDCG@5", (id) => metrics.pipelines[id]?.retrieval?.ndcg_at_5 ?? 0],
    ["Precision@5", (id) => metrics.pipelines[id]?.retrieval?.precision_at_5 ?? 0],
    ["Context recall", (id) => metrics.pipelines[id]?.retrieval?.context_recall ?? 0],
  ];
  const data = measures.map(([name, read]) => ({
    measure: name,
    ...Object.fromEntries(PIPELINES.map((p) => [p.name, Math.round(read(p.id) * 1000) / 10])),
  }));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke={palette.grid} />
        <PolarAngleAxis dataKey="measure" tick={{ fill: palette.text, fontSize: 12 }} />
        <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
        <Tooltip content={<Tip unit="%" />} />
        {PIPELINES.map((p) => (
          <Radar
            key={p.id}
            name={p.name}
            dataKey={p.name}
            stroke={palette[p.id]}
            fill={palette[p.id]}
            fillOpacity={p.id === "agentic" ? 0.22 : 0.08}
            strokeWidth={1.6}
            isAnimationActive={false}
          />
        ))}
      </RadarChart>
    </ResponsiveContainer>
  );
}

export function Efficiency({ metrics, palette }: { metrics: Metrics; palette: Palette }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 16, right: 30, left: -12, bottom: 4 }}>
        <CartesianGrid stroke={palette.grid} />
        <XAxis type="number" dataKey="tokens" name="LLM tokens / question" domain={[0, "auto"]} {...axis(palette)}
          tickFormatter={(v: number) => tokens(v)} />
        <YAxis type="number" dataKey="accuracy" name="Exact match" domain={[50, 100]} ticks={[50, 75, 100]} unit="%" {...axis(palette)} />
        <ZAxis type="number" dataKey="z" range={[240, 240]} />
        <Tooltip
          cursor={{ stroke: palette.grid }}
          content={({ active, payload }) => {
            const point = payload?.[0]?.payload as { name: string; tokens: number; accuracy: number } | undefined;
            if (!active || !point) return null;
            return (
              <div className="chart-tip">
                <div className="font-semibold">{point.name}</div>
                <div className="text-[color:var(--soft)]">
                  {point.accuracy}% exact · {tokens(point.tokens)} tokens / q
                </div>
              </div>
            );
          }}
        />
        {PIPELINES.map((p) => {
          const o = metrics.pipelines[p.id]?.overall;
          if (!o) return null;
          const point = { name: p.name, tokens: Math.round(o.tokens_mean), accuracy: Math.round((100 * o.exact_match) / o.scored), z: 1 };
          return (
            <Scatter key={p.id} name={p.name} data={[point]} fill={palette[p.id]} isAnimationActive={false}>
              <LabelList
                dataKey="name"
                position={p.id === "agentic" ? "left" : p.id === "graphrag" ? "right" : "top"}
                offset={14}
                style={{ fill: palette.text, fontSize: 12.5 }}
              />
            </Scatter>
          );
        })}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function ToolMix({ tools, palette }: { tools: Record<string, number>; palette: Palette }) {
  const data = Object.entries(tools)
    .sort((a, b) => b[1] - a[1])
    .map(([tool, count]) => ({ tool, count }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 36, left: 8, bottom: 0 }}>
        <XAxis type="number" hide />
        <YAxis type="category" dataKey="tool" width={150} {...axis(palette)} tick={{ fill: palette.text, fontSize: 12.5, fontFamily: "IBM Plex Mono, monospace" }} />
        <Tooltip cursor={{ fill: palette.grid }} content={<Tip />} />
        <Bar dataKey="count" name="Calls" fill={palette.agentic} radius={[1, 5, 5, 1]} barSize={14} isAnimationActive={false}>
          <LabelList dataKey="count" position="right" style={{ fill: palette.text, fontSize: 12 }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
