import { useCallback, useEffect, useState } from "react";
import { ApiError, fetchOpsStatus, type KeyRow, type OpsStatus } from "../api";

// The owner's page at /ops: not linked anywhere, and empty without the admin token, which only
// lives in this tab's session storage.
const TOKEN_KEY = "stellium-ops-token";
const REFRESH_MS = 30_000;
const RENDER_MEMORY_MB = 512;

const CONSOLES = [
  ["Cohere usage", "https://dashboard.cohere.com/"],
  ["TigerGraph Savanna", "https://tgcloud.io/"],
  ["Render", "https://dashboard.render.com/"],
  ["Netlify", "https://app.netlify.com/"],
  ["cron-job.org", "https://console.cron-job.org/"],
] as const;

function readToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

function saveToken(token: string) {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // A private window without storage just asks again next time.
  }
}

function ago(epochS: number | null | undefined): string {
  if (!epochS) return "never";
  const s = Math.max(0, Math.round(Date.now() / 1000 - epochS));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${(s / 3600).toFixed(1)} h ago`;
  return `${(s / 86400).toFixed(1)} d ago`;
}

function duration(s: number): string {
  if (s < 3600) return `${Math.round(s / 60)} min`;
  if (s < 86400) return `${(s / 3600).toFixed(1)} h`;
  return `${(s / 86400).toFixed(1)} d`;
}

type Tone = "ok" | "warn" | "bad";
const TONE: Record<Tone, string> = { ok: "#7fbf8e", warn: "var(--accent)", bad: "var(--destructive)" };

function Dot({ tone }: { tone: Tone }) {
  return <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: TONE[tone] }} />;
}

function Card({ label, tone, value, detail }: { label: string; tone: Tone; value: string; detail: string }) {
  return (
    <div className="rounded-2xl border border-[color:var(--hair)] bg-[color:var(--raise)] p-4">
      <div className="flex items-center gap-2 text-[12px] uppercase tracking-[0.12em] text-[color:var(--faint)]">
        <Dot tone={tone} />
        {label}
      </div>
      <div className="mt-2 text-[22px] font-medium tabular-nums">{value}</div>
      <div className="mt-1 text-[13px] text-[color:var(--soft)]">{detail}</div>
    </div>
  );
}

function keyTone(row: KeyRow): Tone {
  if (row.state === "parked") return "bad";
  if (row.state === "resting" || (row.last_status ?? 200) >= 400) return "warn";
  return "ok";
}

function TokenForm({ onSubmit, error }: { onSubmit: (token: string) => void; error: string | null }) {
  const [value, setValue] = useState("");
  return (
    <form
      className="mx-auto mt-[18vh] w-full max-w-sm rounded-2xl border border-[color:var(--hair)] bg-[color:var(--raise)] p-6"
      onSubmit={(event) => {
        event.preventDefault();
        if (value.trim()) onSubmit(value.trim());
      }}
    >
      <div className="text-[13px] uppercase tracking-[0.14em] text-[color:var(--faint)]">Operator</div>
      <label className="mt-4 block text-[14px] text-[color:var(--soft)]" htmlFor="ops-token">
        Admin token
      </label>
      <input
        id="ops-token"
        type="password"
        autoComplete="current-password"
        className="mt-2 w-full rounded-xl border border-[color:var(--hair)] bg-transparent px-3 py-2.5 text-[15px] outline-none focus:border-[color:var(--accent)]"
        value={value}
        onChange={(event) => setValue(event.target.value)}
      />
      {error && <p className="mt-3 text-[13px] text-[color:var(--destructive)]">{error}</p>}
      <button type="submit" className="nav-cta mt-5 w-full justify-center">
        Open
      </button>
    </form>
  );
}

export function OpsPage() {
  const [token, setToken] = useState(readToken);
  const [status, setStatus] = useState<OpsStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadedAt, setLoadedAt] = useState(0);
  const [, tick] = useState(0);

  const load = useCallback(async (current: string) => {
    try {
      setStatus(await fetchOpsStatus(current));
      setLoadedAt(Date.now() / 1000);
      setError(null);
    } catch (exc) {
      if (exc instanceof ApiError && (exc.status === 401 || exc.status === 404)) {
        saveToken("");
        setToken("");
        setStatus(null);
        setError(exc.status === 404 ? "The API has no admin token set." : "That token is not valid.");
      } else {
        setError(exc instanceof Error ? exc.message : "The API did not answer.");
      }
    }
  }, []);

  useEffect(() => {
    document.title = "STELLIUM · Operator";
    if (!token) return;
    void load(token);
    const refresh = window.setInterval(() => void load(token), REFRESH_MS);
    const clock = window.setInterval(() => tick((n) => n + 1), 5_000);
    return () => {
      window.clearInterval(refresh);
      window.clearInterval(clock);
    };
  }, [token, load]);

  if (!token) {
    return (
      <main className="min-h-dvh bg-[color:var(--background)] px-4 text-[color:var(--foreground)]">
        <TokenForm
          error={error}
          onSubmit={(next) => {
            saveToken(next);
            setToken(next);
          }}
        />
      </main>
    );
  }

  const q = status?.questions;
  const live = status?.keys.filter((k) => k.state !== "parked").length ?? 0;
  const memory = status?.service.memory_mb ?? null;

  return (
    <main className="min-h-dvh bg-[color:var(--background)] px-4 pb-16 text-[color:var(--foreground)] md:px-10">
      <header className="mx-auto flex max-w-6xl flex-wrap items-baseline justify-between gap-3 pt-8">
        <div>
          <div className="font-[family-name:var(--font-wordmark)] text-[28px] tracking-[0.3em]">STELLIUM</div>
          <div className="text-[13px] text-[color:var(--faint)]">
            Operator status · refreshes every 30 s · updated {ago(loadedAt)}
          </div>
        </div>
        <div className="flex gap-2">
          <button className="nav-cta" onClick={() => void load(token)}>
            Refresh
          </button>
          <button
            className="nav-cta"
            onClick={() => {
              saveToken("");
              setToken("");
              setStatus(null);
            }}
          >
            Sign out
          </button>
        </div>
      </header>

      {error && (
        <p className="mx-auto mt-6 max-w-6xl rounded-xl border border-[color:var(--destructive)] px-4 py-3 text-[14px]">
          {error}
        </p>
      )}

      {status && q && (
        <div className="mx-auto mt-8 max-w-6xl space-y-8">
          <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Card
              label="Graph"
              tone={status.graph.ok ? "ok" : "bad"}
              value={status.graph.ok ? `${status.graph.latency_ms} ms` : "Unreachable"}
              detail={status.graph.ok ? `${status.graph.venues} venues read live` : (status.graph.error ?? "")}
            />
            <Card
              label="Keep-alive"
              tone={status.keepalive.last_ok_at ? "ok" : "warn"}
              value={ago(status.keepalive.last_ok_at)}
              detail={
                status.keepalive.last_error
                  ? `Last miss ${ago(status.keepalive.last_error_at)}: ${status.keepalive.last_error}`
                  : status.keepalive.last_ok_at
                    ? `Every ${Math.round(status.keepalive.interval_s / 60)} min while the API runs`
                    : `First beat ${Math.round(status.keepalive.interval_s / 60)} min after start`
              }
            />
            <Card
              label="Questions today"
              tone={q.daily_cap && q.spent_today >= q.daily_cap ? "bad" : q.daily_cap && q.spent_today > q.daily_cap * 0.8 ? "warn" : "ok"}
              value={`${q.spent_today} / ${q.daily_cap || "∞"}`}
              detail={`${q.visitors_last_hour} ${q.visitors_last_hour === 1 ? "visitor" : "visitors"} in the last hour · ${q.hourly_cap_per_visitor || "∞"} per visitor per hour · resets 00:00 UTC`}
            />
            <Card
              label="Model keys"
              tone={live === 0 ? "bad" : live <= 1 ? "warn" : "ok"}
              value={`${live} / ${status.keys.length} live`}
              detail="Parked keys hit their monthly cap or were rejected"
            />
            <Card
              label="Memory"
              tone={memory !== null && memory > RENDER_MEMORY_MB * 0.85 ? "warn" : "ok"}
              value={memory === null ? "n/a" : `${Math.round(memory)} MB`}
              detail={`of ${RENDER_MEMORY_MB} MB on the free instance`}
            />
            <Card
              label="Uptime"
              tone="ok"
              value={duration(status.service.uptime_s)}
              detail={`Commit ${status.service.commit ?? "local"} · ${status.service.provider}`}
            />
          </section>

          <section>
            <h2 className="text-[13px] uppercase tracking-[0.14em] text-[color:var(--faint)]">Model keys</h2>
            <p className="mt-1 text-[13px] text-[color:var(--soft)]">
              Calls this API process made since it started. Trial keys allow 1,000 calls a month and 40 a minute;
              Cohere publishes no remaining balance, so check its dashboard for the month's total.
            </p>
            <div className="mt-3 overflow-x-auto rounded-2xl border border-[color:var(--hair)]">
              <table className="w-full min-w-[640px] text-left text-[14px] tabular-nums">
                <thead className="text-[12px] uppercase tracking-[0.1em] text-[color:var(--faint)]">
                  <tr>
                    {["Tier", "Key", "State", "Chat", "Embed", "Errors", "Last status", "This minute", "Last used"].map((h) => (
                      <th key={h} className="px-4 py-3 font-normal">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {status.keys.map((row) => (
                    <tr key={row.alias} className="border-t border-[color:var(--hair)]">
                      <td className="px-4 py-2.5">{row.tier}</td>
                      <td className="px-4 py-2.5 font-[family-name:var(--font-mono)] text-[13px]">{row.alias}</td>
                      <td className="px-4 py-2.5">
                        <span className="inline-flex items-center gap-2">
                          <Dot tone={keyTone(row)} />
                          {row.state}
                        </span>
                      </td>
                      <td className="px-4 py-2.5">{row.chat ?? 0}</td>
                      <td className="px-4 py-2.5">{row.embed ?? 0}</td>
                      <td className="px-4 py-2.5">{row.errors ?? 0}</td>
                      <td className="px-4 py-2.5">{row.last_status ?? "–"}</td>
                      <td className="px-4 py-2.5">
                        {row.minute_remaining !== undefined ? `${row.minute_remaining} / ${row.minute_limit ?? "?"} left` : "–"}
                      </td>
                      <td className="px-4 py-2.5 text-[color:var(--soft)]">{ago(row.last_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 className="text-[13px] uppercase tracking-[0.14em] text-[color:var(--faint)]">Recent questions</h2>
            {status.recent.length === 0 ? (
              <p className="mt-2 text-[14px] text-[color:var(--soft)]">None since the API started.</p>
            ) : (
              <ul className="mt-3 divide-y divide-[color:var(--hair)] rounded-2xl border border-[color:var(--hair)]">
                {status.recent.map((item) => (
                  <li key={`${item.at}-${item.question}`} className="px-4 py-3">
                    <div className="flex flex-wrap items-baseline justify-between gap-2">
                      <span className="text-[15px]">{item.question}</span>
                      <span className="text-[12px] text-[color:var(--faint)] tabular-nums">
                        {ago(item.at)} · {item.intent} · {(item.total_ms / 1000).toFixed(1)} s · {item.llm_tokens.toLocaleString()} tokens
                      </span>
                    </div>
                    {Object.keys(item.pipelines).length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-x-5 gap-y-1 text-[13px] text-[color:var(--soft)]">
                        {Object.entries(item.pipelines).map(([name, run]) => (
                          <span key={name}>
                            <b className="font-medium text-[color:var(--foreground)]">{name}</b> {run.answer} ·{" "}
                            {(run.latency_ms / 1000).toFixed(1)} s
                          </span>
                        ))}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section>
            <h2 className="text-[13px] uppercase tracking-[0.14em] text-[color:var(--faint)]">Balances and settings</h2>
            <p className="mt-1 text-[13px] text-[color:var(--soft)]">
              Credits and monthly totals live in each provider's console.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {CONSOLES.map(([name, href]) => (
                <a key={name} className="nav-cta" href={href} target="_blank" rel="noreferrer">
                  {name}
                </a>
              ))}
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
