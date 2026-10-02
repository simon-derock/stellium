import { describe, expect, it } from "vitest";
import type { CompareResult, PipelineResult } from "./api";
import { chromeFor, citedEvents, contrast, percent, seconds, summarize, traceSteps } from "./format";

const result = (docs: string[], answer = "Chen Ding"): PipelineResult =>
  ({ retrieved_doc_ids: docs, answer }) as unknown as PipelineResult;

describe("formatting", () => {
  it("prints seconds and percentages for the ledger", () => {
    expect(seconds(3776)).toBe("3.8 s");
    expect(seconds(13_900)).toBe("14 s");
    expect(percent(99, 100)).toBe("99%");
    expect(percent(1, 0)).toBe("n/a");
  });

  it("collects cited events once, agent first", () => {
    const compare = {
      rag: result(["Q3", "Q1"]),
      graphrag: result(["Q1"]),
      agentic: result(["Q1", "Q2"]),
    } as unknown as CompareResult;
    expect(citedEvents(compare)).toEqual(["Q1", "Q2", "Q3"]);
    expect(citedEvents(null)).toEqual([]);
    const missed = { ...compare, rag: result(["Q9"], "Not found in corpus") } as CompareResult;
    expect(citedEvents(missed)).toEqual(["Q1", "Q2"]);
  });

  it("summarizes an observation by its value, error or leader", () => {
    expect(summarize({ event: "x", value: "7" })).toBe("value → 7");
    expect(summarize({ error: "no match" })).toBe("error → no match");
    expect(summarize({ top: [{ event: "Men's eight", competitor_count: 56 }] })).toBe(
      "top → Men's eight (56)",
    );
  });
});

describe("trace steps", () => {
  it("pairs each plan with its observation, note and source line", () => {
    const steps = traceSteps([
      { event: "plan", thought: "Rank first.", action: "rank_events", action_input: { year: 2016 } },
      {
        event: "observation",
        latency_ms: 470,
        observation: { top: [{ event: "Women's eight", competitor_count: 56 }], note: "This event is an intermediate result." },
      },
      { event: "plan", thought: "Read nations.", action: "event_attribute", action_input: { attribute: "nation_count" } },
      {
        event: "observation",
        observation: { value: "7", source_check: [{ doc_id: "Q1", line: "nations: 7" }], linking: {} },
      },
    ]);
    expect(steps).toHaveLength(2);
    expect(steps[0]).toMatchObject({ index: 1, action: "rank_events", input: "year: 2016", latencyMs: 470 });
    expect(steps[0]?.note).toContain("intermediate");
    expect(steps[1]).toMatchObject({ observation: "value → 7", sourceLine: "nations: 7" });
  });

  it("shows a final answer as a finish step", () => {
    const [step] = traceSteps([{ event: "plan", thought: "Done.", final_answer: "Chen Long" }]);
    expect(step).toMatchObject({ action: "finish", input: "Chen Long" });
  });
});

describe("theme chrome", () => {
  it("tells light themes from dark ones", () => {
    expect(chromeFor({ "--background": "#f3f1ea", "--accent": "#7a4b25" }).scheme).toBe("light");
    expect(chromeFor({ "--background": "#070604", "--accent": "#b8873a" }).scheme).toBe("dark");
  });

  it("lifts an accent too faint to read, and leaves a good one alone", () => {
    expect(chromeFor({ "--background": "#fbfaff", "--accent": "#ff7a18" }).overrides["--accent"]).toContain("color-mix");
    expect(chromeFor({ "--background": "#070604", "--accent": "#b8873a" }).overrides).toEqual({});
    expect(contrast("#000000", "#ffffff")).toBeCloseTo(21, 0);
  });
});
