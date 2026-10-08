/**
 * The reader of a set's results (outline §6.2 items 3, 4; D134), on the answers the backend writes: each stage's sections
 * typed, a section without fields kept with its reason, a refused answer named.
 */
import { describe, expect, it } from "vitest";
import { parseTrainingSetResults } from "../trainingSetResults";
import { resultsAnswer } from "./trainingResults";

describe("a set's results", () => {
  it("reads each stage's sections, and a section without fields with its reason", () => {
    const a = parseTrainingSetResults(true, resultsAnswer("A"), "fixture_set");
    const b = parseTrainingSetResults(true, resultsAnswer("B"), "fixture_set");
    const val = parseTrainingSetResults(true, resultsAnswer("Bval"), "fixture_val");
    const missing = parseTrainingSetResults(true, resultsAnswer("Cmissing"), "fixture-windows");
    if (!a.ok || !b.ok || !val.ok || !missing.ok) throw new Error("an answer does not parse");
    expect(a.value.stage === "A" && a.value.labelling.ok && Object.keys(a.value.labelling.value).sort()).toEqual(["select", "train"]);
    expect(b.value.stage === "B" && !b.value.validation.ok && b.value.validation.problem).toContain("claimed validation readout");
    expect(val.value.stage === "B" && val.value.validation.ok && val.value.validation.value.lossPerStep).toBe(1.08);
    expect(b.value.stage === "B" && b.value.speed.ok && b.value.speed.value.settings[0].rowMs.max).toBeCloseTo(8);
    expect(missing.value.stage === "C" && !missing.value.rounds.ok && missing.value.rounds.problem).toContain("does not exist");
  });

  it("reads stage C's start readout (frontend §4.3): its source round and selection, or why it has none — one of the two", () => {
    const c = parseTrainingSetResults(true, resultsAnswer("C"), "fixture-windows");
    if (!c.ok || c.value.stage !== "C" || !c.value.rounds.ok) throw new Error("stage C's answer does not parse");
    const written = resultsAnswer("C").sections.rounds.start;
    expect(c.value.rounds.value.start).toEqual({ selection: written.selection, why: null });
    const shape = (change: (start: Record<string, any>) => void, says: string) => {
      const raw = resultsAnswer("C");
      change(raw.sections.rounds.start);
      const parsed = parseTrainingSetResults(true, raw, "fixture-windows");     // a malformed section refuses the answer
      expect(parsed.ok).toBe(false);
      if (!parsed.ok) expect(parsed.problem).toContain(says);
    };
    shape((start) => { start.why = "other windows"; }, "one of the two");
    shape((start) => { start.selection = null; }, "one of the two");
    const missing = resultsAnswer("C");
    delete missing.sections.rounds.start;
    const parsed = parseTrainingSetResults(true, missing, "fixture-windows");
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain("start");
  });

  it("reads stage D's rounds by their airports' all cells, and its start's why — a start with a readout refused", () => {
    const d = parseTrainingSetResults(true, resultsAnswer("D"), "fixture-windows-two");
    if (!d.ok || d.value.stage !== "D" || !d.value.rounds.ok) throw new Error("stage D's answer does not parse");
    const written = resultsAnswer("D").sections.rounds;
    expect(d.value.rounds.value.rounds[0].selection).toEqual(written.rounds[0].selection);
    expect(d.value.rounds.value.start).toEqual({ selection: null, why: written.start.why });
    const read = resultsAnswer("D");
    read.sections.rounds.start = { selection: resultsAnswer("C").sections.rounds.start.selection, why: null };
    const parsed = parseTrainingSetResults(true, read, "fixture-windows-two");
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain("stage C's windows");
  });

  it("names a refused answer, and an answer of another shape", () => {
    const refused = parseTrainingSetResults(false, { ok: false, error: "lists no set x" }, "x");
    expect(refused).toEqual({ ok: false, problem: "No results for x: lists no set x" });
    const broken = resultsAnswer("A");
    broken.sections.labelling.splits.train.labelled = "ten";
    const parsed = parseTrainingSetResults(true, broken, "fixture_set");
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain("labelled");
  });
});
