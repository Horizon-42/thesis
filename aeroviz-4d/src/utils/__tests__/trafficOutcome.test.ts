import { describe, expect, it } from "vitest";
import { TRAFFIC_OUTCOMES, trafficOutcomeName, trafficOutcomeTitle } from "../trafficOutcome";

describe("trafficOutcomeName", () => {
  it.each([
    ["separated_at_baseline", "separated at the first solve (no re-solve)"],
    ["separated", "separated after re-solve"],
    // MD14: the loop keeps the solve with the fewest counted losses, so both say the record is the best of its solves
    ["unresolved", "loss left — the best of its solves is shown (fewest losses)"],
    ["solve_failed", "loss left — the best of its solves is shown (fewest losses)"],
    ["wake_at_fixed_time", "wake loss left (landing time fixed)"],
  ])("names the loop's outcome %s", (raw, name) => {
    expect(trafficOutcomeName(raw)).toBe(name);
  });

  it.each([
    ["BaselineFailed: slot solve (delay 12.0 s): all 1 IAF(s) infeasible", "not optimized: its slot could not be flown"],
    ["BaselineFailed: ETA solve: all 1 IAF(s) infeasible", "not optimized: no ETA"],
    ["block failed: RuntimeError: casadi", "not optimized: block failed"],
    ["BaselineFailed: all 2 IAF(s) infeasible", "not optimized"],
    ["BaselineFailed", "not optimized"],
  ])("names the reason of an aircraft that was not optimized: %s", (raw, name) => {
    expect(trafficOutcomeName(raw)).toBe(name);
  });

  it("shows an outcome it does not know as it is", () => {
    expect(trafficOutcomeName("something_new")).toBe("something_new");
    expect(trafficOutcomeName("RuntimeError: x")).toBe("RuntimeError: x");
    expect(trafficOutcomeName("")).toBe("");
  });

  it("names every outcome of the loop (none is shown raw)", () => {
    for (const raw of TRAFFIC_OUTCOMES) expect(trafficOutcomeName(raw)).not.toBe(raw);
  });
});

describe("trafficOutcomeTitle", () => {
  it("keeps the raw string after the plain name", () => {
    expect(trafficOutcomeTitle("unresolved")).toBe("loss left — the best of its solves is shown (fewest losses) (unresolved)");
    expect(trafficOutcomeTitle("BaselineFailed: ETA solve: x")).toBe("not optimized: no ETA (BaselineFailed: ETA solve: x)");
  });

  it("does not repeat an outcome that is shown raw", () => {
    expect(trafficOutcomeTitle("something_new")).toBe("something_new");
  });
});
