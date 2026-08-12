import { describe, expect, it } from "vitest";

import type { EvaluationRow } from "../../data/evaluationReport";
import {
  OBSERVED_VERDICT_COLORS,
  countVerdicts,
  verdictOfRow,
  verdictsByFlightKey,
} from "../observedVerdictColors";

function row(over: Partial<EvaluationRow> = {}): EvaluationRow {
  return {
    id: "AAL123",
    file: "AAL123_05L_a1b2c3_20260101T000000Z_eval.json",
    solved: true,
    success: true,
    verdict: "pass",
    event_status: "estimated",
    lateral_result: "pass",
    vertical_result: "pass",
    violations: [],
    flight_key: "AAL123_05L_a1b2c3_20260101T000000Z",
    subject: "observed",
    airport: "KRDU",
    runway: "05L",
    benchmark: "lpv",
    bounds: { guidance_lateral_m: 53.375, runway_lateral_m: 22.86,
      effective_lateral_m: 22.86, vertical_lower_m: null, vertical_upper_m: null },
    ...over,
  };
}

describe("verdictOfRow", () => {
  it("passes a flight inside both gates", () => {
    expect(verdictOfRow(row())).toBe("pass");
  });

  it("fails a flight outside a gate", () => {
    expect(verdictOfRow(row({ success: false, verdict: "fail", lateral_result: "fail", violations: ["lateral"] }))).toBe(
      "fail",
    );
  });

  it("maps an explicit indeterminate result to the neutral state", () => {
    expect(verdictOfRow(row({ success: false, verdict: "indeterminate",
      vertical_result: "indeterminate" }))).toBe("undecided");
  });

  it("uses the explicit verdict rather than deriving one from success", () => {
    expect(verdictOfRow(row({ success: false, verdict: "indeterminate" }))).toBe("undecided");
  });
});

describe("verdictsByFlightKey", () => {
  it("indexes on flight_key", () => {
    const map = verdictsByFlightKey([row(), row({ flight_key: "B_23L_dd_20260101T010000Z", success: false, verdict: "fail" })]);
    expect(map.get("AAL123_05L_a1b2c3_20260101T000000Z")).toBe("pass");
    expect(map.get("B_23L_dd_20260101T010000Z")).toBe("fail");
  });

  it("skips rows with no flight_key rather than guessing from the callsign", () => {
    // Callsigns are not unique — 552 distinct callsigns across 996 KRDU arrivals — so
    // falling back to `id` would silently paint one flight's verdict onto its namesake.
    expect(verdictsByFlightKey([row({ flight_key: undefined })]).size).toBe(0);
  });

  it("does not collapse two flights sharing a callsign", () => {
    const map = verdictsByFlightKey([
      row({ flight_key: "AAL1_05L_aaa_20260101T000000Z", success: true }),
      row({ flight_key: "AAL1_05L_bbb_20260102T000000Z", success: false, verdict: "fail" }),
    ]);
    expect(map.size).toBe(2);
    expect(map.get("AAL1_05L_aaa_20260101T000000Z")).toBe("pass");
    expect(map.get("AAL1_05L_bbb_20260102T000000Z")).toBe("fail");
  });
});

describe("countVerdicts", () => {
  it("counts each state", () => {
    expect(countVerdicts(["pass", "pass", "fail", "undecided"])).toEqual({
      pass: 2,
      fail: 1,
      undecided: 1,
    });
  });

  it("reports zeroes for an empty batch", () => {
    expect(countVerdicts([])).toEqual({ pass: 0, fail: 0, undecided: 0 });
  });
});

describe("colours", () => {
  it("gives the three states distinct colours", () => {
    const values = Object.values(OBSERVED_VERDICT_COLORS);
    expect(new Set(values).size).toBe(values.length);
  });
});
