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
    violations: [],
    flight_key: "AAL123_05L_a1b2c3_20260101T000000Z",
    subject: "observed",
    established: true,
    extrapolated: true,
    marginal: false,
    ...over,
  };
}

describe("verdictOfRow", () => {
  it("passes a flight inside both gates", () => {
    expect(verdictOfRow(row())).toBe("pass");
  });

  it("fails a flight outside a gate", () => {
    expect(verdictOfRow(row({ success: false, violations: ["lateral 200 m > 106.75 m"] }))).toBe(
      "fail",
    );
  });

  it("calls a marginal PASS undecidable, not a pass", () => {
    // The whole point of the third state: 71% of real KRDU flights land here, and
    // claiming them as passes would overstate what a 25 ft-quantised measurement knows.
    expect(verdictOfRow(row({ success: true, marginal: true }))).toBe("undecided");
  });

  it("calls a marginal FAIL undecidable, not a failure", () => {
    expect(verdictOfRow(row({ success: false, marginal: true }))).toBe("undecided");
  });

  it("calls a not-established flight undecidable, not a failure", () => {
    // No stabilised final approach to fit means no crossing to judge — that is a
    // statement about the track, not about how the approach was flown.
    expect(
      verdictOfRow(row({ established: false, success: false, violations: ["not_established"] })),
    ).toBe("undecided");
  });

  it("treats an optimizer row (no established/marginal fields) as a plain verdict", () => {
    expect(verdictOfRow({ id: "x", file: null, solved: true, success: true, violations: [] })).toBe(
      "pass",
    );
  });
});

describe("verdictsByFlightKey", () => {
  it("indexes on flight_key", () => {
    const map = verdictsByFlightKey([row(), row({ flight_key: "B_23L_dd_20260101T010000Z", success: false })]);
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
      row({ flight_key: "AAL1_05L_bbb_20260102T000000Z", success: false }),
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
