import { describe, expect, it } from "vitest";

import { isEvaluationReport } from "../evaluationReport";

function reportWith(observed: unknown) {
  return {
    schema_version: "terminal-approach-evaluation-v3",
    methodology: {
      terminal_vertical: {
        reference: "LTP elevation MSL + published FAS TCH",
        trajectory_altitude_datum: "msl",
        target_context_tolerance_m: 0.01,
        lpv: {
          scale_model: "do229_lpv_angular_min_clamped",
          one_sided_minimum_fsd_m: 15,
          normal_fsd_fraction: 0.5,
          effective_threshold_bound_m: 7.5,
          sources: [
            { document: "ICAO Doc 9613", location: "§5.3.3.1.1.1(b)", use: "half FSD" },
          ],
        },
      },
    },
    total: 0,
    solved: 0,
    verdict_counts: { pass: 0, fail: 0, indeterminate: 0 },
    assessment_contexts: [],
    trajectories: [],
    observed,
  };
}

describe("isEvaluationReport observed availability contract", () => {
  it("rejects the retired v2 report contract", () => {
    expect(isEvaluationReport({
      ...reportWith(undefined),
      schema_version: "terminal-approach-evaluation-v2",
    })).toBe(false);
  });

  it("rejects v3 without the auditable LPV vertical methodology", () => {
    expect(isEvaluationReport({
      ...reportWith(undefined),
      methodology: {},
    })).toBe(false);
  });

  it("accepts a consistent pre-filter arrival-candidate aggregate", () => {
    expect(isEvaluationReport(reportWith({
      denominator: "arrival_candidates_excluding_not_landing",
      event_denominator: 3,
      event_estimated: 1,
      event_unavailable: 2,
      event_estimated_rate: 1 / 3,
      excluded_not_landing: 1,
    }))).toBe(true);
  });

  it("rejects the old selected-record availability shape", () => {
    expect(isEvaluationReport(reportWith({
      event_estimated: 1,
      event_unavailable: 0,
      event_estimated_rate: 1,
    }))).toBe(false);
  });
});
