import { describe, expect, it } from "vitest";

import { isEvaluationReport } from "../evaluationReport";

function reportWith(observed: unknown) {
  return {
    schema_version: "terminal-approach-evaluation-v2",
    total: 0,
    solved: 0,
    verdict_counts: { pass: 0, fail: 0, indeterminate: 0 },
    assessment_contexts: [],
    trajectories: [],
    observed,
  };
}

describe("isEvaluationReport observed availability contract", () => {
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
