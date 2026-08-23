import { describe, expect, it } from "vitest";

import {
  EVALUATION_REPORT_SCHEMA_VERSION,
  LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS,
  isEvaluationReport,
  isLegacyEvaluationReport,
  type EvaluationReport,
} from "../evaluationReport";

function reportWith(observed: unknown) {
  return {
    schema_version: EVALUATION_REPORT_SCHEMA_VERSION,
    methodology: {
      terminal_vertical: {
        reference: "LTP elevation MSL + published FAS TCH",
        trajectory_altitude_datum: "msl",
        target_context_tolerance_m: 0.01,
        common_rnav_terminal_acceptance: {
          standard_id: "icao_doc_9613_rnp_apch_fas_22m",
          lower_m: -22,
          upper_m: 22,
          source: {
            document: "ICAO Doc 9613",
            location: "Volume II, Part C, Chapter 5, Section A, §5.3.4.4.7",
            use: "common RNAV terminal vertical bound",
          },
          claim_boundary:
            "terminal final-approach geometry; not touchdown or landing certification",
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

  it("accepts published pre-speed-gate v5 reports and flags them as legacy", () => {
    const v5 = {
      ...reportWith(undefined),
      schema_version: "terminal-approach-evaluation-v5",
    };
    expect(isEvaluationReport(v5)).toBe(true);
    expect(isLegacyEvaluationReport(v5 as unknown as EvaluationReport)).toBe(true);
    expect(
      isLegacyEvaluationReport(reportWith(undefined) as unknown as EvaluationReport),
    ).toBe(false);
    expect(LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS).toContain(
      "terminal-approach-evaluation-v5",
    );
  });

  it("rejects the current schema without the auditable common RNAV vertical methodology", () => {
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
