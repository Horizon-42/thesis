import { EVALUATION_REPORT_SCHEMA_VERSION, type EvaluationReport, type EvaluationRow } from "../evaluationReport";

/**
 * A minimal evaluation report the viewer's guard (`isEvaluationReport`) accepts, holding `rows`. The methodology block is the
 * one the guard checks for (the common RNAV terminal vertical bound); every other field the tests read sits on the rows.
 */
export function reportOf(rows: ReadonlyArray<Partial<EvaluationRow> & Record<string, unknown>>): EvaluationReport {
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
          claim_boundary: "terminal final-approach geometry; not touchdown or landing certification",
        },
      },
    },
    total: rows.length,
    solved: rows.length,
    verdict_counts: { pass: 0, fail: rows.length, indeterminate: 0 },
    assessment_contexts: [],
    trajectories: rows.map((row, n) => ({
      id: `row${n}`, file: null, subject: "optimized", solved: true, verdict: "fail",
      bounds: { lateral_criterion: "runway_half_width_at_threshold", lateral_m: 300, vertical_lower_m: -22, vertical_upper_m: 22,
        speed_lower_ms: 55.2, speed_upper_ms: 60.24 },
      violations: [], ...row,
    })),
  } as unknown as EvaluationReport;
}
