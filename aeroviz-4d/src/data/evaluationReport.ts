/** Backend terminal-approach evaluation report.
 *
 * MUST match `evaluation/metrics.py` → `REPORT_SCHEMA_VERSION`. The frontend cannot
 * import the modeling tree, so this is a mirror; change both together. When the
 * producer bumped v4 → v5 and this literal stayed, `isEvaluationReport` silently
 * rejected every report the pipeline wrote — the panel simply rendered nothing, and
 * the vitest fixtures (also pinned to v4) stayed green throughout. Test fixtures now
 * import this constant instead of repeating the string.
 */
export const EVALUATION_REPORT_SCHEMA_VERSION = "terminal-approach-evaluation-v6";

/**
 * Still-displayable older report versions. v5 predates the stall-anchored
 * crossing-speed gate: its verdicts compose lateral+vertical only and its rows
 * carry none of the speed fields. Published v5 artifacts remain on disk because
 * the record batches behind them were cleaned up, so they cannot be re-graded
 * until the optimizer batch is rerun — the report window labels them as
 * pre-speed-gate instead of refusing to open them (`isLegacyEvaluationReport`).
 * Versions before v5 changed shape, not just grading, and stay rejected.
 */
export const LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS = [
  "terminal-approach-evaluation-v5",
] as const;

export type EvaluationReportSchemaVersion =
  | typeof EVALUATION_REPORT_SCHEMA_VERSION
  | (typeof LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS)[number];

export interface MagnitudeSpread {
  mean: number;
  p95: number;
  max: number;
}

export interface SignedSpread {
  mean_signed: number;
  mean_abs: number;
  p95_abs: number;
  max_abs: number;
}

export type EvaluationSubject = "optimized" | "predicted" | "observed";
export type EvaluationVerdict = "pass" | "fail" | "indeterminate";
export type EvaluationComponentResult = EvaluationVerdict;

export interface EvaluationBounds {
  /** Always "runway_half_width_at_threshold" — the lateral bound is the runway, not
   *  the procedure's containment width. See evaluation/thresholds.py. */
  lateral_criterion: string;
  lateral_m: number;
  vertical_lower_m: number | null;
  vertical_upper_m: number | null;
  /** v6: stall-anchored crossing-speed window (evaluation/speed_gate.py). The three
   *  numbers are null when the record was not speed-gradable (observed subjects,
   *  unsolved records, or a record without source.landing_aero). Absent entirely in
   *  legacy v5 reports. */
  speed_criterion?: string;
  stall_speed_ms?: number | null;
  speed_lower_ms?: number | null;
  speed_upper_ms?: number | null;
}

export interface EvaluationRowReference {
  file: string;
  comparison_status: "compared" | "skipped";
  endpoint_tolerance_m: number;
  start_gap_m: number | null;
  end_gap_m: number | null;
  reference_flight_time_s?: number;
  flight_time_delta_s?: number;
  path_lateral_m?: MagnitudeSpread;
  path_vertical_m?: SignedSpread;
  note?: string;
}

export interface EvaluationRow {
  id: string;
  file: string | null;
  flight_key?: string | null;
  subject: EvaluationSubject;
  airport: string;
  runway: string;
  benchmark: "lpv" | "rnp_apch_lnav_vnav_baro";
  solved: boolean;
  success: boolean;
  verdict: EvaluationVerdict;
  event_status: string;
  lateral_result: EvaluationComponentResult;
  vertical_result: EvaluationComponentResult;
  /** v6: "indeterminate" for observed subjects by policy (no crossing airspeed is
   *  measured); composes into `verdict` for optimized/predicted only. Absent in
   *  legacy v5 reports, whose verdicts never graded speed. */
  speed_result?: EvaluationComponentResult;
  violations: string[];
  bounds: EvaluationBounds;
  lateral_m?: number | null;
  cross_track_m?: number | null;
  along_track_m?: number | null;
  vertical_m?: number | null;
  speed_ms?: number;
  /** v6: airspeed at the graded threshold crossing — the value the speed gate judges
   *  (`speed_ms` above is the record's final-state speed). Null when not speed-gradable;
   *  absent in legacy v5 reports. */
  crossing_speed_ms?: number | null;
  heading_rad?: number;
  final_time_s?: number;
  reason?: string;
  reference?: EvaluationRowReference;
}

export interface EvaluationObservedAggregate {
  denominator: "arrival_candidates_excluding_not_landing";
  event_denominator: number;
  event_estimated: number;
  event_unavailable: number;
  event_estimated_rate: number;
  excluded_not_landing: number;
}

export interface EvaluationReferenceAggregate {
  compared: number;
  flight_time_delta_s: { mean: number; min: number; max: number };
  path_lateral_m: { mean: number; max: number };
  path_vertical_m: { mean_abs: number; max_abs: number };
}

export interface EvaluationReport {
  schema_version: EvaluationReportSchemaVersion;
  methodology: Record<string, unknown>;
  assessment_contexts: Record<string, unknown>[];
  subject: EvaluationSubject | "mixed";
  observed?: EvaluationObservedAggregate;
  total: number;
  measured: number;
  solved: number;
  solve_rate: number;
  verdict_counts: Record<EvaluationVerdict, number>;
  successful: number;
  failed: number;
  indeterminate: number;
  success_rate: number;
  lateral_m: MagnitudeSpread | null;
  vertical_m: SignedSpread | null;
  /** v6: batch speed-gate tallies and crossing-speed spread (see
   *  EvaluationRow.speed_result). Absent in legacy v5 reports. */
  speed_result_counts?: Record<EvaluationVerdict, number>;
  crossing_speed_ms?: MagnitudeSpread | null;
  final_time_s: { mean: number; min: number; max: number } | null;
  reference: EvaluationReferenceAggregate | null;
  trajectories: EvaluationRow[];
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function hasCommonRnavVerticalMethodology(value: unknown): boolean {
  if (!value || typeof value !== "object") return false;
  const methodology = value as Record<string, unknown>;
  const terminal = methodology.terminal_vertical;
  if (!terminal || typeof terminal !== "object") return false;
  const terminalRecord = terminal as Record<string, unknown>;
  const acceptance = terminalRecord.common_rnav_terminal_acceptance;
  if (!acceptance || typeof acceptance !== "object") return false;
  const acceptanceRecord = acceptance as Record<string, unknown>;
  const source = acceptanceRecord.source;
  if (!source || typeof source !== "object") return false;
  const sourceRecord = source as Record<string, unknown>;
  return (
    terminalRecord.reference === "LTP elevation MSL + published FAS TCH" &&
    terminalRecord.trajectory_altitude_datum === "msl" &&
    isFiniteNumber(terminalRecord.target_context_tolerance_m) &&
    terminalRecord.target_context_tolerance_m >= 0 &&
    acceptanceRecord.standard_id === "icao_doc_9613_rnp_apch_fas_22m" &&
    acceptanceRecord.lower_m === -22 &&
    acceptanceRecord.upper_m === 22 &&
    acceptanceRecord.claim_boundary ===
      "terminal final-approach geometry; not touchdown or landing certification" &&
    typeof sourceRecord.document === "string" &&
    sourceRecord.location ===
      "Volume II, Part C, Chapter 5, Section A, §5.3.4.4.7" &&
    typeof sourceRecord.use === "string"
  );
}

function isObservedAggregate(value: unknown): value is EvaluationObservedAggregate {
  if (!value || typeof value !== "object") return false;
  const aggregate = value as Record<string, unknown>;
  const counts = [
    aggregate.event_denominator,
    aggregate.event_estimated,
    aggregate.event_unavailable,
    aggregate.excluded_not_landing,
  ];
  // Shape only. The arithmetic (estimated + unavailable === denominator, and rate ===
  // estimated / denominator) used to be re-derived here — but the producer computes
  // `unavailable` and `rate` FROM `denominator` and `estimated` in the same expression
  // (harvest/observed.py → source_event_availability), so those are identities, not
  // invariants that data could violate. Re-checking them bought nothing and cost a
  // silent whole-report rejection: this predicate gates rendering, so a mismatch would
  // blank the panel with no message rather than name a field. The Python side dropped
  // the same re-derivation; both now check only the denominator LABEL, which is the one
  // thing that says WHICH population the rate describes.
  return (
    aggregate.denominator === "arrival_candidates_excluding_not_landing" &&
    counts.every((count) => Number.isInteger(count) && Number(count) >= 0) &&
    typeof aggregate.event_estimated_rate === "number" &&
    Number.isFinite(aggregate.event_estimated_rate)
  );
}

export function isLegacyEvaluationReport(report: EvaluationReport): boolean {
  return report.schema_version !== EVALUATION_REPORT_SCHEMA_VERSION;
}

export function isEvaluationReport(value: unknown): value is EvaluationReport {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  const counts = candidate.verdict_counts as Record<string, unknown> | undefined;
  const versionAccepted =
    candidate.schema_version === EVALUATION_REPORT_SCHEMA_VERSION ||
    (LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS as readonly string[]).includes(
      String(candidate.schema_version),
    );
  return (
    versionAccepted &&
    hasCommonRnavVerticalMethodology(candidate.methodology) &&
    typeof candidate.total === "number" &&
    typeof candidate.solved === "number" &&
    !!counts &&
    typeof counts.pass === "number" &&
    typeof counts.fail === "number" &&
    typeof counts.indeterminate === "number" &&
    (candidate.observed === undefined || isObservedAggregate(candidate.observed)) &&
    Array.isArray(candidate.assessment_contexts) &&
    Array.isArray(candidate.trajectories) &&
    candidate.trajectories.every((row) => {
      if (!row || typeof row !== "object") return false;
      const record = row as Record<string, unknown>;
      return (
        typeof record.id === "string" &&
        typeof record.solved === "boolean" &&
        ["pass", "fail", "indeterminate"].includes(String(record.verdict)) &&
        !!record.bounds && typeof record.bounds === "object"
      );
    })
  );
}
