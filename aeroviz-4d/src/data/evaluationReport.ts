/**
 * evaluationReport.ts
 * -------------------
 * Types + validation for the backend evaluation report (`python -m evaluation`
 * → `evaluation_report.json`, published verbatim into each comparison category
 * dir by `build_scenario_comparison_czml.py --evaluation-report`).
 *
 * SINGLE SOURCE: every number here was computed by the backend evaluation
 * package — the frontend only formats and plots. Do not derive metrics from
 * these fields; if a new metric is needed, add it to `evaluation/metrics.py`.
 */

export interface EvaluationThresholds {
  lateral_max_m: number;
  vertical_below_max_m: number;
  vertical_above_max_m: number;
}

/** `stats.magnitude_spread` dict — mean / p95 / max of non-negative magnitudes. */
export interface MagnitudeSpread {
  mean: number;
  p95: number;
  max: number;
}

/** `stats.signed_spread` dict — signed mean + |value| spreads. */
export interface SignedSpread {
  mean_signed: number;
  mean_abs: number;
  p95_abs: number;
  max_abs: number;
}

/** A row's observed-track comparison block (fields per `metrics.evaluate_batch`). */
export interface EvaluationRowReference {
  file: string;
  flight_time_s: number;
  flight_time_delta_s?: number;
  path_lateral_m?: MagnitudeSpread;
  path_vertical_m?: SignedSpread;
  note?: string;
}

/**
 * What kind of trajectory a report describes (`arrival.Subject`). Absent means
 * `optimized` — the backend only stamps it when it is not.
 */
export type EvaluationSubject = "optimized" | "predicted" | "observed";

/** One per-trajectory verdict row (`metrics._row`, deviations only when measured). */
export interface EvaluationRow {
  id: string;
  file: string | null;
  solved: boolean;
  success: boolean;
  violations: string[];
  lateral_m?: number;
  vertical_m?: number;
  speed_ms?: number;
  heading_rad?: number;
  final_time_s?: number;
  reason?: string;
  reference?: EvaluationRowReference;
  /**
   * Canonical flight identity (`flight_scenarios.identity.flight_key`) and the ONLY
   * safe join key to a rendered track: `id` is the callsign and is not unique.
   */
  flight_key?: string;
  /** Observed rows only — see `EvaluationObservedAggregate`. */
  subject?: EvaluationSubject;
  established?: boolean;
  /** True when the arrival was the fitted threshold crossing, not the last state. */
  extrapolated?: boolean;
  /** True when the 95 % interval straddles a gate boundary: verdict undecidable. */
  marginal?: boolean;
  lateral_sigma_m?: number;
  vertical_sigma_m?: number;
  glidepath_deg?: number;
  extrapolation_m?: number;
}

/**
 * Observed-batch counts (`metrics.evaluate_batch` → `report.observed`).
 *
 * `solve_rate` is meaningless for observed data — an observed track trivially
 * "has states", so it is 1.0 by construction. `established_rate` is the honest
 * analogue and the UI shows it INSTEAD, never alongside.
 *
 * `marginal` counts flights whose gate verdict the data cannot decide: with
 * altitudes quantised to 25 ft (7.62 m) against a 9.15 m window, the majority of
 * real approaches land in this bucket, so a bare pass rate overstates itself.
 */
export interface EvaluationObservedAggregate {
  established: number;
  not_established: number;
  established_rate: number;
  marginal: number;
}

/** The batch observed-comparison aggregate (`metrics._reference_aggregate`). */
export interface EvaluationReferenceAggregate {
  compared: number;
  flight_time_delta_s: { mean: number; min: number; max: number };
  path_lateral_m: { mean: number; max: number };
  path_vertical_m: { mean_abs: number; max_abs: number };
}

/** The report dict (`metrics.evaluate_batch` — schema documented there). */
export interface EvaluationReport {
  thresholds: EvaluationThresholds;
  /** "mixed" when a batch holds more than one subject; absent on older reports. */
  subject?: EvaluationSubject | "mixed";
  /** Present only for observed batches. */
  observed?: EvaluationObservedAggregate;
  total: number;
  /** Records that yielded an arrival to grade — differs from `solved` only for observed. */
  measured?: number;
  solved: number;
  solve_rate: number;
  successful: number;
  success_rate: number;
  success_rate_among_solved: number | null;
  lateral_m: MagnitudeSpread | null;
  vertical_m: SignedSpread | null;
  final_time_s: { mean: number; min: number; max: number } | null;
  reference: EvaluationReferenceAggregate | null;
  trajectories: EvaluationRow[];
}

export function isEvaluationReport(value: unknown): value is EvaluationReport {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.total === "number" &&
    typeof candidate.solved === "number" &&
    typeof candidate.solve_rate === "number" &&
    typeof candidate.success_rate === "number" &&
    !!candidate.thresholds &&
    typeof candidate.thresholds === "object" &&
    Array.isArray(candidate.trajectories) &&
    candidate.trajectories.every(
      (row) =>
        !!row &&
        typeof row === "object" &&
        typeof (row as Record<string, unknown>).id === "string" &&
        typeof (row as Record<string, unknown>).solved === "boolean" &&
        typeof (row as Record<string, unknown>).success === "boolean",
    )
  );
}
