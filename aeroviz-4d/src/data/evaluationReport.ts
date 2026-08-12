/** Backend terminal-approach evaluation report (schema v2). */

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
export type EvaluationComponentResult = EvaluationVerdict | "not_applicable";

export interface EvaluationBounds {
  guidance_lateral_m: number | null;
  runway_lateral_m: number;
  effective_lateral_m: number | null;
  vertical_lower_m: number | null;
  vertical_upper_m: number | null;
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
  violations: string[];
  bounds: EvaluationBounds;
  lateral_m?: number | null;
  cross_track_m?: number | null;
  along_track_m?: number | null;
  vertical_m?: number | null;
  speed_ms?: number;
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
  schema_version: "terminal-approach-evaluation-v2";
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
  final_time_s: { mean: number; min: number; max: number } | null;
  reference: EvaluationReferenceAggregate | null;
  trajectories: EvaluationRow[];
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
  if (
    aggregate.denominator !== "arrival_candidates_excluding_not_landing" ||
    !counts.every((count) => Number.isInteger(count) && Number(count) >= 0) ||
    typeof aggregate.event_estimated_rate !== "number" ||
    !Number.isFinite(aggregate.event_estimated_rate)
  ) {
    return false;
  }
  const denominator = Number(aggregate.event_denominator);
  const estimated = Number(aggregate.event_estimated);
  const unavailable = Number(aggregate.event_unavailable);
  const expectedRate = denominator === 0 ? 0 : estimated / denominator;
  return (
    estimated + unavailable === denominator &&
    Math.abs(aggregate.event_estimated_rate - expectedRate) <= 1e-12
  );
}

export function isEvaluationReport(value: unknown): value is EvaluationReport {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  const counts = candidate.verdict_counts as Record<string, unknown> | undefined;
  return (
    candidate.schema_version === "terminal-approach-evaluation-v2" &&
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
