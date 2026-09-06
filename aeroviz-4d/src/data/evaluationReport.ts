/** Backend terminal-approach evaluation report.
 *
 * MUST match `evaluation/metrics.py` → `REPORT_SCHEMA_VERSION`. The frontend cannot
 * import the modeling tree, so this is a mirror; change both together. When the
 * producer bumped v4 → v5 and this literal stayed, `isEvaluationReport` silently
 * rejected every report the pipeline wrote — the panel simply rendered nothing, and
 * the vitest fixtures (also pinned to v4) stayed green throughout. Test fixtures now
 * import this constant instead of repeating the string.
 */
export const EVALUATION_REPORT_SCHEMA_VERSION = "terminal-approach-evaluation-v9";

/**
 * Still-displayable older report versions, in two classes.
 *
 * PRIOR speed-gate versions carry the speed fields this file reads; they differ from
 * the current schema in what anchored the window. v6–v8 anchored on the project's
 * own stall model (`stall_speed_ms`, from a modelled Cl_max): v6 at 1 g, v7 at the
 * crossing load factor, v8 measuring the observed load factor from ADS-B kinematics
 * and correcting the observed ground speed by the METAR headwind
 * (`crossing_airspeed_estimate_ms`, `wind`). v9 anchors on the type's PUBLISHED
 * approach speed (`vref_low_ms`/`vref_high_ms`, evaluation/docs/THRESHOLD_SPEED_GATE.md
 * §3.1). Batches on disk stay readable and the window notes the difference
 * (`isPriorSpeedGateReport`).
 */
export const PRIOR_SPEED_GATE_REPORT_SCHEMA_VERSIONS = [
  "terminal-approach-evaluation-v6",
  "terminal-approach-evaluation-v7",
  "terminal-approach-evaluation-v8",
] as const;

/**
 * LEGACY versions predate the stall-anchored crossing-speed gate: v5 verdicts compose
 * lateral+vertical only and its rows carry none of the speed fields. Published v5
 * artifacts remain on disk because the record batches behind them were cleaned up,
 * so they cannot be re-graded until the optimizer batch is rerun — the report window
 * labels them as pre-speed-gate instead of refusing to open them
 * (`isLegacyEvaluationReport`). Versions before v5 changed shape, not just grading,
 * and stay rejected.
 */
export const LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS = [
  "terminal-approach-evaluation-v5",
] as const;

export type EvaluationReportSchemaVersion =
  | typeof EVALUATION_REPORT_SCHEMA_VERSION
  | (typeof PRIOR_SPEED_GATE_REPORT_SCHEMA_VERSIONS)[number]
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
  /** v6+: the crossing-speed window (evaluation/speed_gate.py). The numbers are null
   *  when no window could be resolved (unsolved records, no crossing, no aircraft
   *  type or no published entry for it, or an observed event without a fitted ground
   *  speed). Absent entirely in legacy v5 reports.
   *  v9: the window is anchored on the type's PUBLISHED approach speed —
   *  `reference_typecode` names the type, `mass_basis` says whether the window was
   *  framed at the record's crossing mass ("crossing_mass", computed subjects) or
   *  over the type's published mass range ("type_mass_range", observed subjects),
   *  `vref_low_ms`/`vref_high_ms` are the published speed at the framing masses
   *  (`speed_lower_ms` is the low one lifted by the crossing load factor) and
   *  `reference_sources` the table's source ids for the speed, the MALW and the
   *  minimum mass. v6–v8 (prior) carried `stall_speed_ms`/`stall_speed_at_n_ms`. */
  speed_criterion?: string;
  reference_typecode?: string | null;
  reference_sources?: { approach_speed: string; malw: string; min_mass: string | null } | null;
  mass_basis?: "crossing_mass" | "type_mass_range" | null;
  vref_low_ms?: number | null;
  vref_high_ms?: number | null;
  stall_speed_ms?: number | null;
  stall_speed_at_n_ms?: number | null;
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
  /** v6: composes into `verdict` for every subject. Computed subjects are judged on
   *  the crossing model airspeed; observed baselines on the fitted crossing GROUND
   *  speed as a stated proxy (bounds carry the `_ground_speed_proxy` criterion id).
   *  Indeterminate when the airframe is unresolvable or no crossing speed exists.
   *  Absent in legacy v5 reports, whose verdicts never graded speed. */
  speed_result?: EvaluationComponentResult;
  /** v9: why the speed component is indeterminate (null when it was graded) — on the
   *  row itself, readable even when the composite is a lateral/vertical fail. */
  speed_reason?: string | null;
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
  /** v6-additive (2026-08-24): the event's ADS-B GROUND speed at the crossing
   *  (observed subjects) — the value the speed gate judges for baselines, as a
   *  stated proxy for airspeed (wind unmodelled). Null on rows whose event predates
   *  the field or could not fit a speed. */
  crossing_ground_speed_ms?: number | null;
  /** v7: the load factor the crossing was flown at and where it came from —
   *  "controls_last_step" (the control active over the final rollout step),
   *  "adsb_kinematics" (v8, observed rows: inverted from the flight's own kinematics)
   *  or "assumed_1g" (state-output predictions; observed tracks too short to fit).
   *  Flat on the row like the two crossing speeds. */
  crossing_load_factor?: number;
  crossing_load_factor_source?: string;
  /** v8 (observed rows): ground speed + METAR headwind component — the value the speed
   *  gate judges when a usable report exists (`bounds.speed_criterion` ends in
   *  `_metar_airspeed_estimate`); null when the row fell back to the ground-speed
   *  proxy, and `wind.status`/`wind.reason` say why. */
  crossing_airspeed_estimate_ms?: number | null;
  /** v8 (observed rows): the wind report used, or why none was. */
  wind?: Record<string, number | string | null>;
  /** v8+: the judged value's signed distance to the nearest bound (positive inside
   *  the window, negative outside). v8 also carried `speed_uncertainty_ms`; v9 dropped
   *  it — the verdict is pass/fail against a published window. */
  speed_margin_ms?: number | null;
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
  /** One subject names itself; several are "mixed"; a batch with no records is "empty". */
  subject: EvaluationSubject | "mixed" | "empty";
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
  /** v6-additive: spread of the observed rows' crossing GROUND speeds — the proxy
   *  their speed gate judges when no wind report is usable
   *  (`methodology.observed_crossing_ground_speed`), kept apart from
   *  `crossing_speed_ms`. Null when no row carries one. */
  crossing_ground_speed_ms?: MagnitudeSpread | null;
  /** v8: spread of the observed rows' METAR-corrected airspeed estimates and how many
   *  rows were judged on the estimate vs the proxy. */
  crossing_airspeed_estimate_ms?: MagnitudeSpread | null;
  wind_counts?: { estimated: number; unavailable: number };
  /** v9: why speed could not be judged, per cause (no type on the record, no
   *  published entry for the type, no published minimum mass, no crossing speed). */
  speed_indeterminate_reasons?: Record<string, number>;
  /** v8: `speed_result_counts` split by the criterion each row was judged under, so
   *  an observed batch's estimate-judged and proxy-judged rows are never pooled. */
  speed_result_counts_by_criterion?: Record<string, Record<EvaluationVerdict, number>>;
  /** v7: the load factor the crossings were flown at — spread, how many sat below 1 g
   *  (lower bound clamped to the 1-g floor) and how many were a declared 1 g. */
  crossing_load_factor?: {
    mean: number;
    min: number;
    p95: number;
    max: number;
    below_1g: number;
    adsb_kinematics: number;
    assumed_1g: number;
  } | null;
  final_time_s: { mean: number; min: number; max: number } | null;
  reference: EvaluationReferenceAggregate | null;
  trajectories: EvaluationRow[];
}

/** Compose component verdicts into an overall verdict.
 *
 * MUST match `evaluation.metrics._composite` (a declared mirror — the frontend
 * cannot import the modeling tree): any `fail` dominates, else any
 * `indeterminate`, else `pass`. Used by the Details window's speed-gate toggle
 * to re-derive two-gate verdicts client-side from the per-row component
 * results the report deliberately serializes. */
export function composeVerdict(
  components: readonly EvaluationComponentResult[],
): EvaluationVerdict {
  if (components.includes("fail")) return "fail";
  if (components.includes("indeterminate")) return "indeterminate";
  return "pass";
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

/** Pre-speed-gate (v5): no speed fields, two-gate verdicts. */
export function isLegacyEvaluationReport(report: EvaluationReport): boolean {
  return (LEGACY_EVALUATION_REPORT_SCHEMA_VERSIONS as readonly string[]).includes(
    String(report.schema_version),
  );
}

/** Speed-gated at 1 g (v6): every speed field present, the lower bound not yet
 *  anchored on the crossing load factor. */
export function isPriorSpeedGateReport(report: EvaluationReport): boolean {
  return (PRIOR_SPEED_GATE_REPORT_SCHEMA_VERSIONS as readonly string[]).includes(
    String(report.schema_version),
  );
}

export function isEvaluationReport(value: unknown): value is EvaluationReport {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  const counts = candidate.verdict_counts as Record<string, unknown> | undefined;
  const versionAccepted =
    candidate.schema_version === EVALUATION_REPORT_SCHEMA_VERSION ||
    (PRIOR_SPEED_GATE_REPORT_SCHEMA_VERSIONS as readonly string[]).includes(
      String(candidate.schema_version),
    ) ||
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
