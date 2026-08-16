/**
 * observedTracks.ts
 * -----------------
 * Request/response contract for the observed ADS-B trajectory layer.
 *
 * The observed-track file is large (100+ MB per airport), so LOADING and
 * VISIBILITY are kept independent — you can hold a source in memory without
 * painting it, and re-show it instantly without re-parsing:
 *
 *   • LOADED only in Observe — the observed tracks are Observe's content, and the
 *     runway profile samples them only in Observe (planApproachViewSources),
 *     so no other task needs them in memory. Outside Observe `fileUrl` is "" so the
 *     caller releases the source. (This also matters because useObservedTrajectoryLayer drives
 *     the shared viewer clock from the observed CZML's span; loading it behind a
 *     profile in Optimize/Fly would hijack the optimized playback's clock and make
 *     that track vanish — so we simply don't load it there.)
 *   • baseline styling is painted only in Observe with no comparison. Comparison owns
 *     a separate exact reference roster selected by its committed index.
 *   • sampled ONCE by the backend, after runway + terminal-verdict filtering. The
 *     browser therefore receives no discarded entities and never repeats selection.
 */

import type { LandingsManifest } from "./airportData";
import type { WorkbenchMode } from "../context/AppContext";

export type ObservedVerdict = "pass" | "fail" | "undecided";
export type ObservedVerdictFilter = "all" | ObservedVerdict;

export interface ObservedVerdicts {
  /** Counts over the complete runway-eligible roster, before sampling. */
  counts: Record<ObservedVerdict, number> | null;
  /** Published joins / complete runway-eligible roster. */
  matched: number;
  total: number;
}

export const NO_OBSERVED_VERDICTS: ObservedVerdicts = {
  counts: null,
  matched: 0,
  total: 0,
};

export interface ObservedEvaluationSummary {
  total: number;
  verdict_counts: { pass: number; fail: number; indeterminate: number };
  observed: { event_estimated_rate: number } | null;
  lateral_m: { mean: number } | null;
  vertical_m: { mean_abs: number } | null;
}

interface ObservedVerdictPayload {
  counts: Record<ObservedVerdict, number>;
  byFlightId: Record<string, ObservedVerdict>;
  matched: number;
  total: number;
}

export interface ObservedTrajectoryResponse {
  schemaVersion: "observed-trajectories-v1";
  czml: unknown[];
  verdicts: ObservedVerdictPayload | null;
  evaluation: ObservedEvaluationSummary | null;
}

export interface ObservedTrackInputs {
  mode: WorkbenchMode;
  activeAirportCode: string | null;
  selectedRunway: string | null;
  trajectoryComparison: boolean;
  trajectorySampleCount: number;
  observedVerdictFilter: ObservedVerdictFilter;
  backendUrl: string;
  /** Canonical publication contract; legacy observed manifests are rejected upstream. */
  landingsManifest?: LandingsManifest | null;
  landingsStatus?: "idle" | "loading" | "ready" | "empty" | "error";
}

export interface ObservedTrackVisibilityInputs {
  mode: WorkbenchMode;
  trajectoryComparison: boolean;
}

export interface ObservedTrackPlan {
  /** Observed response URL to load; "" means "release / load nothing". */
  fileUrl: string;
  /** Whether the loaded tracks are painted on the globe. */
  visible: boolean;
}

export function planObservedTracks({
  mode,
  activeAirportCode,
  selectedRunway,
  trajectoryComparison,
  trajectorySampleCount,
  observedVerdictFilter,
  backendUrl,
  landingsManifest,
  landingsStatus,
}: ObservedTrackInputs): ObservedTrackPlan {
  const relevant = !!activeAirportCode && mode === "observe" && !trajectoryComparison;
  let fileUrl = "";
  const publicationReady =
    landingsStatus === undefined || (landingsStatus === "ready" && !!landingsManifest);
  if (relevant && activeAirportCode && publicationReady) {
    const params = new URLSearchParams({
      airport: activeAirportCode,
      limit: String(
        Number.isFinite(trajectorySampleCount)
          ? Math.max(0, Math.floor(trajectorySampleCount))
          : 200,
      ),
      seed: "0",
    });
    if (selectedRunway) params.set("runway", selectedRunway);
    if (observedVerdictFilter !== "all") {
      params.set("verdict", observedVerdictFilter);
    }
    fileUrl = `${backendUrl.replace(/\/+$/, "")}/trajectories?${params.toString()}`;
  }
  const visible = observedTracksVisible({ mode, trajectoryComparison });
  return { fileUrl, visible };
}

export interface ObservedReferenceTrackRequest {
  backendUrl: string;
  airport: string;
  flightKeys: string[];
}

/** Build one backend request for the exact groups sampled from a comparison index. */
export function observedReferenceTracksUrl({
  backendUrl,
  airport,
  flightKeys,
}: ObservedReferenceTrackRequest): string {
  if (!airport || flightKeys.length === 0) return "";
  const params = new URLSearchParams({ airport });
  for (const flightKey of flightKeys) params.append("flight_key", flightKey);
  return `${backendUrl.replace(/\/+$/, "")}/trajectories?${params.toString()}`;
}

export function observedTracksVisible({
  mode,
  trajectoryComparison,
}: ObservedTrackVisibilityInputs): boolean {
  return mode === "observe" && !trajectoryComparison;
}

export function isObservedTrajectoryResponse(value: unknown): value is ObservedTrajectoryResponse {
  if (!isRecord(value) || value.schemaVersion !== "observed-trajectories-v1") return false;
  if (!Array.isArray(value.czml)) return false;
  if (value.verdicts !== null && !isObservedVerdictPayload(value.verdicts)) return false;
  return value.evaluation === null || isObservedEvaluationSummary(value.evaluation);
}

export function decodeObservedVerdicts(
  payload: ObservedVerdictPayload | null,
): {
  summary: ObservedVerdicts;
  byFlightId: ReadonlyMap<string, ObservedVerdict> | null;
} {
  if (!payload) return { summary: NO_OBSERVED_VERDICTS, byFlightId: null };
  return {
    summary: {
      counts: payload.counts,
      matched: payload.matched,
      total: payload.total,
    },
    byFlightId: new Map(Object.entries(payload.byFlightId)),
  };
}

function isObservedVerdictPayload(value: unknown): value is ObservedVerdictPayload {
  if (!isRecord(value) || !isVerdictCounts(value.counts) || !isRecord(value.byFlightId)) {
    return false;
  }
  return (
    nonNegativeInteger(value.matched) &&
    nonNegativeInteger(value.total) &&
    value.matched <= value.total &&
    Object.values(value.byFlightId).every(isObservedVerdict)
  );
}

function isObservedEvaluationSummary(value: unknown): value is ObservedEvaluationSummary {
  if (!isRecord(value) || !isRecord(value.verdict_counts)) return false;
  const observed = value.observed;
  const lateral = value.lateral_m;
  const vertical = value.vertical_m;
  return (
    nonNegativeInteger(value.total) &&
    nonNegativeInteger(value.verdict_counts.pass) &&
    nonNegativeInteger(value.verdict_counts.fail) &&
    nonNegativeInteger(value.verdict_counts.indeterminate) &&
    (observed === null || (isRecord(observed) && finiteNumber(observed.event_estimated_rate))) &&
    (lateral === null || (isRecord(lateral) && finiteNumber(lateral.mean))) &&
    (vertical === null || (isRecord(vertical) && finiteNumber(vertical.mean_abs)))
  );
}

function isVerdictCounts(value: unknown): value is Record<ObservedVerdict, number> {
  return (
    isRecord(value) &&
    nonNegativeInteger(value.pass) &&
    nonNegativeInteger(value.fail) &&
    nonNegativeInteger(value.undecided)
  );
}

function isObservedVerdict(value: unknown): value is ObservedVerdict {
  return value === "pass" || value === "fail" || value === "undecided";
}

function nonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 0;
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
