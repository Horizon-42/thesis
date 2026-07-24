/**
 * observedTracks.ts
 * -----------------
 * Pure decision for how the observed ADS-B track CZML is sourced and shown.
 *
 * The observed-track file is large (100+ MB per airport), so LOADING and
 * VISIBILITY are kept independent — you can hold a source in memory without
 * painting it, and re-show it instantly without re-parsing:
 *
 *   • LOADED only in Observe — the observed tracks are Observe's content, and the
 *     runway profile samples them only in Observe (planApproachViewSources),
 *     so no other task needs them in memory. Outside Observe `fileUrl` is "" so the
 *     caller releases the source. (This also matters because useCzmlLoader drives
 *     the shared viewer clock from the observed CZML's span; loading it behind a
 *     profile in Optimize/Fly would hijack the optimized playback's clock and make
 *     that track vanish — so we simply don't load it there.)
 *   • painted on the globe ONLY in Observe with no comparison — the 3-colour
 *     prediction comparison hides the plain tracks in favour of its own source.
 */

import type { LandingsManifest } from "./airportData";
import type { WorkbenchMode } from "../context/AppContext";

export interface ObservedTrackInputs {
  mode: WorkbenchMode;
  activeAirportCode: string | null;
  selectedRunway: string | null;
  trajectoryComparison: boolean;
  trajectorySampleCount: number;
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
  /** CZML url to load; "" means "release / load nothing". */
  fileUrl: string;
  /** Whether the loaded tracks are painted on the globe. */
  visible: boolean;
  /** Runway entity filter; null shows the canonical file's complete roster. */
  runwayFilter: string | null;
}

export function planObservedTracks({
  mode,
  activeAirportCode,
  selectedRunway,
  trajectoryComparison,
  trajectorySampleCount,
  backendUrl,
  landingsManifest,
  landingsStatus,
}: ObservedTrackInputs): ObservedTrackPlan {
  const relevant = !!activeAirportCode && mode === "observe";
  let fileUrl = "";
  let runwayFilter: string | null = selectedRunway;
  if (relevant && activeAirportCode) {
    if (
      landingsStatus !== undefined &&
      (landingsStatus !== "ready" || !landingsManifest)
    ) {
      // The manifest is the publication commit point. Do not load uncommitted, missing,
      // invalid, or obsolete observed data by guessing a filename.
      runwayFilter = null;
    } else {
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
      fileUrl = `${backendUrl.replace(/\/+$/, "")}/trajectories?${params.toString()}`;
    }
  }
  const visible = observedTracksVisible({ mode, trajectoryComparison });
  return { fileUrl, visible, runwayFilter };
}

export function observedTracksVisible({
  mode,
  trajectoryComparison,
}: ObservedTrackVisibilityInputs): boolean {
  return mode === "observe" && !trajectoryComparison;
}
