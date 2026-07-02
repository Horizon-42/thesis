/**
 * observedTracks.ts
 * -----------------
 * Pure decision for how the observed ADS-B track CZML is sourced and shown.
 *
 * The observed-track file is large (100+ MB per airport), so LOADING and
 * VISIBILITY are kept independent — you can hold a source in memory without
 * painting it, and re-show it instantly without re-parsing:
 *
 *   • LOADED whenever the tracks are relevant — in Observe, or whenever a runway
 *     profile is open (the profile chart samples the loaded data source in ANY
 *     task). When neither holds, `fileUrl` is "" so the caller releases the source.
 *   • painted on the globe ONLY in Observe — the observed tracks are Observe's
 *     content, so switching to Fly / Optimize / Compare stays visually exclusive
 *     even while a profile keeps them loaded. The 3-colour optimizer comparison
 *     hides them too.
 */

import { airportDataUrl, airportLandingsRunwayUrl } from "./airportData";
import type { WorkbenchMode } from "../context/AppContext";

export interface ObservedTrackInputs {
  mode: WorkbenchMode;
  activeAirportCode: string | null;
  selectedRunway: string | null;
  isRunwayProfileOpen: boolean;
  trajectoryComparison: boolean;
}

export interface ObservedTrackPlan {
  /** CZML url to load; "" means "release / load nothing". */
  fileUrl: string;
  /** Whether the loaded tracks are painted on the globe. */
  visible: boolean;
}

export function planObservedTracks({
  mode,
  activeAirportCode,
  selectedRunway,
  isRunwayProfileOpen,
  trajectoryComparison,
}: ObservedTrackInputs): ObservedTrackPlan {
  const relevant = !!activeAirportCode && (mode === "observe" || isRunwayProfileOpen);
  let fileUrl = "";
  if (relevant && activeAirportCode) {
    fileUrl = selectedRunway
      ? airportLandingsRunwayUrl(activeAirportCode, selectedRunway)
      : airportDataUrl(activeAirportCode, "trajectories.czml");
  }
  const visible = mode === "observe" && !trajectoryComparison;
  return { fileUrl, visible };
}
