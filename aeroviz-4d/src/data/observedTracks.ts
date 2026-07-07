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
 *     runway profile samples them only in Observe (planProfileTrajectorySources),
 *     so no other task needs them in memory. Outside Observe `fileUrl` is "" so the
 *     caller releases the source. (This also matters because useCzmlLoader drives
 *     the shared viewer clock from the observed CZML's span; loading it behind a
 *     profile in Optimize/Fly would hijack the optimized playback's clock and make
 *     that track vanish — so we simply don't load it there.)
 *   • painted on the globe ONLY in Observe with no comparison — the 3-colour
 *     optimizer comparison hides the plain tracks in favour of its own source.
 */

import { airportDataUrl, airportLandingsRunwayUrl } from "./airportData";
import type { WorkbenchMode } from "../context/AppContext";

export interface ObservedTrackInputs {
  mode: WorkbenchMode;
  activeAirportCode: string | null;
  selectedRunway: string | null;
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
  trajectoryComparison,
}: ObservedTrackInputs): ObservedTrackPlan {
  const relevant = !!activeAirportCode && mode === "observe";
  let fileUrl = "";
  if (relevant && activeAirportCode) {
    fileUrl = selectedRunway
      ? airportLandingsRunwayUrl(activeAirportCode, selectedRunway)
      : airportDataUrl(activeAirportCode, "trajectories.czml");
  }
  const visible = mode === "observe" && !trajectoryComparison;
  return { fileUrl, visible };
}
