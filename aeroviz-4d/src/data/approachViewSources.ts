/**
 * approachViewSources.ts
 * ---------------------------
 * Which trajectory datasources the approach view may plot, so the approach view
 * mirrors the ACTIVE TASK — it samples a source only when that source is the
 * current tab's content on the globe.
 *
 * Observed ADS-B tracks exist only in Observe and are visible only for Baseline
 * (see planObservedTracks). A profile opened in Fly / Optimize / Compare must
 * therefore never plot that source; it would disagree with the globe and could
 * also let the observed clock interfere with optimized playback.
 * The optimized playback is the Optimize tab's content: it exists only while a
 * trajectory-play result is loaded (WorkbenchMode "optimize") and is that tab's
 * content whenever it does.
 *
 * The single source of truth is therefore "is this source the current tab's
 * globe content", reused by both the approach view hook (what it samples) and the
 * panel (its CZML-linked indicator) — never re-derived per call site.
 *
 * KNOWN GAP: the Observe 3-colour comparison overlay is a separate datasource
 * not yet fed to the approach view; in Observe-with-comparison the approach view plots
 * neither source (the plain observed tracks are hidden, the comparison is not
 * wired in). Wiring the comparison overlay into the approach view is a follow-up.
 */

import {
  observedTracksVisible,
  type ObservedTrackVisibilityInputs,
} from "./observedTracks";

export interface ApproachViewSourceInputs extends ObservedTrackVisibilityInputs {
  /** Whether an optimized-playback datasource is currently loaded. */
  hasOptimizedSource: boolean;
}

export interface ApproachViewSourceSelection {
  /** Plot the observed ADS-B source — it is the current tab's globe content. */
  observed: boolean;
  /** Plot the optimized-playback source — it is the current tab's globe content. */
  optimized: boolean;
}

export function planApproachViewSources(
  inputs: ApproachViewSourceInputs,
): ApproachViewSourceSelection {
  return {
    // Observed tracks are painted only in Observe — mirror that exact decision so
    // the approach view never plots them in a tab that hides them on the globe.
    observed: observedTracksVisible(inputs),
    // The optimized playback belongs to the Optimize tab. It only ever exists there
    // (PilotPanel clears it on leaving), but gate on mode too so a lingering source
    // could never leak into another tab's profile.
    optimized: inputs.mode === "optimize" && inputs.hasOptimizedSource,
  };
}
