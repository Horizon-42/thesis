/**
 * profileTrajectorySources.ts
 * ---------------------------
 * Which trajectory datasources the runway profile may plot, so the profile
 * mirrors the ACTIVE TASK — it samples a source only when that source is the
 * current tab's content on the globe.
 *
 * The subtlety this exists to handle: the observed ADS-B tracks stay LOADED
 * behind an open profile in EVERY task (the profile chart samples the loaded
 * source), but they are Observe's content and are painted on the globe only
 * there (see planObservedTracks). So a profile opened in Fly / Optimize /
 * Compare must NOT plot the observed tracks — they are invisible on the globe
 * in those tabs, and plotting them would show trajectories the tab has hidden.
 * The optimized playback is the Optimize tab's content: it exists only while a
 * trajectory-play result is loaded (WorkbenchMode "optimize") and is that tab's
 * content whenever it does.
 *
 * The single source of truth is therefore "is this source the current tab's
 * globe content", reused by both the profile hook (what it samples) and the
 * panel (its CZML-linked indicator) — never re-derived per call site.
 *
 * KNOWN GAP: the Observe 3-colour comparison overlay is a separate datasource
 * not yet fed to the profile; in Observe-with-comparison the profile plots
 * neither source (the plain observed tracks are hidden, the comparison is not
 * wired in). Wiring the comparison overlay into the profile is a follow-up.
 */

import { planObservedTracks, type ObservedTrackInputs } from "./observedTracks";

export interface ProfileTrajectorySourceInputs extends ObservedTrackInputs {
  /** Whether an optimized-playback datasource is currently loaded. */
  hasOptimizedSource: boolean;
}

export interface ProfileTrajectorySourceSelection {
  /** Plot the observed ADS-B source — it is the current tab's globe content. */
  observed: boolean;
  /** Plot the optimized-playback source — it is the current tab's globe content. */
  optimized: boolean;
}

export function planProfileTrajectorySources(
  inputs: ProfileTrajectorySourceInputs,
): ProfileTrajectorySourceSelection {
  return {
    // Observed tracks are painted only in Observe — mirror that exact decision so
    // the profile never plots them in a tab that hides them on the globe.
    observed: planObservedTracks(inputs).visible,
    // The optimized playback belongs to the Optimize tab. It only ever exists there
    // (PilotPanel clears it on leaving), but gate on mode too so a lingering source
    // could never leak into another tab's profile.
    optimized: inputs.mode === "optimize" && inputs.hasOptimizedSource,
  };
}
