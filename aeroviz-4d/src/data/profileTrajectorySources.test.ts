import { describe, expect, it } from "vitest";
import {
  planProfileTrajectorySources,
  type ProfileTrajectorySourceInputs,
} from "./profileTrajectorySources";

const base: ProfileTrajectorySourceInputs = {
  mode: "observe",
  activeAirportCode: "KRDU",
  selectedRunway: "05L",
  trajectoryComparison: false,
  hasOptimizedSource: false,
};

describe("planProfileTrajectorySources", () => {
  it("plots the observed tracks in Observe (their globe content)", () => {
    expect(planProfileTrajectorySources(base)).toEqual({ observed: true, optimized: false });
  });

  it("does NOT plot the observed tracks in Optimize even though they stay loaded", () => {
    // The reported bug: a profile opened in Optimize was drawing the observed tracks,
    // which are hidden on the globe there. Optimize's content is the optimized playback.
    expect(
      planProfileTrajectorySources({ ...base, mode: "optimize", hasOptimizedSource: true }),
    ).toEqual({ observed: false, optimized: true });
  });

  it.each(["fly", "optimize", "compare"] as const)(
    "never plots the observed tracks in %s",
    (mode) => {
      expect(planProfileTrajectorySources({ ...base, mode }).observed).toBe(false);
    },
  );

  it("hides the observed tracks in Observe while the 3-colour comparison is on", () => {
    // Matches planObservedTracks: the plain observed source is hidden on the globe when
    // the comparison overlay is shown, so the profile must not plot it either.
    expect(
      planProfileTrajectorySources({ ...base, trajectoryComparison: true }).observed,
    ).toBe(false);
  });

  it("plots the optimized playback only in Optimize, and only when it exists", () => {
    expect(
      planProfileTrajectorySources({ ...base, mode: "optimize", hasOptimizedSource: false })
        .optimized,
    ).toBe(false);
    expect(
      planProfileTrajectorySources({ ...base, mode: "optimize", hasOptimizedSource: true })
        .optimized,
    ).toBe(true);
    // A source lingering while the tab is not Optimize is never plotted.
    expect(
      planProfileTrajectorySources({ ...base, mode: "observe", hasOptimizedSource: true })
        .optimized,
    ).toBe(false);
  });
});
