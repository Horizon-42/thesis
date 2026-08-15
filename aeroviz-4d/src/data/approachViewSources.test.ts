import { describe, expect, it } from "vitest";
import {
  planApproachViewSources,
  type ApproachViewSourceInputs,
} from "./approachViewSources";

const base: ApproachViewSourceInputs = {
  mode: "observe",
  trajectoryComparison: false,
  hasOptimizedSource: false,
};

describe("planApproachViewSources", () => {
  it("plots the observed tracks in Observe (their globe content)", () => {
    expect(planApproachViewSources(base)).toEqual({ observed: true, optimized: false });
  });

  it("does NOT plot the observed tracks in Optimize even though they stay loaded", () => {
    // The reported bug: a profile opened in Optimize was drawing the observed tracks,
    // which are hidden on the globe there. Optimize's content is the optimized playback.
    expect(
      planApproachViewSources({ ...base, mode: "optimize", hasOptimizedSource: true }),
    ).toEqual({ observed: false, optimized: true });
  });

  it.each(["fly", "optimize", "compare"] as const)(
    "never plots the observed tracks in %s",
    (mode) => {
      expect(planApproachViewSources({ ...base, mode }).observed).toBe(false);
    },
  );

  it("hides the observed tracks in Observe while the 3-colour comparison is on", () => {
    // Matches planObservedTracks: the plain observed source is hidden on the globe when
    // the comparison overlay is shown, so the approach view must not plot it either.
    expect(
      planApproachViewSources({ ...base, trajectoryComparison: true }).observed,
    ).toBe(false);
  });

  it("plots the optimized playback only in Optimize, and only when it exists", () => {
    expect(
      planApproachViewSources({ ...base, mode: "optimize", hasOptimizedSource: false })
        .optimized,
    ).toBe(false);
    expect(
      planApproachViewSources({ ...base, mode: "optimize", hasOptimizedSource: true })
        .optimized,
    ).toBe(true);
    // A source lingering while the tab is not Optimize is never plotted.
    expect(
      planApproachViewSources({ ...base, mode: "observe", hasOptimizedSource: true })
        .optimized,
    ).toBe(false);
  });
});
