import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const { appState, canonicalEntities } = vi.hoisted(() => {
  const canonicalEntities = [
    { id: "flight-1", show: true },
    { id: "flight-2", show: true },
  ];
  return {
    canonicalEntities,
    appState: {
      viewer: {},
      layers: { trajectories: true },
      mode: "observe",
      trajectoryComparison: false,
      trajectoryComparisonCategory: null,
      trajectoryComparisonKinds: {
        reference: true,
        optimizer: false,
        simulator: true,
        predicted: true,
        lookback: true,
      },
      activeAirportCode: "KRDU",
      selectedRunway: null,
      trajectorySampleCount: 200,
      trajectoryDataSource: {
        show: true,
        entities: { values: canonicalEntities },
      },
    },
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => appState,
}));

vi.mock("cesium", () => ({}));

import { useComparisonTrajectoryLayer } from "../useComparisonTrajectoryLayer";

describe("useComparisonTrajectoryLayer", () => {
  it("does not hide canonical observed entities while comparison mode is inactive", () => {
    renderHook(() => useComparisonTrajectoryLayer());

    expect(appState.trajectoryDataSource.show).toBe(true);
    expect(canonicalEntities.map((entity) => entity.show)).toEqual([true, true]);
  });
});
