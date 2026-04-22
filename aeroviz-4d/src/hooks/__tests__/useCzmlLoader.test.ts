import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

const CZML_URL = "/data/airports/KRDU/trajectories.czml";

const {
  loadCzml,
  mockViewer,
  setTrajectoriesVisible,
  getTrajectoriesVisible,
  setSelectedFlightId,
  makeTime,
} = vi.hoisted(() => {
  const makeTime = (seconds: number): any => ({
    seconds,
    clone: () => makeTime(seconds),
  });

  const loadCzml = vi.fn();
  const setSelectedFlightId = vi.fn();
  let trajectoriesVisible = true;

  const mockViewer = {
    clock: {
      startTime: makeTime(0),
      stopTime: makeTime(10),
      currentTime: makeTime(0),
      clockRange: undefined as unknown,
      multiplier: 1,
      shouldAnimate: false,
    },
    dataSources: {
      add: vi.fn(),
      remove: vi.fn(),
    },
    timeline: {
      zoomTo: vi.fn(),
    },
    trackedEntity: undefined as unknown,
  };

  return {
    loadCzml,
    mockViewer,
    setTrajectoriesVisible: (value: boolean) => {
      trajectoriesVisible = value;
    },
    getTrajectoriesVisible: () => trajectoriesVisible,
    setSelectedFlightId,
    makeTime,
  };
});

vi.mock("cesium", () => ({
  ClockRange: {
    LOOP_STOP: "LOOP_STOP",
  },
  JulianDate: {
    lessThan: (left: { seconds: number }, right: { seconds: number }) =>
      left.seconds < right.seconds,
  },
  CzmlDataSource: class CzmlDataSource {
    entities = { values: [] };
    clock: unknown = null;
    show = true;

    constructor(public name: string) {}

    async load(url: string) {
      const loaded = await loadCzml(url);
      Object.assign(this, loaded);
      return this;
    }
  },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
    layers: { trajectories: getTrajectoriesVisible() },
    setSelectedFlightId,
  }),
}));

import { useCzmlLoader } from "../useCzmlLoader";

function resetViewer() {
  mockViewer.clock.startTime = makeTime(0);
  mockViewer.clock.stopTime = makeTime(10);
  mockViewer.clock.currentTime = makeTime(0);
  mockViewer.clock.clockRange = undefined;
  mockViewer.clock.multiplier = 1;
  mockViewer.clock.shouldAnimate = false;
  mockViewer.trackedEntity = undefined;
  mockViewer.dataSources.add.mockClear();
  mockViewer.dataSources.remove.mockClear();
  mockViewer.timeline.zoomTo.mockClear();
  setSelectedFlightId.mockClear();
  setTrajectoriesVisible(true);
}

describe("useCzmlLoader", () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    resetViewer();
    loadCzml.mockReset();
    warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it("warns and skips Cesium clock work when the CZML has no trajectory entities", async () => {
    loadCzml.mockResolvedValue({
      entities: { values: [{ id: "document" }] },
      clock: { startTime: makeTime(5), stopTime: makeTime(5) },
    });

    const { result } = renderHook(() => useCzmlLoader(CZML_URL));

    await waitFor(() => expect(result.current.isLoaded).toBe(true));

    expect(loadCzml).toHaveBeenCalledWith(CZML_URL);
    expect(result.current.flightIds).toEqual([]);
    expect(result.current.warning).toContain("No trajectory entities");
    expect(mockViewer.dataSources.add).not.toHaveBeenCalled();
    expect(mockViewer.timeline.zoomTo).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalled();
  });

  it("loads entities but warns instead of zooming the timeline for a zero-duration clock", async () => {
    loadCzml.mockResolvedValue({
      entities: { values: [{ id: "flight-1" }] },
      clock: { startTime: makeTime(8), stopTime: makeTime(8) },
    });

    const { result } = renderHook(() => useCzmlLoader(CZML_URL));

    await waitFor(() => expect(result.current.isLoaded).toBe(true));

    expect(loadCzml).toHaveBeenCalledWith(CZML_URL);
    expect(result.current.flightIds).toEqual(["flight-1"]);
    expect(result.current.warning).toContain("has no duration");
    expect(mockViewer.dataSources.add).toHaveBeenCalledTimes(1);
    expect(mockViewer.timeline.zoomTo).not.toHaveBeenCalled();
  });

  it("syncs the Cesium clock and timeline for a valid CZML interval", async () => {
    loadCzml.mockResolvedValue({
      entities: { values: [{ id: "flight-1" }, { id: "flight-2" }] },
      clock: { startTime: makeTime(10), stopTime: makeTime(70) },
    });

    const { result } = renderHook(() => useCzmlLoader(CZML_URL));

    await waitFor(() => expect(result.current.isLoaded).toBe(true));

    expect(loadCzml).toHaveBeenCalledWith(CZML_URL);
    expect(result.current.flightIds).toEqual(["flight-1", "flight-2"]);
    expect(result.current.warning).toBeNull();
    expect(mockViewer.clock.startTime.seconds).toBe(10);
    expect(mockViewer.clock.stopTime.seconds).toBe(70);
    expect(mockViewer.clock.shouldAnimate).toBe(true);
    expect(mockViewer.timeline.zoomTo).toHaveBeenCalledWith(
      mockViewer.clock.startTime,
      mockViewer.clock.stopTime,
    );
  });
});
