import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

const CZML_URL = "/data/airports/KRDU/trajectories.czml";
const fetchMock = vi.fn();

function jsonResponse(body: unknown) {
  return {
    ok: true,
    headers: { get: () => "application/json" },
    text: async () => JSON.stringify(body),
  };
}

function observedResponse(
  czml: unknown[],
  verdicts: unknown = null,
  evaluation: unknown = null,
) {
  return { schemaVersion: "observed-trajectories-v1", czml, verdicts, evaluation };
}

const {
  loadCzml,
  mockViewer,
  setTrajectoriesVisible,
  getTrajectoriesVisible,
  setSelectedFlightId,
  setTrajectoryDataSource,
  makeTime,
  ConstantProperty,
  ColorMaterialProperty,
} = vi.hoisted(() => {
  const makeTime = (seconds: number): any => ({
    seconds,
    clone: () => makeTime(seconds),
  });

  const loadCzml = vi.fn();
  const setSelectedFlightId = vi.fn();
  const setTrajectoryDataSource = vi.fn();
  let trajectoriesVisible = true;

  class ConstantProperty {
    constructor(private readonly value: unknown) {}

    getValue() {
      return this.value;
    }
  }

  class ColorMaterialProperty {
    color: ConstantProperty;

    constructor(color: unknown) {
      this.color = new ConstantProperty(color);
    }
  }

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
    setTrajectoryDataSource,
    makeTime,
    ConstantProperty,
    ColorMaterialProperty,
  };
});

vi.mock("cesium", () => ({
  ClockRange: {
    LOOP_STOP: "LOOP_STOP",
    CLAMPED: "CLAMPED",
  },
  Color: {
    fromCssColorString: (value: string) => value,
  },
  ConstantProperty,
  ColorMaterialProperty,
  JulianDate: {
    lessThan: (left: { seconds: number }, right: { seconds: number }) =>
      left.seconds < right.seconds,
  },
  CzmlDataSource: class CzmlDataSource {
    entities = { values: [] };
    clock: unknown = null;
    show = true;

    constructor(public name: string) {}

    async load(czml: unknown) {
      const loaded = await loadCzml(czml);
      Object.assign(this, loaded);
      return this;
    }
  },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
    layers: { trajectories: getTrajectoriesVisible() },
    autoReplay: true,
    selectedFlightId: null,
    setSelectedFlightId,
    setTrajectoryDataSource,
  }),
}));

import { useObservedTrajectoryLayer } from "../useObservedTrajectoryLayer";

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
  setTrajectoryDataSource.mockClear();
  setTrajectoriesVisible(true);
}

describe("useObservedTrajectoryLayer", () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    resetViewer();
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(jsonResponse(observedResponse([{ id: "document" }])));
    vi.stubGlobal("fetch", fetchMock);
    loadCzml.mockReset();
    warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    warnSpy.mockRestore();
    vi.unstubAllGlobals();
  });

  it("warns and skips Cesium clock work when the CZML has no trajectory entities", async () => {
    loadCzml.mockResolvedValue({
      entities: { values: [{ id: "document" }] },
      clock: { startTime: makeTime(5), stopTime: makeTime(5) },
    });

    const { result } = renderHook(() => useObservedTrajectoryLayer(CZML_URL));

    await waitFor(() => expect(result.current.isLoaded).toBe(true));

    expect(fetchMock).toHaveBeenCalledWith(CZML_URL);
    expect(loadCzml).toHaveBeenCalledWith([{ id: "document" }]);
    expect(result.current.flightIds).toEqual([]);
    expect(result.current.warning).toContain("No trajectory entities");
    expect(mockViewer.dataSources.add).not.toHaveBeenCalled();
    expect(mockViewer.timeline.zoomTo).not.toHaveBeenCalled();
    expect(setTrajectoryDataSource).toHaveBeenCalledWith(null);
    expect(warnSpy).toHaveBeenCalled();
  });

  it("loads entities but warns instead of zooming the timeline for a zero-duration clock", async () => {
    loadCzml.mockResolvedValue({
      entities: { values: [{ id: "flight-1" }] },
      clock: { startTime: makeTime(8), stopTime: makeTime(8) },
    });

    const { result } = renderHook(() => useObservedTrajectoryLayer(CZML_URL));

    await waitFor(() => expect(result.current.isLoaded).toBe(true));

    expect(fetchMock).toHaveBeenCalledWith(CZML_URL);
    expect(loadCzml).toHaveBeenCalledWith([{ id: "document" }]);
    expect(result.current.flightIds).toEqual(["flight-1"]);
    expect(result.current.warning).toContain("has no duration");
    expect(mockViewer.dataSources.add).toHaveBeenCalledTimes(1);
    expect(setTrajectoryDataSource).toHaveBeenLastCalledWith(expect.objectContaining({
      entities: { values: [expect.objectContaining({ id: "flight-1" })] },
    }));
    expect(mockViewer.timeline.zoomTo).not.toHaveBeenCalled();
  });

  it("syncs the Cesium clock and timeline for a valid CZML interval", async () => {
    loadCzml.mockResolvedValue({
      entities: { values: [{ id: "flight-1" }, { id: "flight-2" }] },
      clock: { startTime: makeTime(10), stopTime: makeTime(70) },
    });

    const { result } = renderHook(() => useObservedTrajectoryLayer(CZML_URL));

    await waitFor(() => expect(result.current.isLoaded).toBe(true));

    expect(fetchMock).toHaveBeenCalledWith(CZML_URL);
    expect(loadCzml).toHaveBeenCalledWith([{ id: "document" }]);
    expect(result.current.flightIds).toEqual(["flight-1", "flight-2"]);
    expect(result.current.warning).toBeNull();
    expect(mockViewer.clock.startTime.seconds).toBe(10);
    expect(mockViewer.clock.stopTime.seconds).toBe(70);
    expect(mockViewer.clock.shouldAnimate).toBe(true);
    expect(setTrajectoryDataSource).toHaveBeenLastCalledWith(expect.objectContaining({
      entities: { values: [expect.objectContaining({ id: "flight-1" }), expect.objectContaining({ id: "flight-2" })] },
    }));
    expect(mockViewer.timeline.zoomTo).toHaveBeenCalledWith(
      mockViewer.clock.startTime,
      mockViewer.clock.stopTime,
    );
  });

  it("loads the source but keeps it hidden when visible=false (no reload needed to toggle)", async () => {
    loadCzml.mockResolvedValue({
      entities: { values: [{ id: "flight-1" }] },
      clock: { startTime: makeTime(10), stopTime: makeTime(70) },
    });

    const { result } = renderHook(() => useObservedTrajectoryLayer(CZML_URL, false));
    await waitFor(() => expect(result.current.isLoaded).toBe(true));

    // The file is still loaded (parsed + added) ...
    expect(mockViewer.dataSources.add).toHaveBeenCalledTimes(1);
    // ... but the layer is hidden, so flipping `visible` later only toggles show.
    expect(mockViewer.dataSources.add.mock.calls[0][0].show).toBe(false);
  });

  it("shows the loaded source when visible=true", async () => {
    loadCzml.mockResolvedValue({
      entities: { values: [{ id: "flight-1" }] },
      clock: { startTime: makeTime(10), stopTime: makeTime(70) },
    });

    const { result } = renderHook(() => useObservedTrajectoryLayer(CZML_URL, true));
    await waitFor(() => expect(result.current.isLoaded).toBe(true));

    expect(mockViewer.dataSources.add.mock.calls[0][0].show).toBe(true);
  });

  it("replaces the bounded response when the backend runway URL changes", async () => {
    const runway05Url = `${CZML_URL}?runway=05L`;
    const runway23Url = `${CZML_URL}?runway=23R`;
    fetchMock
      .mockResolvedValueOnce(jsonResponse(observedResponse([
        { id: "document" },
        { id: "flight-05L" },
      ])))
      .mockResolvedValueOnce(jsonResponse(observedResponse([
        { id: "document" },
        { id: "flight-23R" },
      ])));
    loadCzml
      .mockResolvedValueOnce({
        entities: { values: [{ id: "flight-05L", show: true }] },
        clock: { startTime: makeTime(10), stopTime: makeTime(70) },
      })
      .mockResolvedValueOnce({
        entities: { values: [{ id: "flight-23R", show: true }] },
        clock: { startTime: makeTime(10), stopTime: makeTime(70) },
      });

    const { result, rerender } = renderHook(
      ({ url }) => useObservedTrajectoryLayer(url, true),
      { initialProps: { url: runway05Url } },
    );
    await waitFor(() => expect(result.current.flightIds).toEqual(["flight-05L"]));

    rerender({ url: runway23Url });
    await waitFor(() => expect(result.current.flightIds).toEqual(["flight-23R"]));
    expect(fetchMock).toHaveBeenNthCalledWith(1, runway05Url);
    expect(fetchMock).toHaveBeenNthCalledWith(2, runway23Url);
    expect(loadCzml).toHaveBeenCalledTimes(2);
  });

  it("renders every backend-selected verdict track and exposes compact metadata", async () => {
    const entities = [{
      id: "flight-fail",
      show: false,
      path: { material: new ColorMaterialProperty("original") },
    }];
    fetchMock.mockResolvedValue(jsonResponse(observedResponse(
      [{ id: "document" }, { id: "flight-fail" }],
      {
        counts: { pass: 4, fail: 1, undecided: 2 },
        byFlightId: { "flight-fail": "fail" },
        matched: 6,
        total: 7,
      },
      {
        total: 7,
        verdict_counts: { pass: 4, fail: 1, indeterminate: 2 },
        observed: { event_estimated_rate: 0.8 },
        lateral_m: { mean: 12.5 },
        vertical_m: { mean_abs: 4.5 },
      },
    )));
    loadCzml.mockResolvedValue({
      entities: { values: entities },
      clock: { startTime: makeTime(10), stopTime: makeTime(70) },
    });

    const { result } = renderHook(() => useObservedTrajectoryLayer(CZML_URL));
    await waitFor(() => expect(result.current.isLoaded).toBe(true));

    expect(result.current.flightIds).toEqual(["flight-fail"]);
    expect(result.current.observedVerdicts.counts).toEqual({ pass: 4, fail: 1, undecided: 2 });
    expect(result.current.observedEvaluation?.observed?.event_estimated_rate).toBe(0.8);
    expect(entities[0].show).toBe(true);
    expect(entities[0].path.material.color.getValue()).toBe("rgb(230, 70, 70)");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(loadCzml).toHaveBeenCalledTimes(1);
  });
});
