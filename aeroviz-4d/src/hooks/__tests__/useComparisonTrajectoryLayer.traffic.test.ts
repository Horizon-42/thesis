import { act, renderHook, waitFor } from "@testing-library/react";
import * as Cesium from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MAX_FLIGHT_KEYS_PER_REQUEST } from "../../data/observedTracks";
import { DEFAULT_MODEL_BUDGET } from "../../utils/trajectoryRenderModel";

/**
 * The comparison layer with a traffic window's recorded aircraft: the REAL Cesium data sources, a viewer double,
 * and the backend / publication files answered by `router.serve`.
 */

const { appState, router } = vi.hoisted(() => ({
  appState: {
    viewer: null as unknown,
    layers: { trajectories: true },
    mode: "evaluation",
    trajectoryComparison: true,
    trajectoryComparisonCategory: "traffic_m1_runway",
    trajectoryComparisonKinds: {
      reference: true, optimizer: false, simulator: true, predicted: true, lookback: true,
    },
    activeAirportCode: "KRDU",
    selectedRunway: null as string | null,
    trajectorySampleCount: 0,                      // 0: every group of the index is shown
    setSelectedFlightId: () => undefined,
    setTrajectoryDataSource: () => undefined,
  },
  router: { serve: (async (_url: string) => undefined) as (url: string) => Promise<unknown> },
}));

vi.mock("../../context/AppContext", () => ({ useApp: () => appState }));
vi.mock("../../utils/fetchJson", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../utils/fetchJson")>()),
  fetchJson: (url: string) => router.serve(url),
}));

import { useComparisonTrajectoryLayer } from "../useComparisonTrajectoryLayer";

const EPOCH = "2026-04-01T08:00:00Z";
const epochPlus = (seconds: number) =>
  Cesium.JulianDate.addSeconds(Cesium.JulianDate.fromIso8601(EPOCH), seconds, new Cesium.JulianDate());

const A = "AAL100_05L_a00001_20260501T000300Z";      // two commanded flights, one per runway
const B = "BAW200_05R_b00002_20260501T020300Z";

function group(key: string, runway: string, recorded: string[], startOffsetsS: number[]) {
  return {
    group: key, flightId: key.split("_")[0], runway, airport: "KRDU", czml: `comparison_KRDU_${runway}_g.czml`,
    status: "solved", finalTimeS: 300, initialState: null, entities: [`ref-${key}`, `sim-${key}`],
    traffic: { outcome: "separated", recorded: recorded.map((k) => `ref-${k}`), startOffsetsS },
  };
}

function indexOf(groups: unknown[], scene?: unknown) {
  return {
    schemaVersion: "comparison-v2-generation", generation: "g", epoch: EPOCH, startHidden: true,
    referenceSource: "canonicalObserved", evaluationReport: "r.json", groups,
    ...(scene ? { scene } : {}),
  };
}

/** An arrival-window flight: t = 0 at its own entry, `durationS` long, with an aircraft model. */
function flightPacket(id: string, durationS: number) {
  return {
    id, name: id.split("_")[0],
    model: { gltf: "/models/aircraft.glb" },
    position: {
      epoch: EPOCH, interpolationAlgorithm: "LINEAR",
      cartographicDegrees: [0, -78.0, 35.0, 1000, durationS, -77.0, 35.0, 900],
    },
    path: { show: true, leadTime: 0, trailTime: 300, width: 2, material: { solidColor: { color: { rgba: [235, 235, 235, 200] } } } },
  };
}

const NEIGHBOUR_DURATION_S = 300;
const REFERENCE_DURATION_S = 200;

/** Answers the index, the comparison CZMLs and the backend's arrival window; records each backend request's keys. */
function serve(
  groups: unknown[],
  requests: string[][],
  { withClock = true, scene, resultClockS, backendClockS = REFERENCE_DURATION_S }:
    { withClock?: boolean; scene?: unknown; resultClockS?: number; backendClockS?: number } = {},
) {
  router.serve = async (url: string) => {
    if (url.endsWith("/comparison_index.json")) return indexOf(groups, scene);
    if (url.endsWith(".czml")) {
      return [{
        id: "document", name: "result", version: "1.0",
        ...(resultClockS === undefined ? {} : { clock: {
          interval: `${EPOCH}/${epochPlus(resultClockS).toString()}`, currentTime: EPOCH,
          multiplier: 60, range: "LOOP_STOP", step: "SYSTEM_CLOCK_MULTIPLIER",
        } }),
      }];
    }
    const keys = new URL(url, "http://backend").searchParams.getAll("flight_key");
    requests.push(keys);
    return {
      schemaVersion: "observed-trajectories-v2", trackWindow: "arrival", verdicts: null, evaluation: null,
      czml: [
        {
          id: "document", name: "observed", version: "1.0",
          ...(withClock ? {
            clock: {
              interval: `${EPOCH}/${epochPlus(backendClockS).toString()}`, currentTime: EPOCH,
              multiplier: 60, range: "LOOP_STOP", step: "SYSTEM_CLOCK_MULTIPLIER",
            },
          } : {}),
        },
        ...keys.map((key) => flightPacket(key, key === A || key === B ? REFERENCE_DURATION_S : NEIGHBOUR_DURATION_S)),
      ],
    };
  };
}

function viewerDouble() {
  const sources: Cesium.DataSource[] = [];
  return {
    sources,
    dataSources: {
      add: vi.fn((source: Cesium.DataSource) => { sources.push(source); return Promise.resolve(source); }),
      remove: vi.fn((source: Cesium.DataSource) => { sources.splice(sources.indexOf(source), 1); return true; }),
    },
    clock: {} as Record<string, unknown>,
    timeline: { zoomTo: vi.fn() },
    scene: { canvas: document.createElement("canvas") },
    trackedEntity: undefined,
  };
}

type Viewer = ReturnType<typeof viewerDouble>;
/** The neighbours' data source (a CZML data source takes its name from its document packet, so find it by its entities). */
const trafficSource = (viewer: Viewer) =>
  viewer.sources.find((source) => source.entities.values.some((entity) => entity.id.startsWith("traffic-"))) as
    Cesium.CzmlDataSource | undefined;
const trafficIds = (viewer: Viewer) =>
  trafficSource(viewer)!.entities.values.filter((e) => e.id !== "document").map((e) => e.id);
const sampleKeys = (prefix: string, n: number) => Array.from({ length: n }, (_, i) => `${prefix}${i}_05L_c00000_20260501T0000${i % 60}Z`);

let viewer: Viewer;
let requests: string[][];

beforeEach(() => {
  viewer = viewerDouble();
  requests = [];
  appState.viewer = viewer;
  appState.selectedRunway = null;
  appState.trajectorySampleCount = 0;
  appState.trajectoryComparisonCategory = "traffic_m1_runway";
  appState.layers = { trajectories: true };
  appState.trajectoryComparisonKinds = { ...appState.trajectoryComparisonKinds, reference: true };
});

afterEach(() => vi.restoreAllMocks());

describe("useComparisonTrajectoryLayer with a traffic window", () => {
  it("shows the recorded aircraft of the shown groups, and removes a hidden group's with it", async () => {
    const [a1, a2] = sampleKeys("A", 2);
    const [b1] = sampleKeys("B", 1);
    serve([group(A, "05L", [a1, a2], [-10, 20]), group(B, "05R", [b1], [5])], requests);

    const { rerender } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    const first = trafficSource(viewer)!;
    expect(trafficIds(viewer).sort()).toEqual([`traffic-${A}/${a1}`, `traffic-${A}/${a2}`, `traffic-${B}/${b1}`].sort());

    // Hide group A (the runway selector shows only 05R): its neighbours go with it, B's are loaded again.
    appState.selectedRunway = "05R";
    rerender();
    await waitFor(() => {
      expect(trafficSource(viewer)).toBeDefined();
      expect(trafficSource(viewer)).not.toBe(first);
    });
    expect(viewer.dataSources.remove).toHaveBeenCalledWith(first, true);
    expect(trafficIds(viewer)).toEqual([`traffic-${B}/${b1}`]);
  });

  const spanOf = (id: string): [number, number] => {
    const availability = trafficSource(viewer)!.entities.getById(id)!.availability!;
    return [Cesium.JulianDate.secondsDifference(availability.start, epochPlus(0)),
            Cesium.JulianDate.secondsDifference(availability.stop, epochPlus(0))];
  };

  it("puts each recorded aircraft at its offset on the group's clock", async () => {
    const [a1, a2] = sampleKeys("A", 2);
    serve([group(A, "05L", [a1, a2], [20, 60])], requests);

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    // entering 20 s and 60 s after the commanded aircraft, inside the clock (0..200 s): from there
    expect(spanOf(`traffic-${A}/${a1}`)[0]).toBe(20);
    expect(spanOf(`traffic-${A}/${a2}`)[0]).toBe(60);
  });

  it("leaves the clock to the groups' own span and clips the neighbours to it, on purpose", async () => {
    const [a1, a2, a3] = sampleKeys("A", 3);
    // The reference spans 0..200 s; the neighbours (300 s long) enter 100 s before it, 50 s into it, 600 s after it.
    serve([group(A, "05L", [a1, a2, a3], [-100, 50, 600])], requests);

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(viewer.clock.stopTime).toBeDefined());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());

    // the clock is the reference's: neighbours that came before or go on after it do not move it
    expect(Cesium.JulianDate.secondsDifference(viewer.clock.startTime as Cesium.JulianDate, epochPlus(0))).toBe(0);
    expect(Cesium.JulianDate.secondsDifference(viewer.clock.stopTime as Cesium.JulianDate, epochPlus(0))).toBe(REFERENCE_DURATION_S);
    // airborne at the clock start: shown from there, mid-flight; entering inside the clock: from its entry; both end at the stop
    expect(spanOf(`traffic-${A}/${a1}`)).toEqual([0, REFERENCE_DURATION_S]);
    expect(spanOf(`traffic-${A}/${a2}`)).toEqual([50, REFERENCE_DURATION_S]);
    // entering after the clock stop: not shown at all — not even loaded
    expect(trafficSource(viewer)!.entities.getById(`traffic-${A}/${a3}`)).toBeUndefined();
  });

  it("loads only the neighbours that have something left inside the clock, and plans the model budget over those", async () => {
    // 10 enter inside the reference's span, 40 only after its stop: the budget (20) then covers all 10 shown ones.
    // Planned over all 50 it would give a model to about 4 of the 10 and waste the rest on neighbours never drawn.
    const inside = sampleKeys("I", 10);
    const outside = sampleKeys("O", 40);
    serve([group(A, "05L", [...inside, ...outside], [...inside.map(() => 10), ...outside.map(() => 600)])], requests);

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    expect(trafficIds(viewer).sort()).toEqual(inside.map((key) => `traffic-${A}/${key}`).sort());
    const now = Cesium.JulianDate.now();
    expect(trafficSource(viewer)!.entities.values.every((e) => e.model!.show!.getValue(now))).toBe(true);
  });

  it("draws no traffic source at all when every neighbour is outside the clock", async () => {
    const [a1] = sampleKeys("A", 1);
    serve([group(A, "05L", [a1], [600])], requests);

    const { result } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(result.current.isLoaded).toBe(true));
    expect(result.current.error).toBeNull();
    expect(trafficSource(viewer)).toBeUndefined();
  });

  it("refuses to draw recorded aircraft when the comparison carries no clock to cut them to", async () => {
    const [a1] = sampleKeys("A", 1);
    serve([group(A, "05L", [a1], [10])], requests, { withClock: false });

    const { result } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(result.current.error).toMatch(/no clock to cut the recorded aircraft's availability to/));
    expect(trafficSource(viewer)).toBeUndefined();
  });

  it("gives only the references' model budget of the neighbours an aircraft model", async () => {
    const neighbours = sampleKeys("N", DEFAULT_MODEL_BUDGET + 10);
    serve([group(A, "05L", neighbours, neighbours.map(() => 0))], requests);

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    const now = Cesium.JulianDate.now();
    const withModel = trafficSource(viewer)!.entities.values
      .filter((e) => e.id !== "document" && e.model!.show!.getValue(now));
    expect(trafficIds(viewer)).toHaveLength(DEFAULT_MODEL_BUDGET + 10);
    expect(withModel).toHaveLength(DEFAULT_MODEL_BUDGET);
  });

  it("asks the backend for the recorded aircraft in requests of at most the backend's limit", async () => {
    const neighbours = sampleKeys("N", 2 * MAX_FLIGHT_KEYS_PER_REQUEST + 500);
    serve([group(A, "05L", neighbours, neighbours.map(() => 0))], requests);

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    expect(requests.map((keys) => keys.length)).toEqual([1, MAX_FLIGHT_KEYS_PER_REQUEST, MAX_FLIGHT_KEYS_PER_REQUEST, 500]);
    expect(requests[0]).toEqual([A]);                                  // the group's own reference
    expect(requests.slice(1).flat()).toEqual(neighbours);              // every neighbour once, in order
    expect(trafficIds(viewer)).toHaveLength(neighbours.length);
  });

  it("loads nothing when the layer is torn down while the neighbours are still being fetched", async () => {
    const [a1] = sampleKeys("A", 1);
    serve([group(A, "05L", [a1], [0])], requests);
    const answer = router.serve;
    let release: () => void = () => undefined;
    const held = new Promise<void>((resolve) => { release = resolve; });
    let neighboursAsked = false;
    router.serve = async (url: string) => {
      const keys = url.includes("/trajectories?") ? new URL(url, "http://backend").searchParams.getAll("flight_key") : [];
      if (keys.includes(a1)) {                                        // the neighbours' request, held open
        neighboursAsked = true;
        await held;
      }
      return answer(url);
    };
    const load = vi.spyOn(Cesium.CzmlDataSource.prototype, "load");

    const { unmount } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(neighboursAsked).toBe(true));
    unmount();
    await act(async () => { release(); await new Promise((resolve) => setTimeout(resolve, 20)); });

    const loaded = load.mock.calls.flatMap(([czml]) => (czml as Array<{ id: string }>).map((packet) => packet.id));
    expect(loaded.some((id) => id.startsWith("traffic-"))).toBe(false);   // never even parsed
    expect(trafficSource(viewer)).toBeUndefined();
  });

  it("removes the neighbours' data source with the others when the layer is torn down", async () => {
    const [a1] = sampleKeys("A", 1);
    serve([group(A, "05L", [a1], [0])], requests);

    const { unmount } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    const added = [...viewer.sources];
    expect(added).toContain(trafficSource(viewer));
    unmount();

    for (const source of added) expect(viewer.dataSources.remove).toHaveBeenCalledWith(source, true);
    expect(viewer.sources).toHaveLength(0);
  });

  it("shows the neighbours with the Reference switch and the trajectories layer", async () => {
    const [a1] = sampleKeys("A", 1);
    serve([group(A, "05L", [a1], [0])], requests);

    const { rerender } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)?.show).toBe(true));

    appState.trajectoryComparisonKinds = { ...appState.trajectoryComparisonKinds, reference: false };
    rerender();
    await waitFor(() => expect(trafficSource(viewer)!.show).toBe(false));

    appState.trajectoryComparisonKinds = { ...appState.trajectoryComparisonKinds, reference: true };
    rerender();
    await waitFor(() => expect(trafficSource(viewer)!.show).toBe(true));

    appState.layers = { trajectories: false };
    rerender();
    await waitFor(() => expect(trafficSource(viewer)!.show).toBe(false));
    appState.layers = { trajectories: true };
  });

  it("asks for, and draws, no neighbours for a category without traffic", async () => {
    const plain = { ...group(A, "05L", [], []), traffic: undefined };
    serve([plain], requests);

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(viewer.sources.length).toBeGreaterThan(0));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 20)); });
    expect(requests).toEqual([[A]]);
    expect(trafficSource(viewer)).toBeUndefined();
  });
  it("reloads a traffic window's category when the sample count changes (only a scene ignores it)", async () => {
    const [a1] = sampleKeys("A", 1);
    serve([group(A, "05L", [a1], [10]), group(B, "05R", [a1], [20])], requests);

    const { rerender } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    const first = trafficSource(viewer)!;
    appState.trajectorySampleCount = 1;
    rerender();
    await waitFor(() => expect(viewer.dataSources.remove).toHaveBeenCalledWith(first, true));
  });
});

// ── An M2 run as one scene ───────────────────────────────────────────────────

/** A scene group: no `traffic` of its own; its place on the scene clock. */
function sceneGroup(key: string, runway: string, startOffsetS: number) {
  const { traffic: _traffic, ...plain } = group(key, runway, [], []);
  return { ...plain, scene: { startOffsetS, outcome: "separated", delayS: 0 } };
}

function sceneOf(background: string[], startOffsetsS: number[]) {
  return {
    startUtc: "2026-05-21T17:47:18.959Z",
    background: { recorded: background.map((key) => `ref-${key}`), startOffsetsS },
  };
}

const referenceSource = (viewer: Viewer) =>
  viewer.sources.find((source) => source.entities.values.some((entity) => entity.id === A));
const referenceEntity = (viewer: Viewer, key: string) => referenceSource(viewer)!.entities.getById(key)!;
const spanSeconds = (entity: Cesium.Entity): [number, number] => [
  Cesium.JulianDate.secondsDifference(entity.availability!.start, epochPlus(0)),
  Cesium.JulianDate.secondsDifference(entity.availability!.stop, epochPlus(0)),
];

describe("useComparisonTrajectoryLayer with an M2 scene", () => {
  it("shows every group, whatever the sample count and the runway selector", async () => {
    appState.trajectorySampleCount = 1;
    appState.selectedRunway = "05L";
    serve([sceneGroup(A, "05L", 0), sceneGroup(B, "05R", 500)], requests, { scene: sceneOf([], []) });

    const { result } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(result.current.isLoaded).toBe(true));
    expect(requests).toEqual([[A, B]]);
    expect(result.current.flightIds.sort()).toEqual([A, B]);
  });

  it("draws each group's reference on the scene clock, at its offset, and only while it flies", async () => {
    serve([sceneGroup(A, "05L", 0), sceneGroup(B, "05R", 500)], requests, { scene: sceneOf([], []) });

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(referenceSource(viewer)).toBeDefined());
    // the backend serves both at their own entry (0 .. 200 s); B enters 500 s into the scene
    expect(spanSeconds(referenceEntity(viewer, A))).toEqual([0, REFERENCE_DURATION_S]);
    expect(spanSeconds(referenceEntity(viewer, B))).toEqual([500, 500 + REFERENCE_DURATION_S]);
    const lonAt = (key: string, seconds: number) => Cesium.Math.toDegrees(Cesium.Cartographic.fromCartesian(
      referenceEntity(viewer, key).position!.getValue(epochPlus(seconds))!).longitude);
    expect(lonAt(B, 500)).toBeCloseTo(-78.0, 6);                       // B's first sample, 500 s in
    expect(lonAt(B, 600)).toBeCloseTo(-77.5, 4);                       // half way along its 200 s track at 600 s
  });

  it("loads the background once for the whole scene, not once per group", async () => {
    const [bg1, bg2] = sampleKeys("G", 2);
    serve([sceneGroup(A, "05L", 0), sceneGroup(B, "05R", 500)], requests,
      { scene: sceneOf([bg1, bg2], [10, 20]) });

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    expect(trafficIds(viewer).sort()).toEqual([`traffic-scene/${bg1}`, `traffic-scene/${bg2}`].sort());
    expect(requests).toEqual([[A, B], [bg1, bg2]]);                    // one request for the background
  });

  it("puts the background on the scene clock and clips it to the clock like a traffic window's neighbours", async () => {
    const [before, inside, after] = sampleKeys("G", 3);
    // The groups span 0..700 s on the scene clock (B's shifted reference ends last). The background (300 s long)
    // enters 100 s before it, 650 s into it, 900 s after its start.
    serve([sceneGroup(A, "05L", 0), sceneGroup(B, "05R", 500)], requests,
      { scene: sceneOf([before, inside, after], [-100, 650, 900]), resultClockS: 650, backendClockS: 900 });

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    const end = 500 + REFERENCE_DURATION_S;
    expect(spanSeconds(trafficSource(viewer)!.entities.getById(`traffic-scene/${before}`)!)).toEqual([0, REFERENCE_DURATION_S]);
    expect(spanSeconds(trafficSource(viewer)!.entities.getById(`traffic-scene/${inside}`)!)).toEqual([650, end]);
    expect(trafficSource(viewer)!.entities.getById(`traffic-scene/${after}`)).toBeUndefined();     // after the stop: not loaded
  });

  it("takes the viewer clock from the groups' spans on the scene clock, shifted references included", async () => {
    const [bg] = sampleKeys("G", 1);
    // The result files' clock stops at 650 s, the backend's own document clock at 900 s; the shifted reference of B
    // ends at 700 s. The scene clock is 0 .. 700: the longest group span, not the backend's unshifted document.
    serve([sceneGroup(A, "05L", 0), sceneGroup(B, "05R", 500)], requests,
      { scene: sceneOf([bg], [900]), resultClockS: 650, backendClockS: 900 });

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(viewer.clock.stopTime).toBeDefined());
    expect(Cesium.JulianDate.secondsDifference(viewer.clock.startTime as Cesium.JulianDate, epochPlus(0))).toBe(0);
    expect(Cesium.JulianDate.secondsDifference(viewer.clock.stopTime as Cesium.JulianDate, epochPlus(0))).toBe(700);
  });

  it("gives only the references' model budget of the background an aircraft model", async () => {
    const background = sampleKeys("G", DEFAULT_MODEL_BUDGET + 10);
    serve([sceneGroup(A, "05L", 0)], requests, { scene: sceneOf(background, background.map(() => 10)) });

    renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    const now = Cesium.JulianDate.now();
    expect(trafficSource(viewer)!.entities.values.filter((e) => e.model!.show!.getValue(now))).toHaveLength(DEFAULT_MODEL_BUDGET);
  });

  it("shows the background and the references with the Reference switch", async () => {
    const [bg] = sampleKeys("G", 1);
    serve([sceneGroup(A, "05L", 0)], requests, { scene: sceneOf([bg], [10]) });

    const { rerender } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)?.show).toBe(true));
    expect(referenceSource(viewer)!.show).toBe(true);

    appState.trajectoryComparisonKinds = { ...appState.trajectoryComparisonKinds, reference: false };
    rerender();
    await waitFor(() => expect(trafficSource(viewer)!.show).toBe(false));
    expect(referenceSource(viewer)!.show).toBe(false);
  });

  it("draws no background source for a scene without one", async () => {
    serve([sceneGroup(A, "05L", 0)], requests, { scene: sceneOf([], []) });

    const { result } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(result.current.isLoaded).toBe(true));
    expect(result.current.error).toBeNull();
    expect(trafficSource(viewer)).toBeUndefined();
    expect(requests).toEqual([[A]]);
  });

  it("does not reload when the runway selector or the sample count changes: they change nothing in a scene", async () => {
    const [bg] = sampleKeys("G", 1);
    serve([sceneGroup(A, "05L", 0), sceneGroup(B, "05R", 500)], requests, { scene: sceneOf([bg], [10]), resultClockS: 650 });
    const answer = router.serve;
    let fetches = 0;
    router.serve = (url: string) => { fetches += 1; return answer(url); };

    const { rerender, result } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(trafficSource(viewer)).toBeDefined());
    await waitFor(() => expect(viewer.clock.stopTime).toBeDefined());
    const [asked, clockStop, removed] = [fetches, viewer.clock.stopTime, viewer.dataSources.remove.mock.calls.length];

    appState.selectedRunway = "05R";
    rerender();
    appState.trajectorySampleCount = 1;
    rerender();
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 50)); });

    expect(fetches).toBe(asked);                                       // nothing refetched
    expect(viewer.dataSources.remove.mock.calls).toHaveLength(removed);  // nothing torn down
    expect(viewer.clock.stopTime).toBe(clockStop);                     // the clock not reset
    expect(result.current.flightIds.sort()).toEqual([A, B]);           // every group still there
  });

  it("still reloads a scene when its category is left for another", async () => {
    serve([sceneGroup(A, "05L", 0)], requests, { scene: sceneOf([], []) });
    const { rerender } = renderHook(() => useComparisonTrajectoryLayer());
    await waitFor(() => expect(referenceSource(viewer)).toBeDefined());
    const first = referenceSource(viewer);

    appState.trajectoryComparisonCategory = "other_category";
    rerender();
    await waitFor(() => expect(viewer.dataSources.remove).toHaveBeenCalledWith(first, true));
    appState.trajectoryComparisonCategory = "traffic_m1_runway";
  });
});
