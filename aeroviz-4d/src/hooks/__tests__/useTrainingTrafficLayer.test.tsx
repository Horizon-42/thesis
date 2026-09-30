/**
 * A multi-aircraft window in the 3D scene (`useTrainingTrafficLayer`), on real Cesium entities (no WebGL canvas): every
 * aircraft but the one on screen drawn as the sentence read has it, each where it is at the window's time (hidden outside
 * its track), the pair under its minimum joined, the judge's end marked; the camera framed once per window.
 */
import { describe, expect, it, vi } from "vitest";
import { act, render, waitFor } from "@testing-library/react";
import { useLayoutEffect } from "react";
import * as Cesium from "cesium";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({ defaultAirport: "KXXX", airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }] }),
}));

import { AppProvider, useApp, useTrainingCursor } from "../../context/AppContext";
import useTrainingTrackLayer from "../useTrainingTrackLayer";
import { parseTrainingOverlays } from "../../data/trainingOverlays";
import { parseTrainingTrafficSet, parseTrainingWindowGenerationOverlay, trainingWindowSelection } from "../../data/trainingTraffic";
import {
  BACKGROUND_ID, REPLAYED_ID, STRAIGHT_ID, VECTORED_ID, WINDOW_MODEL_ID, mockTrafficSet, mockWindowOverlay, mockWindowOverlays,
} from "../../data/__tests__/trainingTraffic.fixture";

async function setup() {
  const set = parseTrainingTrafficSet(mockTrafficSet());
  const entries = parseTrainingOverlays(mockWindowOverlays());
  if (!set.ok || !entries.ok) throw new Error("the fixtures should parse");
  const overlay = parseTrainingWindowGenerationOverlay(mockWindowOverlay(), entries.value.overlays[0], set.value);
  if (!overlay.ok) throw new Error(overlay.problem);
  const [window] = set.value.windows;
  const focus = vi.fn();
  const view = { set: set.value, window, overlays: [overlay.value], focus };
  const entities = new Cesium.EntityCollection();
  const camera = { heading: 0, flyToBoundingSphere: vi.fn() };
  const viewer = { entities, scene: { requestRender: vi.fn() }, camera, isDestroyed: () => false } as unknown as Cesium.Viewer;
  let app!: ReturnType<typeof useApp>;
  let cursor!: ReturnType<typeof useTrainingCursor>;
  function Scene() {
    app = useApp();
    cursor = useTrainingCursor();
    useTrainingTrackLayer();
    useLayoutEffect(() => {
      app.setViewer(viewer);
      app.setMode("training");
      app.setTrainingWindow(view);
      app.setTrainingSelection(trainingWindowSelection(view.set, window, window.commanded[0]));
    }, []);
    return null;
  }
  render(<AppProvider><Scene /></AppProvider>);
  await waitFor(() => expect(app.airports).toHaveLength(1));
  const at = (id: string) => entities.getById(`training-traffic-at-${id}`)!;
  return { set: set.value, window, entities, camera, at, app: () => app, cursor: () => cursor };
}

describe("useTrainingTrafficLayer", () => {
  it("draws every aircraft but the one on screen, each where it is at the window's time", async () => {
    const { entities, at, cursor } = await setup();
    expect(entities.getById(`training-traffic-track-${VECTORED_ID}`)).toBeUndefined();   // the single-flight layers draw it
    for (const id of [STRAIGHT_ID, REPLAYED_ID, BACKGROUND_ID]) expect(entities.getById(`training-traffic-track-${id}`)).toBeDefined();
    // at the window's 100 s (the aircraft on screen's row 0): the straight-in one (from 110 s) and the background one (from
    // 200 s) are not in the air yet
    expect(cursor().trainingSceneS).toBe(100);
    expect([at(VECTORED_ID).show, at(STRAIGHT_ID).show, at(REPLAYED_ID).show, at(BACKGROUND_ID).show]).toEqual([true, false, true, false]);
    act(() => cursor().setTrainingSceneS(210));
    expect([at(STRAIGHT_ID).show, at(BACKGROUND_ID).show]).toEqual([true, true]);
  });

  it("draws each aircraft by its role: the one on screen told from the commanded, the replayed faded — and swaps them", async () => {
    const { set, window, entities, at, app } = await setup();
    const now = Cesium.JulianDate.now();
    const labelOf = (id: string) => at(id).label!.text!.getValue(now) as string;
    const widthOf = (id: string) => entities.getById(`training-traffic-track-${id}`)!.polyline!.width!.getValue(now) as number;
    expect(labelOf(VECTORED_ID).startsWith("▶ ")).toBe(true);
    expect(at(VECTORED_ID).label!.showBackground!.getValue(now)).toBe(true);
    expect(labelOf(STRAIGHT_ID).startsWith("▶ ")).toBe(false);
    expect([STRAIGHT_ID, REPLAYED_ID, BACKGROUND_ID].map((id) => at(id).name))
      .toEqual([expect.stringMatching(/\(commanded\)$/), expect.stringMatching(/\(replayed\)$/), expect.stringMatching(/\(background\)$/)]);
    expect([widthOf(STRAIGHT_ID), widthOf(REPLAYED_ID), widthOf(BACKGROUND_ID)]).toEqual([2.5, 1.2, 1]);
    // the other commanded aircraft on screen: the roles swap, and the first is drawn as commanded
    act(() => app().setTrainingSelection(trainingWindowSelection(set, window, window.commanded[1])));
    expect(labelOf(STRAIGHT_ID).startsWith("▶ ")).toBe(true);
    expect(at(VECTORED_ID).name).toMatch(/\(commanded\)$/);
    expect(widthOf(VECTORED_ID)).toBe(2.5);
    expect(entities.getById(`training-traffic-track-${STRAIGHT_ID}`)).toBeUndefined();
  });

  it("joins the pair under its minimum while it is, and marks where the judge ended an aircraft — of the sentence read", async () => {
    const { entities, app, cursor } = await setup();
    const loss = `training-traffic-loss-visual-${[VECTORED_ID, STRAIGHT_ID].sort().join("|")}`;
    act(() => app().setTrainingSource({ overlayId: WINDOW_MODEL_ID, sample: 0 }));
    act(() => cursor().setTrainingSceneS(122));
    expect(entities.getById(loss)?.name).toBe("VISUAL loss: closest 3000 m of 5556 m");
    expect(entities.getById(`training-traffic-ended-${VECTORED_ID}`)).toBeDefined();
    act(() => cursor().setTrainingSceneS(127));                     // past the step after its last
    expect(entities.getById(loss)).toBeUndefined();
    // the record: no VISUAL loss, nothing ended
    act(() => app().setTrainingSource(null));
    act(() => cursor().setTrainingSceneS(122));
    expect(entities.getById(`training-traffic-ended-${VECTORED_ID}`)).toBeUndefined();
    expect(entities.values.filter((entity) => entity.id.startsWith("training-traffic-loss-"))).toEqual([]);
  });

  it("frames the window once: another of its aircraft on screen keeps the view", async () => {
    const { set, window, camera, app } = await setup();
    await waitFor(() => expect(camera.flyToBoundingSphere).toHaveBeenCalledTimes(1));
    act(() => app().setTrainingSelection(trainingWindowSelection(set, window, window.commanded[1])));
    expect(camera.flyToBoundingSphere).toHaveBeenCalledTimes(1);
  });
});
