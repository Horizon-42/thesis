/**
 * The Training flight in the 3D scene, on real Cesium entities (no WebGL canvas): the envelopes
 * the switches ask for are built once — each heading word's judged rows on the ground with its rows
 * outside in red, the capture turn and corridor, the tubes; the SELECTED word — one column's, chosen
 * in the sentence bar — lights up its own envelope and nothing of another column's, and is restored
 * without rebuilding anything; a new flight is framed once; the executor's rows outside its words are
 * red on its ground trace; leaving Training removes the layer.
 */
import { useLayoutEffect } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import * as Cesium from "cesium";
import { AppProvider, useApp } from "../../context/AppContext";
import { parseTrainingSample } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import { EXECUTOR_ID, mockExecutorOverlay, mockOverlayEntry } from "../../data/__tests__/trainingOverlays.fixture";
import { parseTrainingExecutorOverlay } from "../../data/trainingOverlays";
import useTrainingTrackLayer, { TRAINING_ENTITY } from "../../hooks/useTrainingTrackLayer";
import {
  TRAINING_CAPTURE_TURN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import TrainingSentenceBar from "../TrainingSentenceBar";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({
    defaultAirport: "KXXX",
    airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }],
  }),
}));

async function setup(position = 0, edit: (raw: any) => void = () => undefined, executor = false) {
  const raw = mockSample();
  edit(raw);
  const parsed = parseTrainingSample(raw);
  if (!parsed.ok) throw new Error(parsed.problem);
  const selection = {
    vocabulary: parsed.value.vocabulary, candidates: parsed.value.candidates, flight: parsed.value.flights[position],
  };
  const read = parseTrainingExecutorOverlay(mockExecutorOverlay(), mockOverlayEntry(EXECUTOR_ID), parsed.value);
  if (!read.ok) throw new Error(read.problem);
  const overlay = read.value;
  const entities = new Cesium.EntityCollection();
  const requestRender = vi.fn();
  const camera = { heading: 0, flyToBoundingSphere: vi.fn() };
  const viewer = { entities, scene: { requestRender }, camera, isDestroyed: () => false } as unknown as Cesium.Viewer;
  let app: ReturnType<typeof useApp>;
  function Scene() {
    app = useApp();
    useTrainingTrackLayer();
    useLayoutEffect(() => {
      app.setViewer(viewer);
      app.setMode("training");
      app.setTrainingSelection(selection);
      if (executor) app.setTrainingExecutor({ overlay, flight: overlay.flights[position] });
    }, []);
    return <TrainingSentenceBar />;
  }
  const view = render(<AppProvider><Scene /></AppProvider>);
  await waitFor(() => expect(app!.airports).toHaveLength(1));
  const colourOf = (id: string) => {
    const entity = entities.getById(id)!;
    const graphics = entity.polygon ?? entity.wall ?? entity.polyline;
    return graphics!.material!.getValue(Cesium.JulianDate.now()).color as Cesium.Color;
  };
  const css = (value: string, alpha = 1) => Cesium.Color.fromCssColorString(value).withAlpha(alpha);
  return { ...view, entities, selection, requestRender, camera, colourOf, css, app: () => app! };
}

const band = (name: RegExp) => screen.getByLabelText(name);
const HEADING_180 = /^heading 180° — the heading the track reaches a lead later/;
const DESCEND_TO_LAND = /^altitude descend to land — a new target/;
const CLEARED = /^approach cleared — cleared to join the final/;

describe("Training envelopes in the 3D scene", () => {
  it("builds the track, the heading words' rows, the capture, the tubes, every candidate and the issue points", async () => {
    const scene = await setup();
    for (const id of [
      TRAINING_ENTITY.track, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis, TRAINING_ENTITY.captureTurn,
      TRAINING_ENTITY.heading(0), TRAINING_ENTITY.heading(1), TRAINING_ENTITY.heading(2), TRAINING_ENTITY.headingOutside(2, 0),
      TRAINING_ENTITY.tube(0), TRAINING_ENTITY.tube(1),
      TRAINING_ENTITY.centreline("09"), TRAINING_ENTITY.runway("27"), TRAINING_ENTITY.issue(0), TRAINING_ENTITY.issue(2),
      TRAINING_ENTITY.clearance, TRAINING_ENTITY.capture, TRAINING_ENTITY.end, TRAINING_ENTITY.groundTrace,
      TRAINING_ENTITY.edge(TRAINING_ENTITY.corridor),
      TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(1), "upper"), TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(1), "lower"),
    ]) {
      expect(scene.entities.getById(id), id).toBeDefined();
    }
    // only the third word has a row outside its band
    expect(scene.entities.getById(TRAINING_ENTITY.headingOutside(0, 0))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.headingOutside(2, 1))).toBeUndefined();
    // the heading words' rows and the capture turn lie on the ground, as the corridor does
    const now = Cesium.JulianDate.now();
    for (const id of [TRAINING_ENTITY.heading(1), TRAINING_ENTITY.headingOutside(2, 0), TRAINING_ENTITY.captureTurn,
                      TRAINING_ENTITY.edge(TRAINING_ENTITY.corridor)]) {
      expect(scene.entities.getById(id)!.polyline!.clampToGround!.getValue(now), id).toBe(true);
    }
    expect(scene.entities.getById(TRAINING_ENTITY.corridor)!.polygon!.height).toBeUndefined();
    // no turn region, turn end, turn path or funnel is drawn any more
    expect(scene.entities.values.map((entity) => entity.id).filter((id) => /turn-end|funnel|-fast$|-slow$/.test(id))).toEqual([]);
  });

  it("draws a heading word's rows in the band colour, its rows outside red, and the capture turn in its verdict colour", async () => {
    const scene = await setup();
    expect(scene.colourOf(TRAINING_ENTITY.heading(2))).toEqual(scene.css(TRAINING_HEADING_BAND_COLOR, 0.85));
    expect(scene.colourOf(TRAINING_ENTITY.headingOutside(2, 0))).toEqual(scene.css(TRAINING_OUTSIDE_COLOR));
    expect(scene.colourOf(TRAINING_ENTITY.captureTurn)).toEqual(scene.css(TRAINING_CAPTURE_TURN_COLOR, 0.85));
    const failed = await setup(0, (raw) => { raw.flights[0].envelopes.approach.captureTurn.check.rateOk = false; });
    expect(failed.colourOf(TRAINING_ENTITY.captureTurn)).toEqual(failed.css(TRAINING_OUTSIDE_COLOR, 0.85));
  });

  it("draws nothing on the ground for a heading word the lead carries to the clearance", async () => {
    const scene = await setup(1);
    expect(scene.entities.getById(TRAINING_ENTITY.heading(0))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.captureTurn)).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.issue(0))).toBeDefined();
  });

  it("carries the labeller's verdict on a tube's edges", async () => {
    const scene = await setup();
    const tubes = scene.selection.flight.envelopes.altitude;
    const outside = tubes.findIndex((tube) => !tube.check.contained);
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(outside), "upper")))
      .toEqual(scene.css(TRAINING_OUTSIDE_COLOR, 0.9));
  });

  it("walls each altitude tube over the aircraft's own ground track, between its exported edges", async () => {
    const scene = await setup();
    const wall = scene.entities.getById(TRAINING_ENTITY.tube(1))!.wall!;
    const tube = scene.selection.flight.envelopes.altitude[1];
    expect(wall.minimumHeights!.getValue(Cesium.JulianDate.now())).toEqual(tube.lowerHaeM);
    expect(wall.maximumHeights!.getValue(Cesium.JulianDate.now())).toEqual(tube.upperHaeM);
    const first = Cesium.Cartographic.fromCartesian(wall.positions!.getValue(Cesium.JulianDate.now())[0]);
    expect(Cesium.Math.toDegrees(first.longitude)).toBeCloseTo(scene.selection.flight.signals.lon[tube.row], 9);
  });

  it("highlights nothing until a word is selected", async () => {
    const scene = await setup();
    expect(scene.entities.getById(TRAINING_ENTITY.focusStretch)).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.focusIssue)).toBeUndefined();
    expect(scene.colourOf(TRAINING_ENTITY.heading(0))).toEqual(scene.css(TRAINING_HEADING_BAND_COLOR, 0.85));
  });

  it("lights up ONLY the selected column's word, lets the rest recede, and restores it all without rebuilding", async () => {
    const scene = await setup();
    const original = [...scene.entities.values];
    fireEvent.click(band(HEADING_180));
    expect(scene.colourOf(TRAINING_ENTITY.heading(2))).toEqual(scene.css(TRAINING_WORD_COLOR));
    // the other heading words recede; the red rows outside stay as they are
    expect(scene.colourOf(TRAINING_ENTITY.heading(0))).toEqual(scene.css(TRAINING_HEADING_BAND_COLOR, 0.85 * 0.3));
    expect(scene.colourOf(TRAINING_ENTITY.headingOutside(2, 0))).toEqual(scene.css(TRAINING_OUTSIDE_COLOR));
    // the altitude word in force at the same step is another column's: never lit, and it recedes
    expect(scene.colourOf(TRAINING_ENTITY.tube(0))).toEqual(scene.css(TRAINING_TUBE_COLOR, 0.28 * 0.3));
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(0), "upper"))).toEqual(scene.css(TRAINING_TUBE_COLOR, 0.9 * 0.3));
    expect(scene.colourOf(TRAINING_ENTITY.corridor)).toEqual(scene.css(TRAINING_CORRIDOR_COLOR, 0.3 * 0.3));
    const label = scene.entities.getById(TRAINING_ENTITY.focusIssue)!.label!.text!.getValue(Cesium.JulianDate.now());
    expect(label).toBe("heading 180° · step 10");
    expect(scene.entities.getById(TRAINING_ENTITY.focusStretch)).toBeDefined();

    fireEvent.click(band(DESCEND_TO_LAND));
    expect(scene.colourOf(TRAINING_ENTITY.tube(1))).toEqual(scene.css(TRAINING_TUBE_COLOR, 0.45));
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.tube(1), "lower"))).toEqual(scene.css(TRAINING_WORD_COLOR));
    expect(scene.colourOf(TRAINING_ENTITY.heading(2))).toEqual(scene.css(TRAINING_HEADING_BAND_COLOR, 0.85 * 0.3));
    // nothing was rebuilt
    original.forEach((entity) => expect(scene.entities.getById(entity.id)).toBe(entity));

    // a second click on the selected word clears it: every envelope back at rest
    fireEvent.click(band(DESCEND_TO_LAND));
    expect(scene.colourOf(TRAINING_ENTITY.tube(1))).toEqual(scene.css(TRAINING_TUBE_COLOR, 0.28));
    expect(scene.colourOf(TRAINING_ENTITY.heading(2))).toEqual(scene.css(TRAINING_HEADING_BAND_COLOR, 0.85));
    expect(scene.entities.getById(TRAINING_ENTITY.focusIssue)).toBeUndefined();
    expect(scene.requestRender).toHaveBeenCalled();
  });

  it("gives the clearance its capture turn and corridor; the dashed capture turn turns yellow, dashed, and comes back", async () => {
    const scene = await setup();
    const line = () => scene.entities.getById(TRAINING_ENTITY.captureTurn)!.polyline!;
    const restColour = scene.colourOf(TRAINING_ENTITY.captureTurn);
    expect(line().material).toBeInstanceOf(Cesium.PolylineDashMaterialProperty);
    fireEvent.click(band(CLEARED));
    expect(line().material).toBeInstanceOf(Cesium.PolylineDashMaterialProperty);
    expect(scene.colourOf(TRAINING_ENTITY.captureTurn)).toEqual(scene.css(TRAINING_WORD_COLOR));
    expect(scene.colourOf(TRAINING_ENTITY.corridor)).toEqual(scene.css(TRAINING_CORRIDOR_COLOR, 0.45));
    expect(scene.colourOf(TRAINING_ENTITY.edge(TRAINING_ENTITY.corridor))).toEqual(scene.css(TRAINING_WORD_COLOR));
    expect(scene.colourOf(TRAINING_ENTITY.corridorAxis)).toEqual(scene.css(TRAINING_WORD_COLOR));
    // the heading words are another column's: they recede
    expect(scene.colourOf(TRAINING_ENTITY.heading(1))).toEqual(scene.css(TRAINING_HEADING_BAND_COLOR, 0.85 * 0.3));
    fireEvent.click(band(CLEARED));
    expect(line().material).toBeInstanceOf(Cesium.PolylineDashMaterialProperty);
    expect(scene.colourOf(TRAINING_ENTITY.captureTurn)).toEqual(restColour);
  });

  it("finds the clearance at step 0 on a straight-in flight", async () => {
    const scene = await setup(1);
    fireEvent.click(band(/^approach cleared — cleared to join the final, issued at step 0/));
    expect(scene.colourOf(TRAINING_ENTITY.corridor)).toEqual(scene.css(TRAINING_CORRIDOR_COLOR, 0.45));
    expect(scene.entities.getById(TRAINING_ENTITY.focusIssue)!.label!.text!.getValue(Cesium.JulianDate.now()))
      .toBe("approach cleared · step 0");
  });

  it("repaints nothing while the cursor moves inside the selected word", async () => {
    const scene = await setup();
    fireEvent.click(band(DESCEND_TO_LAND));
    const marker = scene.entities.getById(TRAINING_ENTITY.focusIssue);
    const renders = scene.requestRender.mock.calls.length;
    act(() => scene.app().setTrainingCursorS(80));
    expect(scene.entities.getById(TRAINING_ENTITY.focusIssue)).toBe(marker);
    expect(scene.requestRender.mock.calls.length).toBe(renders);
  });

  it("frames the selected flight once, not on a switch", async () => {
    const scene = await setup();
    expect(scene.camera.flyToBoundingSphere).toHaveBeenCalledTimes(1);
    act(() => scene.app().setTrainingLayer("corridor", false));
    expect(scene.camera.flyToBoundingSphere).toHaveBeenCalledTimes(1);
  });

  it("draws the executor's rows outside the heading word it was told, red on its ground trace, by the bands' switch", async () => {
    const scene = await setup(0, () => undefined, true);
    expect(scene.entities.getById(TRAINING_ENTITY.executorTrack)).toBeDefined();
    expect(scene.colourOf(TRAINING_ENTITY.executorOutside(0))).toEqual(scene.css(TRAINING_OUTSIDE_COLOR));
    expect(scene.entities.getById(TRAINING_ENTITY.executorOutside(1))).toBeUndefined();
    act(() => scene.app().setTrainingLayer("headingBands", false));
    expect(scene.entities.getById(TRAINING_ENTITY.executorOutside(0))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.executorTrack)).toBeDefined();
    expect(scene.camera.flyToBoundingSphere).toHaveBeenCalledTimes(1);
  });

  it("follows the switches, and leaves nothing behind outside Training", async () => {
    const scene = await setup();
    act(() => scene.app().setTrainingLayer("headingBands", false));
    expect(scene.entities.getById(TRAINING_ENTITY.heading(2))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.headingOutside(2, 0))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.issue(2))).toBeDefined();   // where the words were said stays
    act(() => scene.app().setTrainingLayer("corridor", false));
    expect(scene.entities.getById(TRAINING_ENTITY.corridor)).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.captureTurn)).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.tube(0))).toBeDefined();
    act(() => scene.app().setTrainingLayer("vertical", false));
    expect(scene.entities.getById(TRAINING_ENTITY.tube(0))).toBeUndefined();
    act(() => scene.app().setTrainingLayer("candidates", false));
    expect(scene.entities.getById(TRAINING_ENTITY.runway("27"))).toBeUndefined();
    expect(scene.entities.getById(TRAINING_ENTITY.runway("09"))).toBeDefined();   // the designated one stays
    act(() => scene.app().setMode("observe"));
    expect(scene.entities.values).toHaveLength(0);
    scene.unmount();
  });
});
