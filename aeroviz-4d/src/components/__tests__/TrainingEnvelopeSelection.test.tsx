/**
 * The Training flight in the 3D scene, on real Cesium entities (no WebGL canvas): the envelopes
 * the switches ask for are built once, the ones in force at the shared cursor are repainted
 * yellow and restored without rebuilding anything, and leaving Training removes the layer.
 */
import { useLayoutEffect } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import * as Cesium from "cesium";
import { AppProvider, useApp } from "../../context/AppContext";
import { parseTrainingSample } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import useTrainingTrackLayer, { TRAINING_ENTITY } from "../../hooks/useTrainingTrackLayer";
import { TRAINING_WORD_COLOR } from "../../utils/trainingWordColors";
import TrainingSentenceBar from "../TrainingSentenceBar";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({
    defaultAirport: "KXXX",
    airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }],
  }),
}));

async function setup() {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  const selection = { vocabulary: parsed.value.vocabulary, candidates: parsed.value.candidates, flight: parsed.value.flights[0] };
  const entities = new Cesium.EntityCollection();
  const requestRender = vi.fn();
  const viewer = { entities, scene: { requestRender }, isDestroyed: () => false } as unknown as Cesium.Viewer;
  let app: ReturnType<typeof useApp>;
  function Scene() {
    app = useApp();
    useTrainingTrackLayer();
    useLayoutEffect(() => {
      app.setViewer(viewer);
      app.setMode("training");
      app.setTrainingSelection(selection);
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
  const selected = Cesium.Color.fromCssColorString(TRAINING_WORD_COLOR).withAlpha(0.45);
  return { ...view, entities, selection, requestRender, colourOf, selected, app: () => app! };
}

describe("Training envelopes in the 3D scene", () => {
  it("builds the track, the lateral envelopes, the tubes, every candidate and the issue points", async () => {
    const scene = await setup();
    for (const id of [
      TRAINING_ENTITY.track, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis, TRAINING_ENTITY.captureTurn,
      TRAINING_ENTITY.turn(1), TRAINING_ENTITY.funnel(0), TRAINING_ENTITY.funnel(1), TRAINING_ENTITY.tube(0), TRAINING_ENTITY.tube(1),
      TRAINING_ENTITY.centreline("09"), TRAINING_ENTITY.runway("27"), TRAINING_ENTITY.issue(0), TRAINING_ENTITY.issue(1),
      TRAINING_ENTITY.clearance, TRAINING_ENTITY.capture, TRAINING_ENTITY.end,
    ]) {
      expect(scene.entities.getById(id), id).toBeDefined();
    }
    // a word the flight was already holding at entry has no turn region
    expect(scene.entities.getById(TRAINING_ENTITY.turn(0))).toBeUndefined();
    // the lateral envelopes are draped on the ground: they give no height
    expect(scene.entities.getById(TRAINING_ENTITY.funnel(1))!.polygon!.height).toBeUndefined();
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

  it("paints the envelopes in force at the cursor yellow and restores them, without rebuilding", async () => {
    const scene = await setup();
    const original = [...scene.entities.values];
    expect(scene.colourOf(TRAINING_ENTITY.funnel(0))).toEqual(scene.selected);
    expect(scene.colourOf(TRAINING_ENTITY.tube(0))).toEqual(scene.selected);
    fireEvent.click(screen.getByLabelText(/^heading 180° — a turn/));
    expect(scene.colourOf(TRAINING_ENTITY.turn(1))).toEqual(scene.selected);
    expect(scene.colourOf(TRAINING_ENTITY.funnel(1))).toEqual(scene.selected);
    expect(scene.colourOf(TRAINING_ENTITY.funnel(0))).not.toEqual(scene.selected);
    // after the capture the corridor is what holds the flight laterally
    act(() => scene.app().setTrainingCursorS(60));
    expect(scene.colourOf(TRAINING_ENTITY.corridor)).toEqual(scene.selected);
    expect(scene.colourOf(TRAINING_ENTITY.tube(1))).toEqual(scene.selected);
    expect(scene.colourOf(TRAINING_ENTITY.funnel(1))).not.toEqual(scene.selected);
    original.forEach((entity) => expect(scene.entities.getById(entity.id)).toBe(entity));
    expect(scene.requestRender).toHaveBeenCalled();
  });

  it("follows the switches, and leaves nothing behind outside Training", async () => {
    const scene = await setup();
    act(() => scene.app().setTrainingLayer("lateral", false));
    expect(scene.entities.getById(TRAINING_ENTITY.corridor)).toBeUndefined();
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
