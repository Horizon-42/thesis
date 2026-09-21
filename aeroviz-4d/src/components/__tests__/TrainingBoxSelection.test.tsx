import { useLayoutEffect } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import * as Cesium from "cesium";
import { AppProvider, useApp } from "../../context/AppContext";
import { parseTrainingSample } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import useTrainingTrackLayer from "../../hooks/useTrainingTrackLayer";
import { TRAINING_FLOWN_COLOR, TRAINING_TARGET_COLOR, TRAINING_WORD_COLOR } from "../../utils/trainingWordColors";
import TrainingSentenceBar from "../TrainingSentenceBar";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({
    defaultAirport: "KRDU",
    airports: [{ code: "KRDU", name: "Raleigh-Durham", lat: 35.878659, lon: -78.7873 }],
  }),
}));

async function setup() {
  const parsed = parseTrainingSample(mockSample(), "vocabulary-readback");
  if (!parsed.ok) throw new Error(parsed.problem);
  const selection = {
    vocabulary: parsed.value.vocabulary,
    reading: parsed.value.reading,
    flight: parsed.value.flights[0],
  };
  // Real Cesium entities/materials, without a WebGL canvas.
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
  const colour = (index: number, part: "wall" | "lid" | "target") => {
    const entity = entities.getById(`training-envelope-box-${part === "wall" ? "" : `${part}-`}${index}`)!;
    return (part === "wall" ? entity.wall : entity.polygon)!.material!.getValue(Cesium.JulianDate.now()).color;
  };
  const expectSelected = (index: number) => {
    const selected = Cesium.Color.fromCssColorString(TRAINING_WORD_COLOR);
    expect(colour(index, "wall")).toEqual(selected.withAlpha(0.4));
    expect(colour(index, "lid")).toEqual(selected.withAlpha(0.3));
    expect(colour(index, "target")).toEqual(Cesium.Color.fromCssColorString(TRAINING_TARGET_COLOR).withAlpha(0.6));
    expect(entities.getById(`training-segment-node-${index}`)!.point!.pixelSize!.getValue()).toBe(14);
  };
  return { ...view, entities, selection, requestRender, colour, expectSelected, app: () => app! };
}

describe("Training sentence selection in the 3D scene", () => {
  it("highlights a clicked event's sides, lid and node, restoring the previous box without rebuilding", async () => {
    const scene = await setup();
    scene.expectSelected(0);
    const original = [...scene.entities.values];
    fireEvent.click(screen.getByLabelText("Event 4 at 46 s"));
    scene.expectSelected(3);
    const base = Cesium.Color.fromCssColorString(TRAINING_FLOWN_COLOR);
    expect(scene.colour(0, "wall")).toEqual(base.withAlpha(0.1));
    expect(scene.colour(0, "lid")).toEqual(base.withAlpha(0.07));
    expect(scene.colour(0, "target")).toEqual(Cesium.Color.fromCssColorString(TRAINING_TARGET_COLOR).withAlpha(0.22));
    expect(scene.entities.getById("training-segment-node-0")!.point!.pixelSize!.getValue()).toBe(9);
    original.forEach((entity) => expect(scene.entities.getById(entity.id)).toBe(entity));
    expect(screen.getByText("t = 46 s")).toBeTruthy();
    expect(scene.requestRender).toHaveBeenCalled();
  });

  it("links word-band clicks and keyboard selection to the box in force, including the final endpoint", async () => {
    const scene = await setup();
    fireEvent.click(screen.getByLabelText(/^duration 24 s \(word 12\), 60–84 s$/));
    scene.expectSelected(4);
    fireEvent.keyDown(screen.getByLabelText("Event 6 at 84 s"), { key: "Enter" });
    scene.expectSelected(5);
    act(() => scene.app().setTrainingCursorS(103));
    scene.expectSelected(5);
    act(() => scene.app().setTrainingCursorS(104));
    scene.expectSelected(6);
    act(() => scene.app().setTrainingCursorS(scene.selection.flight.durationS));
    scene.expectSelected(6);
  });

  it("resets on flight changes, reapplies after toggling boxes, and removes the layer on exit", async () => {
    const scene = await setup();
    fireEvent.click(screen.getByLabelText("Event 4 at 46 s"));
    act(() => scene.app().setTrainingLayer("flown", false));
    expect(scene.entities.getById("training-envelope-box-3")).toBeUndefined();
    expect(scene.entities.getById("training-envelope-box-target-3")).toBeUndefined();
    act(() => scene.app().setTrainingLayer("flown", true));
    scene.expectSelected(3);
    act(() => scene.app().setTrainingSelection({
      ...scene.selection,
      flight: { ...scene.selection.flight, flightKey: "another-flight" },
    }));
    scene.expectSelected(0);
    expect(screen.getByText("t = 0 s")).toBeTruthy();
    act(() => scene.app().setMode("observe"));
    expect(scene.entities.values).toHaveLength(0);
    scene.unmount();
  });

  it("places every target plane at the target's absolute altitude within the box footprint", async () => {
    const scene = await setup();
    const { flight } = scene.selection;
    // The fixture's observed track supplies the independent threshold datum.
    const thresholdHae = flight.observed.altHaeM[0] - flight.observed.heightM[0];
    flight.envelope.events.forEach((box, index) => {
      const plane = scene.entities.getById(`training-envelope-box-target-${index}`)!.polygon!;
      const height = plane.height!.getValue();
      expect(height).toBeCloseTo(thresholdHae + box.altitudeTargetM, 6);
      const positions: Cesium.Cartesian3[] = plane.hierarchy!.getValue().positions;
      expect(positions).toHaveLength(box.lon.length);
      positions.forEach((position, point) => {
        const coordinate = Cesium.Cartographic.fromCartesian(position);
        expect(Cesium.Math.toDegrees(coordinate.longitude)).toBeCloseTo(box.lon[point], 6);
        expect(Cesium.Math.toDegrees(coordinate.latitude)).toBeCloseTo(box.lat[point], 6);
        expect(height).toBeGreaterThanOrEqual(box.altHaeLoM[point]);
        expect(height).toBeLessThanOrEqual(box.altHaeHiM[point]);
      });
    });
  });
});
