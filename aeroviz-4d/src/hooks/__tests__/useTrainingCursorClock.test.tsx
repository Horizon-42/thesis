/**
 * THE CURSOR IS ONE TIME ON THE SELECTION'S CLOCK (`AppContext`, `TrainingSelection.clock`): a multi-aircraft window's
 * aircraft are read on the window's clock, so putting another of them on screen keeps the moment — its own time is the
 * window's less its row 0 — while a flight of a read-back set is read on its own, and a new clock starts at the flight's
 * own 0 s. With the real provider.
 */
import { describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({ defaultAirport: "KXXX", airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }] }),
}));

import { AppProvider, useApp, useTrainingCursor } from "../../context/AppContext";
import { parseTrainingSample, trainingSelectionOf } from "../../data/trainingSample";
import { parseTrainingTrafficSet, trainingWindowSelection } from "../../data/trainingTraffic";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockTrafficSet } from "../../data/__tests__/trainingTraffic.fixture";

describe("the Training cursor", () => {
  it("keeps a window's moment across its aircraft, and starts a new clock at the flight's own 0 s", () => {
    const traffic = parseTrainingTrafficSet(mockTrafficSet());
    const sample = parseTrainingSample(mockSample());
    if (!traffic.ok || !sample.ok) throw new Error("the fixtures should parse");
    const [window] = traffic.value.windows;
    const [vectored, straight] = window.commanded.map((one) => trainingWindowSelection(traffic.value, window, one));
    let app!: ReturnType<typeof useApp>;
    let cursor!: ReturnType<typeof useTrainingCursor>;
    function Harness() {
      app = useApp();
      cursor = useTrainingCursor();
      return null;
    }
    render(<AppProvider><Harness /></AppProvider>);

    // a window's first aircraft: its own 0 s is the window's 100 s
    act(() => app.setTrainingSelection(vectored));
    expect([cursor.trainingCursorS, cursor.trainingSceneS]).toEqual([0, 100]);
    act(() => cursor.setTrainingCursorS(20));
    expect([cursor.trainingCursorS, cursor.trainingSceneS]).toEqual([20, 120]);
    // another aircraft of the window, at the same moment: 40 s before its own row 0
    act(() => app.setTrainingSelection(straight));
    expect([cursor.trainingCursorS, cursor.trainingSceneS]).toEqual([-40, 120]);
    act(() => cursor.setTrainingSceneS(200));
    expect(cursor.trainingCursorS).toBe(40);
    // a read-back set's flight: its own clock, from its 0 s
    act(() => app.setTrainingSelection(trainingSelectionOf(sample.value, sample.value.flights[0])));
    expect([cursor.trainingCursorS, cursor.trainingSceneS]).toEqual([0, 0]);
    // back to the window: a new clock again, at the aircraft's own 0 s
    act(() => app.setTrainingSelection(straight));
    expect([cursor.trainingCursorS, cursor.trainingSceneS]).toEqual([0, 160]);
  });
});
