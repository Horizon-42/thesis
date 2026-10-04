/**
 * THE PICK BELONGS TO THE FLIGHT ON SCREEN AT ITS Δ (`AppContext`, `trainingSelectionKey`): a word picked on one flight is
 * not flown again when another flight is selected and the first comes back — nor after a switch to another set with the same
 * flight key, nor after another Δ is read, nor after leaving Training (the panel publishes no selection) and returning. The
 * cursor, in flight time, resets with the flight. With the real provider.
 */
import { describe, expect, it, vi } from "vitest";
import { act, render, waitFor } from "@testing-library/react";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({ defaultAirport: "KXXX", airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }] }),
}));

import { AppProvider, useApp, useTrainingCursor } from "../../context/AppContext";
import useTrainingAutopilot from "../useTrainingAutopilot";
import { nextPick } from "../../data/trainingAutopilot";
import { stageAAnswers, stageASample } from "../../data/__tests__/stageA";
import { trainingSelectionOf } from "../../data/trainingSample";

describe("the live executor's pick", () => {
  it("is reset with the flight on screen and with its Δ, so nothing is flown again unasked", async () => {
    const set = stageASample();
    const first = trainingSelectionOf(set, set.flights[0]);
    // another flight of the same set: the same sample, under another key
    const other = trainingSelectionOf(set, { ...set.flights[0], flightKey: "KXXX:other" });
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const request = JSON.parse(init.body as string);
      return { ok: true, status: 200, text: async () => JSON.stringify(stageAAnswers().find((item) => item.segment.row === request.row)) };
    });
    vi.stubGlobal("fetch", fetchMock);
    let app!: ReturnType<typeof useApp>;
    let cursor!: ReturnType<typeof useTrainingCursor>;
    function Harness() {
      app = useApp();
      cursor = useTrainingCursor();
      useTrainingAutopilot("http://backend.test");
      return null;
    }
    render(<AppProvider><Harness /></AppProvider>);

    act(() => app.setTrainingSelection(first));
    act(() => app.setTrainingIntervalS(2));
    act(() => app.setTrainingPick(nextPick(null, 2, "heading", 52)));
    await waitFor(() => expect(app.trainingAutopilot?.status).toBe("ready"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    act(() => cursor.setTrainingCursorS(40));
    expect(cursor.trainingCursorS).toBe(40);
    const lastFlightsSetter = cursor.setTrainingCursorS;

    // another flight, then back: nothing picked, nothing flown, the cursor at the start
    act(() => app.setTrainingSelection(other));
    // the last flight's setter, called late (a chart's handler of the flight before): it writes nothing on this one
    act(() => lastFlightsSetter(55));
    expect(cursor.trainingCursorS).toBe(0);
    act(() => app.setTrainingSelection(first));
    expect(app.trainingPick).toBeNull();
    expect(app.trainingAutopilot).toBeNull();
    expect(cursor.trainingCursorS).toBe(0);

    // another Δ of the same flight: the word's row means another time, so the pick is dropped
    act(() => app.setTrainingPick(nextPick(null, 2, "heading", 52)));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    act(() => app.setTrainingIntervalS(4));
    expect(app.trainingPick).toBeNull();
    act(() => app.setTrainingIntervalS(2));
    expect(app.trainingPick).toBeNull();

    // another set with the same flight key
    act(() => app.setTrainingPick(nextPick(null, 2, "heading", 52)));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    act(() => app.setTrainingSelection({ ...first, setId: "another_set" }));
    expect(app.trainingPick).toBeNull();
    // leaving Training (the panel unmounts and publishes none) and coming back
    act(() => app.setTrainingSelection(null));
    act(() => app.setTrainingSelection(first));
    await act(async () => undefined);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(app.trainingPick).toBeNull();
    vi.unstubAllGlobals();
  });
});
