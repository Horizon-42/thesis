/**
 * THE PICK BELONGS TO THE FLIGHT ON SCREEN (`AppContext`, `trainingSelectionKey`): a word picked on one flight is not
 * flown again when another flight is selected and the first comes back — nor after a switch to another set with the
 * same flight key, nor after leaving Training (the panel publishes no selection) and returning. The cursor resets with
 * it. With the real provider: this is the bug a pick kept across flights made (it flew again, unasked).
 */
import { describe, expect, it, vi } from "vitest";
import { act, render, waitFor } from "@testing-library/react";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({ defaultAirport: "KXXX", airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }] }),
}));

import { AppProvider, useApp } from "../../context/AppContext";
import useTrainingAutopilot from "../useTrainingAutopilot";
import { nextPick } from "../../data/trainingAutopilot";
import { parseTrainingSample, trainingSelectionOf } from "../../data/trainingSample";
import { STRAIGHT_KEY, VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer } from "../../data/__tests__/trainingAutopilot.fixture";

describe("the live executor's pick", () => {
  it("is reset with the flight on screen, so nothing is flown again unasked", async () => {
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const set = parsed.value;
    const vectored = trainingSelectionOf(set, set.flights.find((item) => item.flightKey === VECTORED_KEY)!);
    const straight = trainingSelectionOf(set, set.flights.find((item) => item.flightKey === STRAIGHT_KEY)!);
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => ({
      ok: true, status: 200, text: async () => JSON.stringify(mockAutopilotAnswer(set, JSON.parse(init.body as string))),
    }));
    vi.stubGlobal("fetch", fetchMock);
    let app!: ReturnType<typeof useApp>;
    function Harness() {
      app = useApp();
      useTrainingAutopilot("http://backend.test");
      return null;
    }
    render(<AppProvider><Harness /></AppProvider>);

    act(() => app.setTrainingSelection(vectored));
    act(() => app.setTrainingPick(nextPick(null, "heading", 8)));
    await waitFor(() => expect(app.trainingAutopilot?.status).toBe("ready"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    act(() => app.setTrainingCursorS(40));

    // another flight, then back: nothing picked, nothing flown, the cursor at the start
    act(() => app.setTrainingSelection(straight));
    act(() => app.setTrainingSelection(vectored));
    expect(app.trainingPick).toBeNull();
    expect(app.trainingAutopilot).toBeNull();
    expect(app.trainingCursorS).toBe(0);
    // another set with the same flight key
    act(() => app.setTrainingPick(nextPick(null, "heading", 8)));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    act(() => app.setTrainingSelection({ ...vectored, setId: "another_set" }));
    expect(app.trainingPick).toBeNull();
    // leaving Training (the panel unmounts and publishes none) and coming back
    act(() => app.setTrainingSelection(null));
    act(() => app.setTrainingSelection(vectored));
    await act(async () => undefined);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(app.trainingPick).toBeNull();
    vi.unstubAllGlobals();
  });
});
