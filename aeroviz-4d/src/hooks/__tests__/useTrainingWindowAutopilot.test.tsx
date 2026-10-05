/**
 * A pick on a window of a window set is flown ONCE, by the window's hook (`useTrainingWindowAutopilot`, `POST
 * /autopilot/window-segment`), with stage A's hook mounted beside it as the panel mounts it: stage A's hook leaves a
 * window's derived flight alone, so no second request races the first. With the real provider.
 */
import { describe, expect, it, vi } from "vitest";
import { act, render, waitFor } from "@testing-library/react";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({ defaultAirport: "KXXX", airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }] }),
}));

import { AppProvider, useApp } from "../../context/AppContext";
import useTrainingAutopilot from "../useTrainingAutopilot";
import useTrainingWindowAutopilot from "../useTrainingWindowAutopilot";
import { nextPick } from "../../data/trainingAutopilot";
import { TRAINING_WINDOW_AUTOPILOT_PATH } from "../../data/trainingWindowAutopilot";
import { stageCAnswers, stageCSample, stageCSelection } from "../../data/__tests__/stageC";

describe("a pick on a window", () => {
  it("is flown once, by the window's hook, with stage A's hook mounted beside it", async () => {
    const [answer] = stageCAnswers();
    const fetchMock = vi.fn(async (_url: string, _init: RequestInit) => ({ ok: true, status: 200, text: async () => JSON.stringify(answer) }));
    vi.stubGlobal("fetch", fetchMock);
    let app!: ReturnType<typeof useApp>;
    function Harness() {
      app = useApp();
      useTrainingAutopilot("http://backend.test");
      useTrainingWindowAutopilot("http://backend.test");
      return null;
    }
    render(<AppProvider><Harness /></AppProvider>);
    const selection = stageCSelection(stageCSample(), 0);
    act(() => app.setTrainingSelection(selection));
    act(() => app.setTrainingIntervalS(4));
    const column = ["runway", "heading", "altitude", "angle", "speed"][answer.segment.column] as Parameters<typeof nextPick>[2];
    act(() => app.setTrainingPick(nextPick(null, 4, column, answer.segment.row)));
    await waitFor(() => expect(app.trainingAutopilot?.status).toBe("ready"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(`http://backend.test${TRAINING_WINDOW_AUTOPILOT_PATH}`);
    vi.unstubAllGlobals();
  });
});
