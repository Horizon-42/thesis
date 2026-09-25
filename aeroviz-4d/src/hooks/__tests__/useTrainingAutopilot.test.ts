/**
 * useTrainingAutopilot: a PICKED word (the Fly button, or a band clicked with the switch on) of the flight on screen is
 * flown by the backend when it is picked — once per pick and attempt, never for the cursor — and only the current pick's
 * answer is ever published. (That a pick is reset with the flight is `useTrainingAutopilotScope.test.tsx`'s.)
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

import { parseTrainingSample, trainingSelectionOf, type TrainingSample } from "../../data/trainingSample";
import { nextPick } from "../../data/trainingAutopilot";
import { VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest } from "../../data/__tests__/trainingAutopilot.fixture";

const { appState, setTrainingAutopilot } = vi.hoisted(() => ({
  appState: { trainingSelection: null as unknown, trainingPick: null as any, trainingCursorS: 0 },
  setTrainingAutopilot: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingAutopilot }),
}));

import useTrainingAutopilot from "../useTrainingAutopilot";

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

const BACKEND = "http://backend.test";

function answering(set: TrainingSample) {
  return async (_url: string, init: RequestInit) => {
    const request = JSON.parse(init.body as string);
    return { ok: true, status: 200, text: async () => JSON.stringify(mockAutopilotAnswer(set, request)) };
  };
}

describe("useTrainingAutopilot", () => {
  let set: TrainingSample;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    set = sample();
    appState.trainingSelection = trainingSelectionOf(set, set.flights.find((item) => item.flightKey === VECTORED_KEY)!);
    appState.trainingPick = null;
    appState.trainingCursorS = 0;
    setTrainingAutopilot.mockClear();
    fetchMock = vi.fn(answering(set));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  const last = () => setTrainingAutopilot.mock.calls[setTrainingAutopilot.mock.calls.length - 1][0];

  it("asks for nothing until a word is picked, whatever the cursor does", () => {
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    appState.trainingCursorS = 22;
    rerender();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(last()).toBeNull();
  });

  it("flies the picked word's segment and publishes the answer", async () => {
    appState.trainingPick = nextPick(null, "heading", 8);
    renderHook(() => useTrainingAutopilot(BACKEND));
    expect(last()).toMatchObject({ status: "flying", request: mockAutopilotRequest(set, VECTORED_KEY, "heading", 8) });
    await waitFor(() => expect(last().status).toBe("ready"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(last().segment.segment).toMatchObject({ row: 8, endRow: 10, stopRow: 12 });
    // the browser's own wait, beside the backend's timing
    expect(last().roundTripS).toBeGreaterThanOrEqual(0);
  });

  it("asks again for another pick and for a new attempt at the same one, never for the cursor moving", async () => {
    appState.trainingPick = nextPick(null, "heading", 8);
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    await waitFor(() => expect(last().status).toBe("ready"));
    appState.trainingCursorS = 40;                         // hovering a chart
    rerender();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    appState.trainingPick = nextPick(appState.trainingPick, "heading", 10);
    rerender();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    appState.trainingPick = nextPick(appState.trainingPick, "heading", 10);   // "Fly again"
    expect(appState.trainingPick.attempt).toBe(1);
    rerender();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(last().status).toBe("ready"));
    expect(last().request.row).toBe(10);
  });

  it("never publishes the answer to a pick that has moved on", async () => {
    let answerFirst: () => void = () => undefined;
    fetchMock.mockImplementationOnce((url: string, init: RequestInit) => new Promise((resolve) => {
      answerFirst = () => resolve(answering(set)(url, init));
    }));
    appState.trainingPick = nextPick(null, "heading", 8);
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    appState.trainingPick = nextPick(appState.trainingPick, "heading", 10);
    rerender();
    await waitFor(() => expect(last().status).toBe("ready"));
    expect(last().request.row).toBe(10);
    await act(async () => answerFirst());
    expect(setTrainingAutopilot.mock.calls.filter(([view]) => view?.status === "ready" && view.request.row === 8)).toHaveLength(0);
  });

  it("publishes the refusal when the backend refuses, or the answer is not the segment on screen", async () => {
    appState.trainingPick = nextPick(null, "heading", 8);
    fetchMock.mockImplementationOnce(async () => ({ ok: false, status: 400, text: async () => JSON.stringify({ ok: false, error: "no spec" }) }));
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    await waitFor(() => expect(last().status).toBe("failed"));
    expect(last().problem).toMatch(/no spec/);
    fetchMock.mockImplementationOnce(async (_url: string, init: RequestInit) => {
      const answer = mockAutopilotAnswer(set, JSON.parse(init.body as string));
      answer.segment.endRow = 11;
      return { ok: true, status: 200, text: async () => JSON.stringify(answer) };
    });
    appState.trainingPick = nextPick(appState.trainingPick, "heading", 8);
    rerender();
    await waitFor(() => expect(last().problem ?? "").toMatch(/ends at step 11/));
  });
});

describe("nextPick", () => {
  it("is a new attempt at the word picked already, and a first attempt at any other", () => {
    const first = nextPick(null, "heading", 8);
    expect(first).toEqual({ column: "heading", row: 8, attempt: 0 });
    expect(nextPick(first, "heading", 8).attempt).toBe(1);
    expect(nextPick(first, "heading", 10).attempt).toBe(0);
    expect(nextPick(first, "speed", 8).attempt).toBe(0);
  });
});
