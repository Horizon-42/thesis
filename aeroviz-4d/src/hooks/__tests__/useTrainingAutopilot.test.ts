/**
 * useTrainingAutopilot: a PICKED word (a click on a band) is flown by the backend when it is picked — once per pick,
 * never for the cursor, again on "fly again" — and only the current pick's answer is ever published.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

import { parseTrainingSample, type TrainingSample } from "../../data/trainingSample";
import { STRAIGHT_KEY, VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest } from "../../data/__tests__/trainingAutopilot.fixture";

const { appState, setTrainingAutopilot } = vi.hoisted(() => ({
  appState: { trainingSelection: null as unknown, trainingPick: null as unknown, trainingCursorS: 0 },
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
    const flight = set.flights.find((item) => item.flightKey === VECTORED_KEY)!;
    appState.trainingSelection = { vocabulary: set.vocabulary, candidates: set.candidates, flight };
    appState.trainingPick = null;
    appState.trainingCursorS = 0;
    setTrainingAutopilot.mockClear();
    fetchMock = vi.fn(answering(set));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  const last = () => setTrainingAutopilot.mock.calls[setTrainingAutopilot.mock.calls.length - 1][0];

  it("asks for nothing until a word is picked, whatever the cursor does", () => {
    const { rerender } = renderHook(() => useTrainingAutopilot(true, "KXXX", set, BACKEND));
    appState.trainingCursorS = 22;
    rerender();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(last()).toBeNull();
  });

  it("flies the picked word's segment and publishes the answer", async () => {
    appState.trainingPick = { flightKey: VECTORED_KEY, column: "heading", row: 8 };
    renderHook(() => useTrainingAutopilot(true, "KXXX", set, BACKEND));
    expect(last()).toMatchObject({ status: "flying", request: mockAutopilotRequest(set, VECTORED_KEY, "heading", 8) });
    await waitFor(() => expect(last().status).toBe("ready"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(last().segment.segment).toMatchObject({ row: 8, endRow: 10, stopRow: 12 });
    // the browser's own wait, beside the backend's timing
    expect(last().roundTripS).toBeGreaterThanOrEqual(0);
  });

  it("asks again for another pick and on fly again, never for the cursor moving", async () => {
    appState.trainingPick = { flightKey: VECTORED_KEY, column: "heading", row: 8 };
    const { rerender, result } = renderHook(() => useTrainingAutopilot(true, "KXXX", set, BACKEND));
    await waitFor(() => expect(last().status).toBe("ready"));
    appState.trainingCursorS = 40;                         // hovering a chart
    rerender();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    appState.trainingPick = { flightKey: VECTORED_KEY, column: "heading", row: 10 };
    rerender();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    act(() => result.current.flyAgain());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(last().status).toBe("ready"));
    expect(last().request.row).toBe(10);
  });

  it("never publishes the answer to a pick that has moved on", async () => {
    let answerFirst: () => void = () => undefined;
    fetchMock.mockImplementationOnce((url: string, init: RequestInit) => new Promise((resolve) => {
      answerFirst = () => resolve(answering(set)(url, init));
    }));
    appState.trainingPick = { flightKey: VECTORED_KEY, column: "heading", row: 8 };
    const { rerender } = renderHook(() => useTrainingAutopilot(true, "KXXX", set, BACKEND));
    appState.trainingPick = { flightKey: VECTORED_KEY, column: "heading", row: 10 };
    rerender();
    await waitFor(() => expect(last().status).toBe("ready"));
    expect(last().request.row).toBe(10);
    await act(async () => answerFirst());
    expect(setTrainingAutopilot.mock.calls.filter(([view]) => view?.status === "ready" && view.request.row === 8)).toHaveLength(0);
  });

  it("does not fly a pick of another flight", () => {
    appState.trainingPick = { flightKey: STRAIGHT_KEY, column: "heading", row: 0 };
    renderHook(() => useTrainingAutopilot(true, "KXXX", set, BACKEND));
    expect(fetchMock).not.toHaveBeenCalled();
    expect(last()).toBeNull();
  });

  it("publishes the refusal when the backend refuses, or the answer is not the segment on screen", async () => {
    appState.trainingPick = { flightKey: VECTORED_KEY, column: "heading", row: 8 };
    fetchMock.mockImplementationOnce(async () => ({ ok: false, status: 400, text: async () => JSON.stringify({ ok: false, error: "no spec" }) }));
    const { result } = renderHook(() => useTrainingAutopilot(true, "KXXX", set, BACKEND));
    await waitFor(() => expect(last().status).toBe("failed"));
    expect(last().problem).toMatch(/no spec/);
    fetchMock.mockImplementationOnce(async (_url: string, init: RequestInit) => {
      const answer = mockAutopilotAnswer(set, JSON.parse(init.body as string));
      answer.segment.endRow = 11;
      return { ok: true, status: 200, text: async () => JSON.stringify(answer) };
    });
    act(() => result.current.flyAgain());
    await waitFor(() => expect(last().status).toBe("failed"));
    expect(last().problem).toMatch(/ends at step 11/);
  });

  it("asks nothing with its switch off", () => {
    appState.trainingPick = { flightKey: VECTORED_KEY, column: "heading", row: 8 };
    renderHook(() => useTrainingAutopilot(false, "KXXX", set, BACKEND));
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
