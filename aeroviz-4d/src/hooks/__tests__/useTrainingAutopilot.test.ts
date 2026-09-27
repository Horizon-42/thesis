/**
 * useTrainingAutopilot: a PICKED word (the Fly button, or a band clicked with the switch on) of the flight on screen is
 * flown by the backend when it is picked — once per pick and attempt, never for the cursor — and only the current pick's
 * answer is ever published — a model's word asked with its sample's sentence, or not at all when that sentence is not
 * published. (That a pick is reset with the flight is `useTrainingAutopilotScope.test.tsx`'s.)
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

import { parseTrainingSample, trainingSelectionOf, type TrainingSample } from "../../data/trainingSample";
import { nextPick } from "../../data/trainingAutopilot";
import { VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest } from "../../data/__tests__/trainingAutopilot.fixture";
import { AUGSTART_BASE_ID, BASE_MODEL_ID, MOCK_MOVE, mockGenerationViews } from "../../data/__tests__/trainingOverlays.fixture";

const { appState, cursor, setTrainingAutopilot } = vi.hoisted(() => ({
  appState: { trainingSelection: null as unknown, trainingPick: null as any, trainingGenerations: [] as unknown[] },
  cursor: { trainingCursorS: 0 },
  setTrainingAutopilot: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingAutopilot }),
  useTrainingCursor: () => ({ trainingCursorS: cursor.trainingCursorS, setTrainingCursorS: () => undefined }),
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
    appState.trainingGenerations = [];
    cursor.trainingCursorS = 0;
    setTrainingAutopilot.mockClear();
    fetchMock = vi.fn(answering(set));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  const last = () => setTrainingAutopilot.mock.calls[setTrainingAutopilot.mock.calls.length - 1][0];

  it("asks for nothing until a word is picked, whatever the cursor does", () => {
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    cursor.trainingCursorS = 22;
    rerender();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(last()).toBeNull();
  });

  it("flies the picked word's segment and publishes the answer", async () => {
    appState.trainingPick = nextPick(null, null, "heading", 8);
    renderHook(() => useTrainingAutopilot(BACKEND));
    expect(last()).toMatchObject({ status: "flying", request: mockAutopilotRequest(set, VECTORED_KEY, "heading", 8) });
    await waitFor(() => expect(last().status).toBe("ready"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(last().segment.segment).toMatchObject({ row: 8, endRow: 10, stopRow: 12 });
    // the browser's own wait, beside the backend's timing
    expect(last().roundTripS).toBeGreaterThanOrEqual(0);
  });

  it("asks again for another pick and for a new attempt at the same one, never for the cursor moving", async () => {
    appState.trainingPick = nextPick(null, null, "heading", 8);
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    await waitFor(() => expect(last().status).toBe("ready"));
    cursor.trainingCursorS = 40;                         // hovering a chart
    rerender();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    appState.trainingPick = nextPick(appState.trainingPick, null, "heading", 10);
    rerender();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    appState.trainingPick = nextPick(appState.trainingPick, null, "heading", 10);   // "Fly again"
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
    appState.trainingPick = nextPick(null, null, "heading", 8);
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    appState.trainingPick = nextPick(appState.trainingPick, null, "heading", 10);
    rerender();
    await waitFor(() => expect(last().status).toBe("ready"));
    expect(last().request.row).toBe(10);
    await act(async () => answerFirst());
    expect(setTrainingAutopilot.mock.calls.filter(([view]) => view?.status === "ready" && view.request.row === 8)).toHaveLength(0);
  });

  it("publishes the refusal when the backend refuses, or the answer is not the segment on screen", async () => {
    appState.trainingPick = nextPick(null, null, "heading", 8);
    fetchMock.mockImplementationOnce(async () => ({ ok: false, status: 400, text: async () => JSON.stringify({ ok: false, error: "no spec" }) }));
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    await waitFor(() => expect(last().status).toBe("failed"));
    expect(last().problem).toMatch(/no spec/);
    fetchMock.mockImplementationOnce(async (_url: string, init: RequestInit) => {
      const answer = mockAutopilotAnswer(set, JSON.parse(init.body as string));
      answer.segment.endRow = 11;
      return { ok: true, status: 200, text: async () => JSON.stringify(answer) };
    });
    appState.trainingPick = nextPick(appState.trainingPick, null, "heading", 8);
    rerender();
    await waitFor(() => expect(last().problem ?? "").toMatch(/ends at step 11/));
  });

  it("asks for a model's word with its sample's sentence — and for nothing while that sentence is not published", async () => {
    appState.trainingPick = nextPick(null, { overlayId: BASE_MODEL_ID, sample: 1 }, "heading", 12);
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    expect(fetchMock).not.toHaveBeenCalled();
    appState.trainingGenerations = mockGenerationViews(set);
    rerender();
    await waitFor(() => expect(last()?.status).toBe("ready"));
    const asked = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(asked.sentence).toMatchObject({ overlayId: BASE_MODEL_ID, sample: 1, firstRow: 4, rows: 76, augmentation: null });
    expect(last().segment.source).toMatchObject({ kind: "model", sample: 1 });
  });

  it("asks for a word of a sample from an augmented start with the move its overlay's flight was flown from", async () => {
    appState.trainingGenerations = mockGenerationViews(set, 0, () => undefined, [AUGSTART_BASE_ID]);
    appState.trainingPick = nextPick(null, { overlayId: AUGSTART_BASE_ID, sample: 0 }, "heading", 12);
    renderHook(() => useTrainingAutopilot(BACKEND));
    await waitFor(() => expect(last()?.status).toBe("ready"));
    const asked = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(asked.sentence).toMatchObject({ overlayId: AUGSTART_BASE_ID, augmentation: MOCK_MOVE });
    expect(last().segment.source).toMatchObject({ kind: "model", augmentation: MOCK_MOVE });
  });
});

describe("nextPick", () => {
  it("is a new attempt at the word picked already, and a first attempt at any other", () => {
    const first = nextPick(null, null, "heading", 8);
    expect(first).toEqual({ source: null, column: "heading", row: 8, attempt: 0 });
    expect(nextPick(first, null, "heading", 8).attempt).toBe(1);
    expect(nextPick(first, null, "heading", 10).attempt).toBe(0);
    expect(nextPick(first, null, "speed", 8).attempt).toBe(0);
    // the same step and column of another sentence is another word
    const model = nextPick(first, { overlayId: BASE_MODEL_ID, sample: 0 }, "heading", 8);
    expect(model).toEqual({ source: { overlayId: BASE_MODEL_ID, sample: 0 }, column: "heading", row: 8, attempt: 0 });
    expect(nextPick(model, { overlayId: BASE_MODEL_ID, sample: 0 }, "heading", 8).attempt).toBe(1);
    expect(nextPick(model, { overlayId: BASE_MODEL_ID, sample: 1 }, "heading", 8).attempt).toBe(0);
  });
});
