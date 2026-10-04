/**
 * useTrainingAutopilot: a PICKED word (the Fly button, or a band clicked) of the flight on screen's closed-loop sentence is
 * flown by the backend when it is picked — once per pick and attempt, never for the cursor — the request names the page and
 * its number, and only the current pick's answer is ever published; an answer for another word is published as a failure.
 * (That a pick is reset with the flight and its Δ is `useTrainingAutopilotScope.test.tsx`'s.)
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

import { nextPick } from "../../data/trainingAutopilot";
import { stageAAnswers, stageASelection } from "../../data/__tests__/stageA";

const { appState, setTrainingAutopilot } = vi.hoisted(() => ({
  appState: { trainingSelection: null as unknown, trainingPick: null as any },
  setTrainingAutopilot: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingAutopilot }),
}));

import useTrainingAutopilot from "../useTrainingAutopilot";

const BACKEND = "http://backend.test";

/** The backend: answers a request for heading row 52 or 126 of the fixture's Δ = 2 s sentence with the Python-written answer. */
function backend(wrong?: (answer: Record<string, any>) => void) {
  return async (_url: string, init: RequestInit) => {
    const request = JSON.parse(init.body as string);
    const answer = stageAAnswers().find((item) => item.segment.row === request.row);
    if (!answer) return { ok: false, status: 400, text: async () => JSON.stringify({ ok: false, error: "no such word" }) };
    wrong?.(answer);
    return { ok: true, status: 200, text: async () => JSON.stringify(answer) };
  };
}

describe("useTrainingAutopilot", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    appState.trainingSelection = stageASelection();
    appState.trainingPick = null;
    setTrainingAutopilot.mockReset();
    fetchMock = vi.fn(backend());
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("asks nothing while no word is picked: the cursor never asks", () => {
    renderHook(() => useTrainingAutopilot(BACKEND));
    expect(fetchMock).not.toHaveBeenCalled();
    expect(setTrainingAutopilot).toHaveBeenLastCalledWith(null);
  });

  it("flies a picked word on the backend and publishes flying, then the answer, on the flight's clock", async () => {
    appState.trainingPick = nextPick(null, 2, "heading", 52);
    renderHook(() => useTrainingAutopilot(BACKEND));
    expect(setTrainingAutopilot).toHaveBeenCalledWith({
      status: "flying",
      request: { airport: "KXXX", setId: "fixture_set", flightKey: "KXXX:test_fixture", rowIntervalS: 2, column: "heading", row: 52 },
    });
    await waitFor(() => expect(setTrainingAutopilot).toHaveBeenLastCalledWith(expect.objectContaining({ status: "ready" })));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BACKEND}/autopilot/segment`);
    const body = JSON.parse(init.body);
    expect(body).toMatchObject({ airport: "KXXX", setId: "fixture_set", column: "heading", row: 52, rowIntervalS: 2 });
    expect(typeof body.clientId).toBe("string");
    expect(Number.isInteger(body.seq)).toBe(true);
    const ready = setTrainingAutopilot.mock.calls[setTrainingAutopilot.mock.calls.length - 1][0];
    expect(ready.segment.segment.end).toBe("segment_end");
  });

  it("flies the same word again on a new attempt, with a higher number", async () => {
    appState.trainingPick = nextPick(null, 2, "heading", 52);
    const { rerender } = renderHook(() => useTrainingAutopilot(BACKEND));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    appState.trainingPick = nextPick(appState.trainingPick, 2, "heading", 52);
    rerender();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const seqs = fetchMock.mock.calls.map(([, init]) => JSON.parse(init.body).seq);
    expect(seqs[1]).toBeGreaterThan(seqs[0]);
  });

  it("publishes the backend's refusal as a failure", async () => {
    appState.trainingPick = nextPick(null, 2, "heading", 999);
    renderHook(() => useTrainingAutopilot(BACKEND));
    await waitFor(() => expect(setTrainingAutopilot).toHaveBeenLastCalledWith(expect.objectContaining({ status: "failed" })));
    const failed = setTrainingAutopilot.mock.calls[setTrainingAutopilot.mock.calls.length - 1][0];
    expect(failed.problem).toBe("the backend refused (400): no such word");
  });

  it("publishes an answer for another word than the one picked as a failure, by name", async () => {
    fetchMock.mockImplementation(backend((answer) => { answer.segment.word += 1; }));
    appState.trainingPick = nextPick(null, 2, "heading", 52);
    renderHook(() => useTrainingAutopilot(BACKEND));
    await waitFor(() => expect(setTrainingAutopilot).toHaveBeenLastCalledWith(expect.objectContaining({ status: "failed" })));
    const failed = setTrainingAutopilot.mock.calls[setTrainingAutopilot.mock.calls.length - 1][0];
    expect(failed.problem).toContain("another sentence of this flight");
  });

  it("publishes nothing for a pick that moved on while its request flew", async () => {
    let release!: () => void;
    fetchMock.mockImplementation((url: string, init: RequestInit) => new Promise((resolve) => {
      release = () => resolve(backend()(url, init));
    }));
    appState.trainingPick = nextPick(null, 2, "heading", 52);
    const { unmount } = renderHook(() => useTrainingAutopilot(BACKEND));
    await act(async () => undefined);
    const published = setTrainingAutopilot.mock.calls.length;
    unmount();
    release();
    await act(async () => undefined);
    // only the unmount's reset follows
    expect(setTrainingAutopilot.mock.calls.slice(published).map(([view]) => view)).toEqual([null]);
  });
});
