/**
 * useTrainingOverlays: the overlays manifest of the open airport, and each kind's overlay over the open set — downloaded
 * once per overlay and sample however often its switch goes off and on, never under another airport's path, published
 * for the selected flight only while shown, and unpublished when the panel goes.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

const { setTrainingExecutor, setTrainingPrior } = vi.hoisted(() => ({
  setTrainingExecutor: vi.fn(),
  setTrainingPrior: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ setTrainingExecutor, setTrainingPrior }),
}));

import useTrainingOverlays from "../useTrainingOverlays";
import { parseTrainingSample, type TrainingSample } from "../../data/trainingSample";
import { STRAIGHT_KEY, VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { EXECUTOR_ID, mockExecutorOverlay, mockOverlays, mockPriorOverlay } from "../../data/__tests__/trainingOverlays.fixture";

const OVERLAYS = "data/airports/KXXX/training/overlays.json";
const EXECUTOR = `data/airports/KXXX/training/${EXECUTOR_ID}/executor.json`;

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

type FakeResponse = { ok: boolean; status?: number; headers: { get: () => string }; text: () => Promise<string> };

const last = (mock: ReturnType<typeof vi.fn>) => mock.mock.calls[mock.mock.calls.length - 1][0];

describe("useTrainingOverlays", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    setTrainingExecutor.mockClear();
    setTrainingPrior.mockClear();
    const files: Record<string, unknown> = {
      [OVERLAYS]: mockOverlays(), [EXECUTOR]: mockExecutorOverlay(),
      "data/airports/KXXX/training/prior_test/prior.json": mockPriorOverlay(),
    };
    fetchMock = vi.fn(async (url: string): Promise<FakeResponse> => {
      const path = Object.keys(files).find((key) => String(url).endsWith(key));
      return path !== undefined
        ? { ok: true, headers: { get: () => "application/json" }, text: async () => JSON.stringify(files[path]) }
        : { ok: false, status: 404, headers: { get: () => "text/html" }, text: async () => "<!doctype html>" };
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  const downloads = (path: string) => fetchMock.mock.calls.filter(([url]) => String(url).endsWith(path)).length;

  it("publishes each overlay for the selected flight, and follows the selection", async () => {
    const set = sample();
    const { rerender } = renderHook(({ key }) => useTrainingOverlays("KXXX", set, key), { initialProps: { key: VECTORED_KEY } });
    await waitFor(() => expect(last(setTrainingExecutor)?.flight.flightKey).toBe(VECTORED_KEY));
    await waitFor(() => expect(last(setTrainingPrior)?.flight.flightKey).toBe(VECTORED_KEY));
    rerender({ key: STRAIGHT_KEY });
    await waitFor(() => expect(last(setTrainingExecutor)?.flight.flightKey).toBe(STRAIGHT_KEY));
    expect(downloads(EXECUTOR)).toBe(1);
  });

  it("downloads an overlay once however often its switch goes off and on, and publishes nothing while off", async () => {
    const set = sample();
    const { result } = renderHook(() => useTrainingOverlays("KXXX", set, VECTORED_KEY));
    await waitFor(() => expect(result.current.executor.load.status).toBe("ready"));
    act(() => result.current.executor.setShown(false));
    await waitFor(() => expect(last(setTrainingExecutor)).toBeNull());
    act(() => result.current.executor.setShown(true));
    await waitFor(() => expect(last(setTrainingExecutor)?.flight.flightKey).toBe(VECTORED_KEY));
    expect(downloads(EXECUTOR)).toBe(1);
  });

  it("fetches nothing under another airport's path while the last airport's set is still open", async () => {
    const set = sample();
    const { rerender } = renderHook(({ airport }) => useTrainingOverlays(airport, set, VECTORED_KEY),
      { initialProps: { airport: "KYYY" } });
    await act(async () => undefined);
    // the set on screen is KXXX's: its overlays are never asked for under KYYY
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/KYYY/training/executor"))).toBe(false);
    rerender({ airport: "KXXX" });
    await waitFor(() => expect(last(setTrainingExecutor)?.flight.flightKey).toBe(VECTORED_KEY));
  });

  it("unpublishes both when the panel goes", async () => {
    const set = sample();
    const { unmount } = renderHook(() => useTrainingOverlays("KXXX", set, VECTORED_KEY));
    await waitFor(() => expect(last(setTrainingExecutor)).not.toBeNull());
    unmount();
    expect(last(setTrainingExecutor)).toBeNull();
    expect(last(setTrainingPrior)).toBeNull();
  });
});
