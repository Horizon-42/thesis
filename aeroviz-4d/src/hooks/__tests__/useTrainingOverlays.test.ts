/**
 * useTrainingOverlays: the overlays manifest of the open airport, and each kind's overlay over the open set — downloaded
 * once per overlay and sample however often its switch goes off and on (a failed download is tried again), never under
 * another airport's path, the one chosen among several kept for its own set, published for the selected flight only
 * while shown, and unpublished when the panel goes.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

const { setTrainingExecutor, setTrainingPrior, setTrainingGenerations } = vi.hoisted(() => ({
  setTrainingExecutor: vi.fn(),
  setTrainingPrior: vi.fn(),
  setTrainingGenerations: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ setTrainingExecutor, setTrainingPrior, setTrainingGenerations }),
}));

import useTrainingOverlays from "../useTrainingOverlays";
import { parseTrainingSample, type TrainingSample } from "../../data/trainingSample";
import { STRAIGHT_KEY, VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import {
  BASE_MODEL_ID, EXECUTOR_ID, POST_TRAINED_ID, mockExecutorOverlay, mockGenerationOverlay, mockOverlays, mockOverlaysWithGenerations,
  mockPriorOverlay,
} from "../../data/__tests__/trainingOverlays.fixture";

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
  let files: Record<string, unknown>;

  beforeEach(() => {
    setTrainingExecutor.mockClear();
    setTrainingPrior.mockClear();
    setTrainingGenerations.mockClear();
    files = {
      [OVERLAYS]: mockOverlays(), [EXECUTOR]: mockExecutorOverlay(),
      "data/airports/KXXX/training/prior_test/prior.json": mockPriorOverlay(),
    };
    fetchMock = vi.fn(async (url: string): Promise<FakeResponse> => {
      const path = Object.keys(files).find((key) => String(url).endsWith(key));
      if (path !== undefined && files[path] instanceof Error) throw files[path];
      return path !== undefined
        ? { ok: true, headers: { get: () => "application/json" }, text: async () => JSON.stringify(files[path]) }
        : { ok: false, status: 404, headers: { get: () => "text/html" }, text: async () => "<!doctype html>" };
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  const downloads = (path: string) => fetchMock.mock.calls.filter(([url]) => String(url).endsWith(path)).length;

  /** The overlays manifest of ``airport``: the same two replays, the latest listed last. */
  const listed = (airport: string) => {
    const manifest: any = mockOverlays();
    manifest.airport = airport;
    manifest.overlays.unshift({ ...manifest.overlays[0], id: "executor_older", file: "executor_older/executor.json" });
    return manifest;
  };

  it("downloads every model's own sentences published for the set, with no switch, and publishes them for the selected flight", async () => {
    files[OVERLAYS] = mockOverlaysWithGenerations();
    files[`data/airports/KXXX/training/${BASE_MODEL_ID}/generation.json`] = mockGenerationOverlay(BASE_MODEL_ID);
    files[`data/airports/KXXX/training/${POST_TRAINED_ID}/generation.json`] = { ...mockGenerationOverlay(POST_TRAINED_ID, true), schema: "x" };
    const set = sample();
    const { result, rerender, unmount } = renderHook(({ key }) => useTrainingOverlays("KXXX", set, key), { initialProps: { key: VECTORED_KEY } });
    await waitFor(() => expect(result.current.generations.map(({ load }) => load.status)).toEqual(["ready", "invalid"]));
    // one that cannot be read says why and is not published; the other is, for the selected flight
    await waitFor(() => expect(last(setTrainingGenerations).map((view: any) => [view.overlay.overlayId, view.flight.flightKey]))
      .toEqual([[BASE_MODEL_ID, VECTORED_KEY]]));
    rerender({ key: STRAIGHT_KEY });
    await waitFor(() => expect(last(setTrainingGenerations)[0].flight.flightKey).toBe(STRAIGHT_KEY));
    expect(downloads(`${BASE_MODEL_ID}/generation.json`)).toBe(1);
    unmount();
    expect(last(setTrainingGenerations)).toEqual([]);
  });

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

  it("tries a download that failed again when its switch goes off and on — and not before", async () => {
    const set = sample();
    files[EXECUTOR] = new Error("the connection dropped");
    const { result } = renderHook(() => useTrainingOverlays("KXXX", set, VECTORED_KEY));
    await waitFor(() => expect(result.current.executor.load).toEqual({ status: "invalid", problem: "the connection dropped" }));
    await act(async () => undefined);
    expect(downloads(EXECUTOR)).toBe(1);
    const again = async () => {
      act(() => result.current.executor.setShown(false));
      act(() => result.current.executor.setShown(true));
      await waitFor(() => expect(result.current.executor.load.status).not.toBe("loading"));
    };
    // downloaded, but not readable
    files[EXECUTOR] = { ...mockExecutorOverlay(), schema: "aeroviz-training-executor-v1" };
    await again();
    expect(result.current.executor.load.status).toBe("invalid");
    // rewritten since: read anew
    files[EXECUTOR] = mockExecutorOverlay();
    await again();
    expect(result.current.executor.load.status).toBe("ready");
    expect(downloads(EXECUTOR)).toBe(3);
  });

  it("keeps only the latest download's answer, even of the same file asked for again", async () => {
    files[OVERLAYS] = listed("KXXX");
    // the latest replay's downloads wait until the test answers them, in the order they were asked
    const answers: Array<(response: FakeResponse) => void> = [];
    const served = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((url: string) => (String(url).endsWith(EXECUTOR)
      ? new Promise<FakeResponse>((resolve) => answers.push(resolve)) : served(url)));
    const answer = (index: number, overlay: unknown) =>
      act(async () => answers[index]({ ok: true, headers: { get: () => "application/json" }, text: async () => JSON.stringify(overlay) }));
    const set = sample();
    const { result } = renderHook(() => useTrainingOverlays("KXXX", set, VECTORED_KEY));
    await waitFor(() => expect(answers).toHaveLength(1));
    // the older replay chosen, then the latest again, while its first download is still out
    act(() => result.current.executor.choose("executor_older"));
    await waitFor(() => expect(result.current.executor.load.status).toBe("invalid"));    // not published here
    act(() => result.current.executor.choose(EXECUTOR_ID));
    await waitFor(() => expect(answers).toHaveLength(2));
    // the first download fails: it is not the latest, and changes nothing
    await answer(0, { ...mockExecutorOverlay(), schema: "aeroviz-training-executor-v1" });
    expect(result.current.executor.load.status).toBe("loading");
    await answer(1, mockExecutorOverlay());
    expect(result.current.executor.load.status).toBe("ready");
    expect(downloads(EXECUTOR)).toBe(2);
  });

  it("keeps the overlay chosen among several for its own set: another airport's starts at the latest", async () => {
    // the same two replays listed at both airports
    files[OVERLAYS] = listed("KXXX");
    files["data/airports/KYYY/training/overlays.json"] = listed("KYYY");
    const set = sample();
    const { result, rerender } = renderHook(({ open }) => useTrainingOverlays(open.airport, open, VECTORED_KEY),
      { initialProps: { open: set } });
    await waitFor(() => expect(result.current.executor.entries.map((item) => item.id)).toEqual(["executor_older", EXECUTOR_ID]));
    expect(result.current.executor.entry?.id).toBe(EXECUTOR_ID);
    act(() => result.current.executor.choose("executor_older"));
    expect(result.current.executor.entry?.id).toBe("executor_older");
    rerender({ open: { ...set, airport: "KYYY" } });
    await waitFor(() => expect(result.current.executor.entries).toHaveLength(2));
    expect(result.current.executor.entry?.id).toBe(EXECUTOR_ID);
    rerender({ open: set });
    await waitFor(() => expect(result.current.executor.entries).toHaveLength(2));
    expect(result.current.executor.entry?.id).toBe("executor_older");
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
