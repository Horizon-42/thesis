import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TrafficJobStatus } from "../../data/trafficJobs";

const { api, setTrafficScene, fetchIndex } = vi.hoisted(() => ({
  api: {
    startTrafficJob: vi.fn(),
    fetchTrafficJob: vi.fn(),
    cancelTrafficJob: vi.fn(),
    beaconCancelTrafficJob: vi.fn(),
  },
  setTrafficScene: vi.fn(),
  fetchIndex: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({ useApp: () => ({ setTrafficScene }) }));
vi.mock("../../data/trafficJobs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../data/trafficJobs")>()),
  ...api,
}));
vi.mock("../../utils/fetchJson", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../utils/fetchJson")>()),
  fetchJson: (url: string) => fetchIndex(url),
}));

import { retryDelayMs, useTrafficJob } from "../useTrafficJob";

const JOB = "20261006T120000123456Z-0123abcd";
const NEXT_JOB = "20261006T130000123456Z-89abcdef";
const REQUEST = { mode: "m1", airport: "KRDU", flightKey: "K" } as const;
const LABEL = "K, 2026-05-21";
const status = (over: Partial<TrafficJobStatus> = {}): TrafficJobStatus => ({
  state: "running", progress: { done: 0, total: 1, current: null }, error: null, ...over });
const INDEX = {
  schemaVersion: "comparison-v2-generation", generation: "g", epoch: "e", startHidden: true,
  referenceSource: "canonicalObserved", evaluationReport: "r.json", groups: [],
};
/** A promise the test settles by hand. */
function deferred<T>() {
  let resolve: (value: T) => void = () => undefined;
  let reject: (error: unknown) => void = () => undefined;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

beforeEach(() => {
  vi.useFakeTimers();
  for (const mock of [...Object.values(api), setTrafficScene, fetchIndex]) mock.mockReset();
  api.startTrafficJob.mockResolvedValue(JOB);
  api.fetchTrafficJob.mockResolvedValue(status());
  api.cancelTrafficJob.mockResolvedValue(status({ state: "cancelled" }));
  fetchIndex.mockResolvedValue(INDEX);
});
afterEach(() => vi.useRealTimers());

/** Run the hook's pending promises (and the timers due within `ms`). */
const settle = (ms = 0) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

async function started() {
  const hook = renderHook(() => useTrafficJob("KRDU"));
  await act(async () => { await hook.result.current.start(REQUEST, LABEL); });
  return hook;
}

describe("useTrafficJob", () => {
  it("starts the job the request names and shows it running at once", async () => {
    const { result } = await started();
    expect(api.startTrafficJob).toHaveBeenCalledWith(REQUEST);
    expect(result.current.view).toEqual({ phase: "running", jobId: JOB, status: status(), connection: null });
    expect(setTrafficScene).toHaveBeenLastCalledWith(null);           // a new job starts without a scene
  });

  it("asks how far the job is every 2 s, and no more often", async () => {
    const { result } = await started();
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(1);
    await settle(1999);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(1);
    api.fetchTrafficJob.mockResolvedValue(status({ progress: { done: 1, total: 4, current: "AAL1_05L" } }));
    await settle(1);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(2);
    expect(result.current.view).toMatchObject({ phase: "running", status: { progress: { done: 1, total: 4 } } });
    await settle(2000);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(3);
    expect(api.fetchTrafficJob).toHaveBeenCalledWith(JOB);
  });

  it("hands the finished job's files to the comparison layer, with the inputs it was started with, and stops asking", async () => {
    const { result } = await started();
    api.fetchTrafficJob.mockResolvedValue(status({ state: "done", summary: { windows: 1 } }));
    await settle(2000);

    expect(fetchIndex).toHaveBeenCalledWith(expect.stringMatching(new RegExp(`/traffic/jobs/${JOB}/files/comparison_index\\.json$`)));
    expect(result.current.view).toMatchObject({ phase: "done", jobId: JOB, index: INDEX, label: LABEL });
    expect(setTrafficScene).toHaveBeenLastCalledWith({
      airportCode: "KRDU", jobId: JOB, baseUrl: expect.stringMatching(new RegExp(`/traffic/jobs/${JOB}/files/$`)) });
    const asked = api.fetchTrafficJob.mock.calls.length;
    await settle(10000);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(asked);
  });

  it("shows a failed job's reason, with no scene", async () => {
    const { result } = await started();
    api.fetchTrafficJob.mockResolvedValue(status({ state: "failed", error: "ValueError: no way" }));
    await settle(2000);
    expect(result.current.view).toEqual({ phase: "failed", error: "ValueError: no way" });
    expect(setTrafficScene).not.toHaveBeenCalledWith(expect.objectContaining({ jobId: JOB }));
  });

  it("fails when the job's comparison index is not one the viewer reads", async () => {
    const { result } = await started();
    api.fetchTrafficJob.mockResolvedValue(status({ state: "done" }));
    fetchIndex.mockResolvedValue({ groups: "nope" });
    await settle(2000);
    expect(result.current.view).toMatchObject({ phase: "failed", error: expect.stringMatching(/comparison index/) });
  });

  it("shows the reason a start was refused (a job is running elsewhere), and starts again afterwards", async () => {
    api.startTrafficJob.mockRejectedValueOnce(new Error("a traffic job is running: cancel it or wait for it"));
    const { result } = renderHook(() => useTrafficJob("KRDU"));
    await act(async () => { await result.current.start(REQUEST, LABEL); });
    expect(result.current.view).toEqual({ phase: "failed", error: "a traffic job is running: cancel it or wait for it" });
    await act(async () => { await result.current.start(REQUEST, LABEL); });
    expect(result.current.view).toMatchObject({ phase: "running", jobId: JOB });
  });
});

describe("Start while a job is in flight", () => {
  it("does nothing while a job runs: no second job, no cancel", async () => {
    const { result } = await started();
    await act(async () => { await result.current.start({ ...REQUEST, flightKey: "L" }, "L"); });
    expect(api.startTrafficJob).toHaveBeenCalledTimes(1);
    expect(api.cancelTrafficJob).not.toHaveBeenCalled();
    expect(result.current.view).toMatchObject({ phase: "running", jobId: JOB });
  });

  it("does nothing while a job is starting", async () => {
    const answer = deferred<string>();
    api.startTrafficJob.mockReturnValue(answer.promise);
    const { result } = renderHook(() => useTrafficJob("KRDU"));
    let first: Promise<void> = Promise.resolve();
    await act(async () => { first = result.current.start(REQUEST, LABEL); });
    await act(async () => { await result.current.start(REQUEST, LABEL); });
    expect(api.startTrafficJob).toHaveBeenCalledTimes(1);
    await act(async () => { answer.resolve(JOB); await first; });
  });

  it("does nothing while a cancel is out; after it is answered a job starts without having waited for anything", async () => {
    const { result } = await started();
    const cancelled = deferred<TrafficJobStatus>();
    api.cancelTrafficJob.mockReturnValue(cancelled.promise);
    act(() => result.current.cancel());
    expect(result.current.view).toEqual({ phase: "cancelling" });

    await act(async () => { await result.current.start({ ...REQUEST, flightKey: "L" }, "L"); });
    expect(api.startTrafficJob).toHaveBeenCalledTimes(1);               // refused: the backend still holds the old job

    await act(async () => { cancelled.resolve(status({ state: "cancelled" })); });
    expect(result.current.view).toEqual({ phase: "cancelled" });
    api.startTrafficJob.mockResolvedValue(NEXT_JOB);
    await act(async () => { await result.current.start(REQUEST, LABEL); });
    expect(api.startTrafficJob).toHaveBeenCalledTimes(2);
    expect(result.current.view).toMatchObject({ phase: "running", jobId: NEXT_JOB });
  });

  it("starts after a finished, a failed and a cancelled job alike, and the old scene is removed first", async () => {
    const { result } = await started();
    api.fetchTrafficJob.mockResolvedValue(status({ state: "done" }));
    await settle(2000);
    expect(result.current.view.phase).toBe("done");
    setTrafficScene.mockClear();

    api.startTrafficJob.mockResolvedValue(NEXT_JOB);
    api.fetchTrafficJob.mockResolvedValue(status());
    await act(async () => { await result.current.start(REQUEST, LABEL); });
    expect(setTrafficScene).toHaveBeenCalledWith(null);
    expect(result.current.view).toMatchObject({ phase: "running", jobId: NEXT_JOB });
  });
});

describe("cancel", () => {
  it("cancels the running job, stops asking and removes the scene", async () => {
    const { result } = await started();
    await act(async () => { result.current.cancel(); });
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
    expect(result.current.view).toEqual({ phase: "cancelled" });
    expect(setTrafficScene).toHaveBeenLastCalledWith(null);
    const asked = api.fetchTrafficJob.mock.calls.length;
    await settle(10000);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(asked);
  });

  it("is asked once: a second click while the first is out sends nothing", async () => {
    const { result } = await started();
    api.cancelTrafficJob.mockReturnValue(deferred<TrafficJobStatus>().promise);
    act(() => { result.current.cancel(); result.current.cancel(); });
    expect(api.cancelTrafficJob).toHaveBeenCalledTimes(1);
  });

  it("ignores a poll's answer that arrives after the cancel", async () => {
    const { result } = await started();
    const late = deferred<TrafficJobStatus>();
    api.fetchTrafficJob.mockReturnValue(late.promise);
    await settle(2000);                                          // a poll is in flight
    await act(async () => { result.current.cancel(); });
    await act(async () => { late.resolve(status({ state: "done" })); });
    expect(result.current.view).toEqual({ phase: "cancelled" });
    expect(fetchIndex).not.toHaveBeenCalled();
  });

  it("does not let its answer touch the view or the scene once the panel is gone", async () => {
    const { result, unmount } = await started();
    const cancelled = deferred<TrafficJobStatus>();
    api.cancelTrafficJob.mockReturnValue(cancelled.promise);
    act(() => result.current.cancel());
    unmount();                                                   // the panel goes while the cancel is out
    setTrafficScene.mockClear();
    await act(async () => { cancelled.resolve(status({ state: "cancelled" })); });
    expect(setTrafficScene).not.toHaveBeenCalled();              // a stale cancel clears nothing of what came after
  });

  it("is a failed view, with Start free again, when the backend cannot be told", async () => {
    const { result } = await started();
    api.cancelTrafficJob.mockRejectedValue(new Error("HTTP 500"));
    await act(async () => { result.current.cancel(); });
    expect(result.current.view).toEqual({ phase: "failed", error: "could not cancel the job: HTTP 500" });
    api.startTrafficJob.mockResolvedValue(NEXT_JOB);
    api.cancelTrafficJob.mockResolvedValue(status({ state: "cancelled" }));
    await act(async () => { await result.current.start(REQUEST, LABEL); });
    expect(result.current.view).toMatchObject({ phase: "running", jobId: NEXT_JOB });
  });

  it("while the job is still starting: the job the backend then names is cancelled, and the view ends cancelled", async () => {
    const answer = deferred<string>();
    api.startTrafficJob.mockReturnValue(answer.promise);
    const { result } = renderHook(() => useTrafficJob("KRDU"));
    let starting: Promise<void> = Promise.resolve();
    await act(async () => { starting = result.current.start(REQUEST, LABEL); });
    act(() => result.current.cancel());
    expect(api.cancelTrafficJob).not.toHaveBeenCalled();         // no id to cancel yet
    expect(result.current.view).toEqual({ phase: "cancelling" });

    await act(async () => { answer.resolve(JOB); await starting; });
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
    expect(api.fetchTrafficJob).not.toHaveBeenCalled();
    expect(result.current.view).toEqual({ phase: "cancelled" });
  });
});

describe("a poll that fails", () => {
  it("keeps the job, says the connection is lost and asks again with a growing wait; Cancel still reaches the job", async () => {
    const { result } = await started();
    api.fetchTrafficJob.mockRejectedValue(new Error("Failed to fetch"));
    await settle(2000);
    expect(result.current.view).toMatchObject({ phase: "running", jobId: JOB, connection: "Failed to fetch" });
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(2);

    await settle(1999);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(2);                 // the second failure waits 2 s ...
    await settle(1);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(3);
    await settle(3999);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(3);                 // ... the third 4 s
    await settle(1);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(4);
    expect(result.current.view.phase).toBe("running");                    // never dropped

    await act(async () => { result.current.cancel(); });
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
  });

  it("recovers when the backend answers again: the connection is well, the poll back to 2 s", async () => {
    const { result } = await started();
    api.fetchTrafficJob.mockRejectedValueOnce(new Error("Failed to fetch"));
    await settle(2000);
    expect(result.current.view).toMatchObject({ connection: "Failed to fetch" });
    api.fetchTrafficJob.mockResolvedValue(status({ progress: { done: 1, total: 2, current: "K" } }));
    await settle(2000);
    expect(result.current.view).toMatchObject({ phase: "running", connection: null, status: { progress: { done: 1 } } });
    const asked = api.fetchTrafficJob.mock.calls.length;
    await settle(2000);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(asked + 1);
  });

  it("finishes normally when the job finished while the connection was down", async () => {
    const { result } = await started();
    api.fetchTrafficJob.mockRejectedValue(new Error("Failed to fetch"));
    await settle(2000);
    api.fetchTrafficJob.mockResolvedValue(status({ state: "done" }));
    await settle(2000);
    expect(result.current.view).toMatchObject({ phase: "done", jobId: JOB });
  });

  it("waits 2 s, 4 s, 8 s, then no longer than 15 s", () => {
    expect([0, 1, 2, 3, 4, 10].map(retryDelayMs)).toEqual([2000, 4000, 8000, 15000, 15000, 15000]);
  });
});

describe("the panel going", () => {
  it("cancels a running job and removes the scene when the panel goes (a change of mode, leaving the task)", async () => {
    const { unmount } = await started();
    setTrafficScene.mockClear();
    unmount();
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
    expect(setTrafficScene).toHaveBeenCalledWith(null);
    const asked = api.fetchTrafficJob.mock.calls.length;
    await settle(10000);
    expect(api.fetchTrafficJob).toHaveBeenCalledTimes(asked);                       // nobody asks any more
  });

  it("removes a finished job's scene when the panel goes, and has nothing to cancel", async () => {
    const { unmount } = await started();
    api.fetchTrafficJob.mockResolvedValue(status({ state: "done" }));
    await settle(2000);
    unmount();
    expect(api.cancelTrafficJob).not.toHaveBeenCalled();
    expect(setTrafficScene).toHaveBeenLastCalledWith(null);
  });

  it("leaves nothing to cancel when no job was started", () => {
    const { unmount } = renderHook(() => useTrafficJob("KRDU"));
    unmount();
    expect(api.cancelTrafficJob).not.toHaveBeenCalled();
  });

  it("cancels a job that was still starting when the panel went", async () => {
    const answer = deferred<string>();
    api.startTrafficJob.mockReturnValue(answer.promise);
    const { result, unmount } = renderHook(() => useTrafficJob("KRDU"));
    let starting: Promise<void> = Promise.resolve();
    await act(async () => { starting = result.current.start(REQUEST, LABEL); });
    unmount();
    await act(async () => { answer.resolve(JOB); await starting; });
    expect(api.cancelTrafficJob).toHaveBeenCalledWith(JOB);
    expect(api.fetchTrafficJob).not.toHaveBeenCalled();
  });

  it("sends a beacon to cancel the running job when the page is unloaded", async () => {
    await started();
    window.dispatchEvent(new Event("pagehide"));
    expect(api.beaconCancelTrafficJob).toHaveBeenCalledWith(JOB);
  });
});
