import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  beaconCloseWorkerSession,
  closeWorkerSession,
  openWorkerSession,
} from "../workerSessionClient";

vi.mock("../pilotClient", () => ({
  AEROVIZ_BACKEND_URL: "http://test-backend:9999",
}));

describe("workerSessionClient", () => {
  const sendBeacon = vi.fn();

  beforeEach(() => {
    Object.defineProperty(globalThis.navigator, "sendBeacon", {
      value: sendBeacon,
      configurable: true,
      writable: true,
    });
    sendBeacon.mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("opens the optimizer session at the optimizer endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await openWorkerSession("optimizer");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-backend:9999/optimization/session/open",
      expect.objectContaining({ method: "POST", keepalive: true }),
    );
  });

  it("closes the comparison session at the comparison endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await closeWorkerSession("comparison");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://test-backend:9999/dynamics-comparison/session/close",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("swallows network errors so a lifecycle hint never breaks the UI", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("backend down")));
    // Must not throw.
    await expect(openWorkerSession("optimizer")).resolves.toBeUndefined();
  });

  it("releases the session via sendBeacon on page unload", () => {
    beaconCloseWorkerSession("optimizer");
    expect(sendBeacon).toHaveBeenCalledWith(
      "http://test-backend:9999/optimization/session/close",
    );
  });
});
