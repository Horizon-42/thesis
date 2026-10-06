import { act, render, screen } from "@testing-library/react";
import * as Cesium from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { SceneTime } from "../../utils/sceneTime";

const { app } = vi.hoisted(() => ({
  app: {
    viewer: null as unknown,
    sceneTime: null as unknown,
    mode: "optimize",
    layers: { trajectories: false },
  },
}));
vi.mock("../../context/AppContext", () => ({ useApp: () => app }));

import SceneTimeReadout from "../SceneTimeReadout";

const EPOCH = "2026-04-01T08:00:00Z";
const SCENE: SceneTime = { startUtc: "2026-05-21T17:47:18.959Z", epoch: EPOCH };

function viewerWithClockAt(seconds: number) {
  const clock = new Cesium.Clock({
    currentTime: Cesium.JulianDate.addSeconds(Cesium.JulianDate.fromIso8601(EPOCH), seconds, new Cesium.JulianDate()),
    shouldAnimate: false,
  });
  return { clock };
}
const moveClockTo = (viewer: { clock: Cesium.Clock }, seconds: number) => {
  viewer.clock.currentTime = Cesium.JulianDate.addSeconds(
    Cesium.JulianDate.fromIso8601(EPOCH), seconds, new Cesium.JulianDate());
  viewer.clock.onTick.raiseEvent(viewer.clock);
};

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["performance", "setTimeout", "clearTimeout", "Date"] });     // the readout throttles on performance.now
  app.sceneTime = SCENE;
  app.mode = "optimize";
  app.layers = { trajectories: false };
});
afterEach(() => vi.useRealTimers());

describe("SceneTimeReadout", () => {
  it("shows the real UTC time at the clock's position, following the clock", () => {
    const viewer = viewerWithClockAt(0);
    app.viewer = viewer;
    render(<SceneTimeReadout />);
    expect(screen.getByText("Scene time (UTC)")).toBeTruthy();
    expect(screen.getByRole("status").textContent).toContain("2026-05-21 17:47:18");

    act(() => { vi.advanceTimersByTime(300); moveClockTo(viewer, 125); });
    expect(screen.getByRole("status").textContent).toContain("2026-05-21 17:49:23");
  });

  it("redraws at most four times a second, however fast the clock ticks", () => {
    const viewer = viewerWithClockAt(0);
    app.viewer = viewer;
    render(<SceneTimeReadout />);
    act(() => { vi.advanceTimersByTime(300); moveClockTo(viewer, 60); });
    expect(screen.getByRole("status").textContent).toContain("17:48:18");
    act(() => { vi.advanceTimersByTime(100); moveClockTo(viewer, 120); });          // 100 ms later: not yet
    expect(screen.getByRole("status").textContent).toContain("17:48:18");
    act(() => { vi.advanceTimersByTime(200); moveClockTo(viewer, 120); });
    expect(screen.getByRole("status").textContent).toContain("17:49:18");
  });

  it("shows nothing without a traffic scene", () => {
    app.viewer = viewerWithClockAt(0);
    app.sceneTime = null;
    render(<SceneTimeReadout />);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("shows a published category's time only while the Trajectories layer is on; a job's scene needs no switch", () => {
    app.viewer = viewerWithClockAt(0);
    app.mode = "evaluation";
    const { rerender } = render(<SceneTimeReadout />);
    expect(screen.queryByRole("status")).toBeNull();
    app.layers = { trajectories: true };
    rerender(<SceneTimeReadout />);
    expect(screen.getByRole("status")).toBeTruthy();
    app.mode = "optimize";
    app.layers = { trajectories: false };
    rerender(<SceneTimeReadout />);
    expect(screen.getByRole("status")).toBeTruthy();
  });

  it("stops listening to the clock when it goes", () => {
    const viewer = viewerWithClockAt(0);
    app.viewer = viewer;
    const { unmount } = render(<SceneTimeReadout />);
    expect(viewer.clock.onTick.numberOfListeners).toBe(1);
    unmount();
    expect(viewer.clock.onTick.numberOfListeners).toBe(0);
  });
});
