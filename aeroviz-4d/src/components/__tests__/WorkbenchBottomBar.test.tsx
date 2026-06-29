import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

vi.mock("cesium", () => ({
  ClockRange: { LOOP_STOP: "LOOP_STOP", CLAMPED: "CLAMPED" },
  ClockStep: { SYSTEM_CLOCK_MULTIPLIER: "SYSTEM_CLOCK_MULTIPLIER" },
}));

const { appState, setPlaybackSpeed, setAutoReplay, makeClock } = vi.hoisted(() => {
  function makeClock() {
    return {
      shouldAnimate: false,
      multiplier: 60,
      clockRange: "LOOP_STOP",
      startTime: { clone: () => ({ cloned: true }) },
      currentTime: { cloned: false },
      onTick: { addEventListener: () => () => {} },
    };
  }
  const appState: any = {
    mode: "observe",
    viewer: { clock: makeClock() },
    playbackSpeed: 60,
    autoReplay: true,
  };
  return {
    appState,
    setPlaybackSpeed: vi.fn(),
    setAutoReplay: vi.fn(),
    makeClock,
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setPlaybackSpeed, setAutoReplay }),
}));

import WorkbenchBottomBar from "../WorkbenchBottomBar";

describe("WorkbenchBottomBar", () => {
  beforeEach(() => {
    appState.mode = "observe";
    appState.viewer = { clock: makeClock() };
    appState.playbackSpeed = 60;
    appState.autoReplay = true;
    vi.clearAllMocks();
  });

  it("renders the transport in every time-based task (e.g. optimize)", () => {
    appState.mode = "optimize";
    render(<WorkbenchBottomBar />);
    expect(screen.getByRole("button", { name: /Play/ })).toBeTruthy();
  });

  it("toggles play/pause on the clock", () => {
    render(<WorkbenchBottomBar />);
    fireEvent.click(screen.getByRole("button", { name: /Play/ }));
    expect(appState.viewer.clock.shouldAnimate).toBe(true);
  });

  it("changes the clock multiplier (in multiplier mode) and playback speed", () => {
    render(<WorkbenchBottomBar />);
    fireEvent.click(screen.getByRole("button", { name: "120×" }));
    expect(setPlaybackSpeed).toHaveBeenCalledWith(120);
    expect(appState.viewer.clock.multiplier).toBe(120);
    expect(appState.viewer.clock.clockStep).toBe("SYSTEM_CLOCK_MULTIPLIER");
  });

  it("highlights the preset matching the LIVE clock multiplier (synced with the native dial)", () => {
    // The native Cesium dial moves viewer.clock.multiplier; the bar reads it on mount.
    appState.viewer.clock.multiplier = 30;
    render(<WorkbenchBottomBar />);
    expect(screen.getByRole("button", { name: "30×" }).classList.contains("active")).toBe(true);
    expect(screen.getByRole("button", { name: "60×" }).classList.contains("active")).toBe(false);
  });

  it("resets the clock to the start time", () => {
    appState.viewer.clock.shouldAnimate = true;
    render(<WorkbenchBottomBar />);
    fireEvent.click(screen.getByRole("button", { name: /Reset/ }));
    expect(appState.viewer.clock.currentTime).toEqual({ cloned: true });
    expect(appState.viewer.clock.shouldAnimate).toBe(false);
  });

  it("sets the clock range from auto-replay", () => {
    render(<WorkbenchBottomBar />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(setAutoReplay).toHaveBeenCalledWith(false);
    expect(appState.viewer.clock.clockRange).toBe("CLAMPED");
  });
});
