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
    pilotTransport: null,
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
    appState.pilotTransport = null;
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

  // ── Fly (pilot) mode: the aircraft runs on a manual sim loop, NOT viewer.clock,
  // so the bar drives the published pilot transport instead of the clock. ────────
  it("in Fly mode, drives the pilot sim transport (Play/Reset), not the clock", () => {
    const togglePlay = vi.fn();
    const reset = vi.fn();
    appState.mode = "fly";
    appState.pilotTransport = { running: false, playPauseDisabled: false, resetDisabled: false, togglePlay, reset };
    render(<WorkbenchBottomBar />);

    // No clock-only controls in fly mode (speed presets + loop toggle).
    expect(screen.queryByRole("button", { name: "120×" })).toBeNull();
    expect(screen.queryByRole("checkbox")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /Play/ }));
    expect(togglePlay).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: /Reset/ }));
    expect(reset).toHaveBeenCalledTimes(1);
    // The clock is untouched (pilot mode isn't clock-driven).
    expect(appState.viewer.clock.shouldAnimate).toBe(false);
  });

  it("in Fly mode, shows Pause while the sim is running", () => {
    appState.mode = "fly";
    appState.pilotTransport = { running: true, playPauseDisabled: false, resetDisabled: false, togglePlay: vi.fn(), reset: vi.fn() };
    render(<WorkbenchBottomBar />);
    expect(screen.getByRole("button", { name: /Pause/ })).toBeTruthy();
  });

  it("in Fly mode, disables the transport until PilotPanel publishes it", () => {
    appState.mode = "fly";
    appState.pilotTransport = null;
    render(<WorkbenchBottomBar />);
    expect((screen.getByRole("button", { name: /Play/ }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: /Reset/ }) as HTMLButtonElement).disabled).toBe(true);
  });
});
