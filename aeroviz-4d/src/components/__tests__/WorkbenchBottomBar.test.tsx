import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

vi.mock("cesium", () => ({
  ClockRange: { LOOP_STOP: "LOOP_STOP", CLAMPED: "CLAMPED" },
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

  it("renders nothing in the procedures (non-time) mode", () => {
    appState.mode = "procedures";
    const { container } = render(<WorkbenchBottomBar />);
    expect(container.firstChild).toBeNull();
  });

  it("toggles play/pause on the clock", () => {
    render(<WorkbenchBottomBar />);
    fireEvent.click(screen.getByRole("button", { name: /Play/ }));
    expect(appState.viewer.clock.shouldAnimate).toBe(true);
  });

  it("changes the clock multiplier and playback speed", () => {
    render(<WorkbenchBottomBar />);
    fireEvent.click(screen.getByRole("button", { name: "120×" }));
    expect(setPlaybackSpeed).toHaveBeenCalledWith(120);
    expect(appState.viewer.clock.multiplier).toBe(120);
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
