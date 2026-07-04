/**
 * WorkbenchBottomBar.tsx
 * ----------------------
 * The transport bar docked bottom-center (above Cesium's timeline, clear of the
 * native clock dial at bottom-left): play/pause, reset, clock-speed presets, and
 * auto-replay. The controls operate generically on `viewer.clock`, so they stay in
 * sync (联动) with Cesium's own animation dial — both read and drive the SAME clock.
 *
 * The play/pause icon and the highlighted speed reflect the LIVE clock state (read
 * each tick), so dragging the native dial updates this bar, and clicking a preset
 * here moves the dial. The discrete presets + Reset + loop toggle are the extras the
 * native dial's continuous shuttle ring doesn't offer.
 */

import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { useEffect, useState } from "react";

const SPEED_OPTIONS: Array<{ label: string; value: number }> = [
  { label: "1×", value: 1 },
  { label: "10×", value: 10 },
  { label: "30×", value: 30 },
  { label: "60×", value: 60 },
  { label: "120×", value: 120 },
];

const TIME_MODES = new Set(["observe", "fly", "optimize", "compare"]);

export default function WorkbenchBottomBar() {
  const { viewer, mode, setPlaybackSpeed, autoReplay, setAutoReplay, pilotTransport } = useApp();
  const [isAnimating, setIsAnimating] = useState<boolean>(false);
  const [liveMultiplier, setLiveMultiplier] = useState<number>(60);

  // Mirror the live clock so the bar stays in sync with the native Cesium dial:
  // whichever control moves viewer.clock, this reflects it (play state + speed).
  useEffect(() => {
    if (!viewer) {
      setIsAnimating(false);
      return;
    }
    setIsAnimating(viewer.clock.shouldAnimate);
    setLiveMultiplier(viewer.clock.multiplier);
    const removeListener = viewer.clock.onTick.addEventListener(() => {
      const animating = viewer.clock.shouldAnimate;
      const multiplier = viewer.clock.multiplier;
      setIsAnimating((prev) => (prev === animating ? prev : animating));
      setLiveMultiplier((prev) => (prev === multiplier ? prev : multiplier));
    });
    return () => removeListener();
  }, [viewer]);

  if (!TIME_MODES.has(mode)) return null;

  // Fly (pilot) mode runs a MANUAL sim loop, not viewer.clock — so the transport
  // drives the simulation here (Play/Pause/Reset only; the speed presets + loop
  // toggle are clock concepts that don't apply). PilotPanel publishes the sim
  // transport as `pilotTransport`; it is briefly null while the panel mounts.
  if (mode === "fly") {
    return (
      <div className="workbench-bottom-bar" aria-label="Flight simulation transport">
        <button
          onClick={() => pilotTransport?.togglePlay()}
          disabled={!pilotTransport || pilotTransport.playPauseDisabled}
        >
          {pilotTransport?.running ? "⏸ Pause" : "▶ Play"}
        </button>
        <button
          onClick={() => pilotTransport?.reset()}
          disabled={!pilotTransport || pilotTransport.resetDisabled}
        >
          ⏮ Reset
        </button>
      </div>
    );
  }

  function handleSpeedChange(speed: number) {
    setPlaybackSpeed(speed);
    if (viewer) {
      // Leave real-time (SYSTEM_CLOCK) mode so the multiplier actually takes effect —
      // the same thing the native dial's speed control does, keeping the two in sync.
      viewer.clock.clockStep = Cesium.ClockStep.SYSTEM_CLOCK_MULTIPLIER;
      viewer.clock.multiplier = speed;
    }
    setLiveMultiplier(speed);
  }

  function handlePlayPause() {
    if (!viewer) return;
    const next = !viewer.clock.shouldAnimate;
    viewer.clock.shouldAnimate = next;
    setIsAnimating(next);
  }

  function handleReset() {
    if (!viewer) return;
    viewer.clock.currentTime = viewer.clock.startTime.clone();
    viewer.clock.shouldAnimate = false;
    setIsAnimating(false);
  }

  function handleAutoReplayChange(next: boolean) {
    setAutoReplay(next);
    if (!viewer) return;
    viewer.clock.clockRange = next ? Cesium.ClockRange.LOOP_STOP : Cesium.ClockRange.CLAMPED;
  }

  return (
    <div className="workbench-bottom-bar" aria-label="Playback transport">
      <button onClick={handlePlayPause}>{isAnimating ? "⏸ Pause" : "▶ Play"}</button>
      <button onClick={handleReset}>⏮ Reset</button>

      <div className="workbench-bottom-bar-speeds">
        {SPEED_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            className={liveMultiplier === opt.value ? "active" : ""}
            onClick={() => handleSpeedChange(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <label className="workbench-bottom-bar-replay">
        <input
          type="checkbox"
          checked={autoReplay}
          onChange={(event) => handleAutoReplayChange(event.target.checked)}
        />
        Auto-replay (loop)
      </label>
    </div>
  );
}
