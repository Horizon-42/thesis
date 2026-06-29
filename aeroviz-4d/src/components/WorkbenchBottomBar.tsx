/**
 * WorkbenchBottomBar.tsx
 * ----------------------
 * The transport bar docked along the bottom (above Cesium's timeline scrubber):
 * play/pause, reset, clock-speed presets, and auto-replay. The controls operate
 * generically on `viewer.clock`, so they drive whichever source set the clock
 * (observed CZML, optimizer comparison, or the pilot sim). Shown only in the
 * time-based tasks; hidden in Procedures.
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
  const { viewer, mode, playbackSpeed, setPlaybackSpeed, autoReplay, setAutoReplay } = useApp();
  const [isAnimating, setIsAnimating] = useState<boolean>(false);

  useEffect(() => {
    if (!viewer) {
      setIsAnimating(false);
      return;
    }
    setIsAnimating(viewer.clock.shouldAnimate);
    const removeListener = viewer.clock.onTick.addEventListener(() => {
      const next = viewer.clock.shouldAnimate;
      setIsAnimating((prev) => (prev === next ? prev : next));
    });
    return () => removeListener();
  }, [viewer]);

  if (!TIME_MODES.has(mode)) return null;

  function handleSpeedChange(speed: number) {
    setPlaybackSpeed(speed);
    if (viewer) viewer.clock.multiplier = speed;
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
            className={playbackSpeed === opt.value ? "active" : ""}
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
