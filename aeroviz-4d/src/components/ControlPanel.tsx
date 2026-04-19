/**
 * ControlPanel.tsx
 * ----------------
 * Floating overlay panel for:
 *   • Play / Pause / Reset the simulation clock
 *   • Adjust playback speed (clock multiplier)
 *   • Toggle data layer visibility
 *
 * All interactions go through AppContext — this component never touches
 * the Cesium Viewer directly.  That keeps the UI logic decoupled from
 * the rendering engine.
 */

import { useApp, type LayerKey } from "../context/AppContext";
import { useEffect, useState } from "react";

/** Predefined speed options shown as buttons */
const SPEED_OPTIONS: Array<{ label: string; value: number }> = [
  { label: "1×",   value: 1   },
  { label: "10×",  value: 10  },
  { label: "30×",  value: 30  },
  { label: "60×",  value: 60  },
  { label: "120×", value: 120 },
];

/** Human-readable names for each layer toggle */
const LAYER_LABELS: Record<LayerKey, string> = {
  terrain:      "Terrain",
  dsmTerrain:   "DSM Terrain",
  runways:      "Runways",
  waypoints:    "Waypoints",
  ocsSurfaces:  "OCS Surfaces",
  trajectories: "Trajectories",
  obstacles:    "Obstacles",
  procedures:   "RNAV Procedures",
};

const ACTIVE_LAYER_KEYS: LayerKey[] = [
  "terrain",
  "dsmTerrain",
  "runways",
  "trajectories",
  "obstacles",
  "procedures",
  "ocsSurfaces",
];

export default function ControlPanel() {
  const { viewer, layers, toggleLayer, playbackSpeed, setPlaybackSpeed } = useApp();
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

    return () => {
      removeListener();
    };
  }, [viewer]);

  // ── Clock control handlers ─────────────────────────────────────────────────

  /** Change the simulation speed */
  function handleSpeedChange(speed: number) {
    setPlaybackSpeed(speed);
    if (!viewer) return;
    viewer.clock.multiplier = speed;
  }

  /** Toggle play / pause */
  function handlePlayPause() {
    if (!viewer) return;
    const next = !viewer.clock.shouldAnimate;
    viewer.clock.shouldAnimate = next;
    setIsAnimating(next);
  }

  /** Reset the clock to the start of the simulation */
  function handleReset() {
    if (!viewer) return;
    viewer.clock.currentTime = viewer.clock.startTime.clone();
    viewer.clock.shouldAnimate = false;
    setIsAnimating(false);
  }

  return (
    <div className="control-panel">
      <h3>AeroViz-4D</h3>

      {/* ── Playback controls ────────────────────────────────────────────── */}
      <section>
        <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
          <button onClick={handlePlayPause}>{isAnimating ? "⏸ Pause" : "▶ Play"}</button>
          <button onClick={handleReset}>⏮ Reset</button>
        </div>

        <div className="speed-buttons">
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
      </section>

      {/* ── Layer toggles ────────────────────────────────────────────────── */}
      <section>
        <h4>Layers</h4>
        {ACTIVE_LAYER_KEYS.map((key) => (
          <label key={key}>
            <input
              type="checkbox"
              checked={layers[key]}
              onChange={() => toggleLayer(key)}
            />
            {LAYER_LABELS[key]}
          </label>
        ))}
      </section>
    </div>
  );
}
