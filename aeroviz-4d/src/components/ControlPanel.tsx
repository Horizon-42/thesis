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
  runways:      "Runways",
  waypoints:    "Waypoints",
  ocsSurfaces:  "OCS Surfaces",
  trajectories: "Trajectories",
};

export default function ControlPanel() {
  const { viewer, layers, toggleLayer, playbackSpeed, setPlaybackSpeed } = useApp();

  // ── Clock control handlers ─────────────────────────────────────────────────

  /** Change the simulation speed */
  function handleSpeedChange(speed: number) {
    // TODO ① — Set viewer.clock.multiplier = speed, then call setPlaybackSpeed(speed).
    //
    // Hint: check `if (!viewer) return;` first — the viewer may not be ready yet.
    //
    // Why update both?
    //   viewer.clock.multiplier  → actually changes the animation speed in Cesium
    //   setPlaybackSpeed(speed)  → updates React state so the active button re-renders
  }

  /** Toggle play / pause */
  function handlePlayPause() {
    // TODO ② — Toggle `viewer.clock.shouldAnimate`.
    //
    // Hint: viewer.clock.shouldAnimate = !viewer.clock.shouldAnimate;
    //
    // Note: This doesn't need React state because the button label doesn't
    // need to change here (you can add that as an improvement later).
  }

  /** Reset the clock to the start of the simulation */
  function handleReset() {
    // TODO ③ — Set viewer.clock.currentTime = viewer.clock.startTime.clone()
    //           and set shouldAnimate = false.
    //
    // Why .clone()?  Cesium's JulianDate is mutable.  Assigning without cloning
    // would make currentTime and startTime point to the same object, so any
    // later mutation of currentTime would silently corrupt startTime.
  }

  return (
    <div className="control-panel">
      <h3>AeroViz-4D</h3>

      {/* ── Playback controls ────────────────────────────────────────────── */}
      <section>
        <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
          <button onClick={handlePlayPause}>▶ / ⏸</button>
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
        {(Object.keys(layers) as LayerKey[]).map((key) => (
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
