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
  satelliteImagery: "Satellite Imagery",
  terrain:      "Terrain",
  airportLocalTerrain: "Airport Local Terrain",
  terrainHillshade: "Terrain Hillshade",
  terrainHeightTint: "Terrain Height Tint",
  runways:      "Runways",
  waypoints:    "Waypoints",
  ocsSurfaces:  "Legacy FAF OCS Debug",
  trajectories: "Trajectories",
  obstacles:    "Obstacles",
  obstacleLabels: "Obstacle Labels",
  procedures:   "RNAV Procedures",
};

const ACTIVE_LAYER_KEYS: LayerKey[] = [
  "satelliteImagery",
  "terrain",
  "airportLocalTerrain",
  "terrainHillshade",
  "terrainHeightTint",
  "runways",
  "trajectories",
  "obstacles",
  "obstacleLabels",
  "procedures",
  "ocsSurfaces",
];

function formatTerrainStatus(status: string): string {
  switch (status) {
    case "active":
      return "Active";
    case "preloading":
      return "Preloading";
    case "loading":
      return "Loading";
    case "missing":
      return "Missing";
    case "error":
      return "Error";
    case "disabled":
    default:
      return "Off";
  }
}

function formatTerrainResolution(resolutionM: number | null): string {
  if (resolutionM === null || !Number.isFinite(resolutionM)) return "Pending";

  const precision = resolutionM < 1 ? 3 : resolutionM < 10 ? 2 : 1;
  return `${Number(resolutionM.toFixed(precision)).toLocaleString()} m spacing`;
}

function formatTerrainSource(kind: string | null, name: string | null): string {
  const normalizedKind = kind && kind !== "unknown" ? kind.toUpperCase() : null;
  if (normalizedKind && name) return `${normalizedKind} (${name})`;
  return normalizedKind ?? name ?? "Pending";
}

function formatTerrainCrs(code: string | null): string {
  return code ?? "Pending";
}

export default function ControlPanel() {
  const {
    viewer,
    layers,
    toggleLayer,
    airportLocalTerrain,
    playbackSpeed,
    setPlaybackSpeed,
    airports,
    activeAirportCode,
    setActiveAirportCode,
  } = useApp();
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

      <section>
        <h4>Airport</h4>
        <label className="control-panel-airport-selector">
          <span>Active Airport</span>
          <select
            className="control-panel-airport-selector-input"
            value={activeAirportCode}
            onChange={(event) => setActiveAirportCode(event.target.value)}
            disabled={airports.length === 0}
          >
            {airports.map((airport) => (
              <option key={airport.code} value={airport.code}>
                {airport.code} - {airport.name}
              </option>
            ))}
          </select>
        </label>
      </section>

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
          <label
            key={key}
            className={key === "obstacleLabels" ? "control-panel-layer-toggle-dependent" : undefined}
          >
            <input
              type="checkbox"
              checked={layers[key]}
              onChange={() => toggleLayer(key)}
            />
            {LAYER_LABELS[key]}
          </label>
        ))}
      </section>

      {layers.airportLocalTerrain ? (
        <section className="control-panel-local-terrain" aria-label="Airport local terrain details">
          <h4>Local Terrain</h4>
          <dl className="control-panel-terrain-details">
            <div>
              <dt>Status</dt>
              <dd>{formatTerrainStatus(airportLocalTerrain.status)}</dd>
            </div>
            <div>
              <dt>Source spacing</dt>
              <dd>{formatTerrainResolution(airportLocalTerrain.horizontalResolutionM)}</dd>
            </div>
            <div>
              <dt>Source</dt>
              <dd>{formatTerrainSource(airportLocalTerrain.sourceKind, airportLocalTerrain.sourceName)}</dd>
            </div>
            <div>
              <dt>CRS</dt>
              <dd title={airportLocalTerrain.sourceCrsName ?? undefined}>
                {formatTerrainCrs(airportLocalTerrain.sourceCrsCode)}
              </dd>
            </div>
          </dl>
        </section>
      ) : null}
    </div>
  );
}
