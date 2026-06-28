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

import * as Cesium from "cesium";
import { useApp, type ComparisonKind, type LayerKey } from "../context/AppContext";
import { useLandingsManifest } from "../hooks/useLandingsManifest";
import { useComparisonCategories } from "../hooks/useComparisonCategories";
import { useEffect, useRef, useState } from "react";

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
  rangeRing:    "Range Ring",
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
  "rangeRing",
];

/** The three comparison trajectories, each a colour-keyed visibility checkbox. */
const COMPARISON_KINDS: Array<{ kind: ComparisonKind; label: string; color: string }> = [
  { kind: "reference", label: "Reference", color: "rgb(235, 235, 235)" },
  { kind: "optimizer", label: "Optimizer", color: "rgb(0, 200, 255)" },
  { kind: "simulator", label: "Simulator", color: "rgb(255, 140, 0)" },
];

const RANGE_RING_MIN_KM = 1;
const RANGE_RING_MAX_KM = 50;
const RANGE_RING_STEP_KM = 0.5;

function clampRangeRingRadiusKm(value: number): number {
  if (!Number.isFinite(value)) return RANGE_RING_MIN_KM;
  return Math.min(RANGE_RING_MAX_KM, Math.max(RANGE_RING_MIN_KM, value));
}

/** Canonical string shown in the number field (no forced decimals). */
function formatRangeRingRadiusDraft(km: number): string {
  return String(km);
}

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
    autoReplay,
    setAutoReplay,
    airports,
    activeAirportCode,
    setActiveAirportCode,
    selectedRunway,
    setSelectedRunway,
    trajectoryComparison,
    setTrajectoryComparison,
    trajectoryComparisonCategory,
    setTrajectoryComparisonCategory,
    trajectoryComparisonKinds,
    setTrajectoryComparisonKind,
    trajectorySampleCount,
    setTrajectorySampleCount,
    rangeRingRadiusKm,
    setRangeRingRadiusKm,
  } = useApp();
  const { manifest: landingsManifest } = useLandingsManifest(activeAirportCode);
  const { categories: comparisonCategories } = useComparisonCategories(activeAirportCode);

  // Default the comparison category to the first available, and keep it valid as airports change.
  useEffect(() => {
    if (comparisonCategories.length === 0) {
      if (trajectoryComparisonCategory !== null) setTrajectoryComparisonCategory(null);
      return;
    }
    const stillValid = comparisonCategories.some((c) => c.dir === trajectoryComparisonCategory);
    if (!stillValid) setTrajectoryComparisonCategory(comparisonCategories[0].dir);
  }, [comparisonCategories, trajectoryComparisonCategory, setTrajectoryComparisonCategory]);
  const landingRunways = landingsManifest?.runways ?? [];
  const [isAnimating, setIsAnimating] = useState<boolean>(false);

  // The range-ring radius field keeps its own draft string so the user can fully
  // clear it (e.g. to retype 10 → 30) without each keystroke being clamped back to
  // the minimum. We only sync the draft from the committed value when the field
  // isn't being edited, and normalize on blur.
  const [rangeRingRadiusDraft, setRangeRingRadiusDraft] = useState<string>(() =>
    formatRangeRingRadiusDraft(rangeRingRadiusKm),
  );
  const rangeRingRadiusFocusedRef = useRef<boolean>(false);

  useEffect(() => {
    if (!rangeRingRadiusFocusedRef.current) {
      setRangeRingRadiusDraft(formatRangeRingRadiusDraft(rangeRingRadiusKm));
    }
  }, [rangeRingRadiusKm]);

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

  /** Toggle auto-replay: loop at the end vs stop and hold the final state */
  function handleAutoReplayChange(next: boolean) {
    setAutoReplay(next);
    if (!viewer) return;
    viewer.clock.clockRange = next
      ? Cesium.ClockRange.LOOP_STOP
      : Cesium.ClockRange.CLAMPED;
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

        {landingRunways.length > 0 ? (
          <label className="control-panel-airport-selector">
            <span>Landing Runway</span>
            <select
              className="control-panel-airport-selector-input"
              value={selectedRunway ?? ""}
              onChange={(event) => setSelectedRunway(event.target.value || null)}
            >
              <option value="">All runways</option>
              {landingRunways.map((entry) => (
                <option key={entry.runway} value={entry.runway}>
                  {entry.runway} ({entry.count})
                </option>
              ))}
            </select>
          </label>
        ) : null}
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

        <label className="control-panel-auto-replay">
          <input
            type="checkbox"
            checked={autoReplay}
            onChange={(event) => handleAutoReplayChange(event.target.checked)}
          />
          Auto-replay (loop)
        </label>
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

      {layers.trajectories ? (
        <section className="control-panel-trajectory-options" aria-label="Trajectory options">
          <h4>Trajectories</h4>
          <label>
            <input
              type="checkbox"
              checked={trajectoryComparison}
              onChange={(event) => setTrajectoryComparison(event.target.checked)}
            />
            Optimizer comparison (3-colour)
          </label>
          {trajectoryComparison ? (
            comparisonCategories.length > 0 ? (
              <label className="control-panel-airport-selector">
                <span>Optimization category</span>
                <select
                  className="control-panel-airport-selector-input"
                  value={trajectoryComparisonCategory ?? ""}
                  onChange={(event) => setTrajectoryComparisonCategory(event.target.value || null)}
                >
                  {comparisonCategories.map((category) => (
                    <option key={category.key} value={category.dir}>
                      {category.label} ({category.groups})
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <p className="control-panel-comparison-empty">
                No comparison data found for this airport.
              </p>
            )
          ) : null}
          <label className="control-panel-airport-selector">
            <span>Sample count (0 = all)</span>
            <input
              type="number"
              min={0}
              step={1}
              className="control-panel-airport-selector-input"
              value={trajectorySampleCount}
              onChange={(event) => {
                const parsed = Number.parseInt(event.target.value, 10);
                setTrajectorySampleCount(Number.isFinite(parsed) && parsed > 0 ? parsed : 0);
              }}
            />
          </label>
          {trajectoryComparison ? (
            <div className="control-panel-comparison-kinds">
              {COMPARISON_KINDS.map(({ kind, label, color }) => (
                <label key={kind}>
                  <input
                    type="checkbox"
                    checked={trajectoryComparisonKinds[kind]}
                    onChange={(event) => setTrajectoryComparisonKind(kind, event.target.checked)}
                  />
                  <i style={{ background: color }} />
                  {label}
                </label>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}

      {layers.rangeRing ? (
        <section className="control-panel-range-ring" aria-label="Range ring radius">
          <h4>Range Ring Radius</h4>
          <div className="control-panel-range-ring-value">{rangeRingRadiusKm.toFixed(1)} km</div>
          <div className="control-panel-range-ring-inputs">
            <input
              type="range"
              min={RANGE_RING_MIN_KM}
              max={RANGE_RING_MAX_KM}
              step={RANGE_RING_STEP_KM}
              value={rangeRingRadiusKm}
              onChange={(event) =>
                setRangeRingRadiusKm(clampRangeRingRadiusKm(Number(event.target.value)))
              }
            />
            <input
              type="number"
              min={RANGE_RING_MIN_KM}
              max={RANGE_RING_MAX_KM}
              step={RANGE_RING_STEP_KM}
              value={rangeRingRadiusDraft}
              onFocus={() => {
                rangeRingRadiusFocusedRef.current = true;
              }}
              onChange={(event) => {
                const raw = event.target.value;
                setRangeRingRadiusDraft(raw);
                // Let the field sit empty/partial mid-edit; only commit a clamped
                // value once it parses to a finite number.
                if (raw.trim() === "") return;
                const parsed = Number(raw);
                if (Number.isFinite(parsed)) {
                  setRangeRingRadiusKm(clampRangeRingRadiusKm(parsed));
                }
              }}
              onBlur={() => {
                rangeRingRadiusFocusedRef.current = false;
                const next = clampRangeRingRadiusKm(Number(rangeRingRadiusDraft));
                setRangeRingRadiusKm(next);
                setRangeRingRadiusDraft(formatRangeRingRadiusDraft(next));
              }}
            />
          </div>
        </section>
      ) : null}

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
