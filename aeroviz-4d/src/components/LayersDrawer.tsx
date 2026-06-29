/**
 * LayersDrawer.tsx
 * ----------------
 * On-demand drawer for the static scene + constraint layer toggles (imagery,
 * terrain family, runways, obstacles, RNAV procedures, OCS debug, range ring) plus
 * the contextual readouts for the layers that have them (range-ring radius, local
 * terrain status). Opened from the workbench top bar; gated on `layersDrawerOpen`.
 *
 * All toggles go through AppContext — this never touches Cesium directly.
 */

import { useApp, type LayerKey } from "../context/AppContext";
import { useEffect, useRef, useState } from "react";

/** Human-readable names for each layer toggle in this drawer. */
const LAYER_LABELS: Record<string, string> = {
  satelliteImagery: "Satellite Imagery",
  terrain: "Terrain",
  airportLocalTerrain: "Airport Local Terrain",
  terrainHillshade: "Terrain Hillshade",
  terrainHeightTint: "Terrain Height Tint",
  runways: "Runways",
  obstacles: "Obstacles",
  obstacleLabels: "Obstacle Labels",
  ocsSurfaces: "Legacy FAF OCS Debug",
  rangeRing: "Range Ring",
};

// `procedures` is intentionally absent — the RNAV procedures master switch lives in
// the Procedures-mode panel itself (ProcedurePanel), not in this layer drawer.
const GEOMETRY_LAYER_KEYS: LayerKey[] = [
  "satelliteImagery",
  "terrain",
  "airportLocalTerrain",
  "terrainHillshade",
  "terrainHeightTint",
  "runways",
  "obstacles",
  "obstacleLabels",
  "ocsSurfaces",
  "rangeRing",
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

export default function LayersDrawer() {
  const {
    layers,
    toggleLayer,
    airportLocalTerrain,
    rangeRingRadiusKm,
    setRangeRingRadiusKm,
    layersDrawerOpen,
    setLayersDrawerOpen,
  } = useApp();

  // The range-ring radius field keeps its own draft string so the user can fully clear
  // it without each keystroke being clamped back to the minimum. We sync the draft from
  // the committed value only when the field isn't focused, and normalize on blur.
  const [rangeRingRadiusDraft, setRangeRingRadiusDraft] = useState<string>(() =>
    formatRangeRingRadiusDraft(rangeRingRadiusKm),
  );
  const rangeRingRadiusFocusedRef = useRef<boolean>(false);

  useEffect(() => {
    if (!rangeRingRadiusFocusedRef.current) {
      setRangeRingRadiusDraft(formatRangeRingRadiusDraft(rangeRingRadiusKm));
    }
  }, [rangeRingRadiusKm]);

  if (!layersDrawerOpen) return null;

  return (
    <aside className="layers-drawer" aria-label="Layers">
      <header className="layers-drawer-header">
        <h3>Layers</h3>
        <button type="button" onClick={() => setLayersDrawerOpen(false)} aria-label="Close layers">
          ✕
        </button>
      </header>

      <section className="layers-drawer-list">
        {GEOMETRY_LAYER_KEYS.map((key) => (
          <label
            key={key}
            className={key === "obstacleLabels" ? "control-panel-layer-toggle-dependent" : undefined}
          >
            <input type="checkbox" checked={layers[key]} onChange={() => toggleLayer(key)} />
            {LAYER_LABELS[key]}
          </label>
        ))}
      </section>

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
    </aside>
  );
}
