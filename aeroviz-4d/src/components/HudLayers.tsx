/**
 * HudLayers.tsx
 * -------------
 * The scene-layer toggles as a collapsible sub-control under the Camera panel (HUD children): one chip per
 * layer, the range-ring radius while the ring is on, a one-line source summary while the local terrain is on.
 * Collapsed by default — the header shows how many layers are on.
 *
 * All toggles go through AppContext — this never touches Cesium directly.
 */

import { useEffect, useRef, useState } from "react";
import { useApp, useRangeRingRadiusKm, type LayerKey } from "../context/AppContext";

// `procedures` is intentionally absent — the RNAV procedures master switch lives in the
// Procedures-mode panel itself (ProcedurePanel), not here.
const LAYER_CHIPS: { key: LayerKey; label: string; title: string }[] = [
  { key: "satelliteImagery", label: "Imagery", title: "Satellite imagery" },
  { key: "terrain", label: "Terrain", title: "Terrain" },
  { key: "airportLocalTerrain", label: "Local terrain", title: "Airport local terrain" },
  { key: "terrainHillshade", label: "Hillshade", title: "Terrain hillshade" },
  { key: "terrainHeightTint", label: "Height tint", title: "Terrain height tint" },
  { key: "runways", label: "Runways", title: "Runways" },
  { key: "obstacles", label: "Obstacles", title: "Obstacles" },
  { key: "obstacleLabels", label: "Labels", title: "Obstacle labels (drawn only while Obstacles is on)" },
  { key: "rangeRing", label: "Range ring", title: "Range ring around the airport" },
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

export default function HudLayers() {
  const { layers, toggleLayer, airportLocalTerrain, setRangeRingRadiusKm } = useApp();
  const rangeRingRadiusKm = useRangeRingRadiusKm();
  const [open, setOpen] = useState(false);

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

  const onCount = LAYER_CHIPS.filter(({ key }) => layers[key]).length;
  const localTerrainSummary = [
    formatTerrainSource(airportLocalTerrain.sourceKind, airportLocalTerrain.sourceName),
    formatTerrainResolution(airportLocalTerrain.horizontalResolutionM),
    airportLocalTerrain.sourceCrsCode ?? "Pending",
  ].join(" · ");

  return (
    <section className="hud-layers" aria-label="Layers">
      <button
        type="button"
        className="hud-layers-head"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <span className="hud-layers-title">Layers</span>
        <span className="hud-layers-count">{onCount}/{LAYER_CHIPS.length}</span>
        <span className="hud-layers-caret" aria-hidden="true">{open ? "▾" : "▸"}</span>
      </button>

      {open ? (
        <>
          <div className="hud-layer-chips">
            {LAYER_CHIPS.map(({ key, label, title }) => (
              <label key={key} className="hud-layer-chip" title={title}>
                <input type="checkbox" checked={layers[key]} onChange={() => toggleLayer(key)} />
                <span>{label}</span>
              </label>
            ))}
          </div>

          {layers.rangeRing ? (
            <div className="hud-layer-ring" aria-label="Range ring radius">
              <span className="hud-toggle-label">Ring</span>
              <input
                type="range"
                className="hud-slider"
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
                className="hud-layer-ring-number"
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
              <span className="hud-toggle-label">km</span>
            </div>
          ) : null}

          {layers.airportLocalTerrain ? (
            <div
              className="hud-layer-terrain-note"
              title={airportLocalTerrain.sourceCrsName ?? localTerrainSummary}
            >
              {localTerrainSummary}
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
