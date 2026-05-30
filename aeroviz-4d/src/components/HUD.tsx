/**
 * HUD.tsx
 * -------
 * Top-right camera overlay with two purposes:
 *   1. Readout — live heading, pitch, altitude, lat/lon
 *   2. Controls — buttons to rotate heading, tilt pitch, and zoom
 *
 * Camera state is read from viewer.scene.postRender (throttled to ~10 Hz)
 * so the display stays live without hammering React's reconciler.
 */

import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";

// ── Constants ────────────────────────────────────────────────────────────────
const HDG_STEP   = 15;  // degrees per heading button click
const PITCH_STEP = 10;  // degrees per pitch button click
const SIDE_VIEW_PITCH_DEG = -8;
const TERRAIN_EXAGGERATION_MAX = 20;
const EXAGGERATION_COMMIT_DELAY_MS = 180;

// ── Compass SVG ──────────────────────────────────────────────────────────────
// The outer ring (labels + ticks) is fixed; only the needle rotates.
function CompassRose({ heading }: { heading: number }) {
  const ticks = [0, 45, 90, 135, 180, 225, 270, 315];

  return (
    <svg viewBox="0 0 80 80" width="72" height="72" className="hud-compass-svg">
      {/* Outer ring */}
      <circle
        cx="40" cy="40" r="36"
        fill="rgba(16,20,30,0.6)"
        stroke="#2a3a5a"
        strokeWidth="1.5"
      />

      {/* Tick marks at 45° intervals */}
      {ticks.map((deg) => {
        const rad = (deg - 90) * (Math.PI / 180);
        const isMajor = deg % 90 === 0;
        const r1 = isMajor ? 29 : 31.5;
        return (
          <line
            key={deg}
            x1={40 + r1   * Math.cos(rad)} y1={40 + r1   * Math.sin(rad)}
            x2={40 + 35.5 * Math.cos(rad)} y2={40 + 35.5 * Math.sin(rad)}
            stroke={isMajor ? "#4a6a9a" : "#2a3a5a"}
            strokeWidth={isMajor ? 1.5 : 1}
          />
        );
      })}

      {/* Cardinal labels (fixed — represent absolute directions) */}
      <text x="40" y="11"  textAnchor="middle" fill="#7eb8f7" fontSize="9" fontWeight="700">N</text>
      <text x="40" y="73"  textAnchor="middle" fill="#4a6a8a" fontSize="8">S</text>
      <text x="72" y="43"  textAnchor="middle" fill="#4a6a8a" fontSize="8">E</text>
      <text x="8"  y="43"  textAnchor="middle" fill="#4a6a8a" fontSize="8">W</text>

      {/* Rotating needle — red tip points toward current heading */}
      <g transform={`rotate(${heading}, 40, 40)`}>
        <polygon points="40,13 43.5,40 40,35 36.5,40" fill="#d94f4f" />
        <polygon points="40,67 43.5,40 40,45 36.5,40" fill="#3a5a8a" />
        <circle cx="40" cy="40" r="3.5" fill="#c8d8ec" stroke="#1a2a40" strokeWidth="1" />
      </g>
    </svg>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
interface CamState {
  heading:  number;  // 0–360 degrees
  pitch:    number;  // degrees (negative = looking down)
  altitude: number;  // metres above ellipsoid
  lat:      number;  // decimal degrees
  lon:      number;  // decimal degrees
}

export default function HUD() {
  const { viewer, airport, airportLocalTerrain, setSelectedFlightId } = useApp();
  const [cam, setCam] = useState<CamState | null>(null);
  const [lighting, setLighting] = useState(true);
  const [exaggeration, setExaggeration] = useState(1);
  const [terrainTilesRemaining, setTerrainTilesRemaining] = useState(0);
  const lastUpdateRef = useRef<number>(0);
  const isEditingExaggerationRef = useRef(false);
  const exaggerationCommitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestExaggerationRef = useRef(1);

  // ── Live readout (throttled to ~10 Hz) ───────────────────────────────────
  useEffect(() => {
    if (!viewer) return;

    const clearExaggerationCommitTimer = () => {
      if (!exaggerationCommitTimerRef.current) return;
      clearTimeout(exaggerationCommitTimerRef.current);
      exaggerationCommitTimerRef.current = null;
    };

    const syncSceneControls = () => {
      const nextLighting = viewer.scene.globe.enableLighting;
      const nextExaggeration = viewer.scene.verticalExaggeration;
      setLighting((current) => (current === nextLighting ? current : nextLighting));
      if (!isEditingExaggerationRef.current && !exaggerationCommitTimerRef.current) {
        latestExaggerationRef.current = nextExaggeration;
        setExaggeration((current) => (
          current === nextExaggeration ? current : nextExaggeration
        ));
      }
    };

    syncSceneControls();

    const read = () => {
      const now = Date.now();
      if (now - lastUpdateRef.current < 100) return;
      lastUpdateRef.current = now;
      syncSceneControls();

      const c = viewer.camera;
      let pos: Cesium.Cartographic;
      try {
        pos = Cesium.Cartographic.fromCartesian(c.position);
      } catch {
        return;
      }

      setCam({
        heading:  ((Cesium.Math.toDegrees(c.heading) % 360) + 360) % 360,
        pitch:    Cesium.Math.toDegrees(c.pitch),
        altitude: pos.height,
        lat:      Cesium.Math.toDegrees(pos.latitude),
        lon:      Cesium.Math.toDegrees(pos.longitude),
      });
    };

    const remove = viewer.scene.postRender.addEventListener(read);
    const removeTerrainLoadListener =
      viewer.scene.globe.tileLoadProgressEvent.addEventListener((remainingTiles: number) => {
        setTerrainTilesRemaining(remainingTiles);
      });

    read();
    return () => {
      clearExaggerationCommitTimer();
      removeTerrainLoadListener();
      remove();
    };
  }, [viewer]);

  // ── Scene toggles ────────────────────────────────────────────────────────
  function toggleLighting() {
    if (!viewer) return;
    const next = !viewer.scene.globe.enableLighting;
    viewer.scene.globe.enableLighting = next;
    setLighting(next);
    viewer.scene.requestRender();
  }

  function commitExaggeration(value: number) {
    if (!viewer) return;
    viewer.scene.verticalExaggeration = value;
    setExaggeration(value);
    latestExaggerationRef.current = value;
    isEditingExaggerationRef.current = false;
    viewer.scene.requestRender();
  }

  function scheduleExaggerationCommit(value: number) {
    if (!viewer) return;
    latestExaggerationRef.current = value;
    isEditingExaggerationRef.current = true;
    setExaggeration(value);

    if (exaggerationCommitTimerRef.current) {
      clearTimeout(exaggerationCommitTimerRef.current);
    }

    exaggerationCommitTimerRef.current = setTimeout(() => {
      exaggerationCommitTimerRef.current = null;
      commitExaggeration(latestExaggerationRef.current);
    }, EXAGGERATION_COMMIT_DELAY_MS);
  }

  function commitPendingExaggeration() {
    if (!viewer) return;
    if (exaggerationCommitTimerRef.current) {
      clearTimeout(exaggerationCommitTimerRef.current);
      exaggerationCommitTimerRef.current = null;
    }
    commitExaggeration(latestExaggerationRef.current);
  }

  // ── Camera controls ──────────────────────────────────────────────────────
  function rotateHeading(deltaDeg: number) {
    if (!viewer) return;
    const c = viewer.camera;
    c.flyTo({
      destination: c.position.clone(),
      orientation: {
        heading: c.heading + Cesium.Math.toRadians(deltaDeg),
        pitch:   c.pitch,
        roll:    c.roll,
      },
      duration: 0.35,
      easingFunction: Cesium.EasingFunction.CUBIC_OUT,
    });
  }

  function adjustPitch(deltaDeg: number) {
    if (!viewer) return;
    const c = viewer.camera;
    c.flyTo({
      destination: c.position.clone(),
      orientation: {
        heading: c.heading,
        pitch: Cesium.Math.clamp(
          c.pitch + Cesium.Math.toRadians(deltaDeg),
          Cesium.Math.toRadians(-89),
          Cesium.Math.toRadians(5),
        ),
        roll: c.roll,
      },
      duration: 0.35,
      easingFunction: Cesium.EasingFunction.CUBIC_OUT,
    });
  }

  function cameraFocusPoint(): Cesium.Cartesian3 | null {
    if (!viewer) return null;
    const canvas = viewer.scene.canvas;
    const center = new Cesium.Cartesian2(
      canvas.clientWidth / 2,
      canvas.clientHeight / 2,
    );
    const pickRay = viewer.camera.getPickRay(center);
    if (!pickRay) return null;

    const globePoint = viewer.scene.globe.pick(pickRay, viewer.scene);
    if (globePoint) return globePoint;
    return viewer.camera.pickEllipsoid(center, viewer.scene.globe.ellipsoid) ?? null;
  }

  function sideView() {
    if (!viewer) return;
    const c = viewer.camera;
    const focus = cameraFocusPoint();
    if (!focus) return;

    const range = Cesium.Cartesian3.distance(c.positionWC, focus);
    c.flyToBoundingSphere(
      new Cesium.BoundingSphere(focus, 1),
      {
        duration: 0.45,
        offset: new Cesium.HeadingPitchRange(
          c.heading,
          Cesium.Math.toRadians(SIDE_VIEW_PITCH_DEG),
          range,
        ),
        easingFunction: Cesium.EasingFunction.CUBIC_OUT,
      },
    );
  }

  // Zoom proportionally to current altitude so each click feels consistent
  // at any scale (100 m minimum to avoid floating-point weirdness).
  function zoom(dir: "in" | "out") {
    if (!viewer) return;
    const c = viewer.camera;
    const pos = Cesium.Cartographic.fromCartesian(c.position);
    const amount = Math.max(100, pos.height * 0.35) * (dir === "in" ? 1 : -1);
    const newPosition = Cesium.Cartesian3.add(
      c.position,
      Cesium.Cartesian3.multiplyByScalar(c.direction, amount, new Cesium.Cartesian3()),
      new Cesium.Cartesian3(),
    );
    c.flyTo({
      destination: newPosition,
      orientation: { heading: c.heading, pitch: c.pitch, roll: c.roll },
      duration: 0.35,
      easingFunction: Cesium.EasingFunction.CUBIC_OUT,
    });
  }

  // Fly back to the airport at the same angle used on startup.
  function resetView() {
    if (!viewer || !airport) return;
    viewer.trackedEntity = undefined;  // stop following any plane
    setSelectedFlightId(null);         // clear the table selection
    viewer.camera.flyToBoundingSphere(
      new Cesium.BoundingSphere(
        Cesium.Cartesian3.fromDegrees(airport.lon, airport.lat, 0),
        airport.height,
      ),
      {
        duration: 1.5,
        offset: new Cesium.HeadingPitchRange(
          Cesium.Math.toRadians(-45),
          Cesium.Math.toRadians(-42),
          airport.height,
        ),
      },
    );
  }

  // ── Formatting helpers ────────────────────────────────────────────────────
  function fmtAlt(m: number): string {
    return m >= 9_999
      ? `${(m / 1000).toFixed(1)} km`
      : `${Math.round(m).toLocaleString()} m`;
  }

  function fmtCoord(deg: number, pos: string, neg: string): string {
    return `${Math.abs(deg).toFixed(4)}°\u2009${deg >= 0 ? pos : neg}`;
  }

  if (!cam) return null;

  const hdgLabel = Math.round(cam.heading).toString().padStart(3, "0") + "°";
  const terrainLoadLabel =
    terrainTilesRemaining > 0 ? `Refining ${terrainTilesRemaining}` : "Ready";
  const localTerrainLabel = (() => {
    switch (airportLocalTerrain.status) {
      case "active":
        return airportLocalTerrain.totalTiles > 0 &&
          airportLocalTerrain.loadedTiles < airportLocalTerrain.totalTiles
          ? `Active ${airportLocalTerrain.loadedTiles}/${airportLocalTerrain.totalTiles}`
          : "Active";
      case "preloading":
        return airportLocalTerrain.totalTiles > 0
          ? `Preload ${airportLocalTerrain.loadedTiles}/${airportLocalTerrain.totalTiles}`
          : "Preloading";
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
  })();

  return (
    <div className="hud">
      <h3 className="hud-title">Camera</h3>

      {/* ── Heading ──────────────────────────────────────────────────────── */}
      <div className="hud-section">
        <h4 className="hud-section-label">Heading</h4>
        <div className="hud-compass-row">
          <button
            className="hud-btn hud-arrow-btn"
            onClick={() => rotateHeading(-HDG_STEP)}
            title={`Rotate left ${HDG_STEP}°`}
          >◁</button>

          <div className="hud-compass-wrap">
            <CompassRose heading={cam.heading} />
            <span className="hud-hdg-value">{hdgLabel}</span>
          </div>

          <button
            className="hud-btn hud-arrow-btn"
            onClick={() => rotateHeading(HDG_STEP)}
            title={`Rotate right ${HDG_STEP}°`}
          >▷</button>
        </div>
      </div>

      {/* ── Pitch ────────────────────────────────────────────────────────── */}
      <div className="hud-section">
        <h4 className="hud-section-label">Pitch</h4>
        <div className="hud-ctrl-row">
          <button
            className="hud-btn hud-sm-btn"
            onClick={() => adjustPitch(PITCH_STEP)}
            title={`Tilt up ${PITCH_STEP}°`}
          >▲</button>
          <span className="hud-ctrl-value">{cam.pitch.toFixed(1)}°</span>
          <button
            className="hud-btn hud-sm-btn"
            onClick={() => adjustPitch(-PITCH_STEP)}
            title={`Tilt down ${PITCH_STEP}°`}
          >▼</button>
        </div>
        <button
          className="hud-btn hud-side-view-btn"
          onClick={sideView}
          title="Keep the current focus and switch to a shallow side-view pitch"
        >
          Side View
        </button>
      </div>

      {/* ── Altitude / zoom ──────────────────────────────────────────────── */}
      <div className="hud-section">
        <h4 className="hud-section-label">Altitude</h4>
        <div className="hud-ctrl-row">
          <button
            className="hud-btn hud-sm-btn"
            onClick={() => zoom("out")}
            title="Zoom out"
          >−</button>
          <span className="hud-ctrl-value hud-alt-value">{fmtAlt(cam.altitude)}</span>
          <button
            className="hud-btn hud-sm-btn"
            onClick={() => zoom("in")}
            title="Zoom in"
          >+</button>
        </div>
      </div>

      {/* ── Scene controls ──────────────────────────────────────────────── */}
      <div className="hud-section">
        <h4 className="hud-section-label">Scene</h4>
        <label className="hud-toggle-row" title="Sun-based terrain lighting">
          <input
            type="checkbox"
            checked={lighting}
            onChange={toggleLighting}
          />
          <span className="hud-toggle-label">Lighting</span>
        </label>
        <div className="hud-slider-row">
          <span className="hud-toggle-label">Terrain</span>
          <input
            type="range"
            min="1"
            max={TERRAIN_EXAGGERATION_MAX}
            step="0.5"
            value={exaggeration}
            onPointerDown={() => { isEditingExaggerationRef.current = true; }}
            onPointerUp={commitPendingExaggeration}
            onMouseUp={commitPendingExaggeration}
            onTouchEnd={commitPendingExaggeration}
            onBlur={commitPendingExaggeration}
            onChange={(e) => scheduleExaggerationCommit(Number(e.target.value))}
            className="hud-slider"
            title="Terrain height exaggeration"
          />
          <span className="hud-slider-value">{exaggeration}x</span>
        </div>
        <div className="hud-terrain-status" aria-live="polite">
          <span className="hud-toggle-label">Load</span>
          <span
            className={
              terrainTilesRemaining > 0
                ? "hud-terrain-status-value is-loading"
                : "hud-terrain-status-value"
            }
          >
            {terrainLoadLabel}
          </span>
        </div>
        <div className="hud-terrain-status" aria-live="polite">
          <span className="hud-toggle-label">Local</span>
          <span
            className={
              airportLocalTerrain.status === "preloading" ||
              airportLocalTerrain.status === "loading"
                ? "hud-terrain-status-value is-loading"
                : airportLocalTerrain.status === "active"
                  ? "hud-terrain-status-value is-ready"
                  : airportLocalTerrain.status === "error"
                    ? "hud-terrain-status-value is-error"
                    : "hud-terrain-status-value"
            }
            title={airportLocalTerrain.error ?? airportLocalTerrain.sourceLabel ?? undefined}
          >
            {localTerrainLabel}
          </span>
        </div>
      </div>

      {/* ── Position readout ─────────────────────────────────────────────── */}
      <div className="hud-divider" />
      <div className="hud-readout">
        <div className="hud-readout-row">
          <span className="hud-readout-label">LAT</span>
          <span className="hud-readout-val">{fmtCoord(cam.lat, "N", "S")}</span>
        </div>
        <div className="hud-readout-row">
          <span className="hud-readout-label">LON</span>
          <span className="hud-readout-val">{fmtCoord(cam.lon, "E", "W")}</span>
        </div>
      </div>

      {/* ── Reset ────────────────────────────────────────────────────────── */}
      <button className="hud-btn hud-reset-btn" onClick={resetView}>
        ⌖ Reset View
      </button>
    </div>
  );
}
