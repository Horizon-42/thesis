/**
 * HUD.tsx
 * -------
 * Top-right camera overlay:
 *   1. Drag controller — a miniature ground disc seen from the camera: drag it sideways to turn the
 *      heading, up/down to tilt the pitch (both orbit the point at screen centre); the strip beside it
 *      zooms (drag up = in, springs back; the wheel works over both). Double-click the disc = north up.
 *   2. Readout — heading, pitch, altitude, lat/lon; scene toggles and terrain status.
 *
 * Camera state is read from viewer.scene.postRender (throttled to ~10 Hz)
 * so the display stays live without hammering React's reconciler.
 */

import { memo, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode, type WheelEvent as ReactWheelEvent } from "react";
import * as Cesium from "cesium";
import {
  useAirportLocalTerrainProgress,
  useApp,
  type AirportLocalTerrainProgress,
  type AirportLocalTerrainState,
} from "../context/AppContext";
import {
  ORBIT_PITCH_MAX_DEG,
  ORBIT_PITCH_MIN_DEG,
  dialFlatten,
  dialPoint,
  orbitDrag,
  quantiseDeg,
  quantiseHeading,
  quantiseReadout,
  sameReadout,
  zoomRange,
  type CameraReadout,
} from "../utils/cameraDial";
import { clamp, toRadians } from "../utils/procedureGeoMath";

// ── Constants ────────────────────────────────────────────────────────────────
const SIDE_VIEW_PITCH_DEG = -8;
const TERRAIN_EXAGGERATION_MAX = 20;
const EXAGGERATION_COMMIT_DELAY_MS = 180;
const DIAL_SIZE = 84;
const DIAL_R = 34;
const ZOOM_STRIP_TRAVEL_PX = 34;
const WHEEL_PX_PER_DELTA = 0.25;
const WHEEL_PX_PER_LINE = 33; // deltaMode 1 (Firefox): deltaY counts lines, not pixels

// ── Drag disc SVG ────────────────────────────────────────────────────────────
// A ground disc as the camera sees it: the compass letters turn with the heading, the disc flattens
// towards the horizon as the pitch rises. The caret at the bottom is the camera, looking up the screen.
const DIAL_LETTERS = [
  { bearing: 0, text: "N" }, { bearing: 90, text: "E" }, { bearing: 180, text: "S" }, { bearing: 270, text: "W" },
];
const DIAL_TICKS = [45, 135, 225, 315];

const OrbitDial = memo(function OrbitDial({ heading, pitch }: { heading: number; pitch: number }) {
  const c = DIAL_SIZE / 2;
  const cy = c - 3;
  const k = dialFlatten(pitch);
  return (
    <svg viewBox={`0 0 ${DIAL_SIZE} ${DIAL_SIZE}`} width={DIAL_SIZE} height={DIAL_SIZE} className="hud-dial-svg">
      <ellipse cx={c} cy={cy} rx={DIAL_R} ry={DIAL_R * k} fill="rgba(16,20,30,0.6)" stroke="#2a3a5a" strokeWidth="1.5" />
      <ellipse cx={c} cy={cy} rx={DIAL_R * 0.5} ry={DIAL_R * 0.5 * k} fill="none" stroke="#22304c" strokeWidth="1" />
      {DIAL_TICKS.map((b) => {
        const p = dialPoint(b, heading, DIAL_R, k);
        return <circle key={b} cx={c + p.x} cy={cy + p.y} r="1.6" fill="#4a6a9a" />;
      })}
      {DIAL_LETTERS.map(({ bearing, text }) => {
        const p = dialPoint(bearing, heading, DIAL_R * 0.74, k);
        const isNorth = bearing === 0;
        return (
          <text
            key={text} x={c + p.x} y={cy + p.y + 3} textAnchor="middle"
            style={{ fill: isNorth ? "#ff6b6b" : "#7f9bbd", opacity: 0.55 + 0.45 * ((p.depth + 1) / 2) }}
            fontSize={isNorth ? 10 : 8.5} fontWeight={isNorth ? 700 : 500}
          >{text}</text>
        );
      })}
      <polygon points={`${c},${DIAL_SIZE - 7} ${c - 4},${DIAL_SIZE - 1} ${c + 4},${DIAL_SIZE - 1}`} fill="#7eb8f7" />
    </svg>
  );
});

// ── Main component ────────────────────────────────────────────────────────────

/** The HUD's "Local" line: the local terrain's phase, and while tiles are still warming, how many of how many. */
export function localTerrainLabel(
  status: AirportLocalTerrainState["status"], { loadedTiles, totalTiles }: AirportLocalTerrainProgress,
): string {
  switch (status) {
    case "active":
      return totalTiles > 0 && loadedTiles < totalTiles ? `Active ${loadedTiles}/${totalTiles}` : "Active";
    case "preloading":
      return totalTiles > 0 ? `Preload ${loadedTiles}/${totalTiles}` : "Preloading";
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

export default function HUD({ children }: { children?: ReactNode } = {}) {
  const { viewer, airport, airportLocalTerrain, setSelectedFlightId } = useApp();
  const terrainProgress = useAirportLocalTerrainProgress();
  const [cam, setCam] = useState<CameraReadout | null>(null);
  const [lighting, setLighting] = useState(true);
  const [exaggeration, setExaggeration] = useState(1);
  const [terrainTilesRemaining, setTerrainTilesRemaining] = useState(0);
  const lastUpdateRef = useRef<number>(0);
  const isEditingExaggerationRef = useRef(false);
  const exaggerationCommitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestExaggerationRef = useRef(1);

  // One drag at a time. The focus and range are read once at pointer-down so the whole drag turns about the
  // same point; heading/pitch are accumulated here (never read back — camera.heading is measured at the
  // camera, the orbit at the focus, and the two differ at long range).
  const dragRef = useRef<{
    kind: "orbit" | "zoom";
    lastX: number;
    lastY: number;
    focus: Cesium.Cartesian3 | null;
    range: number;
    heading: number;
    pitch: number;
  } | null>(null);
  const [zoomThumbPx, setZoomThumbPx] = useState(0);

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

      // While a drag runs the disc shows the drag's own heading/pitch (the camera's read-back differs at long range).
      const dragging = dragRef.current?.kind === "orbit";
      const next = quantiseReadout({
        heading:  ((Cesium.Math.toDegrees(c.heading) % 360) + 360) % 360,
        pitch:    Cesium.Math.toDegrees(c.pitch),
        altitude: pos.height,
        lat:      Cesium.Math.toDegrees(pos.latitude),
        lon:      Cesium.Math.toDegrees(pos.longitude),
      });
      setCam((prev) => {
        const merged = dragging && prev ? { ...next, heading: prev.heading, pitch: prev.pitch } : next;
        return prev && sameReadout(prev, merged) ? prev : merged;
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
  // The point under the screen centre: what a drag orbits and what the range is measured to.
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

  // Distance to the focus; with nothing under the centre (sky) the camera's own height stands in.
  function rangeTo(focus: Cesium.Cartesian3 | null): number {
    const c = viewer!.camera;
    return focus
      ? Cesium.Cartesian3.distance(c.positionWC, focus)
      : Cesium.Cartographic.fromCartesian(c.position).height;
  }

  // Animated re-orientation about the focus, keeping the range (Side view, north up).
  function flyOrient(headingRad: number, pitchRad: number) {
    if (!viewer) return;
    const focus = cameraFocusPoint();
    if (!focus) return;
    viewer.camera.flyToBoundingSphere(
      new Cesium.BoundingSphere(focus, 1),
      {
        duration: 0.45,
        offset: new Cesium.HeadingPitchRange(headingRad, pitchRad, rangeTo(focus)),
        easingFunction: Cesium.EasingFunction.CUBIC_OUT,
      },
    );
  }

  function sideView() {
    if (viewer) flyOrient(viewer.camera.heading, toRadians(SIDE_VIEW_PITCH_DEG));
  }

  function northUp() {
    if (viewer) flyOrient(0, viewer.camera.pitch);
  }

  function startDrag(e: ReactPointerEvent<HTMLElement>, kind: "orbit" | "zoom") {
    if (!viewer || e.button !== 0 || !e.isPrimary) return;
    viewer.camera.cancelFlight(); // a Side / north-up / Reset flight would fight the drag for the camera
    e.currentTarget.setPointerCapture(e.pointerId);
    const focus = cameraFocusPoint();
    dragRef.current = {
      kind,
      lastX: e.clientX,
      lastY: e.clientY,
      focus,
      range: rangeTo(focus),
      heading: Cesium.Math.toDegrees(viewer.camera.heading),
      pitch: clamp(Cesium.Math.toDegrees(viewer.camera.pitch), ORBIT_PITCH_MIN_DEG, ORBIT_PITCH_MAX_DEG),
    };
  }

  function moveDrag(e: ReactPointerEvent<HTMLElement>) {
    const drag = dragRef.current;
    if (!viewer || !drag) return;
    const dx = e.clientX - drag.lastX;
    const dy = e.clientY - drag.lastY;
    drag.lastX = e.clientX;
    drag.lastY = e.clientY;
    const c = viewer.camera;
    if (drag.kind === "orbit") {
      const next = orbitDrag(drag.heading, drag.pitch, dx, dy);
      drag.heading = next.headingDeg;
      drag.pitch = next.pitchDeg;
      const hpr = new Cesium.HeadingPitchRange(toRadians(drag.heading), toRadians(drag.pitch), drag.range);
      if (drag.focus) {
        c.lookAt(drag.focus, hpr);
        c.lookAtTransform(Cesium.Matrix4.IDENTITY);
      } else {
        c.setView({ orientation: { heading: hpr.heading, pitch: hpr.pitch, roll: 0 } });
      }
      // The disc follows the finger now, not the next 10 Hz camera read.
      const heading = quantiseHeading(drag.heading);
      const pitch = quantiseDeg(drag.pitch);
      setCam((prev) => (prev && prev.heading === heading && prev.pitch === pitch ? prev : prev && { ...prev, heading, pitch }));
    } else {
      const nextRange = zoomRange(drag.range, dy);
      c.moveForward(drag.range - nextRange);
      drag.range = nextRange;
      setZoomThumbPx((px) => clamp(px + dy, -ZOOM_STRIP_TRAVEL_PX, ZOOM_STRIP_TRAVEL_PX));
    }
    viewer.scene.requestRender();
  }

  function endDrag() {
    dragRef.current = null;
    setZoomThumbPx(0);
  }

  function wheelZoom(e: ReactWheelEvent<HTMLElement>) {
    if (!viewer) return;
    const range = rangeTo(cameraFocusPoint());
    viewer.camera.moveForward(range - zoomRange(range, e.deltaY * (e.deltaMode === 1 ? WHEEL_PX_PER_LINE : 1) * WHEEL_PX_PER_DELTA));
    viewer.scene.requestRender();
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

  const hdgLabel = (Math.round(cam.heading) % 360).toString().padStart(3, "0") + "°";
  const terrainLoadLabel =
    terrainTilesRemaining > 0 ? `Refining ${terrainTilesRemaining}` : "Ready";

  return (
    <div className="hud">
      <div className="hud-head">
        <h3 className="hud-title">Camera</h3>
        <button
          className="hud-btn hud-chip-btn"
          onClick={sideView}
          title="Keep the current focus and switch to a shallow side-view pitch"
        >Side</button>
        <button
          className="hud-btn hud-chip-btn"
          onClick={resetView}
          title="Fly back to the airport"
        >⌖ Reset</button>
      </div>

      {/* ── Drag controller + readout ─────────────────────────────────────── */}
      <div className="hud-pad-row">
        <div
          className="hud-dial"
          onPointerDown={(e) => startDrag(e, "orbit")}
          onPointerMove={moveDrag}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onLostPointerCapture={endDrag}
          onWheel={wheelZoom}
          onDoubleClick={northUp}
          title="Drag sideways: heading · up/down: pitch · wheel: zoom · double-click: north up"
        >
          <OrbitDial heading={cam.heading} pitch={cam.pitch} />
        </div>
        <div
          className="hud-zoom"
          onPointerDown={(e) => startDrag(e, "zoom")}
          onPointerMove={moveDrag}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onLostPointerCapture={endDrag}
          onWheel={wheelZoom}
          title="Drag up to zoom in, down to zoom out (springs back) · wheel"
        >
          <span className="hud-zoom-sign">+</span>
          <div className="hud-zoom-track">
            <div
              className={zoomThumbPx === 0 ? "hud-zoom-thumb" : "hud-zoom-thumb is-dragging"}
              style={{ transform: `translateY(${zoomThumbPx}px)` }}
            />
          </div>
          <span className="hud-zoom-sign">−</span>
        </div>
        <div className="hud-readout">
          <div className="hud-readout-row">
            <span className="hud-readout-label">HDG</span>
            <span className="hud-readout-val hud-readout-main">{hdgLabel}</span>
          </div>
          <div className="hud-readout-row">
            <span className="hud-readout-label">PIT</span>
            <span className="hud-readout-val">{cam.pitch.toFixed(1)}°</span>
          </div>
          <div className="hud-readout-row">
            <span className="hud-readout-label">ALT</span>
            <span className="hud-readout-val">{fmtAlt(cam.altitude)}</span>
          </div>
        </div>
      </div>
      <div className="hud-position">
        {fmtCoord(cam.lat, "N", "S")}  {fmtCoord(cam.lon, "E", "W")}
      </div>

      {/* ── Scene controls ──────────────────────────────────────────────── */}
      <div className="hud-divider" />
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
        <label className="hud-toggle-row hud-sun-toggle" title="Sun-based terrain lighting">
          <input type="checkbox" checked={lighting} onChange={toggleLighting} />
          <span className="hud-toggle-label">Sun</span>
        </label>
      </div>
      <div className="hud-terrain-status" aria-live="polite">
        <span
          className={
            terrainTilesRemaining > 0
              ? "hud-terrain-status-value is-loading"
              : "hud-terrain-status-value"
          }
          title="Global terrain tiles"
        >
          Tiles {terrainLoadLabel}
        </span>
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
          title={airportLocalTerrain.error ?? airportLocalTerrain.sourceLabel ?? "Airport-local terrain"}
        >
          Local {localTerrainLabel(airportLocalTerrain.status, terrainProgress)}
        </span>
      </div>
      {children}
    </div>
  );
}
