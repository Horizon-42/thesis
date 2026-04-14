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
import { DEFAULT_AIRPORT } from "../hooks/useCesiumViewer";

// ── Constants ────────────────────────────────────────────────────────────────
const HDG_STEP   = 15;  // degrees per heading button click
const PITCH_STEP = 10;  // degrees per pitch button click

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
  const { viewer, setSelectedFlightId } = useApp();
  const [cam, setCam] = useState<CamState | null>(null);
  const lastUpdateRef = useRef<number>(0);

  // ── Live readout (throttled to ~10 Hz) ───────────────────────────────────
  useEffect(() => {
    if (!viewer) return;

    const read = () => {
      const now = Date.now();
      if (now - lastUpdateRef.current < 100) return;
      lastUpdateRef.current = now;

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
    read();
    return () => remove();
  }, [viewer]);

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
    if (!viewer) return;
    viewer.trackedEntity = undefined;  // stop following any plane
    setSelectedFlightId(null);         // clear the table selection
    viewer.camera.flyToBoundingSphere(
      new Cesium.BoundingSphere(
        Cesium.Cartesian3.fromDegrees(DEFAULT_AIRPORT.lon, DEFAULT_AIRPORT.lat, 0),
        DEFAULT_AIRPORT.height,
      ),
      {
        duration: 1.5,
        offset: new Cesium.HeadingPitchRange(
          Cesium.Math.toRadians(-45),
          Cesium.Math.toRadians(-42),
          DEFAULT_AIRPORT.height,
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
