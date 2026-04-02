/**
 * czmlBuilder.ts
 * --------------
 * Pure utility functions for constructing CZML packet objects in TypeScript.
 *
 * Why a separate utility instead of writing CZML directly in Python?
 *   The frontend sometimes needs to generate or patch CZML on-the-fly
 *   (e.g. to add a predicted trajectory before the Python result arrives).
 *   These helpers let you build valid CZML without remembering the exact
 *   array layout every time.
 *
 * These are pure functions (no Cesium import, no side effects) — fully
 * unit-testable in Node.js / Vitest without a browser.
 *
 * 📖 Tutorial: see docs/04-czml-loader.md § "CZML format deep-dive"
 */

import type {
  CzmlDocumentPacket,
  CzmlEntityPacket,
  CzmlSampledPosition,
} from "../types/czml";

// ── Types used as function arguments ─────────────────────────────────────────

/** A single 4D waypoint: time offset in seconds + 3D position */
export interface Waypoint4D {
  /** Seconds elapsed since the epoch (NOT a unix timestamp) */
  offsetSec: number;
  /** Longitude in decimal degrees */
  lon: number;
  /** Latitude in decimal degrees */
  lat: number;
  /** Altitude in metres MSL */
  altM: number;
}

export interface FlightInput {
  /** Unique entity ID — must match `#id.position` in orientationReference */
  id: string;
  /** Human-readable callsign, e.g. "United 123" */
  callsign: string;
  /** ICAO aircraft type code, e.g. "B738" */
  type: string;
  /** 4D waypoints produced by the scheduling algorithm */
  waypoints: Waypoint4D[];
  /** RGBA colour [0-255] for the trail polyline */
  trailColor?: [number, number, number, number];
}

// ── Document packet ───────────────────────────────────────────────────────────

/**
 * Build the mandatory "document" packet (always the first element in CZML).
 *
 * @param startIso  - ISO 8601 UTC start time, e.g. "2026-04-01T08:00:00Z"
 * @param endIso    - ISO 8601 UTC end time
 * @param multiplier - clock speed multiplier (60 = 1 real second = 1 sim minute)
 */
export function buildDocumentPacket(
  startIso: string,
  endIso: string,
  multiplier = 60
): CzmlDocumentPacket {
  return {
    id: "document",
    name: "AeroViz-4D Trajectories",
    version: "1.0",
    clock: {
      interval: `${startIso}/${endIso}`,
      currentTime: startIso,
      multiplier,
      range: "LOOP_STOP",
      step: "SYSTEM_CLOCK_MULTIPLIER",
    },
  };
}

// ── Position array ────────────────────────────────────────────────────────────

/**
 * Convert an array of Waypoint4D into a CZML SampledPositionProperty object.
 *
 * The CZML `cartographicDegrees` field is a flat array:
 *   [t0, lon0, lat0, alt0,  t1, lon1, lat1, alt1, ...]
 *
 * @param epoch    - ISO 8601 reference time (all `offsetSec` are relative to this)
 * @param waypoints - sorted by offsetSec ascending
 */
export function buildSampledPosition(
  epoch: string,
  waypoints: Waypoint4D[]
): CzmlSampledPosition {
  // TODO ① — Build the `cartographicDegrees` flat array.
  //
  // Steps:
  //   1. Create an empty array: const flat: number[] = [];
  //   2. For each waypoint, push four values in this order:
  //        flat.push(wp.offsetSec, wp.lon, wp.lat, wp.altM);
  //   3. Return the CzmlSampledPosition object with:
  //        epoch, cartographicDegrees: flat,
  //        interpolationAlgorithm: "LAGRANGE",
  //        interpolationDegree: 3,
  //        forwardExtrapolationType: "HOLD"
  //
  // Example input: waypoints = [{ offsetSec: 0, lon: -119.38, lat: 49.95, altM: 4500 }]
  // Expected flat: [0, -119.38, 49.95, 4500]
  //
  // Hint: If waypoints is empty, return a position with cartographicDegrees: []
  //       (Cesium will simply not render the entity).

  throw new Error("TODO: implement buildSampledPosition");
}

// ── Entity packet ─────────────────────────────────────────────────────────────

/**
 * Build a full CZML entity packet for one aircraft.
 *
 * This packages together the model, position, orientation, trail, and label
 * into a single CzmlEntityPacket object ready to be serialised as JSON.
 *
 * @param flight   - flight metadata + 4D waypoints
 * @param epochIso - ISO 8601 reference epoch (same as the document packet startTime)
 */
export function buildFlightPacket(
  flight: FlightInput,
  epochIso: string
): CzmlEntityPacket {
  const color = flight.trailColor ?? [255, 165, 0, 200]; // default orange

  // TODO ② — Return a CzmlEntityPacket with these fields:
  //
  //   id:          flight.id
  //   name:        flight.callsign
  //   description: a short HTML string, e.g. `<b>${callsign}</b><br/>Type: ${type}`
  //
  //   model: {
  //     gltf:            "/models/aircraft.glb",
  //     scale:           3.0,
  //     minimumPixelSize: 32,
  //     maximumScale:    20_000,
  //     runAnimations:   true,
  //   }
  //
  //   position: buildSampledPosition(epochIso, flight.waypoints)
  //
  //   orientation: { velocityReference: `#${flight.id}.position` }
  //   (This tells Cesium to compute heading/pitch from the velocity vector automatically)
  //
  //   path: {
  //     show:      true,
  //     leadTime:  0,          // don't draw ahead of current time
  //     trailTime: 300,        // draw 300 s of past trail
  //     width:     2,
  //     material:  { solidColor: { color: { rgba: color } } }
  //   }
  //
  //   label: {
  //     text:          flight.callsign,
  //     font:          "12px sans-serif",
  //     fillColor:     { rgba: [255, 255, 255, 255] },
  //     outlineColor:  { rgba: [0, 0, 0, 255] },
  //     outlineWidth:  2,
  //     style:         "FILL_AND_OUTLINE",
  //     verticalOrigin: "BOTTOM",
  //     pixelOffset:   { cartesian2: [0, -30] }
  //   }

  throw new Error("TODO: implement buildFlightPacket");
}

// ── Top-level builder ─────────────────────────────────────────────────────────

/**
 * Assemble a complete CZML array from a list of flights.
 *
 * @param flights    - array of flight inputs
 * @param startIso   - simulation start (ISO 8601 UTC)
 * @param endIso     - simulation end (ISO 8601 UTC)
 * @param multiplier - clock speed
 * @returns          a CZML array: [documentPacket, ...entityPackets]
 */
export function buildCzml(
  flights: FlightInput[],
  startIso: string,
  endIso: string,
  multiplier = 60
): [CzmlDocumentPacket, ...CzmlEntityPacket[]] {
  // TODO ③ — Compose the result using buildDocumentPacket and buildFlightPacket.
  //
  //   const doc = buildDocumentPacket(startIso, endIso, multiplier);
  //   const entities = flights.map(f => buildFlightPacket(f, startIso));
  //   return [doc, ...entities];

  throw new Error("TODO: implement buildCzml");
}
