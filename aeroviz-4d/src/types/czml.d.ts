/**
 * czml.d.ts
 * ---------
 * Minimal TypeScript type definitions for the CZML packet format.
 *
 * What is CZML?
 *   CZML (Cesium Language) is a JSON array where:
 *     - The FIRST element is always the "document" packet (global clock settings).
 *     - Every SUBSEQUENT element describes one entity (aircraft, waypoint, etc.)
 *       and its properties over time.
 *
 * Full spec: https://github.com/AnalyticalGraphicsInc/czml-writer/wiki/CZML-Guide
 *
 * We only type the fields we actually use — a full CZML type would be enormous.
 */

/** The "document" packet — must be the first element in any CZML array */
export interface CzmlDocumentPacket {
  id: "document";
  name: string;
  version: "1.0";
  clock?: {
    /**
     * ISO 8601 interval string: "startISO/endISO"
     * e.g. "2026-04-01T08:00:00Z/2026-04-01T09:00:00Z"
     */
    interval: string;
    currentTime: string;
    /**
     * How many simulation seconds pass per real second.
     * multiplier: 60 means 1 real second = 1 simulation minute.
     */
    multiplier: number;
    range: "UNBOUNDED" | "CLAMPED" | "LOOP_STOP";
    step: "SYSTEM_CLOCK" | "SYSTEM_CLOCK_MULTIPLIER" | "TICK_DEPENDENT";
  };
}

/** A sampled (time-varying) position using cartographic degrees */
export interface CzmlSampledPosition {
  /**
   * The reference epoch (ISO 8601).
   * All time offsets in `cartographicDegrees` are in SECONDS from this epoch.
   */
  epoch: string;
  /**
   * Flat array: [t0, lon0, lat0, alt0,  t1, lon1, lat1, alt1, ...]
   *   t   = seconds since epoch
   *   lon = longitude in decimal degrees
   *   lat = latitude  in decimal degrees
   *   alt = altitude  in METERS above WGS84 ellipsoid
   */
  cartographicDegrees: number[];
  interpolationAlgorithm?: "LINEAR" | "LAGRANGE" | "HERMITE";
  interpolationDegree?: number;
  forwardExtrapolationType?: "NONE" | "HOLD" | "EXTRAPOLATE";
}

/** An entity packet representing one aircraft */
export interface CzmlEntityPacket {
  id: string;
  name?: string;
  description?: string;
  model?: {
    gltf: string;         // URL to .glb file
    scale?: number;
    minimumPixelSize?: number;
    maximumScale?: number;
    runAnimations?: boolean;
  };
  position?: CzmlSampledPosition;
  /** Use velocityReference so Cesium auto-computes heading from the velocity vector */
  orientation?: {
    velocityReference: string; // e.g. "#UAL123.position"
  };
  path?: {
    show?: boolean;
    leadTime?: number;   // seconds ahead to draw the future path
    trailTime?: number;  // seconds behind to draw the past trail
    width?: number;
    material?: {
      solidColor: { color: { rgba: [number, number, number, number] } };
    };
  };
  label?: {
    text: string;
    font?: string;
    fillColor?: { rgba: [number, number, number, number] };
    outlineColor?: { rgba: [number, number, number, number] };
    outlineWidth?: number;
    style?: "FILL" | "OUTLINE" | "FILL_AND_OUTLINE";
    verticalOrigin?: "TOP" | "CENTER" | "BOTTOM";
    pixelOffset?: { cartesian2: [number, number] };
  };
}

/** The complete CZML document: first element is DocumentPacket, rest are EntityPackets */
export type CzmlDocument = [CzmlDocumentPacket, ...CzmlEntityPacket[]];
