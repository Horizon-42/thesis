/**
 * ocsGeometry.ts
 * --------------
 * Pure utility functions for computing the 3D geometry of PANS-OPS
 * Obstacle Clearance Surfaces (OCS) from approach procedure parameters.
 *
 * These functions do NOT touch Cesium — they work purely with numbers and
 * arrays so they are easy to unit-test without a browser or WebGL context.
 * The caller (useOcsLayer hook) converts the output into Cesium entities.
 *
 * Coordinate convention used throughout this file:
 *   lon  = longitude in decimal degrees (WGS84)
 *   lat  = latitude  in decimal degrees (WGS84)
 *   altM = altitude  in metres MSL (above WGS84 ellipsoid)
 *
 * 📖 Tutorial: see docs/03-ocs-geometry.md
 */

// ── Constants ────────────────────────────────────────────────────────────────

/** Metres per degree of latitude (approximately constant globally) */
const METRES_PER_DEG_LAT = 111_320;

/** Metres per degree of longitude at a given latitude */
function metresPerDegLon(latDeg: number): number {
  return METRES_PER_DEG_LAT * Math.cos((latDeg * Math.PI) / 180);
}

// ── Types ─────────────────────────────────────────────────────────────────────

/** A single point in 3D geographic space */
export interface GeoPoint3D {
  lon: number;
  lat: number;
  altM: number;
}

/** Input parameters for buildFinalApproachOCS */
export interface OCSParams {
  /** Final Approach Fix (FAF) */
  faf: GeoPoint3D;
  /** Runway threshold (landing end) */
  threshold: GeoPoint3D;
  /**
   * Half-width of the PRIMARY protection area (metres).
   * Typical values:
   *   ILS Cat I approach → 75 m each side
   *   RNAV approach      → depends on RNP value, often 185 m (0.1 NM)
   */
  primaryHalfWidthM: number;
  /**
   * Additional width of the SECONDARY protection area beyond the primary edge.
   * The secondary area slopes at 7:1 (7 metres horizontal per 1 metre vertical).
   * Typical value: 75 m (making total protected width 150 m each side)
   */
  secondaryWidthM: number;
}

/** A polygon defined as an ordered array of 3D corner points (not closed — caller closes) */
export type Polygon3D = GeoPoint3D[];

/** Output of buildFinalApproachOCS */
export interface OCSGeometry {
  /** The flat primary protection area (trapezoid, 4 corners) */
  primaryPolygon: Polygon3D;
  /** Left secondary area (triangle/trapezoid, sloping outward at 7:1) */
  secondaryLeft: Polygon3D;
  /** Right secondary area */
  secondaryRight: Polygon3D;
}

// ── Core functions ────────────────────────────────────────────────────────────

/**
 * Compute the bearing FROM point A TO point B in radians (0 = north, clockwise).
 *
 * Formula (forward azimuth on a flat-Earth approximation, valid for short distances):
 *   Δx = (lonB − lonA) × metersPerDegLon(latA)
 *   Δy = (latB − latA) × METRES_PER_DEG_LAT
 *   bearing = atan2(Δx, Δy)
 *
 * @returns bearing in radians, range (−π, π]
 */
export function bearingRad(
  lonA: number, latA: number,
  lonB: number, latB: number
): number {
  // ① — Implement this formula.
  //
  // Step-by-step:
  //   1. dx = (lonB - lonA) * metresPerDegLon(latA)
  //   2. dy = (latB - latA) * METRES_PER_DEG_LAT
  //   3. return Math.atan2(dx, dy)
  //
  // Example:
  //   bearingRad(-119.38, 49.95, -119.38, 49.90)
  //   → approximately -π (pointing south, i.e. negative y direction)
  //
  // Hint: atan2(y, x) but here we want atan2(east_component, north_component)
  // so the argument ORDER is atan2(dx, dy) — east first, north second.

  let dx: number = (lonB - lonA) * metresPerDegLon(latA);
  let dy: number = (latB - latA) * METRES_PER_DEG_LAT;
  return Math.atan2(dx, dy);
}

/**
 * Offset a geographic point by `distanceM` metres in direction `bearingRad`.
 *
 * This is a flat-Earth approximation.  It is accurate to within ~0.1%
 * for distances under 20 km, which is more than enough for approach OCS geometry.
 *
 * Formula:
 *   newLon = lon + (distanceM × sin(bearing)) / metresPerDegLon(lat)
 *   newLat = lat + (distanceM × cos(bearing)) / METRES_PER_DEG_LAT
 */
export function offsetPoint(
  lon: number,
  lat: number,
  altM: number,
  bearingRadians: number,
  distanceM: number
): GeoPoint3D {
  // ② — Implement the flat-Earth offset formula above.
  //
  // Example:
  //   offsetPoint(-119.38, 49.95, 3000, Math.PI / 2, 1000)
  //   → moves 1000 m due EAST
  //   → lon increases by roughly 1000 / metresPerDegLon(49.95) ≈ 0.0135°
  //   → lat stays the same
  //   → altM stays the same

  let newLon: number = lon + (distanceM * Math.sin(bearingRadians)) / metresPerDegLon(lat);
  let newLat: number = lat + (distanceM * Math.cos(bearingRadians)) / METRES_PER_DEG_LAT;
  let newAlt: number = altM;

  return { lon: newLon, lat: newLat, altM: newAlt };
}

/**
 * Build the PANS-OPS final approach segment OCS polygon geometry.
 *
 * Returns three polygons:
 *   1. Primary area — a flat trapezoid between FAF and threshold,
 *      `primaryHalfWidthM` metres each side of the centreline,
 *      at the altitudes of FAF and threshold respectively.
 *
 *   2. Secondary left/right — each is a trapezoid outboard of the primary edge.
 *      The outboard edge is `secondaryWidthM` further out.
 *      The outboard altitude is LOWER by `secondaryWidthM / 7` metres
 *      (the ICAO 7:1 obstacle clearance slope ratio).
 *
 * Why 7:1?
 *   PANS-OPS specifies that the secondary area provides only partial obstacle
 *   clearance, diminishing linearly from full clearance at the primary edge
 *   to zero clearance at the outer edge.  The 7:1 ratio encodes this.
 */
export function buildFinalApproachOCS(params: OCSParams): OCSGeometry {
  const { faf, threshold, primaryHalfWidthM, secondaryWidthM } = params;

  // A. Approach centreline bearing (FAF → threshold).
  const bearing = bearingRad(faf.lon, faf.lat, threshold.lon, threshold.lat);

  // B. Perpendiculars to the centreline.
  const perpLeft = bearing - Math.PI / 2;
  const perpRight = bearing + Math.PI / 2;

  // C. Primary polygon corners — flat slab at FAF/threshold altitudes.
  const fafLeft = offsetPoint(
    faf.lon, faf.lat, faf.altM, perpLeft, primaryHalfWidthM,
  );
  const fafRight = offsetPoint(
    faf.lon, faf.lat, faf.altM, perpRight, primaryHalfWidthM,
  );
  const thrLeft = offsetPoint(
    threshold.lon, threshold.lat, threshold.altM, perpLeft, primaryHalfWidthM,
  );
  const thrRight = offsetPoint(
    threshold.lon, threshold.lat, threshold.altM, perpRight, primaryHalfWidthM,
  );

  // D. Secondary outer corners — drop by secondaryWidthM/7 at FAF,
  // land at threshold elevation at the threshold end.
  const outerAltAtFaf = faf.altM - secondaryWidthM / 7;
  const outerAltAtThr = threshold.altM;
  const outerOffset = primaryHalfWidthM + secondaryWidthM;

  const secFafLeft = offsetPoint(
    faf.lon, faf.lat, outerAltAtFaf, perpLeft, outerOffset,
  );
  const secFafRight = offsetPoint(
    faf.lon, faf.lat, outerAltAtFaf, perpRight, outerOffset,
  );
  const secThrLeft = offsetPoint(
    threshold.lon, threshold.lat, outerAltAtThr, perpLeft, outerOffset,
  );
  const secThrRight = offsetPoint(
    threshold.lon, threshold.lat, outerAltAtThr, perpRight, outerOffset,
  );

  return {
    // Primary: ordered FAF-left → FAF-right → threshold-right → threshold-left
    primaryPolygon: [fafLeft, fafRight, thrRight, thrLeft],
    // Secondary left: inner FAF edge → outer FAF edge → outer threshold edge → inner threshold edge
    secondaryLeft: [fafLeft, secFafLeft, secThrLeft, thrLeft],
    secondaryRight: [fafRight, secFafRight, secThrRight, thrRight],
  };
}
