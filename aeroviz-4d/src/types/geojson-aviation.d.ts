/**
 * geojson-aviation.d.ts
 * ---------------------
 * TypeScript type definitions for the GeoJSON feature properties used
 * by this project.  These are NOT standard GeoJSON types — they are the
 * custom `properties` objects attached to each Feature in our data files.
 *
 * Why bother typing these?
 *   GeoJSON's built-in Feature type uses `properties: Record<string, unknown>`,
 *   which gives you zero type safety.  By narrowing the type at the point
 *   where you read the data, every downstream access is type-checked.
 */

/** Properties on each feature in runway.geojson */
export interface RunwayProperties {
  /** ICAO airport code, e.g. "CYLW" */
  airport_ident: string;
  /** Lower-end runway identifier, e.g. "16" */
  le_ident: string;
  /** Higher-end runway identifier, e.g. "34" */
  he_ident: string;
  /** Total runway length in feet */
  length_ft: number;
  /** Runway width in feet */
  width_ft: number;
  /** Surface material code: "ASP" = asphalt, "CON" = concrete, etc. */
  surface: string;
  /** 1 if the runway has edge lights, 0 otherwise */
  lighted: 0 | 1;
  /** Elevation at the lower-end threshold (feet MSL) */
  le_elevation_ft: number;
  /** Elevation at the higher-end threshold (feet MSL) */
  he_elevation_ft: number;
}

/** Properties on each feature in waypoints.geojson */
export interface WaypointProperties {
  /** Published name, e.g. "KEVOL" */
  name: string;
  /**
   * Role in the approach procedure:
   *   IAF = Initial Approach Fix
   *   IF  = Intermediate Fix
   *   FAF = Final Approach Fix
   *   MAPt = Missed Approach Point
   */
  type: "IAF" | "IF" | "FAF" | "MAPt" | string;
  /** Minimum crossing altitude in feet MSL (may be undefined for advisory waypoints) */
  min_alt_ft?: number;
  /** The procedure this waypoint belongs to, e.g. "RNAV(GNSS) Z RWY 34" */
  procedure: string;
  /** Order in the procedure sequence (1 = first fix after IAF) */
  sequence: number;
}
