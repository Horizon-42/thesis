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

/** Properties on each feature in obstacles.geojson (FAA DOF data) */
export interface ObstacleProperties {
  /** FAA OAS number, e.g. "37-000306" */
  oas_number: string;
  /** True if verification status is "O" (verified) */
  verified: boolean;
  /** Two-letter country code, e.g. "US", "CA" */
  country: string;
  /** Two-letter state code, e.g. "NC" (blank for international) */
  state: string;
  /** City name from DOF record */
  city: string;
  /** Obstacle type, e.g. "TOWER", "BLDG", "WINDMILL" */
  obstacle_type: string;
  /** Number of obstacles at this location */
  quantity: number;
  /** Height above ground level in feet */
  agl_ft: number;
  /** Height above mean sea level in feet */
  amsl_ft: number;
  /** Height above ground level in metres */
  agl_m: number;
  /** Height above mean sea level in metres */
  amsl_m: number;
  /** FAA lighting code: R/D/H/M/S/F/C/W/L/N/U */
  lighting: string;
  /** Horizontal accuracy code (1-9) */
  horizontal_accuracy: string;
  /** Vertical accuracy code (A-I) */
  vertical_accuracy: string;
  /** Marking code: P/W/M/F/S/N/U */
  marking: string;
  /** Data source tag */
  source: string;
}

/** One sampled point along a procedure route, used for 4D gates/profile data. */
export interface ProcedureRouteSample {
  sequence: number;
  fixIdent: string;
  legType: string;
  role: string;
  altitudeFt: number | null;
  geometryAltitudeFt: number;
  distanceFromStartM: number;
  timeSeconds: number;
  sourceLine: number;
}

/** Shared properties for CIFP-derived procedure features. */
export interface ProcedureBaseProperties {
  featureType: "procedure-route" | "procedure-fix";
  airport: string;
  procedureType: string;
  procedureIdent: string;
  procedureName: string;
  branch: string;
  runway: string | null;
  routeId: string;
  source: string;
  sourceCycle: string | null;
  researchUseOnly: boolean;
}

/** Properties on each LineString feature in procedures.geojson. */
export interface ProcedureRouteProperties extends ProcedureBaseProperties {
  featureType: "procedure-route";
  nominalSpeedKt: number;
  tunnel: {
    lateralHalfWidthNm: number;
    verticalHalfHeightFt: number;
    sampleSpacingM: number;
  };
  samples: ProcedureRouteSample[];
  warnings: string[];
}

/** Properties on each Point feature in procedures.geojson. */
export interface ProcedureFixProperties extends ProcedureBaseProperties {
  featureType: "procedure-fix";
  name: string;
  sequence: number;
  legType: string;
  role: string;
  altitudeFt: number | null;
  geometryAltitudeFt: number;
  distanceFromStartM: number;
  timeSeconds: number;
  sourceLine: number;
}

export type ProcedureFeatureProperties =
  | ProcedureRouteProperties
  | ProcedureFixProperties;

export interface ProcedureFeature {
  type: "Feature";
  geometry:
    | {
        type: "LineString";
        coordinates: Array<[number, number, number]>;
      }
    | {
        type: "Point";
        coordinates: [number, number, number];
      };
  properties: ProcedureFeatureProperties;
}

export interface ProcedureFeatureCollection {
  type: "FeatureCollection";
  metadata?: Record<string, unknown>;
  features: ProcedureFeature[];
}
