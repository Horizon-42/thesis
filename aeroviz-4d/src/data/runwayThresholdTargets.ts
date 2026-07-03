import { airportDataUrl } from "./airportData";
import { fetchJson } from "../utils/fetchJson";
import type { RunwayProperties } from "../types/geojson-aviation";
import {
  procedureDetailsIndexUrl,
  type ProcedureDetailsIndexManifest,
} from "./procedureDetails";
import {
  EARTH_RADIUS_M,
  FEET_TO_METERS as METRES_PER_FOOT,
  toDegrees,
  toRadians,
} from "../utils/procedureGeoMath";
import { normalizeRunwayIdent } from "../utils/runwayIdent";

interface RunwayFeatureProperties extends RunwayProperties {
  zone_type?: string;
  runway_ident?: string;
}

interface RunwayPolygonFeature {
  type: "Feature";
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
  properties: RunwayFeatureProperties;
}

interface RunwayFeatureCollection {
  type: "FeatureCollection";
  features: RunwayPolygonFeature[];
}

export interface RunwayThresholdTarget {
  id: string;
  runwayIdent: string;
  runwayPairIdent: string;
  lon: number;
  lat: number;
  altM: number;
  psiDeg: number;
}

export async function fetchRunwayThresholdTargets(
  airportCode: string,
): Promise<RunwayThresholdTarget[]> {
  const collection = await fetchJson<RunwayFeatureCollection>(
    airportDataUrl(airportCode, "runway.geojson"),
  );
  // CIFP landing thresholds from the procedure-details index (authoritative — the geojson
  // runway_surface edges are PAVEMENT ends, up to ~970 m from the displaced landing threshold).
  // Airports without procedure data fall back to the pavement geometry.
  const index = await fetchJson<ProcedureDetailsIndexManifest>(
    procedureDetailsIndexUrl(airportCode),
  ).catch(() => null);
  return buildRunwayThresholdTargets(collection, index);
}

export function buildRunwayThresholdTargets(
  collection: RunwayFeatureCollection,
  index: ProcedureDetailsIndexManifest | null = null,
): RunwayThresholdTarget[] {
  // The CIFP threshold per runway (the displaced landing threshold). Position + elevation come
  // from here when present; the HEADING stays pavement-derived — the runway axis direction is
  // accurate in the polygon even where its ends are not.
  const cifpByIdent = new Map<
    string,
    { lon: number; lat: number; elevationFt: number | null }
  >();
  for (const runway of index?.runways ?? []) {
    if (runway.threshold) {
      cifpByIdent.set(normalizeRunwayIdent(runway.runwayIdent), runway.threshold);
    }
  }

  return collection.features
    .filter((feature) =>
      feature.geometry.type === "Polygon" &&
      feature.properties.zone_type === "runway_surface"
    )
    .flatMap((feature) => {
      const centers = thresholdCenters(feature);
      if (!centers) return [];

      const lePsiDeg = simulatorPsiDeg(
        centers.le.lon,
        centers.le.lat,
        centers.he.lon,
        centers.he.lat,
      );
      const hePsiDeg = simulatorPsiDeg(
        centers.he.lon,
        centers.he.lat,
        centers.le.lon,
        centers.le.lat,
      );
      const leIdent = normalizeRunwayIdent(feature.properties.le_ident);
      const heIdent = normalizeRunwayIdent(feature.properties.he_ident);
      const leCifp = cifpByIdent.get(leIdent) ?? null;
      const heCifp = cifpByIdent.get(heIdent) ?? null;
      return [
        {
          id: leIdent,
          runwayIdent: leIdent,
          runwayPairIdent: feature.properties.runway_ident ?? "",
          lon: leCifp?.lon ?? centers.le.lon,
          lat: leCifp?.lat ?? centers.le.lat,
          altM: leCifp?.elevationFt != null
            ? leCifp.elevationFt * METRES_PER_FOOT
            : feature.properties.le_elevation_ft * METRES_PER_FOOT,
          psiDeg: lePsiDeg,
        },
        {
          id: heIdent,
          runwayIdent: heIdent,
          runwayPairIdent: feature.properties.runway_ident ?? "",
          lon: heCifp?.lon ?? centers.he.lon,
          lat: heCifp?.lat ?? centers.he.lat,
          altM: heCifp?.elevationFt != null
            ? heCifp.elevationFt * METRES_PER_FOOT
            : feature.properties.he_elevation_ft * METRES_PER_FOOT,
          psiDeg: hePsiDeg,
        },
      ];
    })
    .sort((left, right) => left.runwayIdent.localeCompare(right.runwayIdent));
}

function thresholdCenters(feature: RunwayPolygonFeature): {
  le: { lon: number; lat: number };
  he: { lon: number; lat: number };
} | null {
  const ring = feature.geometry.coordinates[0] ?? [];
  if (ring.length < 4) return null;

  const [leLeft, leRight, heRight, heLeft] = ring;
  if (!leLeft || !leRight || !heRight || !heLeft) return null;

  return {
    le: midpoint(leLeft, leRight),
    he: midpoint(heLeft, heRight),
  };
}

function midpoint(
  left: number[],
  right: number[],
): { lon: number; lat: number } {
  return {
    lon: (left[0] + right[0]) / 2,
    lat: (left[1] + right[1]) / 2,
  };
}

function simulatorPsiDeg(
  fromLon: number,
  fromLat: number,
  toLon: number,
  toLat: number,
): number {
  const meanLat = toRadians((fromLat + toLat) / 2);
  const east = toRadians(toLon - fromLon) * EARTH_RADIUS_M * Math.cos(meanLat);
  const north = toRadians(toLat - fromLat) * EARTH_RADIUS_M;
  return normalizeDegrees(toDegrees(Math.atan2(north, east)));
}

function normalizeDegrees(value: number): number {
  return ((value % 360) + 360) % 360;
}
