import { airportDataUrl } from "./airportData";
import { fetchJson } from "../utils/fetchJson";
import type { RunwayProperties } from "../types/geojson-aviation";
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
  return buildRunwayThresholdTargets(collection);
}

export function buildRunwayThresholdTargets(
  collection: RunwayFeatureCollection,
): RunwayThresholdTarget[] {
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
      return [
        {
          id: normalizeRunwayIdent(feature.properties.le_ident),
          runwayIdent: normalizeRunwayIdent(feature.properties.le_ident),
          runwayPairIdent: feature.properties.runway_ident ?? "",
          lon: centers.le.lon,
          lat: centers.le.lat,
          altM: feature.properties.le_elevation_ft * METRES_PER_FOOT,
          psiDeg: lePsiDeg,
        },
        {
          id: normalizeRunwayIdent(feature.properties.he_ident),
          runwayIdent: normalizeRunwayIdent(feature.properties.he_ident),
          runwayPairIdent: feature.properties.runway_ident ?? "",
          lon: centers.he.lon,
          lat: centers.he.lat,
          altM: feature.properties.he_elevation_ft * METRES_PER_FOOT,
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
