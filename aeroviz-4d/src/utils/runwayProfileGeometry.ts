import type {
  ProcedureFeature,
  ProcedureFeatureCollection,
  ProcedureRouteProperties,
  RunwayProperties,
} from "../types/geojson-aviation";

const EARTH_RADIUS_M = 6_378_137;

interface RunwayFeatureProperties extends RunwayProperties {
  zone_type?: string;
  runway_ident?: string;
  le_displaced_threshold_ft?: number;
  he_displaced_threshold_ft?: number;
  lateral_offset_m?: number;
}

interface RunwayPolygonFeature {
  type: "Feature";
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
  properties: RunwayFeatureProperties;
}

export interface RunwayFeatureCollection {
  type: "FeatureCollection";
  features: RunwayPolygonFeature[];
}

export interface RunwayFrame {
  runwayIdent: string;
  thresholdLon: number;
  thresholdLat: number;
  thresholdAltM: number;
  approachUnitEast: number;
  approachUnitNorth: number;
  leftUnitEast: number;
  leftUnitNorth: number;
}

export interface RunwayProfilePoint {
  xM: number;
  yM: number;
  zM: number;
}

export interface HorizontalPlateRoute {
  routeId: string;
  procedureName: string;
  procedureFamily: string;
  branchType: string;
  halfWidthM: number;
  points: RunwayProfilePoint[];
}

function toRadians(value: number): number {
  return (value * Math.PI) / 180;
}

function toDegrees(value: number): number {
  return (value * 180) / Math.PI;
}

function mod360(value: number): number {
  return ((value % 360) + 360) % 360;
}

function angularDifferenceDeg(left: number, right: number): number {
  const delta = Math.abs(mod360(left) - mod360(right));
  return Math.min(delta, 360 - delta);
}

function distance2d(
  left: { east: number; north: number },
  right: { east: number; north: number },
): number {
  return Math.hypot(right.east - left.east, right.north - left.north);
}

function headingFromVectorDeg(east: number, north: number): number {
  return mod360(toDegrees(Math.atan2(east, north)));
}

function parseRunwayHeadingDeg(runwayIdent: string): number | null {
  const normalized = normalizeRunwayIdent(runwayIdent);
  const digits = normalized.replace(/^RW/, "").match(/^\d{1,2}/)?.[0];
  if (!digits) return null;
  const heading = Number.parseInt(digits, 10) * 10;
  return heading === 360 ? 0 : heading;
}

function normalizeUnitVector(east: number, north: number): { east: number; north: number } {
  const length = Math.hypot(east, north);
  if (length === 0) {
    throw new Error("Cannot normalize a zero-length runway vector");
  }
  return {
    east: east / length,
    north: north / length,
  };
}

function pointToEastNorth(
  lon: number,
  lat: number,
  originLon: number,
  originLat: number,
): { east: number; north: number } {
  const dLon = toRadians(lon - originLon);
  const dLat = toRadians(lat - originLat);
  const meanLat = toRadians((lat + originLat) / 2);
  return {
    east: dLon * EARTH_RADIUS_M * Math.cos(meanLat),
    north: dLat * EARTH_RADIUS_M,
  };
}

function uniquePolygonVertices(feature: RunwayPolygonFeature): Array<{ lon: number; lat: number }> {
  const ring = feature.geometry.coordinates[0] ?? [];
  const unique = ring.slice(0, ring.length > 1 ? -1 : ring.length);
  if (unique.length !== 4) {
    throw new Error(
      `Runway polygon for ${feature.properties.runway_ident ?? "unknown"} does not have 4 vertices`,
    );
  }
  return unique.map(([lon, lat]) => ({ lon, lat }));
}

function edgeDataFromPolygon(feature: RunwayPolygonFeature) {
  const vertices = uniquePolygonVertices(feature);
  const origin = vertices[0];
  const eastNorthVertices = vertices.map((vertex) => ({
    ...pointToEastNorth(vertex.lon, vertex.lat, origin.lon, origin.lat),
    lon: vertex.lon,
    lat: vertex.lat,
  }));

  return eastNorthVertices.map((start, index) => {
    const end = eastNorthVertices[(index + 1) % eastNorthVertices.length];
    return {
      start,
      end,
      lengthM: distance2d(start, end),
      center: {
        east: (start.east + end.east) / 2,
        north: (start.north + end.north) / 2,
        lon: (start.lon + end.lon) / 2,
        lat: (start.lat + end.lat) / 2,
      },
    };
  });
}

function selectRunwayThresholdCenters(feature: RunwayPolygonFeature): {
  leCenter: { lon: number; lat: number; east: number; north: number };
  heCenter: { lon: number; lat: number; east: number; north: number };
} {
  const edges = edgeDataFromPolygon(feature)
    .sort((left, right) => left.lengthM - right.lengthM)
    .slice(0, 2);

  if (edges.length !== 2) {
    throw new Error("Could not determine runway threshold edges");
  }

  const [centerA, centerB] = edges.map((edge) => edge.center);
  const leHeading = parseRunwayHeadingDeg(feature.properties.le_ident);
  const heHeading = parseRunwayHeadingDeg(feature.properties.he_ident);
  const headingAB = headingFromVectorDeg(centerB.east - centerA.east, centerB.north - centerA.north);
  const headingBA = headingFromVectorDeg(centerA.east - centerB.east, centerA.north - centerB.north);

  const abScore = Math.min(
    leHeading === null ? Number.POSITIVE_INFINITY : angularDifferenceDeg(headingAB, leHeading),
    heHeading === null ? Number.POSITIVE_INFINITY : angularDifferenceDeg(headingBA, heHeading),
  );
  const baScore = Math.min(
    leHeading === null ? Number.POSITIVE_INFINITY : angularDifferenceDeg(headingBA, leHeading),
    heHeading === null ? Number.POSITIVE_INFINITY : angularDifferenceDeg(headingAB, heHeading),
  );

  return abScore <= baScore
    ? { leCenter: centerA, heCenter: centerB }
    : { leCenter: centerB, heCenter: centerA };
}

function findRunwaySurfaceFeature(
  runwayCollection: RunwayFeatureCollection,
  runwayIdent: string,
): RunwayPolygonFeature {
  const normalizedRunway = normalizeRunwayIdent(runwayIdent);
  const feature = runwayCollection.features.find((candidate) => {
    const candidateProps = candidate.properties;
    if (candidate.geometry.type !== "Polygon") return false;
    if (candidateProps.zone_type !== "runway_surface") return false;
    return (
      normalizeRunwayIdent(candidateProps.le_ident) === normalizedRunway ||
      normalizeRunwayIdent(candidateProps.he_ident) === normalizedRunway
    );
  });

  if (!feature) {
    throw new Error(`Runway surface for ${normalizedRunway} was not found`);
  }

  return feature;
}

function isProcedureRouteFeature(feature: ProcedureFeature): feature is ProcedureFeature & {
  geometry: { type: "LineString"; coordinates: Array<[number, number, number]> };
  properties: ProcedureRouteProperties;
} {
  return (
    feature.geometry.type === "LineString" &&
    feature.properties.featureType === "procedure-route"
  );
}

function pointInsidePlateRoute(
  point: Pick<RunwayProfilePoint, "xM" | "yM">,
  route: HorizontalPlateRoute,
): boolean {
  for (let index = 0; index < route.points.length - 1; index += 1) {
    const start = route.points[index];
    const end = route.points[index + 1];
    const dx = end.xM - start.xM;
    const dy = end.yM - start.yM;
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared === 0) continue;
    const t = Math.max(
      0,
      Math.min(
        1,
        ((point.xM - start.xM) * dx + (point.yM - start.yM) * dy) / lengthSquared,
      ),
    );
    const closestX = start.xM + t * dx;
    const closestY = start.yM + t * dy;
    if (Math.hypot(point.xM - closestX, point.yM - closestY) <= route.halfWidthM) {
      return true;
    }
  }
  return false;
}

export function normalizeRunwayIdent(runwayIdent: string): string {
  const trimmed = runwayIdent.trim().toUpperCase();
  return trimmed.startsWith("RW") ? trimmed : `RW${trimmed}`;
}

export function buildRunwayFrame(
  runwayCollection: RunwayFeatureCollection,
  runwayIdent: string,
): RunwayFrame {
  const normalizedRunway = normalizeRunwayIdent(runwayIdent);
  const feature = findRunwaySurfaceFeature(runwayCollection, normalizedRunway);
  const { leCenter, heCenter } = selectRunwayThresholdCenters(feature);

  const selectedIsLowerEnd = normalizeRunwayIdent(feature.properties.le_ident) === normalizedRunway;
  const threshold = selectedIsLowerEnd ? leCenter : heCenter;
  const opposite = selectedIsLowerEnd ? heCenter : leCenter;
  const thresholdAltM =
    ((selectedIsLowerEnd
      ? feature.properties.le_elevation_ft
      : feature.properties.he_elevation_ft) ??
      0) * 0.3048;

  const runwayInteriorUnit = normalizeUnitVector(
    opposite.east - threshold.east,
    opposite.north - threshold.north,
  );
  const approachUnit = {
    east: -runwayInteriorUnit.east,
    north: -runwayInteriorUnit.north,
  };
  const leftUnit = {
    east: -approachUnit.north,
    north: approachUnit.east,
  };

  return {
    runwayIdent: normalizedRunway,
    thresholdLon: threshold.lon,
    thresholdLat: threshold.lat,
    thresholdAltM,
    approachUnitEast: approachUnit.east,
    approachUnitNorth: approachUnit.north,
    leftUnitEast: leftUnit.east,
    leftUnitNorth: leftUnit.north,
  };
}

export function projectPositionToRunwayFrame(
  frame: RunwayFrame,
  lon: number,
  lat: number,
  altM: number,
): RunwayProfilePoint {
  const local = pointToEastNorth(lon, lat, frame.thresholdLon, frame.thresholdLat);
  return {
    xM: local.east * frame.approachUnitEast + local.north * frame.approachUnitNorth,
    yM: local.east * frame.leftUnitEast + local.north * frame.leftUnitNorth,
    zM: altM - frame.thresholdAltM,
  };
}

export function buildHorizontalPlateRoutes(
  procedureCollection: ProcedureFeatureCollection,
  frame: RunwayFrame,
  runwayIdent: string,
): HorizontalPlateRoute[] {
  const normalizedRunway = normalizeRunwayIdent(runwayIdent);
  return procedureCollection.features
    .filter(isProcedureRouteFeature)
    .filter((feature) => {
      const props = feature.properties;
      if (normalizeRunwayIdent(props.runwayIdent ?? props.runway ?? "") !== normalizedRunway) {
        return false;
      }
      if (!(props.procedureFamily ?? "UNKNOWN").toUpperCase().startsWith("RNAV")) {
        return false;
      }
      return (props.branchType ?? "final").toLowerCase() !== "missed";
    })
    .map((feature) => ({
      routeId: feature.properties.routeId,
      procedureName: feature.properties.procedureName,
      procedureFamily: feature.properties.procedureFamily ?? "UNKNOWN",
      branchType: feature.properties.branchType ?? "final",
      halfWidthM: (feature.properties.tunnel?.lateralHalfWidthNm ?? 0.3) * 1852,
      points: feature.geometry.coordinates.map(([lon, lat, altM]) =>
        projectPositionToRunwayFrame(frame, lon, lat, altM ?? frame.thresholdAltM),
      ),
    }))
    .filter((route) => route.points.length >= 2);
}

export function pointIsInsideHorizontalPlate(
  point: Pick<RunwayProfilePoint, "xM" | "yM">,
  routes: HorizontalPlateRoute[],
): boolean {
  return routes.some((route) => pointInsidePlateRoute(point, route));
}
