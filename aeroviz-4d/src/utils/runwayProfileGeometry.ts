import type { ProcedureRouteViewModel } from "../data/procedureRoutes";
import type { DisplayAltitudeConstraint } from "../data/altitudeConstraints";
import type {
  ProcedureRenderBundle,
  ProcedureSegmentRenderBundle,
} from "../data/procedureRenderBundle";
import type {
  ProcedureProtectionSurface,
  ProtectionSurfaceKind,
} from "../data/procedureProtectionSurfaces";
import type { RunwayProperties } from "../types/geojson-aviation";

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

export interface HorizontalPlateRoutePoint extends RunwayProfilePoint {
  fixIdent: string;
  role: string;
  altitudeConstraint: DisplayAltitudeConstraint | null;
}

export interface HorizontalPlateAssessmentSegment {
  segmentId: string;
  primaryHalfWidthM: number;
  secondaryHalfWidthM: number | null;
  points: RunwayProfilePoint[];
  finalVerticalReference?: HorizontalPlateFinalVerticalReference;
  lnavVnavOcs?: HorizontalPlateLnavVnavOcs;
  precisionSurfaces?: HorizontalPlatePrecisionSurface[];
}

export interface HorizontalPlateFinalVerticalReference {
  kind: "FINAL_VERTICAL_REFERENCE";
  label: string;
  gpaDeg: number;
  tchFt: number | null;
  estimatedFromThreshold: boolean;
  halfWidthM: number;
  points: RunwayProfilePoint[];
}

export interface HorizontalPlateLnavVnavOcs {
  kind: "LNAV_VNAV_OCS";
  label: string;
  gpaDeg: number;
  tchFt: number;
  primaryHalfWidthM: number;
  secondaryHalfWidthM: number | null;
  points: RunwayProfilePoint[];
}

export interface HorizontalPlatePrecisionSurface {
  kind: "PRECISION_SURFACE";
  label: string;
  surfaceType: string;
  gpaDeg: number;
  tchFt: number;
  points: RunwayProfilePoint[];
}

export interface HorizontalPlateRoute {
  routeId: string;
  branchId: string;
  procedureUid?: string;
  branchKey?: string;
  procedureName: string;
  procedureFamily: string;
  procedureIdent: string;
  branchIdent: string;
  transitionIdent: string | null;
  branchType: string;
  defaultVisible: boolean;
  halfWidthM: number;
  points: HorizontalPlateRoutePoint[];
  assessmentSegments?: HorizontalPlateAssessmentSegment[];
  protectionSurfaces?: ProcedureProtectionSurface[];
}

export interface RunwayReferenceMark {
  xM: number;
  yM: number;
  zM: number;
  label: string;
  detail: string;
  priority: number;
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

function isSelectedRnavRunwayFeature(
  props: { runwayIdent?: string | null; procedureFamily?: string; branchType?: string },
  runwayIdent: string,
): boolean {
  if (normalizeRunwayIdent(props.runwayIdent ?? "") !== runwayIdent) {
    return false;
  }
  if (!(props.procedureFamily ?? "UNKNOWN").toUpperCase().startsWith("RNAV")) {
    return false;
  }
  return (props.branchType ?? "final").toLowerCase() !== "missed";
}

function priorityForFixRole(role: string, branchType?: string): number {
  const normalizedRole = role.toUpperCase();
  const rolePriority =
    normalizedRole === "FAF"
      ? 6
      : normalizedRole === "MAPT"
        ? 5
        : normalizedRole === "IF"
          ? 4
          : normalizedRole === "IAF"
            ? 3
            : 2;
  return rolePriority + ((branchType ?? "final").toLowerCase() === "final" ? 1 : 0);
}

function upsertReferenceMark(
  marksByKey: Map<string, RunwayReferenceMark>,
  mark: RunwayReferenceMark,
): void {
  const bucketX = Math.round(mark.xM / 75);
  const key = `${mark.label}|${mark.detail}|${bucketX}`;
  const existing = marksByKey.get(key);

  if (!existing) {
    marksByKey.set(key, mark);
    return;
  }

  marksByKey.set(key, {
    xM: (existing.xM + mark.xM) / 2,
    yM: (existing.yM + mark.yM) / 2,
    zM: (existing.zM + mark.zM) / 2,
    label: mark.label,
    detail: mark.detail,
    priority: Math.max(existing.priority, mark.priority),
  });
}

function buildProjectedRoutePoints(
  route: ProcedureRouteViewModel,
  frame: RunwayFrame,
): HorizontalPlateRoutePoint[] {
  return route.points.map((point) => ({
    ...projectPositionToRunwayFrame(frame, point.lon, point.lat, point.altM),
    fixIdent: point.fixIdent,
    role: point.role,
    altitudeConstraint: point.altitudeConstraint,
  }));
}

function renderBundleBranchKey(packageId: string, branchKey: string): string {
  return `${packageId}:branch:${branchKey.toUpperCase()}`;
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
  procedureRoutes: ProcedureRouteViewModel[],
  frame: RunwayFrame,
  runwayIdent: string,
): HorizontalPlateRoute[] {
  const normalizedRunway = normalizeRunwayIdent(runwayIdent);
  return procedureRoutes
    .filter((route) => isSelectedRnavRunwayFeature(route, normalizedRunway))
    .map((route) => ({
      routeId: route.routeId,
      branchId: renderBundleBranchKey(route.procedureUid, route.branchKey),
      procedureUid: route.procedureUid,
      branchKey: route.branchKey,
      procedureName: route.procedureName,
      procedureIdent: route.procedureIdent,
      procedureFamily: route.procedureFamily ?? "UNKNOWN",
      branchIdent: route.branchIdent,
      transitionIdent: route.transitionIdent ?? null,
      branchType: route.branchType ?? "final",
      defaultVisible: route.defaultVisible,
      halfWidthM: (route.tunnel?.lateralHalfWidthNm ?? 0.3) * 1852,
      points: buildProjectedRoutePoints(route, frame),
    }))
    .filter((route) => route.points.length >= 2);
}

function maxHalfWidthM(samples: Array<{ halfWidthNm: number }> | undefined): number | null {
  if (!samples || samples.length === 0) return null;
  return Math.max(...samples.map((sample) => sample.halfWidthNm)) * 1852;
}

function maxSurfacePrimaryHalfWidthM(surface: ProcedureProtectionSurface | null): number | null {
  const samples = surface?.lateral.widthSamples ?? [];
  if (samples.length === 0) return null;
  return Math.max(...samples.map((sample) => sample.primaryHalfWidthNm)) * 1852;
}

function maxSurfaceSecondaryHalfWidthM(surface: ProcedureProtectionSurface | null): number | null {
  const samples = (surface?.lateral.widthSamples ?? [])
    .map((sample) => sample.secondaryOuterHalfWidthNm)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (samples.length === 0) return null;
  return Math.max(...samples) * 1852;
}

function surfacePoints(
  surface: ProcedureProtectionSurface,
  frame: RunwayFrame,
): RunwayProfilePoint[] {
  return surface.centerline.geoPositions.map((point) =>
    projectPositionToRunwayFrame(frame, point.lonDeg, point.latDeg, point.altM),
  );
}

function protectionSurfaceForSegment(
  surfaces: ProcedureProtectionSurface[],
  segmentId: string,
  kind: ProtectionSurfaceKind,
): ProcedureProtectionSurface | null {
  return surfaces.find((surface) => surface.segmentId === segmentId && surface.kind === kind) ?? null;
}

function protectionSurfacesForSegment(
  surfaces: ProcedureProtectionSurface[],
  segmentId: string,
  kind: ProtectionSurfaceKind,
): ProcedureProtectionSurface[] {
  return surfaces.filter((surface) => surface.segmentId === segmentId && surface.kind === kind);
}

function primaryAssessmentSurface(
  surfaces: ProcedureProtectionSurface[],
  segmentId: string,
): ProcedureProtectionSurface | null {
  return (
    protectionSurfaceForSegment(surfaces, segmentId, "FINAL_LNAV_OEA") ??
    protectionSurfaceForSegment(surfaces, segmentId, "MISSED_SECTION_1") ??
    protectionSurfaceForSegment(surfaces, segmentId, "MISSED_SECTION_2_STRAIGHT")
  );
}

function precisionSurfaceType(surface: ProcedureProtectionSurface): string {
  return surface.surfaceId.slice(surface.segmentId.length + 1).replace(/-/g, "_").toUpperCase();
}

const FINAL_VERTICAL_REFERENCE_DEFAULT_HALF_WIDTH_NM = 0.15;
const FINAL_VERTICAL_REFERENCE_PROTECTION_WIDTH_RATIO = 0.5;
const MIN_PROFILE_ASSESSMENT_SEGMENT_LENGTH_M = 30;

// Mirrors the 3D procedure layer's GPA reference construction so the embedded 2D
// profile exposes the same estimated vertical geometry instead of only OCS data.
function isFinalProfileSegment(segmentBundle: ProcedureSegmentRenderBundle): boolean {
  return segmentBundle.segment.segmentType?.startsWith("FINAL") === true;
}

function finalVerticalReferenceGeoPositions(segmentBundle: ProcedureSegmentRenderBundle) {
  if (!isFinalProfileSegment(segmentBundle)) return [];
  const gpaDeg = segmentBundle.segment.verticalRule?.gpaDeg;
  if (typeof gpaDeg !== "number" || !Number.isFinite(gpaDeg) || gpaDeg <= 0) return [];

  if (segmentBundle.lnavVnavOcs?.centerline.geoPositions.length) {
    return segmentBundle.lnavVnavOcs.centerline.geoPositions;
  }

  const centerline = segmentBundle.segmentGeometry.centerline;
  if (centerline.geoPositions.length < 2 || centerline.geodesicLengthNm <= 0) return [];
  const samples =
    segmentBundle.segmentGeometry.stationAxis.samples.length >= 2
      ? segmentBundle.segmentGeometry.stationAxis.samples.map((sample) => ({
          stationNm: sample.stationNm,
          geoPosition: sample.geoPosition,
        }))
      : centerline.geoPositions.map((geoPosition, index) => ({
          stationNm:
            centerline.geodesicLengthNm *
            (index / Math.max(centerline.geoPositions.length - 1, 1)),
          geoPosition,
        }));
  if (samples.length < 2) return [];

  const thresholdSample = samples[samples.length - 1];
  const thresholdElevationFtMsl = thresholdSample.geoPosition.altM / 0.3048;
  const thresholdReferenceAltitudeFtMsl =
    thresholdElevationFtMsl + (segmentBundle.segment.verticalRule?.tchFt ?? 0);
  const gpaRad = (gpaDeg * Math.PI) / 180;
  const totalStationNm = thresholdSample.stationNm;

  return samples.map((sample) => {
    const distanceBeforeThresholdNm = Math.max(0, totalStationNm - sample.stationNm);
    const altitudeFtMsl =
      thresholdReferenceAltitudeFtMsl +
      (Math.tan(gpaRad) * distanceBeforeThresholdNm * 1852) / 0.3048;
    return {
      lonDeg: sample.geoPosition.lonDeg,
      latDeg: sample.geoPosition.latDeg,
      altM: altitudeFtMsl * 0.3048,
    };
  });
}

function finalVerticalReferenceHalfWidthM(segmentBundle: ProcedureSegmentRenderBundle): number {
  const protectionWidthSamples =
    segmentBundle.finalOea?.primary.halfWidthNmSamples ??
    segmentBundle.segmentGeometry.primaryEnvelope?.halfWidthNmSamples ??
    [];
  const maxProtectionHalfWidthNm =
    protectionWidthSamples.length > 0
      ? Math.max(...protectionWidthSamples.map((sample) => sample.halfWidthNm))
      : FINAL_VERTICAL_REFERENCE_DEFAULT_HALF_WIDTH_NM * 2;
  return (
    Math.max(
      FINAL_VERTICAL_REFERENCE_DEFAULT_HALF_WIDTH_NM,
      maxProtectionHalfWidthNm * FINAL_VERTICAL_REFERENCE_PROTECTION_WIDTH_RATIO,
    ) * 1852
  );
}

function profilePolylineLengthM(points: RunwayProfilePoint[]): number {
  let lengthM = 0;
  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1];
    const current = points[index];
    lengthM += Math.hypot(current.xM - previous.xM, current.yM - previous.yM);
  }
  return lengthM;
}

export function attachRenderBundleAssessmentSegments(
  routes: HorizontalPlateRoute[],
  renderBundles: ProcedureRenderBundle[],
  frame: RunwayFrame,
  runwayIdent: string,
): HorizontalPlateRoute[] {
  const normalizedRunway = normalizeRunwayIdent(runwayIdent);
  const segmentsByRouteKey = new Map<string, HorizontalPlateAssessmentSegment[]>();
  const protectionSurfacesByRouteKey = new Map<string, ProcedureProtectionSurface[]>();

  renderBundles.forEach((bundle) => {
    bundle.branchBundles
      .filter((branch) => normalizeRunwayIdent(branch.runwayId ?? "") === normalizedRunway)
      .forEach((branch) => {
        const branchProtectionSurfaces = branch.protectionSurfaces ?? [];
        protectionSurfacesByRouteKey.set(branch.branchId, branchProtectionSurfaces);
        const assessmentSegments = branch.segmentBundles
          .map((segmentBundle): HorizontalPlateAssessmentSegment | null => {
            const centerline = segmentBundle.segmentGeometry.centerline;
            if (centerline.geoPositions.length < 2) return null;
            const points = centerline.geoPositions.map((point) =>
              projectPositionToRunwayFrame(frame, point.lonDeg, point.latDeg, point.altM),
            );
            if (profilePolylineLengthM(points) < MIN_PROFILE_ASSESSMENT_SEGMENT_LENGTH_M) {
              return null;
            }

            const assessmentSurface = primaryAssessmentSurface(
              branchProtectionSurfaces,
              segmentBundle.segment.segmentId,
            );
            const primaryHalfWidthM =
              maxSurfacePrimaryHalfWidthM(assessmentSurface) ??
              maxHalfWidthM(
                segmentBundle.segmentGeometry.primaryEnvelope?.halfWidthNmSamples,
              ) ??
              segmentBundle.segment.xttNm * 2 * 1852;
            const secondaryHalfWidthM =
              maxSurfaceSecondaryHalfWidthM(assessmentSurface) ??
              maxHalfWidthM(
                segmentBundle.segmentGeometry.secondaryEnvelope?.halfWidthNmSamples,
              ) ??
              (segmentBundle.segment.secondaryEnabled ? segmentBundle.segment.xttNm * 3 * 1852 : null);
            const finalVerticalReferenceGeo = finalVerticalReferenceGeoPositions(segmentBundle);
            const gpaDeg = segmentBundle.segment.verticalRule?.gpaDeg;
            const tchFt = segmentBundle.segment.verticalRule?.tchFt;
            const finalVerticalReference =
              finalVerticalReferenceGeo.length >= 2 &&
              typeof gpaDeg === "number" &&
              Number.isFinite(gpaDeg)
                ? {
                    kind: "FINAL_VERTICAL_REFERENCE" as const,
                    label: `GPA ${gpaDeg.toFixed(1)} deg`,
                    gpaDeg,
                    tchFt: typeof tchFt === "number" && Number.isFinite(tchFt) ? tchFt : null,
                    estimatedFromThreshold:
                      segmentBundle.lnavVnavOcs === null &&
                      typeof segmentBundle.segment.verticalRule?.tchFt !== "number",
                    halfWidthM: finalVerticalReferenceHalfWidthM(segmentBundle),
                    points: finalVerticalReferenceGeo.map((point) =>
                      projectPositionToRunwayFrame(frame, point.lonDeg, point.latDeg, point.altM),
                    ),
                  }
                : undefined;
            const lnavVnavSurface = protectionSurfaceForSegment(
              branchProtectionSurfaces,
              segmentBundle.segment.segmentId,
              "FINAL_LNAV_VNAV_OCS",
            );
            const lnavVnavGpaDeg =
              typeof segmentBundle.segment.verticalRule?.gpaDeg === "number"
                ? segmentBundle.segment.verticalRule.gpaDeg
                : segmentBundle.lnavVnavOcs?.verticalProfile.gpaDeg;
            const lnavVnavTchFt =
              typeof segmentBundle.segment.verticalRule?.tchFt === "number"
                ? segmentBundle.segment.verticalRule.tchFt
                : segmentBundle.lnavVnavOcs?.verticalProfile.tchFt;
            const lnavVnavOcs =
              lnavVnavSurface &&
              lnavVnavSurface.centerline.geoPositions.length >= 2 &&
              typeof lnavVnavGpaDeg === "number" &&
              Number.isFinite(lnavVnavGpaDeg) &&
              typeof lnavVnavTchFt === "number" &&
              Number.isFinite(lnavVnavTchFt)
                ? {
                  kind: "LNAV_VNAV_OCS" as const,
                  label: "LNAV/VNAV OCS",
                  gpaDeg: lnavVnavGpaDeg,
                  tchFt: lnavVnavTchFt,
                  primaryHalfWidthM:
                    maxSurfacePrimaryHalfWidthM(lnavVnavSurface) ??
                    primaryHalfWidthM,
                  secondaryHalfWidthM:
                    maxSurfaceSecondaryHalfWidthM(lnavVnavSurface) ??
                    secondaryHalfWidthM,
                  points: surfacePoints(lnavVnavSurface, frame),
                }
                : segmentBundle.lnavVnavOcs
                  ? {
                      kind: "LNAV_VNAV_OCS" as const,
                      label: "LNAV/VNAV OCS",
                      gpaDeg: segmentBundle.lnavVnavOcs.verticalProfile.gpaDeg,
                      tchFt: segmentBundle.lnavVnavOcs.verticalProfile.tchFt,
                      primaryHalfWidthM:
                        maxHalfWidthM(segmentBundle.lnavVnavOcs.primary.halfWidthNmSamples) ??
                        primaryHalfWidthM,
                      secondaryHalfWidthM:
                        maxHalfWidthM(segmentBundle.lnavVnavOcs.secondaryOuter.halfWidthNmSamples) ??
                        secondaryHalfWidthM,
                      points: segmentBundle.lnavVnavOcs.centerline.geoPositions.map((point) =>
                        projectPositionToRunwayFrame(frame, point.lonDeg, point.latDeg, point.altM),
                      ),
                    }
                  : undefined;
            const precisionSurfacesFromProtection = protectionSurfacesForSegment(
              branchProtectionSurfaces,
              segmentBundle.segment.segmentId,
              "FINAL_PRECISION_DEBUG",
            )
              .filter((surface) => surface.centerline.geoPositions.length >= 2)
              .map((surface) => ({
                kind: "PRECISION_SURFACE" as const,
                label: `${precisionSurfaceType(surface).replace(/_/g, " ")} estimate`,
                surfaceType: precisionSurfaceType(surface),
                gpaDeg:
                  typeof segmentBundle.segment.verticalRule?.gpaDeg === "number"
                    ? segmentBundle.segment.verticalRule.gpaDeg
                    : 0,
                tchFt:
                  typeof segmentBundle.segment.verticalRule?.tchFt === "number"
                    ? segmentBundle.segment.verticalRule.tchFt
                    : 0,
                points: surfacePoints(surface, frame),
              }));
            const precisionSurfaces = precisionSurfacesFromProtection.length > 0
              ? precisionSurfacesFromProtection
              : (segmentBundle.precisionFinalSurfaces ?? [])
              .filter((surface) => surface.centerline.geoPositions.length >= 2)
              .map((surface) => ({
                kind: "PRECISION_SURFACE" as const,
                label: `${surface.surfaceType.replace(/_/g, " ")} estimate`,
                surfaceType: surface.surfaceType,
                gpaDeg: surface.verticalProfile.gpaDeg,
                tchFt: surface.verticalProfile.tchFt,
                points: surface.centerline.geoPositions.map((point) =>
                  projectPositionToRunwayFrame(frame, point.lonDeg, point.latDeg, point.altM),
                ),
              }));

            return {
              segmentId: segmentBundle.segment.segmentId,
              primaryHalfWidthM,
              secondaryHalfWidthM,
              points,
              finalVerticalReference,
              lnavVnavOcs,
              precisionSurfaces,
            };
          })
          .filter((segment): segment is HorizontalPlateAssessmentSegment => segment !== null);

        if (assessmentSegments.length === 0) return;
        segmentsByRouteKey.set(branch.branchId, assessmentSegments);
      });
  });

  return routes.map((route) => {
    if (!route.procedureUid || !route.branchKey) return route;
    const routeKey = renderBundleBranchKey(route.procedureUid, route.branchKey);
    const assessmentSegments = segmentsByRouteKey.get(routeKey);
    const protectionSurfaces = protectionSurfacesByRouteKey.get(routeKey);
    if (!assessmentSegments && !protectionSurfaces) return route;
    return {
      ...route,
      ...(assessmentSegments ? { assessmentSegments } : {}),
      ...(protectionSurfaces ? { protectionSurfaces } : {}),
    };
  });
}

export function buildRunwayReferenceMarks(
  procedureRoutes: ProcedureRouteViewModel[],
  frame: RunwayFrame,
  runwayIdent: string,
): RunwayReferenceMark[] {
  const plateRoutes = buildHorizontalPlateRoutes(procedureRoutes, frame, runwayIdent);

  return buildRunwayReferenceMarksFromPlateRoutes(plateRoutes, runwayIdent);
}

export function buildRunwayReferenceMarksFromPlateRoutes(
  plateRoutes: HorizontalPlateRoute[],
  runwayIdent: string,
): RunwayReferenceMark[] {
  const normalizedRunway = normalizeRunwayIdent(runwayIdent);
  const marksByKey = new Map<string, RunwayReferenceMark>();

  plateRoutes.forEach((route) => {
    route.points.forEach((sample) => {
      upsertReferenceMark(marksByKey, {
        xM: sample.xM,
        yM: sample.yM,
        zM: sample.zM,
        label: sample.fixIdent,
        detail: sample.role,
        priority: priorityForFixRole(sample.role, route.branchType),
      });
    });
  });

  const thresholdKey = `${normalizedRunway}|Threshold|0`;
  if (!marksByKey.has(thresholdKey)) {
    marksByKey.set(thresholdKey, {
      xM: 0,
      yM: 0,
      zM: 0,
      label: normalizedRunway,
      detail: "Threshold",
      priority: 10,
    });
  }

  return [...marksByKey.values()].sort((left, right) => {
    if (left.priority === right.priority) return right.xM - left.xM;
    return right.priority - left.priority;
  });
}

export function pointIsInsideHorizontalPlate(
  point: Pick<RunwayProfilePoint, "xM" | "yM">,
  routes: HorizontalPlateRoute[],
): boolean {
  return routes.some((route) => pointInsidePlateRoute(point, route));
}
