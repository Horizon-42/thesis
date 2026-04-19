/**
 * procedureGeometry.ts
 * --------------------
 * Pure helpers for display-oriented RNAV procedure tunnel geometry.
 *
 * These functions do not depend on Cesium. The hook converts the output into
 * Cesium entities. The tunnel is an approximate visualization volume, not a
 * TERPS/PANS-OPS containment surface.
 */

const METRES_PER_DEG_LAT = 111_320;
const NM_TO_METRES = 1852;
const FEET_TO_METRES = 0.3048;

export const DEFAULT_TUNNEL_HALF_WIDTH_M = 0.3 * NM_TO_METRES;
export const DEFAULT_TUNNEL_HALF_HEIGHT_M = 300 * FEET_TO_METRES;
export const DEFAULT_TUNNEL_SAMPLE_SPACING_M = 250;
export const DEFAULT_NOMINAL_SPEED_KT = 140;

export interface ProcedurePoint3D {
  lon: number;
  lat: number;
  altM: number;
}

export interface TunnelSection {
  center: ProcedurePoint3D;
  leftBottom: ProcedurePoint3D;
  leftTop: ProcedurePoint3D;
  rightBottom: ProcedurePoint3D;
  rightTop: ProcedurePoint3D;
  distanceFromStartM: number;
  timeSeconds: number;
}

export interface BuildTunnelOptions {
  halfWidthM?: number;
  halfHeightM?: number;
  sampleSpacingM?: number;
  nominalSpeedKt?: number;
}

function metresPerDegLon(latDeg: number): number {
  return METRES_PER_DEG_LAT * Math.cos((latDeg * Math.PI) / 180);
}

export function distanceMeters(a: ProcedurePoint3D, b: ProcedurePoint3D): number {
  const radiusM = 6_371_008.8;
  const phi1 = (a.lat * Math.PI) / 180;
  const phi2 = (b.lat * Math.PI) / 180;
  const dPhi = ((b.lat - a.lat) * Math.PI) / 180;
  const dLambda = ((b.lon - a.lon) * Math.PI) / 180;
  const hav =
    Math.sin(dPhi / 2) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2) ** 2;
  return radiusM * 2 * Math.atan2(Math.sqrt(hav), Math.sqrt(1 - hav));
}

export function bearingRad(a: ProcedurePoint3D, b: ProcedurePoint3D): number {
  const dx = (b.lon - a.lon) * metresPerDegLon(a.lat);
  const dy = (b.lat - a.lat) * METRES_PER_DEG_LAT;
  return Math.atan2(dx, dy);
}

export function offsetPoint(
  point: ProcedurePoint3D,
  bearingRadians: number,
  distanceM: number,
  altitudeOffsetM = 0,
): ProcedurePoint3D {
  return {
    lon: point.lon + (distanceM * Math.sin(bearingRadians)) / metresPerDegLon(point.lat),
    lat: point.lat + (distanceM * Math.cos(bearingRadians)) / METRES_PER_DEG_LAT,
    altM: point.altM + altitudeOffsetM,
  };
}

function interpolatePoint(a: ProcedurePoint3D, b: ProcedurePoint3D, t: number): ProcedurePoint3D {
  return {
    lon: a.lon + (b.lon - a.lon) * t,
    lat: a.lat + (b.lat - a.lat) * t,
    altM: a.altM + (b.altM - a.altM) * t,
  };
}

function densifyRoute(
  route: ProcedurePoint3D[],
  sampleSpacingM: number,
): Array<{ point: ProcedurePoint3D; distanceFromStartM: number }> {
  if (route.length === 0) return [];
  if (route.length === 1) return [{ point: route[0], distanceFromStartM: 0 }];

  const samples: Array<{ point: ProcedurePoint3D; distanceFromStartM: number }> = [
    { point: route[0], distanceFromStartM: 0 },
  ];
  let cumulativeM = 0;

  for (let i = 0; i < route.length - 1; i++) {
    const start = route[i];
    const end = route[i + 1];
    const segmentM = distanceMeters(start, end);
    const interiorCount = Math.max(0, Math.floor(segmentM / sampleSpacingM) - 1);

    for (let sampleIndex = 1; sampleIndex <= interiorCount; sampleIndex++) {
      const offsetM = sampleIndex * sampleSpacingM;
      const t = offsetM / segmentM;
      samples.push({
        point: interpolatePoint(start, end, t),
        distanceFromStartM: cumulativeM + offsetM,
      });
    }

    cumulativeM += segmentM;
    samples.push({ point: end, distanceFromStartM: cumulativeM });
  }

  return samples;
}

function localBearing(
  samples: Array<{ point: ProcedurePoint3D; distanceFromStartM: number }>,
  index: number,
): number {
  if (samples.length === 1) return 0;
  if (index === 0) return bearingRad(samples[0].point, samples[1].point);
  if (index === samples.length - 1) {
    return bearingRad(samples[index - 1].point, samples[index].point);
  }
  return bearingRad(samples[index - 1].point, samples[index + 1].point);
}

export function buildTunnelSections(
  route: ProcedurePoint3D[],
  options: BuildTunnelOptions = {},
): TunnelSection[] {
  if (route.length < 2) return [];

  const halfWidthM = options.halfWidthM ?? DEFAULT_TUNNEL_HALF_WIDTH_M;
  const halfHeightM = options.halfHeightM ?? DEFAULT_TUNNEL_HALF_HEIGHT_M;
  const sampleSpacingM = options.sampleSpacingM ?? DEFAULT_TUNNEL_SAMPLE_SPACING_M;
  const nominalSpeedKt = options.nominalSpeedKt ?? DEFAULT_NOMINAL_SPEED_KT;
  const speedMps = (nominalSpeedKt * NM_TO_METRES) / 3600;
  const samples = densifyRoute(route, sampleSpacingM);

  return samples.map((sample, index) => {
    const bearing = localBearing(samples, index);
    const leftBearing = bearing - Math.PI / 2;
    const rightBearing = bearing + Math.PI / 2;
    const left = offsetPoint(sample.point, leftBearing, halfWidthM);
    const right = offsetPoint(sample.point, rightBearing, halfWidthM);

    return {
      center: sample.point,
      leftBottom: { ...left, altM: left.altM - halfHeightM },
      leftTop: { ...left, altM: left.altM + halfHeightM },
      rightBottom: { ...right, altM: right.altM - halfHeightM },
      rightTop: { ...right, altM: right.altM + halfHeightM },
      distanceFromStartM: sample.distanceFromStartM,
      timeSeconds: sample.distanceFromStartM / speedMps,
    };
  });
}
