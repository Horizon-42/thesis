export const EARTH_RADIUS_M = 6_378_137;
export const FEET_TO_METERS = 0.3048;
export const METERS_PER_NM = 1852;

export interface CartesianPoint {
  x: number;
  y: number;
  z: number;
}

export interface GeoPoint {
  lonDeg: number;
  latDeg: number;
  altM: number;
}

export function toRadians(value: number): number {
  return (value * Math.PI) / 180;
}

export function toDegrees(value: number): number {
  return (value * 180) / Math.PI;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function toCartesian(point: GeoPoint): CartesianPoint {
  const lon = toRadians(point.lonDeg);
  const lat = toRadians(point.latDeg);
  const radius = EARTH_RADIUS_M + point.altM;
  const cosLat = Math.cos(lat);
  return {
    x: radius * cosLat * Math.cos(lon),
    y: radius * cosLat * Math.sin(lon),
    z: radius * Math.sin(lat),
  };
}

export function haversineDistanceM(start: GeoPoint, end: GeoPoint): number {
  const lat1 = toRadians(start.latDeg);
  const lat2 = toRadians(end.latDeg);
  const dLat = toRadians(end.latDeg - start.latDeg);
  const dLon = toRadians(end.lonDeg - start.lonDeg);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function distanceNm(start: GeoPoint, end: GeoPoint): number {
  return haversineDistanceM(start, end) / METERS_PER_NM;
}

export function interpolateGreatCircle(start: GeoPoint, end: GeoPoint, ratio: number): GeoPoint {
  const lat1 = toRadians(start.latDeg);
  const lon1 = toRadians(start.lonDeg);
  const lat2 = toRadians(end.latDeg);
  const lon2 = toRadians(end.lonDeg);
  const centralAngle = haversineDistanceM(start, end) / EARTH_RADIUS_M;

  if (centralAngle < 1e-9) {
    return {
      lonDeg: start.lonDeg + (end.lonDeg - start.lonDeg) * ratio,
      latDeg: start.latDeg + (end.latDeg - start.latDeg) * ratio,
      altM: start.altM + (end.altM - start.altM) * ratio,
    };
  }

  const sinCentral = Math.sin(centralAngle);
  const a = Math.sin((1 - ratio) * centralAngle) / sinCentral;
  const b = Math.sin(ratio * centralAngle) / sinCentral;
  const x = a * Math.cos(lat1) * Math.cos(lon1) + b * Math.cos(lat2) * Math.cos(lon2);
  const y = a * Math.cos(lat1) * Math.sin(lon1) + b * Math.cos(lat2) * Math.sin(lon2);
  const z = a * Math.sin(lat1) + b * Math.sin(lat2);
  const lat = Math.atan2(z, Math.hypot(x, y));
  const lon = Math.atan2(y, x);

  return {
    lonDeg: toDegrees(lon),
    latDeg: toDegrees(lat),
    altM: start.altM + (end.altM - start.altM) * ratio,
  };
}

export function bearingRad(start: GeoPoint, end: GeoPoint): number {
  const lat1 = toRadians(start.latDeg);
  const lat2 = toRadians(end.latDeg);
  const dLon = toRadians(end.lonDeg - start.lonDeg);
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x =
    Math.cos(lat1) * Math.sin(lat2) -
    Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return Math.atan2(y, x);
}

export function offsetPoint(point: GeoPoint, bearing: number, distanceM: number): GeoPoint {
  const angularDistance = distanceM / EARTH_RADIUS_M;
  const lat1 = toRadians(point.latDeg);
  const lon1 = toRadians(point.lonDeg);
  const lat2 = Math.asin(
    Math.sin(lat1) * Math.cos(angularDistance) +
      Math.cos(lat1) * Math.sin(angularDistance) * Math.cos(bearing),
  );
  const lon2 =
    lon1 +
    Math.atan2(
      Math.sin(bearing) * Math.sin(angularDistance) * Math.cos(lat1),
      Math.cos(angularDistance) - Math.sin(lat1) * Math.sin(lat2),
    );
  return {
    lonDeg: toDegrees(lon2),
    latDeg: toDegrees(lat2),
    altM: point.altM,
  };
}

export function localBearing(points: GeoPoint[], index: number): number {
  if (points.length < 2) return 0;
  if (index === 0) return bearingRad(points[0], points[1]);
  if (index === points.length - 1) {
    return bearingRad(points[index - 1], points[index]);
  }
  return bearingRad(points[index - 1], points[index + 1]);
}
