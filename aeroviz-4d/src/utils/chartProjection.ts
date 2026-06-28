/**
 * Project geographic fix positions (lon/lat) onto a published approach chart's
 * page coordinate system.
 *
 * FAA NACO chart PDFs are not georeferenced, so we calibrate: a handful of fixes
 * whose page positions are measured once (see `chartCalibration`) define an
 * affine map from a local equirectangular projection (metres east/north of a
 * reference) to PDF page units. The plan view of an instrument approach is a
 * conformal map at this scale, so an affine (translation + rotation + scale +
 * the small lon/lat anisotropy) reproduces every other fix's page position to
 * within a pixel or two — enough to drop interactive markers exactly on the
 * charted fixes.
 *
 * Page units are PDF.js viewport units at scale 1 (= PDF points). Multiply by
 * the current render scale to get canvas pixels.
 */
import { EARTH_RADIUS_M, toRadians } from "./procedureGeoMath";

export interface LonLat {
  lon: number;
  lat: number;
}

/** A measured correspondence: a named fix at a known page position. */
export interface ChartReferencePoint extends LonLat {
  ident: string;
  pageX: number;
  pageY: number;
}

/** Affine (E,N metres) -> (pageX,pageY): x = a*E + b*N + c, y = d*E + e*N + f. */
interface Affine {
  a: number; b: number; c: number;
  d: number; e: number; f: number;
}

export interface ChartProjection {
  origin: LonLat;
  affine: Affine;
  /** RMS page-unit residual of the fit (sanity / calibration quality). */
  rmsResidual: number;
}

function toLocalMetres(point: LonLat, origin: LonLat): [number, number] {
  const east = toRadians(point.lon - origin.lon) * EARTH_RADIUS_M * Math.cos(toRadians(origin.lat));
  const north = toRadians(point.lat - origin.lat) * EARTH_RADIUS_M;
  return [east, north];
}

/** Solve a 3x3 system M·x = r by Cramer's rule (M is a unit-scaled normal matrix). */
function solve3(m: number[][], r: number[]): [number, number, number] {
  const det = (a: number[][]) =>
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
    a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  const d = det(m);
  // Inputs are centered + scaled to unit RMS, so the non-degenerate determinant
  // is O(1) and collinear points give a determinant near machine epsilon.
  if (Math.abs(d) < 1e-9) throw new Error("chart projection: degenerate calibration (collinear points?)");
  const col = (k: number) => m.map((row, i) => row.map((v, j) => (j === k ? r[i] : v)));
  return [det(col(0)) / d, det(col(1)) / d, det(col(2)) / d];
}

/** Least-squares fit of one output coordinate over normalized inputs (e,n). */
function fitComponent(en: Array<[number, number]>, values: number[]): [number, number, number] {
  let sEE = 0, sEN = 0, sE = 0, sNN = 0, sN = 0, s1 = 0;
  let bE = 0, bN = 0, b1 = 0;
  en.forEach(([e, n], i) => {
    const v = values[i];
    sEE += e * e; sEN += e * n; sE += e; sNN += n * n; sN += n; s1 += 1;
    bE += e * v; bN += n * v; b1 += v;
  });
  const m = [
    [sEE, sEN, sE],
    [sEN, sNN, sN],
    [sE, sN, s1],
  ];
  return solve3(m, [bE, bN, b1]);
}

/**
 * Fit the lon/lat -> page-unit projection from >= 3 non-collinear reference
 * points (more is better; the fit averages out click error). The local metres
 * are centered and scaled to unit RMS before the fit (well-conditioned +
 * meaningful degeneracy test), then the coefficients are de-normalized back to
 * raw (E,N) so {@link projectLonLat} can use lon/lat directly.
 */
export function fitChartProjection(refs: ChartReferencePoint[]): ChartProjection {
  if (refs.length < 3) throw new Error("chart projection needs at least 3 reference points");
  const origin: LonLat = { lon: refs[0].lon, lat: refs[0].lat };
  const en = refs.map((r) => toLocalMetres(r, origin));

  const meanE = en.reduce((s, [e]) => s + e, 0) / en.length;
  const meanN = en.reduce((s, [, n]) => s + n, 0) / en.length;
  const rms = Math.sqrt(
    en.reduce((s, [e, n]) => s + (e - meanE) ** 2 + (n - meanN) ** 2, 0) / en.length,
  );
  if (rms < 1e-6) throw new Error("chart projection: coincident reference points");
  const norm = en.map(([e, n]): [number, number] => [(e - meanE) / rms, (n - meanN) / rms]);

  // De-normalize the fitted coefficients from (e',n') back to raw (E,N):
  // x = a'·(E-meanE)/rms + b'·(N-meanN)/rms + c'.
  const denorm = ([a1, b1, c1]: [number, number, number]): [number, number, number] => [
    a1 / rms,
    b1 / rms,
    c1 - (a1 * meanE + b1 * meanN) / rms,
  ];
  const [a, b, c] = denorm(fitComponent(norm, refs.map((r) => r.pageX)));
  const [d, e, f] = denorm(fitComponent(norm, refs.map((r) => r.pageY)));
  const affine: Affine = { a, b, c, d, e, f };

  let sumSq = 0;
  refs.forEach((r, i) => {
    const [eM, nM] = en[i];
    const dx = a * eM + b * nM + c - r.pageX;
    const dy = d * eM + e * nM + f - r.pageY;
    sumSq += dx * dx + dy * dy;
  });
  return { origin, affine, rmsResidual: Math.sqrt(sumSq / refs.length) };
}

/** Project a geographic point to page units (multiply by render scale for pixels). */
export function projectLonLat(projection: ChartProjection, point: LonLat): { x: number; y: number } {
  const [e, n] = toLocalMetres(point, projection.origin);
  const { a, b, c, d, e: ee, f } = projection.affine;
  return { x: a * e + b * n + c, y: d * e + ee * n + f };
}
