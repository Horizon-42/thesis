import { describe, expect, it } from "vitest";
import {
  fitChartProjection,
  projectLonLat,
  type ChartReferencePoint,
} from "../chartProjection";

// Three non-collinear fixes near KRDU with arbitrary page positions. With
// exactly three points the affine is exactly determined, so the fit must pass
// through all three (zero residual) and reproject each to its page position.
const REFS: ChartReferencePoint[] = [
  { ident: "A", lon: -79.0, lat: 35.7, pageX: 100, pageY: 300 },
  { ident: "B", lon: -78.9, lat: 35.8, pageX: 200, pageY: 200 },
  { ident: "C", lon: -78.95, lat: 35.65, pageX: 120, pageY: 380 },
];

describe("chartProjection", () => {
  it("fits 3 non-collinear references exactly and reprojects them", () => {
    const projection = fitChartProjection(REFS);
    expect(projection.rmsResidual).toBeLessThan(1e-6);
    for (const ref of REFS) {
      const p = projectLonLat(projection, ref);
      expect(p.x).toBeCloseTo(ref.pageX, 6);
      expect(p.y).toBeCloseTo(ref.pageY, 6);
    }
  });

  it("interpolates a new point monotonically between references", () => {
    const projection = fitChartProjection(REFS);
    // A point east of A and B should project further right than A.
    const east = projectLonLat(projection, { lon: -78.85, lat: 35.8 });
    const atB = projectLonLat(projection, { lon: -78.9, lat: 35.8 });
    expect(east.x).toBeGreaterThan(atB.x);
  });

  it("rejects fewer than 3 references", () => {
    expect(() => fitChartProjection(REFS.slice(0, 2))).toThrow();
  });

  it("throws on a degenerate (collinear) calibration", () => {
    const collinear: ChartReferencePoint[] = [
      { ident: "A", lon: -79.0, lat: 35.7, pageX: 100, pageY: 300 },
      { ident: "B", lon: -78.9, lat: 35.75, pageX: 200, pageY: 250 },
      { ident: "C", lon: -78.8, lat: 35.8, pageX: 300, pageY: 200 },
    ];
    expect(() => fitChartProjection(collinear)).toThrow();
  });
});
