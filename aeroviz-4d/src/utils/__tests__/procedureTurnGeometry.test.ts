import { describe, expect, it } from "vitest";
import { toCartesian, type GeoPoint } from "../procedureGeoMath";
import type { PolylineGeometry3D } from "../procedureSegmentGeometry";
import { buildTfTurnJunctions } from "../procedureTurnGeometry";

const angledGeoPositions: GeoPoint[] = [
  { lonDeg: -78.95, latDeg: 35.8, altM: 900 },
  { lonDeg: -78.9, latDeg: 35.8, altM: 850 },
  { lonDeg: -78.9, latDeg: 35.85, altM: 800 },
];

const angledCenterline: PolylineGeometry3D = {
  geoPositions: angledGeoPositions,
  worldPositions: angledGeoPositions.map(toCartesian),
  geodesicLengthNm: 5.4,
  isArc: false,
};

describe("procedure turn geometry", () => {
  it("builds visual fill patches around a TF turn junction without claiming FB/FO compliance", () => {
    const junctions = buildTfTurnJunctions(
      "segment:initial",
      angledCenterline,
      2,
      3,
      {
        insetNm: 0.4,
        minTurnAngleDeg: 5,
      },
    );

    expect(junctions).toHaveLength(1);
    const junction = junctions[0];
    expect(junction.segmentId).toBe("segment:initial");
    expect(junction.turnPointIndex).toBe(1);
    expect(junction.turnAngleDeg).toBeGreaterThan(80);
    expect(junction.turnAngleDeg).toBeLessThan(100);
    expect(junction.turnDirection).toBe("LEFT");
    expect(junction.constructionStatus).toBe("VISUAL_FILL_ONLY");
    expect(junction.primaryPatch.halfWidthNm).toBe(2);
    expect(junction.secondaryPatch?.halfWidthNm).toBe(3);
    expect(junction.primaryPatch.ribbon.leftGeoBoundary).toHaveLength(3);
    expect(junction.primaryPatch.ribbon.rightGeoBoundary).toHaveLength(3);
  });

  it("does not emit turn patches for nearly straight TF sequences", () => {
    const straightGeoPositions: GeoPoint[] = [
      { lonDeg: -78.95, latDeg: 35.8, altM: 900 },
      { lonDeg: -78.9, latDeg: 35.8, altM: 850 },
      { lonDeg: -78.85, latDeg: 35.8, altM: 800 },
    ];
    const straightCenterline: PolylineGeometry3D = {
      geoPositions: straightGeoPositions,
      worldPositions: straightGeoPositions.map(toCartesian),
      geodesicLengthNm: 5,
      isArc: false,
    };

    expect(
      buildTfTurnJunctions("segment:straight", straightCenterline, 2, 3, {
        minTurnAngleDeg: 5,
      }),
    ).toEqual([]);
  });
});
