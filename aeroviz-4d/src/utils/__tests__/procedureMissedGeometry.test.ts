import { describe, expect, it } from "vitest";
import type {
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
} from "../../data/procedurePackage";
import type { SegmentGeometryBundle } from "../procedureSegmentGeometry";
import { buildMissedCourseGuides, buildMissedSectionSurface } from "../procedureMissedGeometry";

const missedSegment: ProcedureSegment = {
  segmentId: "segment:missed-s1",
  branchId: "branch:R",
  segmentType: "MISSED_S1",
  navSpec: "RNP_APCH",
  startFixId: "fix:RW",
  endFixId: "fix:MIS1",
  legIds: ["leg:R:040"],
  xttNm: 1,
  attNm: 1,
  secondaryEnabled: true,
  widthChangeMode: "NONE",
  transitionRule: null,
  verticalRule: { kind: "MISSED_CLIMB_SURFACE" },
  constructionFlags: {},
  sourceRefs: [],
  legacy: {
    rawSegmentType: "missed_s1",
    sequenceRange: [40, 40],
  },
};

const envelope = {
  geometryId: "segment:missed-s1:primary",
  envelopeType: "PRIMARY" as const,
  leftBoundary: [],
  rightBoundary: [],
  leftGeoBoundary: [
    { lonDeg: -78.8, latDeg: 35.87, altM: 250 },
    { lonDeg: -78.74, latDeg: 35.91, altM: 550 },
  ],
  rightGeoBoundary: [
    { lonDeg: -78.79, latDeg: 35.87, altM: 250 },
    { lonDeg: -78.73, latDeg: 35.91, altM: 550 },
  ],
  halfWidthNmSamples: [{ stationNm: 0, halfWidthNm: 2 }],
};

const geometryBundle: SegmentGeometryBundle = {
  segmentId: "segment:missed-s1",
  centerline: {
    worldPositions: [],
    geoPositions: [
      { lonDeg: -78.8, latDeg: 35.87, altM: 250 },
      { lonDeg: -78.74, latDeg: 35.91, altM: 550 },
    ],
    geodesicLengthNm: 4,
    isArc: false,
  },
  stationAxis: { samples: [], totalLengthNm: 4 },
  primaryEnvelope: envelope,
  secondaryEnvelope: { ...envelope, geometryId: "segment:missed-s1:secondary", envelopeType: "SECONDARY" },
  turnJunctions: [],
  diagnostics: [],
};

const fixes = new Map<string, ProcedurePackageFix>([
  [
    "fix:RW",
    {
      fixId: "fix:RW",
      ident: "RW",
      role: ["MAP"],
      latDeg: 35.87,
      lonDeg: -78.8,
      altFtMsl: 800,
      annotations: [],
      sourceRefs: [],
    },
  ],
]);

const caLeg: ProcedurePackageLeg = {
  legId: "leg:missed:ca",
  segmentId: "segment:missed-s1",
  legType: "CA",
  rawPathTerminator: "CA",
  startFixId: "fix:RW",
  endFixId: "fix:RW",
  outboundCourseDeg: 305,
  requiredAltitude: { kind: "AT", minFtMsl: 1000, maxFtMsl: 1000, sourceText: "1000 ft" },
  requiredSpeed: null,
  navSpecAtLeg: "RNP_APCH",
  xttNm: 1,
  attNm: 1,
  secondaryEnabled: true,
  notes: [],
  sourceRefs: [],
  legacy: {
    sequence: 35,
    constructionMethod: "course_to_altitude",
    roleAtEnd: "UNKNOWN",
    qualityStatus: "exact",
    renderedInPlanView: true,
  },
};

describe("procedure missed geometry", () => {
  it("wraps missed section one envelopes as an independent surface object", () => {
    const result = buildMissedSectionSurface(missedSegment, geometryBundle);

    expect(result.diagnostics).toEqual([]);
    expect(result.geometry).toMatchObject({
      segmentId: "segment:missed-s1",
      surfaceType: "MISSED_SECTION1_ENVELOPE",
      primary: expect.objectContaining({ geometryId: "segment:missed-s1:primary" }),
      secondaryOuter: expect.objectContaining({ geometryId: "segment:missed-s1:secondary" }),
    });
  });

  it("diagnoses missed sections that have no primary envelope", () => {
    const result = buildMissedSectionSurface(missedSegment, {
      ...geometryBundle,
      primaryEnvelope: undefined,
    });

    expect(result.geometry).toBeNull();
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "SOURCE_INCOMPLETE",
        severity: "WARN",
      }),
    ]);
  });

  it("builds a conservative CA course guide without constructing a termination point", () => {
    const result = buildMissedCourseGuides(missedSegment, [caLeg], fixes);

    expect(result.diagnostics).toEqual([]);
    expect(result.geometries).toHaveLength(1);
    expect(result.geometries[0]).toMatchObject({
      segmentId: "segment:missed-s1",
      legId: "leg:missed:ca",
      legType: "CA",
      startFixId: "fix:RW",
      courseDeg: 305,
      guideLengthNm: 3,
      requiredAltitudeFtMsl: 1000,
      constructionStatus: "COURSE_DIRECTION_ONLY",
    });
    expect(result.geometries[0].geoPositions[0]).toMatchObject({
      lonDeg: -78.8,
      latDeg: 35.87,
    });
    expect(result.geometries[0].geoPositions[1].lonDeg).not.toBeCloseTo(-78.8, 4);
    expect(result.geometries[0].geoPositions[1].latDeg).not.toBeCloseTo(35.87, 4);
  });

  it("diagnoses CA course guides when the course is missing", () => {
    const result = buildMissedCourseGuides(
      missedSegment,
      [{ ...caLeg, outboundCourseDeg: undefined }],
      fixes,
    );

    expect(result.geometries).toEqual([]);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "SOURCE_INCOMPLETE",
        legId: "leg:missed:ca",
        severity: "WARN",
      }),
    ]);
  });
});
