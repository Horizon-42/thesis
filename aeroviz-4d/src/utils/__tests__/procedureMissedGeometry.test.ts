import { describe, expect, it } from "vitest";
import type {
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
} from "../../data/procedurePackage";
import type { SegmentGeometryBundle } from "../procedureSegmentGeometry";
import {
  buildMissedCaCenterlines,
  buildMissedCaEndpoints,
  buildMissedCaMahfConnectors,
  buildMissedCaSegmentGeometry,
  buildMissedConnectorSurfaces,
  buildMissedCourseGuides,
  buildMissedSectionSurface,
  buildMissedTurnDebugPrimitives,
  buildMissedTurnDebugPoint,
} from "../procedureMissedGeometry";

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
  verticalRule: { kind: "MISSED_CLIMB_SURFACE", climbGradientFtPerNm: 200 },
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
      sectionKind: "SECTION_1",
      constructionStatus: "SOURCE_BACKED",
      lateralWidthRule: {
        ruleId: "MISSED_SEGMENT_XTT_PRIMARY_SECONDARY",
        ruleStatus: "SEGMENT_TOLERANCE_ESTIMATE",
        primaryHalfWidthNm: 2,
        secondaryOuterHalfWidthNm: 2,
        transitionStatus: "SECTION_1_TERMINAL_WIDTH",
      },
      verticalProfile: {
        constructionStatus: "SOURCE_CLIMB_GRADIENT",
        climbGradientFtPerNm: 200,
      },
      primary: expect.objectContaining({ geometryId: "segment:missed-s1:primary" }),
      secondaryOuter: expect.objectContaining({ geometryId: "segment:missed-s1:secondary" }),
    });
  });

  it("diagnoses missed section vertical profiles when climb gradient is missing", () => {
    const result = buildMissedSectionSurface(
      { ...missedSegment, verticalRule: { kind: "MISSED_CLIMB_SURFACE" } },
      geometryBundle,
    );

    expect(result.geometry).toMatchObject({
      verticalProfile: {
        constructionStatus: "CENTERLINE_ALTITUDE_ONLY",
        climbGradientFtPerNm: null,
      },
    });
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "SOURCE_INCOMPLETE",
      }),
    ]);
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

  it("estimates a CA endpoint from course, target altitude, start elevation, and climb gradient", () => {
    const result = buildMissedCaEndpoints(missedSegment, [caLeg], fixes, {
      climbGradientFtPerNm: 250,
    });

    expect(result.diagnostics).toEqual([]);
    expect(result.geometries).toHaveLength(1);
    expect(result.geometries[0]).toMatchObject({
      segmentId: "segment:missed-s1",
      legId: "leg:missed:ca",
      startFixId: "fix:RW",
      courseDeg: 305,
      startAltitudeFtMsl: 800,
      targetAltitudeFtMsl: 1000,
      climbGradientFtPerNm: 250,
      distanceNm: 0.8,
      constructionStatus: "ESTIMATED_ENDPOINT",
    });
    expect(result.geometries[0].geoPositions[0]).toMatchObject({
      lonDeg: -78.8,
      latDeg: 35.87,
      altM: 800 * 0.3048,
    });
    expect(result.geometries[0].geoPositions[1].altM).toBeCloseTo(1000 * 0.3048, 8);
    expect(result.geometries[0].geoPositions[1].lonDeg).not.toBeCloseTo(-78.8, 4);
    expect(result.geometries[0].notes[0]).toContain("explicit climb gradient");
  });

  it("estimates a CA endpoint with the documented default climb model", () => {
    const result = buildMissedCaEndpoints(missedSegment, [caLeg], fixes);

    expect(result.diagnostics).toEqual([]);
    expect(result.geometries[0]).toMatchObject({
      climbGradientFtPerNm: 200,
      distanceNm: 1,
      constructionStatus: "ESTIMATED_ENDPOINT",
    });
    expect(result.geometries[0].notes[0]).toContain("default 200 ft/NM climb model");
  });

  it("diagnoses CA endpoints when the climb model is disabled and no explicit gradient exists", () => {
    const result = buildMissedCaEndpoints(missedSegment, [caLeg], fixes, {
      useDefaultClimbGradient: false,
    });

    expect(result.geometries).toEqual([]);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "CA_ENDPOINT_NOT_CONSTRUCTIBLE",
        legId: "leg:missed:ca",
        severity: "WARN",
      }),
    ]);
  });

  it("diagnoses CA endpoints when required source semantics are missing", () => {
    const result = buildMissedCaEndpoints(
      missedSegment,
      [
        {
          ...caLeg,
          outboundCourseDeg: undefined,
          requiredAltitude: null,
        },
      ],
      fixes,
    );

    expect(result.geometries).toEqual([]);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "CA_ENDPOINT_NOT_CONSTRUCTIBLE",
        message: expect.stringContaining("outbound course"),
      }),
    ]);
  });

  it("builds sampled CA centerlines from estimated endpoints", () => {
    const endpointResult = buildMissedCaEndpoints(missedSegment, [caLeg], fixes, {
      climbGradientFtPerNm: 200,
    });
    const centerlines = buildMissedCaCenterlines(endpointResult.geometries, {
      samplingStepNm: 0.25,
    });

    expect(centerlines).toHaveLength(1);
    expect(centerlines[0]).toMatchObject({
      segmentId: "segment:missed-s1",
      legId: "leg:missed:ca",
      sourceEndpointStatus: "ESTIMATED_ENDPOINT",
      constructionStatus: "ESTIMATED_CENTERLINE",
      isArc: false,
    });
    expect(centerlines[0].geodesicLengthNm).toBeCloseTo(1, 8);
    expect(centerlines[0].geoPositions).toHaveLength(5);
    expect(centerlines[0].geoPositions[0].lonDeg).toBeCloseTo(-78.8, 8);
    expect(centerlines[0].geoPositions[0].latDeg).toBeCloseTo(35.87, 8);
    expect(centerlines[0].geoPositions[centerlines[0].geoPositions.length - 1]).toEqual(
      endpointResult.geometries[0].geoPositions[1],
    );
  });

  it("backfills a CA-only missed surface when a later DF leg has no constructible start fix", () => {
    const dfLeg: ProcedurePackageLeg = {
      ...caLeg,
      legId: "leg:missed:df",
      legType: "DF",
      rawPathTerminator: "DF",
      startFixId: "fix:",
      endFixId: "fix:DUHAM",
      outboundCourseDeg: undefined,
      requiredAltitude: { kind: "AT", minFtMsl: 2200, maxFtMsl: 2200, sourceText: "2200 ft" },
      legacy: {
        ...caLeg.legacy,
        sequence: 45,
        constructionMethod: "direct_to_fix",
        roleAtEnd: "Route",
      },
    };
    const baseGeometry: SegmentGeometryBundle = {
      ...geometryBundle,
      centerline: {
        worldPositions: [],
        geoPositions: [],
        geodesicLengthNm: 0,
        isArc: false,
      },
      stationAxis: { samples: [], totalLengthNm: 0 },
      primaryEnvelope: undefined,
      secondaryEnvelope: undefined,
      diagnostics: [
        {
          severity: "WARN",
          segmentId: missedSegment.segmentId,
          legId: caLeg.legId,
          code: "UNSUPPORTED_LEG_TYPE",
          message: "CA leg is not constructible by the base segment kernel.",
          sourceRefs: [],
        },
        {
          severity: "ERROR",
          segmentId: missedSegment.segmentId,
          legId: dfLeg.legId,
          code: "SOURCE_INCOMPLETE",
          message: "DF leg requires positioned start and end fixes.",
          sourceRefs: [],
        },
      ],
    };
    const endpointResult = buildMissedCaEndpoints(missedSegment, [caLeg], fixes);
    const centerlines = buildMissedCaCenterlines(endpointResult.geometries, {
      samplingStepNm: 0.25,
    });

    const backfillResult = buildMissedCaSegmentGeometry(
      missedSegment,
      [caLeg, dfLeg],
      baseGeometry,
      centerlines,
    );

    expect(backfillResult.backfilled).toBe(true);
    expect(backfillResult.geometry.centerline.geoPositions).toHaveLength(5);
    expect(backfillResult.geometry.primaryEnvelope?.geometryId).toBe(
      "segment:missed-s1:ca-estimated-primary",
    );
    expect(backfillResult.geometry.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "ESTIMATED_CA_GEOMETRY", legId: caLeg.legId }),
        expect.objectContaining({ code: "SOURCE_INCOMPLETE", legId: dfLeg.legId }),
      ]),
    );
    expect(backfillResult.geometry.diagnostics).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "UNSUPPORTED_LEG_TYPE", legId: caLeg.legId }),
      ]),
    );

    const surfaceResult = buildMissedSectionSurface(missedSegment, backfillResult.geometry);
    expect(surfaceResult.geometry).toMatchObject({
      surfaceType: "MISSED_SECTION1_ENVELOPE",
      constructionStatus: "ESTIMATED_CA",
      primary: expect.objectContaining({
        geometryId: "segment:missed-s1:ca-estimated-primary",
      }),
    });
  });

  it("estimates continuity connectors from CA endpoints to later MAHF fixes", () => {
    const endpointResult = buildMissedCaEndpoints(missedSegment, [caLeg], fixes);
    const connectorFixes = new Map(fixes);
    connectorFixes.set("fix:DUHAM", {
      fixId: "fix:DUHAM",
      ident: "DUHAM",
      role: ["MAHF"],
      latDeg: 35.94,
      lonDeg: -78.72,
      altFtMsl: 3000,
      annotations: [],
      sourceRefs: [],
    });
    const mahfLeg: ProcedurePackageLeg = {
      ...caLeg,
      legId: "leg:missed:hm",
      legType: "HM",
      rawPathTerminator: "HM",
      startFixId: "fix:DUHAM",
      endFixId: "fix:DUHAM",
      requiredAltitude: null,
      legacy: {
        ...caLeg.legacy,
        sequence: 50,
        constructionMethod: "hold_to_manual",
        roleAtEnd: "MAHF",
      },
    };

    const connectors = buildMissedCaMahfConnectors(
      endpointResult.geometries,
      [caLeg, mahfLeg],
      connectorFixes,
      { samplingStepNm: 1 },
    );

    expect(connectors).toHaveLength(1);
    expect(connectors[0]).toMatchObject({
      sourceLegId: "leg:missed:ca",
      targetFixId: "fix:DUHAM",
      targetFixIdent: "DUHAM",
      targetFixRole: "MAHF",
      constructionStatus: "ESTIMATED_CONNECTOR",
    });
    expect(connectors[0].geoPositions[0]).toEqual(endpointResult.geometries[0].geoPositions[1]);
    const connectorEnd = connectors[0].geoPositions[connectors[0].geoPositions.length - 1];
    expect(connectorEnd).toMatchObject({
      lonDeg: -78.72,
      latDeg: 35.94,
    });
    expect(connectorEnd.altM).toBeCloseTo(914.4, 8);
  });

  it("builds an estimated connector surface from CA endpoint to a later MAHF fix", () => {
    const endpointResult = buildMissedCaEndpoints(missedSegment, [caLeg], fixes);
    const connectorFixes = new Map(fixes);
    connectorFixes.set("fix:DUHAM", {
      fixId: "fix:DUHAM",
      ident: "DUHAM",
      role: ["MAHF"],
      latDeg: 35.94,
      lonDeg: -78.72,
      altFtMsl: null,
      annotations: [],
      sourceRefs: [],
    });
    const mahfLeg: ProcedurePackageLeg = {
      ...caLeg,
      legId: "leg:missed:hm",
      legType: "HM",
      rawPathTerminator: "HM",
      startFixId: "fix:DUHAM",
      endFixId: "fix:DUHAM",
      requiredAltitude: { kind: "AT", minFtMsl: 2200, maxFtMsl: 2200, sourceText: "2200 ft" },
      legacy: {
        ...caLeg.legacy,
        sequence: 50,
        constructionMethod: "hold_to_manual",
        roleAtEnd: "MAHF",
      },
    };

    const surfaces = buildMissedConnectorSurfaces(
      endpointResult.geometries,
      [caLeg, mahfLeg],
      connectorFixes,
      new Map([[missedSegment.segmentId, missedSegment]]),
      {
        samplingStepNm: 1,
        sourceMissedSurfaceBySegmentId: new Map([
          [
            missedSegment.segmentId,
            buildMissedSectionSurface(missedSegment, geometryBundle).geometry!,
          ],
        ]),
      },
    );

    expect(surfaces).toHaveLength(1);
    expect(surfaces[0]).toMatchObject({
      sourceLegId: "leg:missed:ca",
      targetFixId: "fix:DUHAM",
      targetFixIdent: "DUHAM",
      constructionStatus: "ESTIMATED_CONNECTOR_SURFACE",
      lateralWidthRule: {
        ruleId: "MISSED_CONNECTOR_TERMINAL_WIDTH",
        ruleStatus: "SOURCE_SURFACE_TERMINAL_WIDTH",
        transitionStatus: "TERMINAL_WIDTH_HELD_TO_MAHF",
      },
      primary: expect.objectContaining({
        geometryId: "segment:missed-s1:ca-mahf-connector-primary:leg:missed:ca",
      }),
      secondaryOuter: expect.objectContaining({
        geometryId: "segment:missed-s1:ca-mahf-connector-secondary:leg:missed:ca",
      }),
    });
    expect(surfaces[0].primary.halfWidthNmSamples[0].halfWidthNm).toBe(2);
    expect(surfaces[0].secondaryOuter?.halfWidthNmSamples[0].halfWidthNm).toBe(2);
    const lastVerticalSample =
      surfaces[0].verticalProfile.samples[surfaces[0].verticalProfile.samples.length - 1];
    expect(lastVerticalSample.altitudeFtMsl).toBeCloseTo(2200, 8);
  });

  it("builds debug-only turning missed anchor and estimated primitives for flagged section two segments", () => {
    const turningSegment: ProcedureSegment = {
      ...missedSegment,
      segmentId: "segment:missed-s2",
      segmentType: "MISSED_S2",
      startFixId: "fix:RW",
      legIds: ["leg:missed:hm"],
      constructionFlags: { isTurningMissedApproach: true },
    };
    const hmLeg: ProcedurePackageLeg = {
      ...caLeg,
      legId: "leg:missed:hm",
      segmentId: "segment:missed-s2",
      legType: "HM",
      rawPathTerminator: "HM",
      outboundCourseDeg: 305,
      turnDirection: "RIGHT",
    };

    const result = buildMissedTurnDebugPoint(turningSegment, [hmLeg], fixes);

    expect(result.diagnostics).toEqual([]);
    expect(result.geometry).toMatchObject({
      segmentId: "segment:missed-s2",
      debugType: "TURNING_MISSED_ANCHOR",
      anchorFixId: "fix:RW",
      triggerLegTypes: ["HM"],
      constructionStatus: "DEBUG_MARKER_ONLY",
      geoPosition: expect.objectContaining({ lonDeg: -78.8, latDeg: 35.87 }),
    });

    const primitiveResult = buildMissedTurnDebugPrimitives(turningSegment, [hmLeg], fixes);
    expect(primitiveResult.diagnostics).toEqual([
      expect.objectContaining({
        code: "SOURCE_INCOMPLETE",
        legId: "leg:missed:hm",
      }),
    ]);
    expect(primitiveResult.geometries.map((geometry) => geometry.debugType)).toEqual([
      "TIA_BOUNDARY",
      "EARLY_TURN_BASELINE",
      "LATE_TURN_BASELINE",
      "NOMINAL_TURN_PATH",
      "WIND_SPIRAL",
    ]);
    expect(primitiveResult.geometries[0]).toMatchObject({
      constructionStatus: "DEBUG_ESTIMATE_ONLY",
      turnTrigger: "TURN_AT_FIX",
      courseDeg: 305,
      turnDirection: "RIGHT",
    });
    expect(primitiveResult.geometries[0].geoPositions.length).toBeGreaterThan(10);
  });
});
