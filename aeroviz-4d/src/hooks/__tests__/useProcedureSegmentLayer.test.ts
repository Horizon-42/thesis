import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { loadProcedureRenderBundleData } from "../../data/procedureRenderBundle";

const {
  entities,
  mockViewer,
  getProceduresVisible,
  setProceduresVisible,
  getProcedureVisibility,
  setProcedureBranchVisible,
  getProcedureAnnotationEnabled,
  setProcedureAnnotationEnabled,
  getProcedureDisplayLevel,
  setProcedureDisplayLevel,
} = vi.hoisted(() => {
  const entities: any[] = [];
  let proceduresVisible = true;
  let procedureVisibility: Record<string, boolean> = {};
  let procedureAnnotationEnabled = false;
  let procedureDisplayLevel = "PROTECTION";
  const mockViewer = {
    entities: {
      values: entities,
      getById: vi.fn((id: string) => entities.find((entity) => entity.id === id)),
      add: vi.fn((entity: any) => {
        entities.push(entity);
        return entity;
      }),
      removeById: vi.fn((id: string) => {
        const index = entities.findIndex((entity) => entity.id === id);
        if (index >= 0) entities.splice(index, 1);
        return index >= 0;
      }),
    },
  };

  return {
    entities,
    mockViewer,
    getProceduresVisible: () => proceduresVisible,
    setProceduresVisible: (value: boolean) => {
      proceduresVisible = value;
    },
    getProcedureVisibility: () => procedureVisibility,
    setProcedureBranchVisible: (branchId: string, visible: boolean) => {
      procedureVisibility = { ...procedureVisibility, [branchId]: visible };
    },
    getProcedureAnnotationEnabled: () => procedureAnnotationEnabled,
    setProcedureAnnotationEnabled: (value: boolean) => {
      procedureAnnotationEnabled = value;
    },
    getProcedureDisplayLevel: () => procedureDisplayLevel,
    setProcedureDisplayLevel: (value: string) => {
      procedureDisplayLevel = value;
    },
  };
});

vi.mock("cesium", () => {
  class Color {
    constructor(public name: string) {}
    withAlpha(alpha: number) {
      return { name: this.name, alpha };
    }
  }

  return {
    Cartesian3: {
      fromDegrees: (lon: number, lat: number, alt: number) => ({ lon, lat, alt }),
      fromDegreesArrayHeights: (values: number[]) => values,
    },
    Color: {
      CYAN: new Color("CYAN"),
      DEEPSKYBLUE: new Color("DEEPSKYBLUE"),
      YELLOW: new Color("YELLOW"),
      LIME: new Color("LIME"),
      MAGENTA: new Color("MAGENTA"),
      ORANGE: new Color("ORANGE"),
      WHITE: new Color("WHITE"),
      BLACK: new Color("BLACK"),
    },
    PolygonHierarchy: class PolygonHierarchy {
      constructor(public positions: unknown[]) {}
    },
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
    activeAirportCode: "KRDU",
    layers: { procedures: getProceduresVisible() },
    procedureVisibility: getProcedureVisibility(),
    procedureAnnotationEnabled: getProcedureAnnotationEnabled(),
    procedureDisplayLevel: getProcedureDisplayLevel(),
  }),
}));

vi.mock("../../data/procedureRenderBundle", () => ({
  loadProcedureRenderBundleData: vi.fn(),
}));

import { useProcedureSegmentLayer } from "../useProcedureSegmentLayer";

const geoPositions = [
  { lonDeg: -78.84, latDeg: 35.84, altM: 670 },
  { lonDeg: -78.8, latDeg: 35.87, altM: 244 },
];

const ribbon = {
  geometryId: "ribbon",
  leftBoundary: [],
  rightBoundary: [],
  leftGeoBoundary: [
    { lonDeg: -78.85, latDeg: 35.84, altM: 670 },
    { lonDeg: -78.81, latDeg: 35.87, altM: 244 },
  ],
  rightGeoBoundary: [
    { lonDeg: -78.83, latDeg: 35.84, altM: 670 },
    { lonDeg: -78.79, latDeg: 35.87, altM: 244 },
  ],
  halfWidthNmSamples: [
    { stationNm: 0, halfWidthNm: 0.6 },
    { stationNm: 3, halfWidthNm: 0.6 },
  ],
};

const turnJunction = {
  geometryId: "turn",
  segmentId: "segment:final",
  turnPointIndex: 1,
  stationNm: 1,
  turnAngleDeg: 45,
  turnDirection: "LEFT",
  constructionStatus: "VISUAL_FILL_ONLY",
  primaryPatch: {
    geometryId: "turn-primary",
    envelopeType: "PRIMARY",
    ribbon,
    halfWidthNm: 0.6,
  },
  secondaryPatch: {
    geometryId: "turn-secondary",
    envelopeType: "SECONDARY",
    ribbon,
    halfWidthNm: 0.9,
  },
};

const interSegmentTurnJunction = {
  geometryId: "KRDU-R05LY-RW05L:branch:R:junction:segment:intermediate->segment:final",
  branchId: "KRDU-R05LY-RW05L:branch:R",
  fromSegmentId: "segment:intermediate",
  toSegmentId: "segment:final",
  joinGapNm: 0,
  turnAngleDeg: 35,
  turnDirection: "RIGHT",
  constructionStatus: "VISUAL_FILL_ONLY",
  primaryPatch: {
    geometryId: "inter-turn-primary",
    envelopeType: "PRIMARY",
    ribbon,
    halfWidthNm: 0.6,
  },
  secondaryPatch: {
    geometryId: "inter-turn-secondary",
    envelopeType: "SECONDARY",
    ribbon,
    halfWidthNm: 0.9,
  },
};

const renderBundleData = {
  index: {},
  documents: [],
  packages: [
    {
      packageId: "KRDU-R05LY-RW05L",
      sharedFixes: [
        {
          fixId: "fix:INIT",
          ident: "INIT",
          role: ["IF"],
          lonDeg: -78.86,
          latDeg: 35.82,
          altFtMsl: null,
          sourceRefs: [],
        },
        {
          fixId: "fix:FAF",
          ident: "FAF",
          role: ["FAF"],
          lonDeg: -78.84,
          latDeg: 35.84,
          altFtMsl: null,
          sourceRefs: [],
        },
        {
          fixId: "fix:RW",
          ident: "RW",
          role: ["MAP", "RWY"],
          lonDeg: -78.8,
          latDeg: 35.87,
          altFtMsl: 800,
          sourceRefs: [],
        },
        {
          fixId: "fix:DUHAM",
          ident: "DUHAM",
          role: ["MAHF"],
          lonDeg: -78.72,
          latDeg: 35.94,
          altFtMsl: 3000,
          sourceRefs: [],
        },
      ],
    },
  ],
  renderBundles: [
    {
      packageId: "KRDU-R05LY-RW05L",
      procedureId: "R05LY",
      procedureName: "RNAV(GPS) Y RW05L",
      airportId: "KRDU",
      diagnostics: [],
      branchBundles: [
        {
          branchId: "KRDU-R05LY-RW05L:branch:R",
          branchName: "RW05L",
          branchRole: "STRAIGHT_IN",
          runwayId: "RW05L",
          turnJunctions: [interSegmentTurnJunction],
          missedCaMahfConnectors: [
            {
              sourceSegmentId: "segment:final",
              sourceLegId: "leg:missed:ca",
              sourceEndpointStatus: "ESTIMATED_ENDPOINT",
              targetFixId: "fix:DUHAM",
              targetFixIdent: "DUHAM",
              targetFixRole: "MAHF",
              constructionStatus: "ESTIMATED_CONNECTOR",
              notes: ["Connector test note."],
              geoPositions: [
                { lonDeg: -78.8, latDeg: 35.87, altM: 304.8 },
                { lonDeg: -78.72, latDeg: 35.94, altM: 914.4 },
              ],
              worldPositions: [],
              geodesicLengthNm: 5.5,
              isArc: false,
            },
          ],
          segmentBundles: [
            {
              segment: {
                segmentId: "segment:transition",
                segmentType: "INITIAL",
                verticalRule: { kind: "NONE" },
              },
              legs: [
                {
                  legId: "leg:transition:init",
                  segmentId: "segment:transition",
                  legType: "IF",
                  endFixId: "fix:INIT",
                  requiredAltitude: { kind: "AT_OR_ABOVE", minFtMsl: 3000, sourceText: "3000 ft" },
                  sourceRefs: [],
                },
              ],
              diagnostics: [],
              segmentGeometry: {
                segmentId: "segment:transition",
                centerline: {
                  geoPositions,
                  worldPositions: [],
                  geodesicLengthNm: 3,
                  isArc: false,
                },
                stationAxis: { samples: [], totalLengthNm: 3 },
                primaryEnvelope: ribbon,
                secondaryEnvelope: ribbon,
                turnJunctions: [],
                diagnostics: [],
              },
              finalOea: null,
              lnavVnavOcs: null,
              precisionFinalSurfaces: [],
              finalSurfaceStatus: null,
              alignedConnector: null,
              missedSectionSurface: null,
              missedCourseGuides: [],
              missedCaEndpoints: [],
              missedCaCenterlines: [],
              missedTurnDebugPoint: null,
              missedTurnDebugPrimitives: [],
            },
            {
              segment: {
                segmentId: "segment:final",
                segmentType: "FINAL_LNAV",
                verticalRule: { kind: "BARO_GLIDEPATH", gpaDeg: 3, tchFt: 50 },
              },
              legs: [
                {
                  legId: "leg:final:faf",
                  segmentId: "segment:final",
                  legType: "TF",
                  endFixId: "fix:FAF",
                  requiredAltitude: { kind: "AT", minFtMsl: 2200, maxFtMsl: 2200, sourceText: "2200 ft" },
                  sourceRefs: [],
                },
                {
                  legId: "leg:final:rw",
                  segmentId: "segment:final",
                  legType: "TF",
                  endFixId: "fix:RW",
                  requiredAltitude: { kind: "AT", minFtMsl: 800, maxFtMsl: 800, sourceText: "800 ft" },
                  sourceRefs: [],
                },
              ],
              diagnostics: [],
              segmentGeometry: {
                segmentId: "segment:final",
                centerline: {
                  geoPositions,
                  worldPositions: [],
                  geodesicLengthNm: 3,
                  isArc: false,
                },
                stationAxis: { samples: [], totalLengthNm: 3 },
                primaryEnvelope: ribbon,
                secondaryEnvelope: ribbon,
                turnJunctions: [turnJunction],
                diagnostics: [],
              },
              finalOea: {
                primary: ribbon,
                secondaryOuter: ribbon,
              },
              lnavVnavOcs: {
                geometryId: "segment:final:lnav-vnav-ocs",
                segmentId: "segment:final",
                surfaceType: "LNAV_VNAV_OCS",
                constructionStatus: "GPA_TCH_SLOPE_ESTIMATE",
                centerline: {
                  geoPositions,
                  worldPositions: [],
                  geodesicLengthNm: 3,
                  isArc: false,
                },
                primary: ribbon,
                secondaryOuter: ribbon,
                verticalProfile: {
                  gpaDeg: 3,
                  tchFt: 50,
                  thresholdElevationFtMsl: 800,
                  thresholdStationNm: 3,
                  samples: [],
                },
                notes: [],
              },
              precisionFinalSurfaces: [
                {
                  geometryId: "segment:final:lpv-w",
                  segmentId: "segment:final",
                  surfaceType: "LPV_W",
                  constructionStatus: "GPA_TCH_DEBUG_ESTIMATE",
                  centerline: {
                    geoPositions,
                    worldPositions: [],
                    geodesicLengthNm: 3,
                    isArc: false,
                  },
                  ribbon,
                  widthScale: 1,
                  verticalProfile: {
                    gpaDeg: 3,
                    tchFt: 50,
                    thresholdElevationFtMsl: 800,
                    thresholdStationNm: 3,
                  },
                  notes: [],
                },
              ],
              finalSurfaceStatus: {
                segmentId: "segment:final",
                requestedModes: ["LPV", "LNAV/VNAV", "LNAV"],
                constructedSurfaceTypes: ["LNAV_FINAL_OEA", "LNAV_VNAV_OCS", "LPV_W"],
                missingSurfaceTypes: ["LPV_X", "LPV_Y"],
                constructionStatus: "COLLAPSED_TO_LNAV_BASELINE",
                notes: [],
              },
              alignedConnector: {
                primary: ribbon,
                secondaryOuter: ribbon,
              },
              missedSectionSurface: {
                segmentId: "segment:final",
                surfaceType: "MISSED_SECTION1_ENVELOPE",
                constructionStatus: "ESTIMATED_CA",
                primary: ribbon,
                secondaryOuter: ribbon,
              },
              missedCourseGuides: [
                {
                  segmentId: "segment:final",
                  legId: "leg:missed:ca",
                  legType: "CA",
                  startFixId: "fix:RW",
                  courseDeg: 305,
                  guideLengthNm: 3,
                  requiredAltitudeFtMsl: 1000,
                  constructionStatus: "COURSE_DIRECTION_ONLY",
                  geoPositions,
                  worldPositions: [],
                },
              ],
              missedCaEndpoints: [
                {
                  segmentId: "segment:final",
                  legId: "leg:missed:ca",
                  startFixId: "fix:RW",
                  courseDeg: 305,
                  startAltitudeFtMsl: 800,
                  targetAltitudeFtMsl: 1000,
                  climbGradientFtPerNm: 200,
                  distanceNm: 1,
                  constructionStatus: "ESTIMATED_ENDPOINT",
                  geoPositions,
                  worldPositions: [],
                  notes: [],
                },
              ],
              missedCaCenterlines: [
                {
                  segmentId: "segment:final",
                  legId: "leg:missed:ca",
                  sourceEndpointStatus: "ESTIMATED_ENDPOINT",
                  constructionStatus: "ESTIMATED_CENTERLINE",
                  geoPositions,
                  worldPositions: [],
                  geodesicLengthNm: 1,
                  isArc: false,
                  notes: [],
                },
              ],
              missedTurnDebugPoint: {
                segmentId: "segment:final",
                debugType: "TURNING_MISSED_ANCHOR",
                anchorFixId: "fix:RW",
                triggerLegTypes: ["HM"],
                constructionStatus: "DEBUG_MARKER_ONLY",
                geoPosition: geoPositions[1],
                worldPosition: {},
              },
              missedTurnDebugPrimitives: [
                {
                  primitiveId: "segment:final:turning-missed:tia-boundary",
                  segmentId: "segment:final",
                  legId: "leg:missed:hm",
                  debugType: "TIA_BOUNDARY",
                  constructionStatus: "DEBUG_ESTIMATE_ONLY",
                  turnTrigger: "TURN_AT_FIX",
                  turnCase: "NOMINAL",
                  anchorFixId: "fix:RW",
                  courseDeg: 305,
                  turnDirection: "RIGHT",
                  geoPositions,
                  worldPositions: [],
                  notes: [],
                },
              ],
            },
            {
              segment: {
                segmentId: "segment:missed-s1",
                segmentType: "MISSED_S1",
                verticalRule: { kind: "MISSED_CLIMB_SURFACE", climbGradientFtPerNm: 200 },
              },
              legs: [
                {
                  legId: "leg:missed:ca",
                  segmentId: "segment:missed-s1",
                  legType: "CA",
                  startFixId: "fix:RW",
                  endFixId: "fix:RW",
                  outboundCourseDeg: 305,
                  requiredAltitude: { kind: "AT", minFtMsl: 1000, maxFtMsl: 1000, sourceText: "1000 ft" },
                  sourceRefs: [],
                },
              ],
              diagnostics: [],
              segmentGeometry: {
                segmentId: "segment:missed-s1",
                centerline: {
                  geoPositions: [
                    { lonDeg: -78.8, latDeg: 35.87, altM: 244 },
                    { lonDeg: -78.815, latDeg: 35.88, altM: 304.8 },
                  ],
                  worldPositions: [],
                  geodesicLengthNm: 1,
                  isArc: false,
                },
                stationAxis: { samples: [], totalLengthNm: 1 },
                primaryEnvelope: ribbon,
                secondaryEnvelope: ribbon,
                turnJunctions: [],
                diagnostics: [],
              },
              finalOea: null,
              lnavVnavOcs: null,
              precisionFinalSurfaces: [],
              finalSurfaceStatus: null,
              alignedConnector: null,
              missedSectionSurface: null,
              missedCourseGuides: [],
              missedCaEndpoints: [
                {
                  segmentId: "segment:missed-s1",
                  legId: "leg:missed:ca",
                  startFixId: "fix:RW",
                  courseDeg: 305,
                  startAltitudeFtMsl: 800,
                  targetAltitudeFtMsl: 1000,
                  climbGradientFtPerNm: 200,
                  distanceNm: 1,
                  constructionStatus: "ESTIMATED_ENDPOINT",
                  geoPositions: [
                    { lonDeg: -78.8, latDeg: 35.87, altM: 244 },
                    { lonDeg: -78.815, latDeg: 35.88, altM: 304.8 },
                  ],
                  worldPositions: [],
                  notes: [],
                },
              ],
              missedCaCenterlines: [],
              missedTurnDebugPoint: null,
              missedTurnDebugPrimitives: [],
            },
            {
              segment: {
                segmentId: "segment:missed-s2",
                segmentType: "MISSED_S2",
                verticalRule: { kind: "MISSED_CLIMB_SURFACE", climbGradientFtPerNm: 200 },
              },
              legs: [
                {
                  legId: "leg:missed:hm",
                  segmentId: "segment:missed-s2",
                  legType: "HM",
                  startFixId: "fix:DUHAM",
                  endFixId: "fix:DUHAM",
                  requiredAltitude: null,
                  sourceRefs: [],
                  legacy: { roleAtEnd: "MAHF" },
                },
              ],
              diagnostics: [],
              segmentGeometry: {
                segmentId: "segment:missed-s2",
                centerline: {
                  geoPositions: [],
                  worldPositions: [],
                  geodesicLengthNm: 0,
                  isArc: false,
                },
                stationAxis: { samples: [], totalLengthNm: 0 },
                turnJunctions: [],
                diagnostics: [],
              },
              finalOea: null,
              lnavVnavOcs: null,
              precisionFinalSurfaces: [],
              finalSurfaceStatus: null,
              alignedConnector: null,
              missedSectionSurface: null,
              missedCourseGuides: [],
              missedCaEndpoints: [],
              missedCaCenterlines: [],
              missedTurnDebugPoint: null,
              missedTurnDebugPrimitives: [],
            },
          ],
        },
      ],
    },
  ],
};

describe("useProcedureSegmentLayer", () => {
  beforeEach(() => {
    entities.length = 0;
    mockViewer.entities.add.mockClear();
    mockViewer.entities.removeById.mockClear();
    mockViewer.entities.getById.mockClear();
    vi.mocked(loadProcedureRenderBundleData).mockReset();
    vi.mocked(loadProcedureRenderBundleData).mockResolvedValue(renderBundleData as any);
    setProceduresVisible(true);
    setProcedureAnnotationEnabled(false);
    setProcedureDisplayLevel("PROTECTION");
    setProcedureBranchVisible("KRDU-R05LY-RW05L:branch:R", true);
  });

  it("renders segment centerline, envelopes, OEA, and connector entities", async () => {
    renderHook(() => useProcedureSegmentLayer());

    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    expect(loadProcedureRenderBundleData).toHaveBeenCalledWith("KRDU");
    expect(entities.some((entity) => String(entity.id).endsWith("-centerline"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-envelope-primary"))).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-envelope-primary") &&
          entity.__aeroVizProcedureAnnotation?.parameters.some(
            (parameter: { label: string; value: string }) =>
              parameter.label === "Vertical meaning" &&
              parameter.value === "Not OCS; not a vertical clearance surface",
          ),
      ),
    ).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-oea-primary"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-lnav-vnav-ocs-primary"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-final-vertical-reference-band"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-final-vertical-reference"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-vertical-profile") && entity.polygon)).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-vertical-profile") &&
          entity.polygon.material.name === "DEEPSKYBLUE" &&
          entity.__aeroVizProcedureAnnotation?.kind === "SEGMENT_VERTICAL_PROFILE",
      ),
    ).toBe(true);
    const finalVerticalProfile = entities.find((entity) =>
      String(entity.id).endsWith("-vertical-profile") && entity.polygon,
    );
    expect(finalVerticalProfile?.polygon.hierarchy.positions).toHaveLength(10);
    expect(
      finalVerticalProfile?.__aeroVizProcedureAnnotation?.parameters.some(
        (parameter: { label: string; value: string }) =>
          parameter.label === "Fixes" &&
          parameter.value.includes("RW -> CA endpoint -> DUHAM"),
      ),
    ).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-final-vertical-reference-band") &&
          entity.polygon.material.name === "CYAN",
      ),
    ).toBe(true);
    const finalVerticalBand = entities.find((entity) =>
      String(entity.id).includes("-final-vertical-reference-band"),
    );
    expect(finalVerticalBand?.polygon.hierarchy.positions).toHaveLength(4);
    expect(
      Math.abs(
        finalVerticalBand.polygon.hierarchy.positions[0].lon -
          finalVerticalBand.polygon.hierarchy.positions[2].lon,
      ),
    ).toBeGreaterThan(0.003);
    expect(entities.some((entity) => String(entity.id).includes("-altitude-leg:final:faf"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-altitude-leg:transition:init"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-altitude-leg:transition:init-link"))).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-altitude-leg:transition:init") &&
          entity.point?.color.name === "CYAN",
      ),
    ).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-altitude-leg:final:faf") &&
          entity.point?.color.name === "LIME",
      ),
    ).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-altitude-leg:transition:init") &&
          entity.__aeroVizProcedureAnnotation?.kind === "ALTITUDE_CONSTRAINT",
      ),
    ).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-altitude-leg:transition:init-link") &&
          entity.__aeroVizProcedureAnnotation?.kind === "ALTITUDE_CONSTRAINT",
      ),
    ).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-lnav-vnav-ocs-primary") &&
          entity.polygon.material.name === "LIME",
      ),
    ).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-precision-lpv-w"))).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-precision-lpv-w") &&
          entity.polygon.material.name === "MAGENTA",
      ),
    ).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-final-surface-status"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-connector-primary"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-connector-primary-boundary"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-missed-surface-primary"))).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-missed-surface-primary") &&
          String(entity.name).includes("CA estimated missed section primary") &&
          entity.polygon.material.name === "ORANGE",
      ),
    ).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-ca-course-guide-"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-ca-centerline-"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-ca-endpoint-"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-ca-mahf-connector-"))).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).includes("-ca-mahf-connector-") &&
          entity.__aeroVizProcedureAnnotation?.kind === "CA_MAHF_CONNECTOR" &&
          entity.polyline.material.name === "ORANGE",
      ),
    ).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-turning-missed-anchor"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-turning-missed-tia-boundary"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-turn-1-primary"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes(":junction:"))).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).endsWith("-centerline") &&
          entity.__aeroVizProcedureAnnotation?.kind === "SEGMENT_CENTERLINE",
      ),
    ).toBe(true);
  });

  it("adds annotation labels when annotation mode is enabled", async () => {
    setProcedureAnnotationEnabled(true);

    renderHook(() => useProcedureSegmentLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    expect(entities.some((entity) => String(entity.id).startsWith("procedure-annotation-label-"))).toBe(true);
    expect(
      entities.some(
        (entity) =>
          String(entity.id).startsWith("procedure-annotation-label-") &&
          entity.show === true &&
          entity.__aeroVizProcedureAnnotation?.title.includes("centerline"),
      ),
    ).toBe(true);
  });

  it("filters procedure entities by cumulative display level", async () => {
    const { rerender } = renderHook(() => useProcedureSegmentLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    const centerline = entities.find((entity) => String(entity.id).endsWith("-centerline"));
    const ocs = entities.find((entity) => String(entity.id).includes("-lnav-vnav-ocs-primary"));
    const finalVerticalReference = entities.find((entity) => String(entity.id).endsWith("-final-vertical-reference"));
    const finalVerticalBand = entities.find((entity) => String(entity.id).endsWith("-final-vertical-reference-band"));
    const verticalProfile = entities.find((entity) => String(entity.id).endsWith("-vertical-profile") && entity.polygon);
    const caMahfConnector = entities.find((entity) => String(entity.id).includes("-ca-mahf-connector-"));
    const finalAltitude = entities.find((entity) => String(entity.id).includes("-altitude-leg:final:faf"));
    const turnFill = entities.find((entity) => String(entity.id).includes("-turn-1-primary"));
    const debugSurface = entities.find((entity) => String(entity.id).includes("-precision-lpv-w"));

    expect(centerline.show).toBe(true);
    expect(finalAltitude.show).toBe(true);
    expect(ocs.show).toBe(false);
    expect(finalVerticalReference.show).toBe(false);
    expect(finalVerticalBand.show).toBe(false);
    expect(verticalProfile.show).toBe(false);
    expect(caMahfConnector.show).toBe(false);
    expect(turnFill.show).toBe(false);
    expect(debugSurface.show).toBe(false);

    setProcedureDisplayLevel("ESTIMATED");
    rerender();

    expect(ocs.show).toBe(true);
    expect(finalVerticalReference.show).toBe(true);
    expect(finalVerticalBand.show).toBe(true);
    expect(verticalProfile.show).toBe(true);
    expect(caMahfConnector.show).toBe(true);
    expect(turnFill.show).toBe(false);
    expect(debugSurface.show).toBe(false);

    setProcedureDisplayLevel("DEBUG");
    rerender();

    expect(ocs.show).toBe(true);
    expect(turnFill.show).toBe(true);
    expect(debugSurface.show).toBe(true);
  });

  it("syncs layer visibility without reloading render bundle data", async () => {
    const { rerender } = renderHook(() => useProcedureSegmentLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    setProceduresVisible(false);
    rerender();

    expect(loadProcedureRenderBundleData).toHaveBeenCalledTimes(1);
    expect(entities.every((entity) => entity.show === false)).toBe(true);
  });

  it("uses the canonical branch id for Procedure Panel branch selection", async () => {
    const { rerender } = renderHook(() => useProcedureSegmentLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    setProcedureBranchVisible("KRDU-R05LY-RW05L:branch:R", false);
    rerender();

    expect(loadProcedureRenderBundleData).toHaveBeenCalledTimes(1);
    expect(entities.every((entity) => entity.show === false)).toBe(true);
  });
});
