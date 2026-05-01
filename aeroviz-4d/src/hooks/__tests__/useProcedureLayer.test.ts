import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

function jsonResponse(body: unknown) {
  return {
    ok: true,
    headers: { get: () => "application/json" },
    text: async () => JSON.stringify(body),
  };
}

const {
  entities,
  mockViewer,
  getProceduresVisible,
  setProceduresVisible,
  getProcedureVisibility,
  setProcedureRouteVisible,
  fetchMock,
} = vi.hoisted(() => {
  const entities: any[] = [];
  let proceduresVisible = true;
  let procedureVisibility: Record<string, boolean> = {};
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
  const fetchMock = vi.fn();

  return {
    entities,
    mockViewer,
    getProceduresVisible: () => proceduresVisible,
    setProceduresVisible: (value: boolean) => {
      proceduresVisible = value;
    },
    getProcedureVisibility: () => procedureVisibility,
    setProcedureRouteVisible: (routeId: string, visible: boolean) => {
      procedureVisibility = { ...procedureVisibility, [routeId]: visible };
    },
    fetchMock,
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
    Cartesian2: class Cartesian2 {
      constructor(public x: number, public y: number) {}
    },
    Cartesian3: {
      fromDegrees: (lon: number, lat: number, alt: number) => ({ lon, lat, alt }),
      fromDegreesArrayHeights: (values: number[]) => values,
    },
    Color: {
      CYAN: new Color("CYAN"),
      YELLOW: new Color("YELLOW"),
      DEEPSKYBLUE: new Color("DEEPSKYBLUE"),
      ORANGE: new Color("ORANGE"),
      BLACK: new Color("BLACK"),
      WHITE: new Color("WHITE"),
    },
    DistanceDisplayCondition: class DistanceDisplayCondition {
      constructor(public near: number, public far: number) {}
    },
    LabelStyle: {
      FILL_AND_OUTLINE: "FILL_AND_OUTLINE",
    },
    PolygonHierarchy: class PolygonHierarchy {
      constructor(public positions: unknown[]) {}
    },
    VerticalOrigin: {
      BOTTOM: "BOTTOM",
    },
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
    activeAirportCode: "KRDU",
    layers: { procedures: getProceduresVisible() },
    procedureVisibility: getProcedureVisibility(),
  }),
}));

import { useProcedureLayer } from "../useProcedureLayer";

const sampleIndex = {
  airport: "KRDU",
  airportName: "Raleigh Durham Intl",
  sourceCycle: "2603",
  researchUseOnly: true,
  runways: [
    {
      runwayIdent: "RW05L",
      chartName: "RW05L",
      procedureUids: ["KRDU-R05LY-RW05L"],
      procedures: [
        {
          procedureUid: "KRDU-R05LY-RW05L",
          procedureIdent: "R05LY",
          chartName: "RNAV(GPS) Y RW05L",
          procedureFamily: "RNAV_GPS",
          variant: "Y",
          approachModes: ["GPS"],
          runwayIdent: "RW05L",
          defaultBranchId: "branch:R",
        },
      ],
    },
  ],
};

const sampleDocument = {
  schemaVersion: "1.0",
  modelType: "procedure-detail",
  procedureUid: "KRDU-R05LY-RW05L",
  provenance: {
    assemblyMode: "test",
    researchUseOnly: true,
    sources: [{ sourceId: "cifp", kind: "CIFP", cycle: "2603" }],
    warnings: [],
  },
  airport: { icao: "KRDU", faa: "RDU", name: "Raleigh Durham Intl" },
  runway: {
    ident: "RW05L",
    landingThresholdFixRef: "fix:RW05L",
    threshold: { lon: -78.8019631, lat: 35.87445, elevationFt: 435 },
  },
  procedure: {
    procedureType: "R",
    procedureFamily: "RNAV_GPS",
    procedureIdent: "R05LY",
    chartName: "RNAV(GPS) Y RW05L",
    variant: "Y",
    runwayIdent: "RW05L",
    baseBranchIdent: "R",
    approachModes: ["GPS"],
  },
  fixes: [
    {
      fixId: "fix:SCHOO",
      ident: "SCHOO",
      kind: "waypoint",
      position: { lon: -78.9251472, lat: 35.7734139 },
      elevationFt: null,
      roleHints: ["IF"],
      sourceRefs: [],
    },
    {
      fixId: "fix:WEPAS",
      ident: "WEPAS",
      kind: "waypoint",
      position: { lon: -78.8829556, lat: 35.8087667 },
      elevationFt: null,
      roleHints: ["FAF"],
      sourceRefs: [],
    },
    {
      fixId: "fix:RW05L",
      ident: "RW05L",
      kind: "runway",
      position: { lon: -78.8019631, lat: 35.87445 },
      elevationFt: 435,
      roleHints: ["MAPt"],
      sourceRefs: [],
    },
  ],
  branches: [
    {
      branchId: "branch:R",
      branchKey: "R",
      branchIdent: "R",
      procedureType: "R",
      transitionIdent: null,
      branchRole: "final",
      sequenceOrder: 0,
      mergeFixRef: null,
      continuesWithBranchId: null,
      defaultVisible: true,
      warnings: [],
      legs: [
        {
          legId: "leg:10",
          sequence: 10,
          segmentType: "final",
          path: {
            pathTerminator: "IF",
            constructionMethod: "track",
            startFixRef: null,
            endFixRef: "fix:SCHOO",
          },
          termination: { kind: "fix", fixRef: "fix:SCHOO" },
          constraints: {
            altitude: { qualifier: "AT", valueFt: 3000, rawText: "3000" },
            speedKt: null,
            geometryAltitudeFt: 3000,
          },
          roleAtEnd: "IF",
          sourceRefs: [],
          quality: { status: "parsed", sourceLine: 10, renderedInPlanView: true },
        },
        {
          legId: "leg:20",
          sequence: 20,
          segmentType: "final",
          path: {
            pathTerminator: "TF",
            constructionMethod: "track",
            startFixRef: "fix:SCHOO",
            endFixRef: "fix:WEPAS",
          },
          termination: { kind: "fix", fixRef: "fix:WEPAS" },
          constraints: {
            altitude: { qualifier: "AT", valueFt: 2200, rawText: "2200" },
            speedKt: null,
            geometryAltitudeFt: 2200,
          },
          roleAtEnd: "FAF",
          sourceRefs: [],
          quality: { status: "parsed", sourceLine: 20, renderedInPlanView: true },
        },
        {
          legId: "leg:30",
          sequence: 30,
          segmentType: "final",
          path: {
            pathTerminator: "TF",
            constructionMethod: "track",
            startFixRef: "fix:WEPAS",
            endFixRef: "fix:RW05L",
          },
          termination: { kind: "fix", fixRef: "fix:RW05L" },
          constraints: {
            altitude: { qualifier: "AT", valueFt: 798, rawText: "798" },
            speedKt: null,
            geometryAltitudeFt: 798,
          },
          roleAtEnd: "MAPt",
          sourceRefs: [],
          quality: { status: "parsed", sourceLine: 30, renderedInPlanView: true },
        },
      ],
    },
  ],
  verticalProfiles: [],
  validation: {
    expectedRunwayIdent: "RW05L",
    expectedIF: "SCHOO",
    expectedFAF: "WEPAS",
    expectedMAPt: "RW05L",
    expectedMissedHoldFix: null,
    knownSimplifications: [],
  },
  displayHints: {
    nominalSpeedKt: 140,
    defaultVisibleBranchIds: ["branch:R"],
    tunnelDefaults: {
      lateralHalfWidthNm: 0.3,
      verticalHalfHeightFt: 300,
      sampleSpacingM: 10000,
      mode: "visualApproximation",
    },
  },
};

describe("useProcedureLayer", () => {
  beforeEach(() => {
    entities.length = 0;
    mockViewer.entities.add.mockClear();
    mockViewer.entities.removeById.mockClear();
    mockViewer.entities.getById.mockClear();
    setProceduresVisible(true);
    setProcedureRouteVisible("KRDU-R05LY-R", true);
    fetchMock.mockReset();
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith("/procedure-details/index.json")) return Promise.resolve(jsonResponse(sampleIndex));
      if (url.endsWith("/procedure-details/KRDU-R05LY-RW05L.json")) {
        return Promise.resolve(jsonResponse(sampleDocument));
      }
      return Promise.reject(new Error(`Unexpected URL ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads procedure route, tunnel, and fix entities", async () => {
    renderHook(() => useProcedureLayer());

    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    expect(fetchMock).toHaveBeenCalledWith("/data/airports/KRDU/procedure-details/index.json");
    expect(fetchMock).toHaveBeenCalledWith(
      "/data/airports/KRDU/procedure-details/KRDU-R05LY-RW05L.json",
    );
    expect(entities.some((entity) => String(entity.id).endsWith("-line"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-tunnel-"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-fix-"))).toBe(true);
  });

  it("syncs procedure visibility without reloading data", async () => {
    const { rerender } = renderHook(() => useProcedureLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    setProceduresVisible(false);
    rerender();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(entities.every((entity) => entity.show === false)).toBe(true);
  });

  it("syncs individual route visibility without reloading data", async () => {
    const { rerender } = renderHook(() => useProcedureLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    setProcedureRouteVisible("KRDU-R05LY-R", false);
    rerender();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(entities.every((entity) => entity.show === false)).toBe(true);
  });

  it("removes only procedure entities on cleanup", async () => {
    const { unmount } = renderHook(() => useProcedureLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());
    const addedIds = entities.map((entity) => entity.id);

    unmount();

    addedIds.forEach((id) => {
      expect(mockViewer.entities.removeById).toHaveBeenCalledWith(id);
    });
  });
});
