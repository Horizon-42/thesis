import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function jsonResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    headers: { get: () => "application/json" },
    text: async () => JSON.stringify(body),
  };
}

function notFoundResponse() {
  return {
    ok: false,
    status: 404,
    headers: { get: () => "application/json" },
    text: async () => "",
  };
}

const {
  fetchMock,
  setActiveAirportCode,
  appState,
} = vi.hoisted(() => ({
  fetchMock: vi.fn(),
  setActiveAirportCode: vi.fn(),
  appState: {
    airports: [
      { code: "KRDU", name: "Raleigh-Durham International Airport", lat: 35.87, lon: -78.79 },
      { code: "CYVR", name: "Vancouver International Airport", lat: 49.19, lon: -123.18 },
    ],
    activeAirportCode: "KRDU",
    setActiveAirportCode: vi.fn(),
    layers: { procedures: true },
    procedureVisibility: {},
    procedureDisplayLevel: "PROTECTION",
  },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    airports: appState.airports,
    activeAirportCode: appState.activeAirportCode,
    setActiveAirportCode,
    layers: appState.layers,
    procedureVisibility: appState.procedureVisibility,
    procedureDisplayLevel: appState.procedureDisplayLevel,
  }),
}));

import ProcedureDetailsPage from "../ProcedureDetailsPage";

const sampleIndex = {
  airport: "KRDU",
  airportName: "Raleigh-Durham International Airport",
  sourceCycle: "2603",
  researchUseOnly: true,
  runways: [
    {
      runwayIdent: "RW05L",
      chartName: "RNAV(GPS) Y RWY 05L",
      procedureUids: ["KRDU-R05LY-RW05L"],
      procedures: [
        {
          procedureUid: "KRDU-R05LY-RW05L",
          procedureIdent: "R05LY",
          chartName: "RNAV(GPS) Y RWY 05L",
          procedureFamily: "RNAV_GPS",
          variant: "Y",
          approachModes: ["LPV", "LNAV/VNAV", "LNAV"],
          runwayIdent: "RW05L",
          defaultBranchId: "R",
        },
      ],
    },
  ],
};

const sampleCharts = {
  airport: "KRDU",
  researchUseOnly: true,
  charts: [
    {
      chartId: "chart:KRDU:00516RY5L.PDF-nameddest-RDU.pdf",
      procedureUid: "KRDU-R05LY-RW05L",
      procedureIdent: "R05LY",
      runwayIdent: "RW05L",
      title: "RNAV(GPS) Y RWY 05L",
      originalFileName: "00516RY5L.PDF#nameddest=(RDU).pdf",
      sourcePath: "/tmp/00516RY5L.PDF#nameddest=(RDU).pdf",
      url: "/data/airports/KRDU/charts/00516RY5L.PDF-nameddest-RDU.pdf",
    },
  ],
};

const sampleDocument = {
  schemaVersion: "1.0.0",
  modelType: "rnav-procedure-runway",
  procedureUid: "KRDU-R05LY-RW05L",
  provenance: {
    assemblyMode: "cifp_primary_export",
    researchUseOnly: true,
    sources: [],
    warnings: [],
  },
  airport: {
    icao: "KRDU",
    faa: "RDU",
    name: "Raleigh-Durham International Airport",
  },
  runway: {
    ident: "RW05L",
    landingThresholdFixRef: "fix:RW05L",
    threshold: {
      lon: -78.80196389,
      lat: 35.87445,
      elevationFt: 798,
    },
  },
  procedure: {
    procedureType: "SIAP",
    procedureFamily: "RNAV_GPS",
    procedureIdent: "R05LY",
    chartName: "RNAV(GPS) Y RWY 05L",
    variant: "Y",
    runwayIdent: "RW05L",
    baseBranchIdent: "R",
    approachModes: ["LPV", "LNAV/VNAV", "LNAV"],
  },
  fixes: [
    {
      fixId: "fix:SCHOO",
      ident: "SCHOO",
      kind: "named_fix",
      position: { lon: -78.92647222, lat: 35.77341389 },
      elevationFt: null,
      roleHints: ["IF"],
      sourceRefs: ["src:cifp-detail"],
    },
    {
      fixId: "fix:WEPAS",
      ident: "WEPAS",
      kind: "final_approach_fix",
      position: { lon: -78.88295556, lat: 35.80876667 },
      elevationFt: null,
      roleHints: ["FAF"],
      sourceRefs: ["src:cifp-detail"],
    },
    {
      fixId: "fix:RW05L",
      ident: "RW05L",
      kind: "runway_threshold",
      position: { lon: -78.80196389, lat: 35.87445 },
      elevationFt: 798,
      roleHints: ["MAPt"],
      sourceRefs: ["src:cifp-detail"],
    },
  ],
  branches: [
    {
      branchId: "branch:R",
      branchIdent: "R",
      branchRole: "final",
      sequenceOrder: 1,
      mergeFixRef: null,
      continuesWithBranchId: null,
      defaultVisible: true,
      warnings: [],
      legs: [
        {
          legId: "leg:R:010",
          sequence: 10,
          segmentType: "intermediate",
          path: {
            pathTerminator: "IF",
            constructionMethod: "if_to_fix",
            startFixRef: null,
            endFixRef: "fix:SCHOO",
          },
          termination: { kind: "fix", fixRef: "fix:SCHOO" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 3000, rawText: "3000 ft" },
            speedKt: null,
            geometryAltitudeFt: 3000,
          },
          roleAtEnd: "IF",
          sourceRefs: ["src:cifp-detail"],
          quality: { status: "exact", sourceLine: 1, renderedInPlanView: true },
        },
        {
          legId: "leg:R:020",
          sequence: 20,
          segmentType: "final",
          path: {
            pathTerminator: "TF",
            constructionMethod: "track_to_fix",
            startFixRef: "fix:SCHOO",
            endFixRef: "fix:WEPAS",
          },
          termination: { kind: "fix", fixRef: "fix:WEPAS" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 2200, rawText: "2200 ft" },
            speedKt: null,
            geometryAltitudeFt: 2200,
          },
          roleAtEnd: "FAF",
          sourceRefs: ["src:cifp-detail"],
          quality: { status: "exact", sourceLine: 2, renderedInPlanView: true },
        },
        {
          legId: "leg:R:030",
          sequence: 30,
          segmentType: "final",
          path: {
            pathTerminator: "TF",
            constructionMethod: "track_to_fix",
            startFixRef: "fix:WEPAS",
            endFixRef: "fix:RW05L",
          },
          termination: { kind: "fix", fixRef: "fix:RW05L" },
          constraints: {
            altitude: null,
            speedKt: null,
            geometryAltitudeFt: 798,
          },
          roleAtEnd: "MAPt",
          sourceRefs: ["src:cifp-detail"],
          quality: { status: "exact", sourceLine: 3, renderedInPlanView: true },
        },
      ],
    },
  ],
  verticalProfiles: [
    {
      profileId: "profile:R05LY:RW05L",
      appliesToModes: ["LPV", "LNAV/VNAV", "LNAV"],
      branchId: "branch:R",
      fromFixRef: "fix:SCHOO",
      toFixRef: "fix:RW05L",
      basis: "cifp_leg_constraints",
      glidepathAngleDeg: 3,
      thresholdCrossingHeightFt: null,
      constraintSamples: [
        {
          fixRef: "fix:SCHOO",
          ident: "SCHOO",
          role: "IF",
          distanceFromStartM: 0,
          altitudeFt: 3000,
          geometryAltitudeFt: 3000,
          sourceLine: 1,
        },
        {
          fixRef: "fix:WEPAS",
          ident: "WEPAS",
          role: "FAF",
          distanceFromStartM: 5555,
          altitudeFt: 2200,
          geometryAltitudeFt: 2200,
          sourceLine: 2,
        },
        {
          fixRef: "fix:RW05L",
          ident: "RW05L",
          role: "MAPt",
          distanceFromStartM: 15882,
          altitudeFt: null,
          geometryAltitudeFt: 798,
          sourceLine: 3,
        },
      ],
      warnings: [],
    },
  ],
  validation: {
    expectedRunwayIdent: "RW05L",
    expectedIF: "fix:SCHOO",
    expectedFAF: "fix:WEPAS",
    expectedMAPt: "fix:RW05L",
    expectedMissedHoldFix: null,
    knownSimplifications: [],
  },
  displayHints: {
    nominalSpeedKt: 140,
    defaultVisibleBranchIds: ["branch:R"],
    tunnelDefaults: {
      lateralHalfWidthNm: 0.3,
      verticalHalfHeightFt: 300,
      sampleSpacingM: 250,
      mode: "visualApproximation",
    },
  },
};

const rfSampleDocument = {
  ...sampleDocument,
  branches: [
    {
      ...sampleDocument.branches[0],
      legs: sampleDocument.branches[0].legs.map((leg) =>
        leg.legId === "leg:R:020"
          ? {
              ...leg,
              path: {
                ...leg.path,
                pathTerminator: "RF",
                constructionMethod: "radius_to_fix",
                turnDirection: "LEFT",
                arcRadiusNm: 2,
                centerFixRef: "fix:CENTER",
                centerLatDeg: 35.82,
                centerLonDeg: -78.86,
              },
            }
          : leg,
      ),
    },
  ],
};

const missedSectionDocument = {
  ...sampleDocument,
  fixes: [
    ...sampleDocument.fixes,
    {
      fixId: "fix:MIS1",
      ident: "MIS1",
      kind: "named_fix",
      position: { lon: -78.76, lat: 35.9 },
      elevationFt: null,
      roleHints: ["UNKNOWN"],
      sourceRefs: ["src:cifp-detail"],
    },
    {
      fixId: "fix:HOLD",
      ident: "HOLD",
      kind: "missed_hold_fix",
      position: { lon: -78.7, lat: 35.95 },
      elevationFt: null,
      roleHints: ["MAHF"],
      sourceRefs: ["src:cifp-detail"],
    },
  ],
  branches: [
    {
      ...sampleDocument.branches[0],
      legs: [
        ...sampleDocument.branches[0].legs,
        {
          legId: "leg:R:035",
          sequence: 35,
          segmentType: "missed",
          path: {
            pathTerminator: "CA",
            constructionMethod: "course_to_altitude",
            startFixRef: "fix:RW05L",
            endFixRef: "fix:RW05L",
            courseDeg: 305,
          },
          termination: { kind: "fix", fixRef: "fix:RW05L" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 1000, rawText: "1000 ft" },
            speedKt: null,
            geometryAltitudeFt: 1000,
          },
          roleAtEnd: "UNKNOWN",
          sourceRefs: ["src:cifp-detail"],
          quality: { status: "exact", sourceLine: 35, renderedInPlanView: true },
        },
        {
          legId: "leg:R:040",
          sequence: 40,
          segmentType: "missed",
          path: {
            pathTerminator: "DF",
            constructionMethod: "direct_to_fix",
            startFixRef: "fix:RW05L",
            endFixRef: "fix:MIS1",
          },
          termination: { kind: "fix", fixRef: "fix:MIS1" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 1500, rawText: "1500 ft" },
            speedKt: null,
            geometryAltitudeFt: 1500,
          },
          roleAtEnd: "UNKNOWN",
          sourceRefs: ["src:cifp-detail"],
          quality: { status: "exact", sourceLine: 4, renderedInPlanView: true },
        },
        {
          legId: "leg:R:050",
          sequence: 50,
          segmentType: "missed",
          path: {
            pathTerminator: "HM",
            constructionMethod: "hold_to_manual",
            startFixRef: "fix:MIS1",
            endFixRef: "fix:HOLD",
            courseDeg: 305,
            turnDirection: "RIGHT",
          },
          termination: { kind: "fix", fixRef: "fix:HOLD" },
          constraints: {
            altitude: { qualifier: "at", valueFt: 3000, rawText: "3000 ft" },
            speedKt: null,
            geometryAltitudeFt: 3000,
          },
          roleAtEnd: "MAHF",
          sourceRefs: ["src:cifp-detail"],
          quality: { status: "exact", sourceLine: 5, renderedInPlanView: false },
        },
      ],
    },
  ],
};

describe("ProcedureDetailsPage", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    setActiveAirportCode.mockReset();
    appState.activeAirportCode = "KRDU";
    appState.layers = { procedures: true };
    appState.procedureVisibility = {};
    appState.procedureDisplayLevel = "PROTECTION";
    window.history.replaceState({}, "", "/procedure-details");
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads the active airport procedure details and shows the local chart link", async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith("/procedure-details/index.json")) return Promise.resolve(jsonResponse(sampleIndex));
      if (url.endsWith("/charts/index.json")) return Promise.resolve(jsonResponse(sampleCharts));
      if (url.endsWith("/procedure-details/KRDU-R05LY-RW05L.json")) {
        return Promise.resolve(jsonResponse(sampleDocument));
      }
      return Promise.resolve(notFoundResponse());
    });

    render(<ProcedureDetailsPage />);

    await waitFor(() => expect(screen.getAllByText("RNAV(GPS) Y RWY 05L").length).toBeGreaterThan(0));
    await waitFor(() =>
      expect(screen.getByRole("link", { name: "Open FAA Procedure Search" })).toBeTruthy(),
    );

    expect(fetchMock).toHaveBeenCalledWith("/data/airports/KRDU/procedure-details/index.json");
    expect(screen.getByRole("link", { name: "Open FAA Procedure Search" }).getAttribute("href")).toContain(
      "nasrId=RDU",
    );
    expect(screen.getByRole("link", { name: "Open Local Chart PDF" }).getAttribute("href")).toBe(
      "/data/airports/KRDU/charts/00516RY5L.PDF-nameddest-RDU.pdf",
    );

    fireEvent.click(screen.getAllByText("WEPAS")[0]);
    expect(await screen.findByText("final_approach_fix")).toBeTruthy();
    expect(screen.getAllByText(/DEFAULT_TOLERANCE/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Final surfaces").length).toBeGreaterThan(0);
    expect(screen.getByText("WEPAS 2,200 ft")).toBeTruthy();
    expect(screen.queryByText("GPA 3.0 deg est")).toBeNull();
    expect(screen.getByText(/Missing LPV_W, LPV_X, LPV_Y, LNAV_VNAV_OCS/)).toBeTruthy();
    expect(screen.getByText(/FINAL_VERTICAL_SURFACE_UNIMPLEMENTED/)).toBeTruthy();
  });

  it("uses procedure panel visibility and display level for 2D procedure charts", async () => {
    appState.procedureVisibility = { "KRDU-R05LY-RW05L:branch:R": false };
    appState.procedureDisplayLevel = "ESTIMATED";
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith("/procedure-details/index.json")) return Promise.resolve(jsonResponse(sampleIndex));
      if (url.endsWith("/charts/index.json")) return Promise.resolve(jsonResponse(sampleCharts));
      if (url.endsWith("/procedure-details/KRDU-R05LY-RW05L.json")) {
        return Promise.resolve(jsonResponse(sampleDocument));
      }
      return Promise.resolve(notFoundResponse());
    });

    render(<ProcedureDetailsPage />);

    expect(await screen.findByText("No positioned fixes available for the plan view yet.")).toBeTruthy();
    expect(screen.getByText("No altitude-supported samples are available for the vertical profile yet.")).toBeTruthy();

    appState.procedureVisibility = { "KRDU-R05LY-RW05L:branch:R": true };
    render(<ProcedureDetailsPage />);

    expect(await screen.findByText("GPA 3.0 deg est")).toBeTruthy();
    expect(screen.getByText("WEPAS 2,200 ft")).toBeTruthy();
  });

  it("switches procedure-details distance axes from nautical miles to metres", async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith("/procedure-details/index.json")) return Promise.resolve(jsonResponse(sampleIndex));
      if (url.endsWith("/charts/index.json")) return Promise.resolve(jsonResponse(sampleCharts));
      if (url.endsWith("/procedure-details/KRDU-R05LY-RW05L.json")) {
        return Promise.resolve(jsonResponse(sampleDocument));
      }
      return Promise.resolve(notFoundResponse());
    });

    render(<ProcedureDetailsPage />);

    expect(await screen.findByText("East offset from origin (NM)")).toBeTruthy();
    expect(screen.getByText("Along-track distance from MAPT / runway (NM)")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "m" }));

    expect(screen.getByText("East offset from origin (m)")).toBeTruthy();
    expect(screen.getByText("Along-track distance from MAPT / runway (m)")).toBeTruthy();
  });

  it("shows RF radius, turn direction, and center metadata in the focused sequence", async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith("/procedure-details/index.json")) return Promise.resolve(jsonResponse(sampleIndex));
      if (url.endsWith("/charts/index.json")) return Promise.resolve(jsonResponse(sampleCharts));
      if (url.endsWith("/procedure-details/KRDU-R05LY-RW05L.json")) {
        return Promise.resolve(jsonResponse(rfSampleDocument));
      }
      return Promise.resolve(notFoundResponse());
    });

    render(<ProcedureDetailsPage />);

    expect((await screen.findAllByText("RF")).length).toBeGreaterThan(0);
    expect(screen.getByText("Turn LEFT")).toBeTruthy();
    expect(screen.getByText("Radius 2.00 NM")).toBeTruthy();
    expect(screen.getByText("Center CENTER")).toBeTruthy();
    expect(screen.getByText("RF center CENTER")).toBeTruthy();
  });

  it("marks the missed approach section split in plan and profile views", async () => {
    appState.procedureDisplayLevel = "DEBUG";
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith("/procedure-details/index.json")) return Promise.resolve(jsonResponse(sampleIndex));
      if (url.endsWith("/charts/index.json")) return Promise.resolve(jsonResponse(sampleCharts));
      if (url.endsWith("/procedure-details/KRDU-R05LY-RW05L.json")) {
        return Promise.resolve(jsonResponse(missedSectionDocument));
      }
      return Promise.resolve(notFoundResponse());
    });

    render(<ProcedureDetailsPage />);

    expect((await screen.findAllByText("S1/S2")).length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("CA 305 deg").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("CA end est").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("DF leg").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("HM leg").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Turn debug").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("Turn TIA")).toBeTruthy();
    expect(screen.getAllByText("Missed section status").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/WARN UNSUPPORTED_LEG_TYPE/).length).toBeGreaterThan(0);
    expect(screen.getByText(/WARN TURNING_MISSED_UNIMPLEMENTED/)).toBeTruthy();
  });

  it("shows a friendly empty state when the richer dataset is missing", async () => {
    appState.activeAirportCode = "CYVR";
    fetchMock.mockResolvedValue(notFoundResponse());

    render(<ProcedureDetailsPage />);

    expect(
      await screen.findByText("No procedure-details dataset yet for CYVR"),
    ).toBeTruthy();
  });
});
