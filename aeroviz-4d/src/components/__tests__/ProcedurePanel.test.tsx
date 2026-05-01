import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

function jsonResponse(body: unknown) {
  return {
    ok: true,
    headers: { get: () => "application/json" },
    text: async () => JSON.stringify(body),
  };
}

const {
  fetchMock,
  navigateWithinApp,
  toggleLayer,
  setProcedureBranchVisible,
  setProcedureBranchesVisible,
  getProcedureVisibility,
  setSelectedProfileRunwayIdent,
  setRunwayProfileOpen,
  getSelectedProfileRunwayIdent,
  getRunwayProfileOpen,
} = vi.hoisted(() => {
  let procedureVisibility: Record<string, boolean> = {};
  let selectedProfileRunwayIdent: string | null = null;
  let isRunwayProfileOpen = false;
  return {
    fetchMock: vi.fn(),
    navigateWithinApp: vi.fn(),
    toggleLayer: vi.fn(),
    setProcedureBranchVisible: vi.fn((branchId: string, visible: boolean) => {
      procedureVisibility = { ...procedureVisibility, [branchId]: visible };
    }),
    setProcedureBranchesVisible: vi.fn((branchIds: string[], visible: boolean) => {
      const next = { ...procedureVisibility };
      branchIds.forEach((branchId) => {
        next[branchId] = visible;
      });
      procedureVisibility = next;
    }),
    setSelectedProfileRunwayIdent: vi.fn((runwayIdent: string | null) => {
      selectedProfileRunwayIdent = runwayIdent;
    }),
    setRunwayProfileOpen: vi.fn((open: boolean) => {
      isRunwayProfileOpen = open;
    }),
    getProcedureVisibility: () => procedureVisibility,
    getSelectedProfileRunwayIdent: () => selectedProfileRunwayIdent,
    getRunwayProfileOpen: () => isRunwayProfileOpen,
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    layers: { procedures: true },
    activeAirportCode: "KRDU",
    toggleLayer,
    procedureVisibility: getProcedureVisibility(),
    setProcedureBranchVisible,
    setProcedureBranchesVisible,
    selectedProfileRunwayIdent: getSelectedProfileRunwayIdent(),
    setSelectedProfileRunwayIdent,
    isRunwayProfileOpen: getRunwayProfileOpen(),
    setRunwayProfileOpen,
  }),
}));

vi.mock("../../utils/navigation", () => ({
  navigateWithinApp,
}));

import ProcedurePanel from "../ProcedurePanel";

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
    {
      runwayIdent: "RW23R",
      chartName: "RW23R",
      procedureUids: ["KRDU-R23RY-RW23R"],
      procedures: [
        {
          procedureUid: "KRDU-R23RY-RW23R",
          procedureIdent: "R23RY",
          chartName: "RNAV(GPS) Y RW23R",
          procedureFamily: "RNAV_GPS",
          variant: "Y",
          approachModes: ["GPS"],
          runwayIdent: "RW23R",
          defaultBranchId: "branch:R",
        },
      ],
    },
  ],
};

function makeLeg(sequence: number, fixId: string, role: string) {
  return {
    legId: `leg:${sequence}`,
    sequence,
    segmentType: "final",
    path: {
      pathTerminator: sequence === 10 ? "IF" : "TF",
      constructionMethod: "track",
      startFixRef: null,
      endFixRef: fixId,
    },
    termination: { kind: "fix", fixRef: fixId },
    constraints: {
      altitude: { qualifier: "AT", valueFt: 2000 - sequence, rawText: String(2000 - sequence) },
      speedKt: null,
      geometryAltitudeFt: 2000 - sequence,
    },
    roleAtEnd: role,
    sourceRefs: [],
    quality: { status: "parsed", sourceLine: sequence, renderedInPlanView: true },
  };
}

function makeDocument(options: {
  procedureUid: string;
  runwayIdent: string;
  procedureIdent: string;
  chartName: string;
  branches: Array<{
    branchIdent: string;
    branchRole: string;
    defaultVisible: boolean;
    warnings: string[];
  }>;
}) {
  const fixes = [
    {
      fixId: "fix:A",
      ident: "FIXA",
      kind: "waypoint",
      position: { lon: -78.9, lat: 35.7 },
      elevationFt: null,
      roleHints: ["IF"],
      sourceRefs: [],
    },
    {
      fixId: "fix:B",
      ident: options.runwayIdent,
      kind: "runway",
      position: { lon: -78.8, lat: 35.8 },
      elevationFt: 400,
      roleHints: ["MAPt"],
      sourceRefs: [],
    },
  ];

  return {
    schemaVersion: "1.0",
    modelType: "procedure-detail",
    procedureUid: options.procedureUid,
    provenance: {
      assemblyMode: "test",
      researchUseOnly: true,
      sources: [{ sourceId: "cifp", kind: "CIFP", cycle: "2603" }],
      warnings: [],
    },
    airport: { icao: "KRDU", faa: "RDU", name: "Raleigh Durham Intl" },
    runway: {
      ident: options.runwayIdent,
      landingThresholdFixRef: "fix:B",
      threshold: { lon: -78.8, lat: 35.8, elevationFt: 400 },
    },
    procedure: {
      procedureType: "R",
      procedureFamily: "RNAV_GPS",
      procedureIdent: options.procedureIdent,
      chartName: options.chartName,
      variant: "Y",
      runwayIdent: options.runwayIdent,
      baseBranchIdent: "R",
      approachModes: ["GPS"],
    },
    fixes,
    branches: options.branches.map((branch, index) => ({
      branchId: `branch:${branch.branchIdent}`,
      branchKey: branch.branchIdent,
      branchIdent: branch.branchIdent,
      procedureType: "R",
      transitionIdent: branch.branchRole === "transition" ? branch.branchIdent : null,
      branchRole: branch.branchRole,
      sequenceOrder: index,
      mergeFixRef: null,
      continuesWithBranchId: null,
      defaultVisible: branch.defaultVisible,
      warnings: branch.warnings,
      legs: [makeLeg(10, "fix:A", "IF"), makeLeg(20, "fix:B", "MAPt")],
    })),
    verticalProfiles: [],
    validation: {
      expectedRunwayIdent: options.runwayIdent,
      expectedIF: null,
      expectedFAF: null,
      expectedMAPt: options.runwayIdent,
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
}

const sampleDocuments = {
  "KRDU-R05LY-RW05L": makeDocument({
    procedureUid: "KRDU-R05LY-RW05L",
    runwayIdent: "RW05L",
    procedureIdent: "R05LY",
    chartName: "RNAV(GPS) Y RW05L",
    branches: [
      {
        branchIdent: "R",
        branchRole: "final",
        defaultVisible: true,
        warnings: ["Skipped unsupported leg CA at sequence 040"],
      },
      { branchIdent: "AOTTOS", branchRole: "transition", defaultVisible: false, warnings: [] },
    ],
  }),
  "KRDU-R23RY-RW23R": makeDocument({
    procedureUid: "KRDU-R23RY-RW23R",
    runwayIdent: "RW23R",
    procedureIdent: "R23RY",
    chartName: "RNAV(GPS) Y RW23R",
    branches: [{ branchIdent: "R", branchRole: "final", defaultVisible: true, warnings: [] }],
  }),
};

describe("ProcedurePanel", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    toggleLayer.mockClear();
    setProcedureBranchVisible.mockClear();
    setProcedureBranchesVisible.mockClear();
    setSelectedProfileRunwayIdent.mockClear();
    setRunwayProfileOpen.mockClear();
    navigateWithinApp.mockClear();
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith("/procedure-details/index.json")) return Promise.resolve(jsonResponse(sampleIndex));
      const match = url.match(/\/procedure-details\/(.+)\.json$/);
      if (match) {
        const document = sampleDocuments[match[1] as keyof typeof sampleDocuments];
        if (document) return Promise.resolve(jsonResponse(document));
      }
      return Promise.reject(new Error(`Unexpected URL ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("groups procedure branches by runway and exposes branch toggles", async () => {
    render(<ProcedurePanel />);

    await waitFor(() => expect(screen.getByText("RW05L")).toBeTruthy());

    expect(fetchMock).toHaveBeenCalledWith("/data/airports/KRDU/procedure-details/index.json");
    expect(screen.getByText("KRDU CIFP 2603")).toBeTruthy();
    expect(screen.getByText("3 branches")).toBeTruthy();
    expect(screen.getByText("2 runways")).toBeTruthy();
    expect(screen.getByText("5 warnings")).toBeTruthy();
    expect(screen.getByText("RNAV(GPS) Y RW05L")).toBeTruthy();
    expect(screen.getByText("AOTTOS")).toBeTruthy();
  });

  it("updates route visibility when branch checkbox changes", async () => {
    render(<ProcedurePanel />);
    await waitFor(() => expect(screen.getByText("AOTTOS")).toBeTruthy());

    expect(fetchMock).toHaveBeenCalledWith("/data/airports/KRDU/procedure-details/index.json");

    const labels = screen.getAllByText("AOTTOS");
    const label = labels[0].closest("label");
    const checkbox = label?.querySelector("input");
    expect(checkbox).toBeTruthy();

    fireEvent.click(checkbox as HTMLInputElement);

    expect(setProcedureBranchVisible).toHaveBeenCalledWith(
      "KRDU-R05LY-RW05L:branch:AOTTOS",
      true,
    );
  });

  it("opens the runway trajectory profile for a runway group", async () => {
    render(<ProcedurePanel />);
    await waitFor(() => expect(screen.getByText("RW05L")).toBeTruthy());

    fireEvent.click(screen.getAllByRole("button", { name: "Profile" })[0]);

    expect(setSelectedProfileRunwayIdent).toHaveBeenCalledWith("RW05L");
    expect(setRunwayProfileOpen).toHaveBeenCalledWith(true);
  });

  it("opens procedure details from the panel footer", async () => {
    render(<ProcedurePanel />);
    await waitFor(() => expect(screen.getByText("RW05L")).toBeTruthy());

    fireEvent.click(screen.getByRole("button", { name: "Procedure Details" }));

    expect(navigateWithinApp).toHaveBeenCalledWith("/procedure-details?airport=KRDU");
  });
});
