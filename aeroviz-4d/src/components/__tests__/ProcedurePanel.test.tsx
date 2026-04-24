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
  setProcedureRouteVisible,
  setProcedureRoutesVisible,
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
    setProcedureRouteVisible: vi.fn((routeId: string, visible: boolean) => {
      procedureVisibility = { ...procedureVisibility, [routeId]: visible };
    }),
    setProcedureRoutesVisible: vi.fn((routeIds: string[], visible: boolean) => {
      const next = { ...procedureVisibility };
      routeIds.forEach((routeId) => {
        next[routeId] = visible;
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
    setProcedureRouteVisible,
    setProcedureRoutesVisible,
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

const sampleGeoJson = {
  type: "FeatureCollection",
  metadata: {
    sourceCycle: "2603",
  },
  features: [
    {
      type: "Feature",
      geometry: { type: "LineString", coordinates: [] },
      properties: {
        featureType: "procedure-route",
        routeId: "KRDU-R05LY-R",
        runwayIdent: "RW05L",
        procedureIdent: "R05LY",
        procedureName: "RNAV(GPS) Y RW05L",
        procedureFamily: "RNAV_GPS",
        branchIdent: "R",
        branchType: "final",
        defaultVisible: true,
        warnings: ["Skipped unsupported leg CA at sequence 040"],
      },
    },
    {
      type: "Feature",
      geometry: { type: "LineString", coordinates: [] },
      properties: {
        featureType: "procedure-route",
        routeId: "KRDU-R05LY-AOTTOS",
        runwayIdent: "RW05L",
        procedureIdent: "R05LY",
        procedureName: "RNAV(GPS) Y RW05L",
        procedureFamily: "RNAV_GPS",
        branchIdent: "AOTTOS",
        branchType: "transition",
        defaultVisible: false,
        warnings: [],
      },
    },
    {
      type: "Feature",
      geometry: { type: "LineString", coordinates: [] },
      properties: {
        featureType: "procedure-route",
        routeId: "KRDU-R23RY-R",
        runwayIdent: "RW23R",
        procedureIdent: "R23RY",
        procedureName: "RNAV(GPS) Y RW23R",
        procedureFamily: "RNAV_GPS",
        branchIdent: "R",
        branchType: "final",
        defaultVisible: true,
        warnings: [],
      },
    },
  ],
};

describe("ProcedurePanel", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    toggleLayer.mockClear();
    setProcedureRouteVisible.mockClear();
    setProcedureRoutesVisible.mockClear();
    setSelectedProfileRunwayIdent.mockClear();
    setRunwayProfileOpen.mockClear();
    navigateWithinApp.mockClear();
    fetchMock.mockResolvedValue(jsonResponse(sampleGeoJson));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("groups procedure routes by runway and exposes branch toggles", async () => {
    render(<ProcedurePanel />);

    await waitFor(() => expect(screen.getByText("RW05L")).toBeTruthy());

    expect(fetchMock).toHaveBeenCalledWith("/data/airports/KRDU/procedures.geojson");
    expect(screen.getByText("KRDU CIFP 2603")).toBeTruthy();
    expect(screen.getByText("3 branches")).toBeTruthy();
    expect(screen.getByText("2 runways")).toBeTruthy();
    expect(screen.getByText("1 warnings")).toBeTruthy();
    expect(screen.getByText("RNAV(GPS) Y RW05L")).toBeTruthy();
    expect(screen.getByText("AOTTOS")).toBeTruthy();
  });

  it("updates route visibility when branch checkbox changes", async () => {
    render(<ProcedurePanel />);
    await waitFor(() => expect(screen.getByText("AOTTOS")).toBeTruthy());

    expect(fetchMock).toHaveBeenCalledWith("/data/airports/KRDU/procedures.geojson");

    const labels = screen.getAllByText("AOTTOS");
    const label = labels[0].closest("label");
    const checkbox = label?.querySelector("input");
    expect(checkbox).toBeTruthy();

    fireEvent.click(checkbox as HTMLInputElement);

    expect(setProcedureRouteVisible).toHaveBeenCalledWith("KRDU-R05LY-AOTTOS", true);
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
