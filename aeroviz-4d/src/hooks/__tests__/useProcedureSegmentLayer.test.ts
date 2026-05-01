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
      ORANGE: new Color("ORANGE"),
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
  halfWidthNmSamples: [],
};

const renderBundleData = {
  index: {},
  documents: [],
  packages: [],
  renderBundles: [
    {
      packageId: "KRDU-R05LY-RW05L",
      procedureId: "R05LY",
      procedureName: "RNAV(GPS) Y RW05L",
      airportId: "KRDU",
      diagnostics: [],
      branchBundles: [
        {
          branchId: "branch:R",
          branchName: "RW05L",
          branchRole: "STRAIGHT_IN",
          runwayId: "RW05L",
          segmentBundles: [
            {
              segment: { segmentId: "segment:final", segmentType: "FINAL_LNAV" },
              legs: [],
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
                diagnostics: [],
              },
              finalOea: {
                primary: ribbon,
                secondaryOuter: ribbon,
              },
              alignedConnector: {
                primary: ribbon,
                secondaryOuter: ribbon,
              },
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
    setProcedureBranchVisible("branch:R", true);
  });

  it("renders segment centerline, envelopes, OEA, and connector entities", async () => {
    renderHook(() => useProcedureSegmentLayer());

    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    expect(loadProcedureRenderBundleData).toHaveBeenCalledWith("KRDU");
    expect(entities.some((entity) => String(entity.id).endsWith("-centerline"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-envelope-primary"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-oea-primary"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-connector-primary"))).toBe(true);
  });

  it("syncs layer visibility without reloading render bundle data", async () => {
    const { rerender } = renderHook(() => useProcedureSegmentLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    setProceduresVisible(false);
    rerender();

    expect(loadProcedureRenderBundleData).toHaveBeenCalledTimes(1);
    expect(entities.every((entity) => entity.show === false)).toBe(true);
  });
});
