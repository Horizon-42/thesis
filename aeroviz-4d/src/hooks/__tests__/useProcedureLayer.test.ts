import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

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

const sampleGeoJson = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      geometry: {
        type: "LineString",
        coordinates: [
          [-78.9251472, 35.7734139, 914.4],
          [-78.8829556, 35.8087667, 670.56],
          [-78.8019631, 35.87445, 243.23],
        ],
      },
      properties: {
        featureType: "procedure-route",
        routeId: "KRDU-R05LY-R",
        procedureName: "RNAV(GPS) Y RW05L",
        defaultVisible: true,
        nominalSpeedKt: 140,
        tunnel: {
          lateralHalfWidthNm: 0.3,
          verticalHalfHeightFt: 300,
          sampleSpacingM: 10000,
        },
      },
    },
    {
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [-78.8829556, 35.8087667, 670.56],
      },
      properties: {
        featureType: "procedure-fix",
        routeId: "KRDU-R05LY-R",
        name: "WEPAS",
        sequence: 20,
        role: "FAF",
      },
    },
  ],
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
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => sampleGeoJson,
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads procedure route, tunnel, and fix entities", async () => {
    renderHook(() => useProcedureLayer());

    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    expect(fetchMock).toHaveBeenCalledWith("/data/airports/KRDU/procedures.geojson");
    expect(entities.some((entity) => String(entity.id).endsWith("-line"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-tunnel-"))).toBe(true);
    expect(entities.some((entity) => String(entity.id).includes("-fix-"))).toBe(true);
  });

  it("syncs procedure visibility without reloading data", async () => {
    const { rerender } = renderHook(() => useProcedureLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    setProceduresVisible(false);
    rerender();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(entities.every((entity) => entity.show === false)).toBe(true);
  });

  it("syncs individual route visibility without reloading data", async () => {
    const { rerender } = renderHook(() => useProcedureLayer());
    await waitFor(() => expect(mockViewer.entities.add).toHaveBeenCalled());

    setProcedureRouteVisible("KRDU-R05LY-R", false);
    rerender();

    expect(fetchMock).toHaveBeenCalledTimes(1);
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
