import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  addEntity,
  removeEntity,
  requestRender,
  mockViewer,
} = vi.hoisted(() => {
  const addEntity = vi.fn((entity: unknown) => entity);
  const removeEntity = vi.fn();
  const requestRender = vi.fn();
  const mockViewer = {
    isDestroyed: vi.fn(() => false),
    entities: {
      add: addEntity,
      remove: removeEntity,
    },
    scene: {
      requestRender,
    },
  };
  return { addEntity, removeEntity, requestRender, mockViewer };
});

vi.mock("cesium", () => ({
  Cartesian2: class Cartesian2 {
    constructor(public x: number, public y: number) {}
  },
  Cartesian3: {
    fromDegrees: (lon: number, lat: number, alt = 0) => ({ lon, lat, alt }),
  },
  Math: {
    toRadians: (degrees: number) => degrees * globalThis.Math.PI / 180,
  },
  ConstantPositionProperty: class ConstantPositionProperty {
    constructor(public value: unknown) {}
  },
  ConstantProperty: class ConstantProperty {
    constructor(public value: unknown) {}
  },
  Color: {
    WHITE: "white",
    BLACK: "black",
    fromCssColorString: () => ({ withAlpha: () => "color" }),
  },
  LabelStyle: { FILL_AND_OUTLINE: "FILL_AND_OUTLINE" },
  VerticalOrigin: { BOTTOM: "BOTTOM" },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
  }),
}));

import {
  buildTargetGatePositions,
  usePilotTargetGate,
} from "../usePilotTargetGate";

describe("buildTargetGatePositions", () => {
  it("builds a closed vertical threshold gate", () => {
    const positions = buildTargetGatePositions({
      lon: -78.8,
      lat: 35.8,
      altM: 100,
      headingDeg: 90,
    }) as unknown as Array<{ lon: number; lat: number; alt: number }>;

    expect(positions).toHaveLength(5);
    expect(positions[0]).toEqual(positions[4]);
    expect(positions[0].alt).toBe(104);
    expect(positions[1].alt).toBe(146);
    expect(positions[2].alt).toBe(146);
    expect(Math.abs(positions[0].lon - positions[2].lon)).toBeGreaterThan(0.0005);
  });
});

describe("usePilotTargetGate", () => {
  beforeEach(() => {
    addEntity.mockClear();
    removeEntity.mockClear();
    requestRender.mockClear();
  });

  it("adds and removes the threshold gate entities", () => {
    const target = {
      runwayThresholdId: "RW05L",
      runwayIdent: "RW05L",
      lon: -78.802,
      lat: 35.874,
      altM: 111.86,
      headingDeg: 45,
    };

    const { rerender } = renderHook(
      ({ enabled }) => usePilotTargetGate({ enabled, target }),
      { initialProps: { enabled: true } },
    );

    expect(addEntity).toHaveBeenCalledTimes(2);
    expect(addEntity.mock.calls[0][0]).toEqual(
      expect.objectContaining({
        id: "trajectory-play-target-threshold-gate",
        polyline: expect.objectContaining({
          positions: expect.arrayContaining([
            expect.objectContaining({ alt: 115.86 }),
          ]),
        }),
      }),
    );
    expect(addEntity.mock.calls[1][0]).toEqual(
      expect.objectContaining({
        id: "trajectory-play-target-threshold-marker",
        label: expect.objectContaining({ text: "TARGET RW05L" }),
      }),
    );

    rerender({ enabled: false });

    expect(removeEntity).toHaveBeenCalledTimes(2);
    expect(requestRender).toHaveBeenCalled();
  });
});
