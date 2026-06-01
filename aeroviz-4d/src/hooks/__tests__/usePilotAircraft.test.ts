import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  setSelectedFlightId,
  mockViewer,
  capturedHeadingPitchRolls,
} = vi.hoisted(() => {
  const addEntity = vi.fn((entity: unknown) => entity);
  const removeEntity = vi.fn();
  const requestRender = vi.fn();
  const setSelectedFlightId = vi.fn();
  const capturedHeadingPitchRolls: Array<{ heading: number; pitch: number; roll: number }> = [];
  const mockViewer = {
    isDestroyed: vi.fn(() => false),
    trackedEntity: undefined as any,
    entities: {
      add: addEntity,
      remove: removeEntity,
    },
    scene: {
      requestRender,
    },
  };
  return {
    setSelectedFlightId,
    mockViewer,
    capturedHeadingPitchRolls,
  };
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
  Transforms: {
    headingPitchRollQuaternion: (_position: unknown, hpr: unknown) => ({ hpr }),
  },
  HeadingPitchRoll: class HeadingPitchRoll {
    constructor(
      public heading: number,
      public pitch: number,
      public roll: number,
    ) {
      capturedHeadingPitchRolls.push({ heading, pitch, roll });
    }
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
    setSelectedFlightId,
  }),
}));

import { usePilotAircraft } from "../usePilotAircraft";

describe("usePilotAircraft", () => {
  beforeEach(() => {
    capturedHeadingPitchRolls.length = 0;
  });

  it("converts simulator psi degrees to Cesium heading sign", () => {
    // Simulator psi increases from east toward north, while Cesium HPR heading increases from east toward south.
    renderHook(() =>
      usePilotAircraft({
        enabled: true,
        pose: {
          lon: 0,
          lat: 0,
          altM: 1000,
          headingDeg: 10,
          flightPathDeg: 0,
          bankDeg: 0,
        },
        trail: [],
        follow: false,
      }),
    );

    expect(capturedHeadingPitchRolls[0].heading).toBeCloseTo(-10 * Math.PI / 180);
  });

  it("converts simulator bank degrees to Cesium roll sign", () => {
    // Positive simulator bank turns left in ENU; Cesium positive roll visually banks right for the same heading frame.
    renderHook(() =>
      usePilotAircraft({
        enabled: true,
        pose: {
          lon: 0,
          lat: 0,
          altM: 1000,
          headingDeg: 0,
          flightPathDeg: 0,
          bankDeg: 12,
        },
        trail: [],
        follow: false,
      }),
    );

    expect(capturedHeadingPitchRolls[0].roll).toBeCloseTo(-12 * Math.PI / 180);
  });
});
