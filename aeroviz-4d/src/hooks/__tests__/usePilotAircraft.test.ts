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

    setValue(value: unknown) {
      this.value = value;
    }
  },
  ConstantProperty: class ConstantProperty {
    constructor(public value: unknown) {}

    setValue(value: unknown) {
      this.value = value;
    }
  },
  ColorMaterialProperty: class ColorMaterialProperty {
    constructor(public color: unknown) {}
  },
  CallbackProperty: class CallbackProperty {
    constructor(
      public callback: () => unknown,
      public isConstant: boolean,
    ) {}

    getValue() {
      return this.callback();
    }
  },
  Color: {
    WHITE: "white",
    BLACK: "black",
    fromCssColorString: (color: string) => ({
      withAlpha: (alpha: number) => `${color}:${alpha}`,
    }),
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
    mockViewer.trackedEntity = undefined;
    mockViewer.entities.add.mockClear();
    mockViewer.entities.remove.mockClear();
    mockViewer.scene.requestRender.mockClear();
    setSelectedFlightId.mockClear();
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
          attackDeg: 0,
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
          attackDeg: 0,
        },
        trail: [],
        follow: false,
      }),
    );

    expect(capturedHeadingPitchRolls[0].roll).toBeCloseTo(-12 * Math.PI / 180);
  });

  it("adds attack angle to flight path pitch for aircraft attitude", () => {
    renderHook(() =>
      usePilotAircraft({
        enabled: true,
        pose: {
          lon: 0,
          lat: 0,
          altM: 1000,
          headingDeg: 0,
          flightPathDeg: 2,
          bankDeg: 0,
          attackDeg: 4,
        },
        trail: [],
        follow: false,
      }),
    );

    expect(capturedHeadingPitchRolls[0].pitch).toBeCloseTo(6 * Math.PI / 180);
  });

  it("keeps Cesium properties stable while updating aircraft pose and trail points", () => {
    const firstPose = {
      lon: -114,
      lat: 51,
      altM: 1100,
      headingDeg: 4,
      flightPathDeg: 1,
      bankDeg: 0,
      attackDeg: 0,
    };
    const secondPose = {
      lon: -113.99,
      lat: 51.01,
      altM: 1120,
      headingDeg: 8,
      flightPathDeg: 2,
      bankDeg: 1,
      attackDeg: 3,
    };
    const thirdPose = {
      lon: -113.98,
      lat: 51.02,
      altM: 1130,
      headingDeg: 9,
      flightPathDeg: 2,
      bankDeg: 1,
      attackDeg: 3,
    };
    const firstTrail = [firstPose, secondPose];
    const secondTrail = [firstPose, secondPose, thirdPose];

    const { rerender } = renderHook(
      ({ pose, trail }) =>
        usePilotAircraft({
          enabled: true,
          pose,
          trail,
          follow: false,
        }),
      {
        initialProps: {
          pose: firstPose,
          trail: firstTrail,
        },
      },
    );

    const aircraft = mockViewer.entities.add.mock.calls[0][0] as any;
    const trail = mockViewer.entities.add.mock.calls[1][0] as any;
    const positionProperty = aircraft.position;
    const orientationProperty = aircraft.orientation;
    const trailPositionsProperty = trail.polyline.positions;

    rerender({
      pose: secondPose,
      trail: secondTrail,
    });

    expect(aircraft.position).toBe(positionProperty);
    expect(aircraft.orientation).toBe(orientationProperty);
    expect(trail.polyline.positions).toBe(trailPositionsProperty);
    expect(positionProperty.value).toEqual({
      lon: secondPose.lon,
      lat: secondPose.lat,
      alt: secondPose.altM,
    });
    expect(trailPositionsProperty.getValue()).toEqual([
      { lon: firstPose.lon, lat: firstPose.lat, alt: firstPose.altM },
      { lon: secondPose.lon, lat: secondPose.lat, alt: secondPose.altM },
      { lon: thirdPose.lon, lat: thirdPose.lat, alt: thirdPose.altM },
    ]);
  });

  it("draws replay trail segments with separate colors", () => {
    const point0 = {
      lon: -114,
      lat: 51,
      altM: 1100,
      headingDeg: 4,
      flightPathDeg: 1,
      bankDeg: 0,
      attackDeg: 0,
      segmentIndex: 0,
    };
    const point1 = {
      ...point0,
      lon: -113.99,
      lat: 51.01,
      segmentIndex: 0,
    };
    const point2 = {
      ...point0,
      lon: -113.98,
      lat: 51.02,
      segmentIndex: 1,
    };
    const point3 = {
      ...point0,
      lon: -113.97,
      lat: 51.03,
      segmentIndex: 1,
    };

    renderHook(() =>
      usePilotAircraft({
        enabled: true,
        pose: point3,
        trail: [point0, point1, point2, point3],
        follow: false,
      }),
    );

    const firstTrailSegment = mockViewer.entities.add.mock.calls[1][0] as any;
    const secondTrailSegment = mockViewer.entities.add.mock.calls[2][0] as any;

    expect(firstTrailSegment.polyline.material.color).toBe("#67e8f9:0.78");
    expect(secondTrailSegment.polyline.material.color).toBe("#f97316:0.78");
    expect(firstTrailSegment.polyline.positions.getValue()).toEqual([
      { lon: point0.lon, lat: point0.lat, alt: point0.altM },
      { lon: point1.lon, lat: point1.lat, alt: point1.altM },
    ]);
    expect(secondTrailSegment.polyline.positions.getValue()).toEqual([
      { lon: point1.lon, lat: point1.lat, alt: point1.altM },
      { lon: point2.lon, lat: point2.lat, alt: point2.altM },
      { lon: point3.lon, lat: point3.lat, alt: point3.altM },
    ]);
  });
});
