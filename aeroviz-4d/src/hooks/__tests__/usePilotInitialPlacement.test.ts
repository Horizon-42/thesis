import type * as Cesium from "cesium";
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  addEntity,
  removeEntity,
  requestRender,
  setInputAction,
  destroyHandler,
  setSelectedFlightId,
  mockViewer,
} = vi.hoisted(() => {
  const addEntity = vi.fn((entity: unknown) => entity);
  const removeEntity = vi.fn();
  const requestRender = vi.fn();
  const setInputAction = vi.fn();
  const destroyHandler = vi.fn();
  const setSelectedFlightId = vi.fn();
  const mockViewer = {
    isDestroyed: vi.fn(() => false),
    trackedEntity: undefined as unknown,
    entities: {
      add: addEntity,
      remove: removeEntity,
    },
    scene: {
      canvas: { style: {} },
      screenSpaceCameraController: {
        enableLook: true,
        enableRotate: true,
        enableTilt: true,
        enableTranslate: true,
      },
      pickPositionSupported: false,
      globe: {
        pick: vi.fn(),
        ellipsoid: {},
      },
      requestRender,
    },
    camera: {
      getPickRay: vi.fn(),
      pickEllipsoid: vi.fn(),
    },
  };
  return {
    addEntity,
    removeEntity,
    requestRender,
    setInputAction,
    destroyHandler,
    setSelectedFlightId,
    mockViewer,
  };
});

vi.mock("cesium", () => ({
  Cartesian2: class Cartesian2 {
    constructor(public x: number, public y: number) {}
  },
  Cartesian3: {
    fromDegrees: (lon: number, lat: number, alt = 0) => ({ lon, lat, alt }),
  },
  Cartographic: {
    fromCartesian: (cartesian: { latitudeRad: number; longitudeRad: number }) => ({
      latitude: cartesian.latitudeRad,
      longitude: cartesian.longitudeRad,
    }),
  },
  Math: {
    toDegrees: (radians: number) => radians * 180 / globalThis.Math.PI,
    toRadians: (degrees: number) => degrees * globalThis.Math.PI / 180,
  },
  Transforms: {
    headingPitchRollQuaternion: () => ({ quaternion: true }),
  },
  HeadingPitchRoll: class HeadingPitchRoll {
    constructor(
      public heading: number,
      public pitch: number,
      public roll: number,
    ) {}
  },
  ConstantPositionProperty: class ConstantPositionProperty {
    constructor(public value: unknown) {}
  },
  ConstantProperty: class ConstantProperty {
    constructor(public value: unknown) {}
  },
  PolygonHierarchy: class PolygonHierarchy {
    constructor(public positions: unknown[]) {}
  },
  PolygonGraphics: class PolygonGraphics {
    constructor(public options: unknown) {}
  },
  Color: {
    WHITE: "white",
    BLACK: "black",
    fromCssColorString: () => ({ withAlpha: () => "color" }),
  },
  ColorBlendMode: { MIX: "MIX" },
  HeightReference: { CLAMP_TO_GROUND: "CLAMP_TO_GROUND" },
  LabelStyle: { FILL_AND_OUTLINE: "FILL_AND_OUTLINE" },
  ScreenSpaceEventHandler: class ScreenSpaceEventHandler {
    constructor(public canvas: unknown) {}
    setInputAction = setInputAction;
    destroy = destroyHandler;
  },
  ScreenSpaceEventType: {
    LEFT_DOWN: "LEFT_DOWN",
    LEFT_UP: "LEFT_UP",
    MOUSE_MOVE: "MOUSE_MOVE",
  },
  VerticalOrigin: {
    BOTTOM: "BOTTOM",
  },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
    setSelectedFlightId,
  }),
}));

import {
  pickPilotPlacementPosition,
  usePilotInitialPlacement,
} from "../usePilotInitialPlacement";

describe("pickPilotPlacementPosition", () => {
  it("uses scene depth picking when available", () => {
    const pickPosition = vi.fn(() => cartesianFromDegrees(-78.7873, 35.8787));
    const globePick = vi.fn();
    const pickEllipsoid = vi.fn();

    const position = pickPilotPlacementPosition(
      makeViewer({ pickPositionSupported: true, pickPosition, globePick, pickEllipsoid }),
      screenPosition(),
    );

    expect(position?.lon).toBeCloseTo(-78.7873);
    expect(position?.lat).toBeCloseTo(35.8787);
    expect(globePick).not.toHaveBeenCalled();
    expect(pickEllipsoid).not.toHaveBeenCalled();
  });

  it("falls back to globe ray picking when depth picking misses", () => {
    const pickPosition = vi.fn(() => undefined);
    const globePick = vi.fn(() => cartesianFromDegrees(-79.1, 36.2));
    const pickEllipsoid = vi.fn();

    const position = pickPilotPlacementPosition(
      makeViewer({ pickPositionSupported: true, pickPosition, globePick, pickEllipsoid }),
      screenPosition(),
    );

    expect(position?.lon).toBeCloseTo(-79.1);
    expect(position?.lat).toBeCloseTo(36.2);
    expect(globePick).toHaveBeenCalled();
    expect(pickEllipsoid).not.toHaveBeenCalled();
  });

  it("falls back to ellipsoid picking when no terrain ray intersects", () => {
    const pickEllipsoid = vi.fn(() => cartesianFromDegrees(-80.25, 34.5));

    const position = pickPilotPlacementPosition(
      makeViewer({
        pickPositionSupported: false,
        getPickRay: vi.fn(() => undefined),
        pickEllipsoid,
      }),
      screenPosition(),
    );

    expect(position?.lon).toBeCloseTo(-80.25);
    expect(position?.lat).toBeCloseTo(34.5);
    expect(pickEllipsoid).toHaveBeenCalled();
  });
});

describe("usePilotInitialPlacement", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("keeps the initial aircraft preview after placement mode ends", () => {
    const props = {
      enabled: true,
      previewVisible: true,
      initialState: {
        lon: -78.7873,
        lat: 35.878659,
        altM: 1000,
        speedMps: 120,
        headingDeg: 0,
        flightPathDeg: 0,
        massKg: 78000,
        aircraftType: "A320" as const,
      },
      placementGuidance: null,
      onPositionChange: vi.fn(),
      onFinish: vi.fn(),
      onCancel: vi.fn(),
    };

    const { rerender } = renderHook(
      (nextProps: typeof props) => usePilotInitialPlacement(nextProps),
      { initialProps: props },
    );

    expect(addEntity).toHaveBeenCalledTimes(2);

    rerender({ ...props, enabled: false, previewVisible: true });
    expect(removeEntity).not.toHaveBeenCalled();

    rerender({ ...props, enabled: false, previewVisible: false });
    expect(removeEntity).toHaveBeenCalledTimes(2);
    expect(requestRender).toHaveBeenCalled();
  });

  it("draws the selected runway final approach placement band", () => {
    renderHook(() =>
      usePilotInitialPlacement({
        enabled: true,
        previewVisible: false,
        initialState: {
          lon: -78.7873,
          lat: 35.878659,
          altM: 1000,
          speedMps: 120,
          headingDeg: 0,
          flightPathDeg: 0,
          massKg: 78000,
          aircraftType: "A320",
        },
        placementGuidance: {
          runway: {
            id: "RW05L",
            runwayIdent: "RW05L",
            runwayPairIdent: "05L/23R",
            lon: -78.802,
            lat: 35.874,
            altM: 111.86,
            psiDeg: 45,
          },
          aircraft: {
            code: "A320",
            name: "Airbus A320-200",
            category: "narrow_body",
            massKg: 78000,
            wingAreaM2: 122.6,
            maxThrustN: 240000,
            approachThrustGuessN: 40000,
            terminalSpeedKt: 145,
            terminalSpeedMinKt: 135,
            terminalSpeedMaxKt: 155,
            finalApproachMinNm: 5,
            finalApproachMaxNm: 10,
            finalApproachLateralHalfWidthNm: 0.8,
            finalApproachGlideAngleDeg: 3,
            thresholdCrossingHeightM: 15,
          },
        },
        onPositionChange: vi.fn(),
        onFinish: vi.fn(),
        onCancel: vi.fn(),
      }),
    );

    expect(addEntity).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "pilot-initial-placement-guidance",
        polygon: expect.any(Object),
      }),
    );
  });
});

function makeViewer({
  pickPositionSupported,
  pickPosition = vi.fn(),
  getPickRay = vi.fn(() => ({ origin: "camera" })),
  globePick = vi.fn(),
  pickEllipsoid = vi.fn(),
}: {
  pickPositionSupported: boolean;
  pickPosition?: ReturnType<typeof vi.fn>;
  getPickRay?: ReturnType<typeof vi.fn>;
  globePick?: ReturnType<typeof vi.fn>;
  pickEllipsoid?: ReturnType<typeof vi.fn>;
}): Cesium.Viewer {
  return {
    scene: {
      pickPositionSupported,
      pickPosition,
      globe: {
        pick: globePick,
        ellipsoid: {},
      },
    },
    camera: {
      getPickRay,
      pickEllipsoid,
    },
  } as unknown as Cesium.Viewer;
}

function cartesianFromDegrees(lon: number, lat: number): unknown {
  return {
    longitudeRad: lon * globalThis.Math.PI / 180,
    latitudeRad: lat * globalThis.Math.PI / 180,
  };
}

function screenPosition(): Cesium.Cartesian2 {
  return { x: 12, y: 24 } as Cesium.Cartesian2;
}
