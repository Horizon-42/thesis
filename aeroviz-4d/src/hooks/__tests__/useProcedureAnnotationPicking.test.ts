import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  attachProcedureAnnotation,
  type ProcedureEntityAnnotation,
} from "../../data/procedureAnnotations";

const annotation: ProcedureEntityAnnotation = {
  entityId: "procedure-entity-1",
  label: "FINAL_LNAV",
  title: "Final centerline",
  kind: "SEGMENT_CENTERLINE",
  status: "SOURCE_BACKED",
  airportId: "KRDU",
  runwayId: "RW32",
  procedureUid: "KRDU-R32-RW32",
  procedureId: "R32",
  procedureName: "RNAV(GPS) RWY 32",
  branchId: "branch:R",
  branchName: "RW32",
  branchRole: "STRAIGHT_IN",
  meaning: "test",
  parameters: [],
  diagnostics: [],
  sourceRefs: [],
};

const { setInputAction, setSelectedProcedureAnnotation, mockViewer } = vi.hoisted(() => {
  const setInputAction = vi.fn();
  const setSelectedProcedureAnnotation = vi.fn();
  const mockViewer = {
    clock: { currentTime: "now" },
    entities: {
      getById: vi.fn(() => ({ position: { getValue: () => ({ x: 1, y: 2, z: 3 }) } })),
      removeById: vi.fn(),
      add: vi.fn(),
    },
    scene: {
      canvas: {},
      pick: vi.fn(),
      pickPositionSupported: false,
    },
  };
  return { setInputAction, setSelectedProcedureAnnotation, mockViewer };
});

vi.mock("cesium", () => ({
  Color: {
    WHITE: { withAlpha: () => "white" },
    ORANGE: { withAlpha: () => "orange" },
  },
  ScreenSpaceEventType: {
    LEFT_CLICK: "LEFT_CLICK",
    LEFT_DOUBLE_CLICK: "LEFT_DOUBLE_CLICK",
  },
  ScreenSpaceEventHandler: class ScreenSpaceEventHandler {
    constructor(public canvas: unknown) {}
    setInputAction = setInputAction;
    destroy = vi.fn();
  },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
    layers: { procedures: true },
    procedureAnnotationEnabled: true,
    setSelectedProcedureAnnotation,
  }),
}));

import { useProcedureAnnotationPicking } from "../useProcedureAnnotationPicking";

describe("useProcedureAnnotationPicking", () => {
  it("selects procedure annotations on single click", () => {
    const pickedEntity = attachProcedureAnnotation({ id: "procedure-entity-1" }, annotation);
    mockViewer.scene.pick.mockReturnValue({ id: pickedEntity });

    renderHook(() => useProcedureAnnotationPicking());

    expect(setInputAction).toHaveBeenCalledWith(expect.any(Function), "LEFT_CLICK");
    const handler = setInputAction.mock.calls[0][0] as (event: { position: unknown }) => void;
    handler({ position: { x: 12, y: 24 } });

    expect(setSelectedProcedureAnnotation).toHaveBeenCalledWith(annotation);
    expect(mockViewer.entities.add).toHaveBeenCalledWith(
      expect.objectContaining({ id: "procedure-annotation-selected-highlight" }),
    );
  });
});
