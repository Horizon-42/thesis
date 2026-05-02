import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ProcedureEntityAnnotation } from "../../data/procedureAnnotations";

const { setSelectedProcedureAnnotation, navigateWithinApp } = vi.hoisted(() => ({
  setSelectedProcedureAnnotation: vi.fn(),
  navigateWithinApp: vi.fn(),
}));

const annotation: ProcedureEntityAnnotation = {
  entityId: "entity-1",
  label: "LNAV/VNAV OCS",
  title: "RNAV(GPS) RWY 32 LNAV/VNAV OCS primary",
  kind: "LNAV_VNAV_OCS",
  status: "ESTIMATED",
  airportId: "KRDU",
  runwayId: "RW32",
  procedureUid: "KRDU-R32-RW32",
  procedureId: "R32",
  procedureName: "RNAV(GPS) RWY 32",
  branchId: "branch:R",
  branchName: "RW32",
  branchRole: "STRAIGHT_IN",
  segmentId: "segment:final",
  segmentType: "FINAL_LNAV_VNAV",
  meaning: "Sloping LNAV/VNAV obstacle clearance surface estimate.",
  parameters: [
    { label: "GPA", value: "3.5 deg" },
    { label: "TCH", value: "50 ft" },
  ],
  diagnostics: ["test diagnostic"],
  sourceRefs: ["src:cifp-detail"],
};

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    selectedProcedureAnnotation: annotation,
    setSelectedProcedureAnnotation,
    activeAirportCode: "KRDU",
  }),
}));

vi.mock("../../utils/navigation", () => ({
  navigateWithinApp,
}));

import ProcedureAnnotationPopup from "../ProcedureAnnotationPopup";

describe("ProcedureAnnotationPopup", () => {
  it("renders annotation context and actions", () => {
    render(<ProcedureAnnotationPopup />);

    expect(screen.getByText("RNAV(GPS) RWY 32 LNAV/VNAV OCS primary")).toBeTruthy();
    expect(screen.getByText("Estimated")).toBeTruthy();
    expect(screen.getByText("3.5 deg")).toBeTruthy();
    expect(screen.getByText("test diagnostic")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Procedure Details" }));
    expect(navigateWithinApp).toHaveBeenCalledWith("/procedure-details?airport=KRDU");

    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(setSelectedProcedureAnnotation).toHaveBeenCalledWith(null);
  });
});
