import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { RunwayTrajectoryProfileState } from "../../hooks/useRunwayTrajectoryProfile";
import RunwayTrajectoryProfilePanel from "../RunwayTrajectoryProfilePanel";

const appMock = vi.hoisted(() => ({
  runwayProfileViewMode: "side-xz" as "side-xz" | "top-xy",
}));

const profileMock = vi.hoisted(() => ({
  state: null as RunwayTrajectoryProfileState | null,
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    activeAirportCode: "KRDU",
    isRunwayProfileOpen: true,
    selectedProfileRunwayIdent: "RW23R",
    runwayProfileViewMode: appMock.runwayProfileViewMode,
    setRunwayProfileOpen: vi.fn(),
    setRunwayProfileViewMode: vi.fn(),
    trajectoryDataSource: null,
  }),
}));

const closeXRoute = {
  routeId: "KRDU-R23RY-ABUTTS",
  procedureName: "RNAV(GPS) Y RWY 23R",
  procedureFamily: "RNAV_GPS",
  procedureIdent: "R23RY",
  branchIdent: "BUTTS",
  transitionIdent: "BUTTS",
  branchType: "transition",
  defaultVisible: true,
  halfWidthM: 555.6,
  points: [
    { xM: 30000, yM: 0, zM: 1600, fixIdent: "BUTTS", role: "IF" },
    { xM: 29200, yM: 0, zM: 1500, fixIdent: "WARMS", role: "IF" },
    { xM: 20500, yM: 0, zM: 1200, fixIdent: "DABKE", role: "IF" },
    { xM: 0, yM: 0, zM: 0, fixIdent: "RW23R", role: "MAPt" },
  ],
};

const shorterRoute = {
  ...closeXRoute,
  routeId: "KRDU-R23RY-DABKE",
  points: [
    { xM: 20500, yM: 0, zM: 1200, fixIdent: "DABKE", role: "IF" },
    { xM: 0, yM: 0, zM: 0, fixIdent: "RW23R", role: "MAPt" },
  ],
};

function makeProfileState(
  plateRoutes: RunwayTrajectoryProfileState["plateRoutes"],
): RunwayTrajectoryProfileState {
  return {
    isLoading: false,
    error: null,
    currentTimeIso: "2026-05-01T00:00:00.000Z",
    runwayFrame: null,
    plateRoutes,
    referenceMarks: [
      { xM: 0, yM: 0, zM: 0, label: "RW23R", detail: "Threshold", priority: 10 },
      { xM: 30000, yM: 0, zM: 1600, label: "BUTTS", detail: "IF", priority: 4 },
      { xM: 29200, yM: 0, zM: 1500, label: "WARMS", detail: "IF", priority: 4 },
      { xM: 20500, yM: 0, zM: 1200, label: "DABKE", detail: "IF", priority: 4 },
    ],
    procedureNames: ["RNAV(GPS) Y RWY 23R"],
    sourceCycle: "2603",
    aircraftTracks: [],
  };
}

vi.mock("../../hooks/useRunwayTrajectoryProfile", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../hooks/useRunwayTrajectoryProfile")>();
  return {
    ...actual,
    useRunwayTrajectoryProfile: () => profileMock.state,
  };
});

describe("RunwayTrajectoryProfilePanel", () => {
  beforeEach(() => {
    appMock.runwayProfileViewMode = "side-xz";
    profileMock.state = makeProfileState([closeXRoute]);
  });

  it("draws every side-view route fix even when nearby labels are de-conflicted", () => {
    const { container } = render(<RunwayTrajectoryProfilePanel />);

    const routePointCount = closeXRoute.points.length;
    const thresholdCount = 1;
    const referencePoints = container.querySelectorAll(".runway-profile-reference-point");
    const referenceLabels = container.querySelectorAll(".runway-profile-reference-label");

    expect(referencePoints).toHaveLength(routePointCount + thresholdCount);
    expect(referenceLabels.length).toBeLessThan(referencePoints.length);
    expect(container.textContent).toContain("BUTTS");
    expect(container.textContent).not.toContain("WARMS");
  });

  it("removes side-view route fix dots when the active route set changes", () => {
    const { container, rerender } = render(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll('[data-fix-ident="WARMS"]')).toHaveLength(1);

    profileMock.state = makeProfileState([shorterRoute]);
    rerender(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll('[data-fix-ident="WARMS"]')).toHaveLength(0);
    expect(container.querySelectorAll('[data-fix-ident="DABKE"]')).toHaveLength(1);
  });

  it("draws every top-view route fix even when nearby labels are de-conflicted", () => {
    appMock.runwayProfileViewMode = "top-xy";

    const { container } = render(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll('[data-fix-ident="BUTTS"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-fix-ident="WARMS"]')).toHaveLength(1);
    expect(container.textContent).toContain("BUTTS");
    expect(container.textContent).not.toContain("WARMS");
  });

  it("switches profile distance axes from nautical miles to metres", () => {
    const { container } = render(<RunwayTrajectoryProfilePanel />);

    expect(container.textContent).toContain("x: approach distance from threshold (NM)");

    fireEvent.click(screen.getByRole("button", { name: "m" }));

    expect(container.textContent).toContain("x: approach distance from threshold (m)");
    expect(container.textContent).toContain("m");
  });
});
