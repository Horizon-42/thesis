import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { RunwayTrajectoryProfileState } from "../../hooks/useRunwayTrajectoryProfile";
import RunwayTrajectoryProfilePanel from "../RunwayTrajectoryProfilePanel";

const appMock = vi.hoisted(() => ({
  runwayProfileViewMode: "side-xz" as "side-xz" | "top-xy",
  procedureDisplayLevel: "PROTECTION" as "CORE" | "PROTECTION" | "ESTIMATED" | "DEBUG",
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
    procedureDisplayLevel: appMock.procedureDisplayLevel,
  }),
}));

const closeXRoute = {
  routeId: "KRDU-R23RY-ABUTTS",
  branchId: "branch:ABUTTS",
  procedureName: "RNAV(GPS) Y RWY 23R",
  procedureFamily: "RNAV_GPS",
  procedureIdent: "R23RY",
  branchIdent: "BUTTS",
  transitionIdent: "BUTTS",
  branchType: "transition",
  defaultVisible: true,
  halfWidthM: 555.6,
  points: [
    {
      xM: 30000,
      yM: 0,
      zM: 1600,
      fixIdent: "BUTTS",
      role: "IF",
      altitudeConstraint: { kind: "AT_OR_ABOVE" as const, minFtMsl: 5200, sourceText: "5200 ft" },
    },
    {
      xM: 29200,
      yM: 0,
      zM: 1500,
      fixIdent: "WARMS",
      role: "IF",
      altitudeConstraint: null,
    },
    {
      xM: 20500,
      yM: 0,
      zM: 1200,
      fixIdent: "DABKE",
      role: "IF",
      altitudeConstraint: { kind: "AT_OR_BELOW" as const, maxFtMsl: 3900, sourceText: "3900 ft" },
    },
    {
      xM: 0,
      yM: 0,
      zM: 0,
      fixIdent: "RW23R",
      role: "MAPt",
      altitudeConstraint: null,
    },
  ],
};

const shorterRoute = {
  ...closeXRoute,
  routeId: "KRDU-R23RY-DABKE",
  branchId: "branch:DABKE",
  points: [
    {
      xM: 20500,
      yM: 0,
      zM: 1200,
      fixIdent: "DABKE",
      role: "IF",
      altitudeConstraint: { kind: "AT_OR_BELOW" as const, maxFtMsl: 3900, sourceText: "3900 ft" },
    },
    {
      xM: 0,
      yM: 0,
      zM: 0,
      fixIdent: "RW23R",
      role: "MAPt",
      altitudeConstraint: null,
    },
  ],
};

const assessedRoute = {
  ...closeXRoute,
  assessmentSegments: [
    {
      segmentId: "KRDU-R23RY-RW23R:branch:ABUTTS:segment:final:1",
      primaryHalfWidthM: 370.4,
      secondaryHalfWidthM: 740.8,
      points: [
        { xM: 20500, yM: 0, zM: 950 },
        { xM: 0, yM: 0, zM: 80 },
      ],
      finalVerticalReference: {
        kind: "FINAL_VERTICAL_REFERENCE" as const,
        label: "GPA 3.0 deg",
        gpaDeg: 3,
        tchFt: null,
        estimatedFromThreshold: true,
        halfWidthM: 185.2,
        points: [
          { xM: 20500, yM: 0, zM: 980 },
          { xM: 0, yM: 0, zM: 15 },
        ],
      },
      lnavVnavOcs: {
        kind: "LNAV_VNAV_OCS" as const,
        label: "LNAV/VNAV OCS",
        gpaDeg: 3,
        tchFt: 50,
        primaryHalfWidthM: 370.4,
        secondaryHalfWidthM: 740.8,
        points: [
          { xM: 20500, yM: 0, zM: 950 },
          { xM: 0, yM: 0, zM: 80 },
        ],
      },
    },
  ],
};

const assessedFinalRoute = {
  ...assessedRoute,
  routeId: "KRDU-R23RY-FINAL",
  branchId: "branch:R",
  branchIdent: "R",
  transitionIdent: null,
  branchType: "final",
  points: [
    {
      xM: 20500,
      yM: 0,
      zM: 1200,
      fixIdent: "DABKE",
      role: "FAF",
      altitudeConstraint: { kind: "AT" as const, minFtMsl: 3900, maxFtMsl: 3900, sourceText: "3900 ft" },
    },
    {
      xM: 0,
      yM: 0,
      zM: 0,
      fixIdent: "RW23R",
      role: "MAPt",
      altitudeConstraint: null,
    },
  ],
};

function makeProfileState(
  plateRoutes: RunwayTrajectoryProfileState["plateRoutes"],
  aircraftTracks: RunwayTrajectoryProfileState["aircraftTracks"] = [],
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
    aircraftTracks,
  };
}

function makeAircraftTrack(
  overrides: Partial<RunwayTrajectoryProfileState["aircraftTracks"][number]["current"]> = {},
): RunwayTrajectoryProfileState["aircraftTracks"][number] {
  const current = {
    xM: 20_000,
    yM: 185.2,
    zM: 900,
    timeIso: "2026-05-01T00:00:00.000Z",
    segmentAssessment: {
      routeId: "KRDU-R23RY-ABUTTS",
      branchId: "branch:ABUTTS",
      activeSegmentId: "branch:ABUTTS:profile-segment:2",
      segmentIndex: 1,
      stationM: 9_260,
      crossTrackErrorM: 185.2,
      verticalErrorM: 30.48,
      containment: "PRIMARY" as const,
      closestPoint: { xM: 20_000, yM: 0, zM: 900 },
      events: [{ kind: "LATERAL_CONTAINMENT" as const, label: "PRIMARY" }],
    },
    ...overrides,
  };

  return {
    flightId: "AAL123",
    color: "#38bdf8",
    isSelected: true,
    current,
    trail: [current],
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
    appMock.procedureDisplayLevel = "PROTECTION";
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

  it("marks source altitude constraints in the vertical profile", () => {
    const { container } = render(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll(".runway-profile-altitude-constraint-point")).toHaveLength(2);
    expect(container.querySelectorAll(".runway-profile-altitude-constraint-station-line")).toHaveLength(2);
    expect(container.querySelectorAll(".runway-profile-altitude-constraint-link")).toHaveLength(2);
    expect(container.querySelectorAll(".runway-profile-altitude-constraint-point.is-at-or-above")).toHaveLength(1);
    expect(container.querySelectorAll(".runway-profile-altitude-constraint-point.is-at-or-below")).toHaveLength(1);
    expect(container.textContent).toContain("BUTTS >= 5,200 ft");
    expect(container.textContent).toContain("DABKE <= 3,900 ft");
    expect(container.textContent).not.toContain("WARMS 1,500 ft");
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

  it("gates profile protection geometry by procedure display level", () => {
    appMock.runwayProfileViewMode = "top-xy";
    appMock.procedureDisplayLevel = "CORE";
    const { container, rerender } = render(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll(".runway-profile-route-band")).toHaveLength(0);

    appMock.procedureDisplayLevel = "PROTECTION";
    rerender(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll(".runway-profile-route-band")).toHaveLength(1);
    expect(container.querySelector("clipPath#runway-profile-plot-clip-top")).toBeTruthy();
    expect(
      container.querySelector(".runway-profile-route-band")?.getAttribute("clip-path"),
    ).toBe("url(#runway-profile-plot-clip-top)");
  });

  it("gates profile vertical references and segment debug labels by procedure display level", () => {
    profileMock.state = makeProfileState([assessedRoute]);
    appMock.procedureDisplayLevel = "PROTECTION";
    const { container, rerender } = render(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll(".runway-profile-final-vertical-reference-line")).toHaveLength(0);
    expect(container.querySelectorAll(".runway-profile-lnav-vnav-ocs-line")).toHaveLength(0);
    expect(container.querySelectorAll(".runway-profile-segment-debug-label")).toHaveLength(0);

    appMock.procedureDisplayLevel = "ESTIMATED";
    rerender(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll(".runway-profile-final-vertical-reference-line")).toHaveLength(1);
    expect(container.querySelectorAll(".runway-profile-lnav-vnav-ocs-line")).toHaveLength(1);
    expect(container.textContent).toContain("GPA 3.0 deg");
    expect(container.querySelectorAll(".runway-profile-segment-debug-label")).toHaveLength(0);

    appMock.procedureDisplayLevel = "DEBUG";
    rerender(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll(".runway-profile-final-vertical-reference-line")).toHaveLength(1);
    expect(container.querySelectorAll(".runway-profile-segment-debug-label")).toHaveLength(1);
  });

  it("keeps final-route GPA visible in vertical profile when transitions are active", () => {
    appMock.procedureDisplayLevel = "ESTIMATED";
    profileMock.state = makeProfileState([closeXRoute, assessedFinalRoute]);

    const { container } = render(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll('[data-route-id="KRDU-R23RY-ABUTTS"]')).toHaveLength(4);
    expect(container.querySelectorAll(".runway-profile-final-vertical-reference-line")).toHaveLength(1);
    expect(container.textContent).toContain("GPA 3.0 deg");
  });

  it("switches profile distance axes from nautical miles to metres", () => {
    const { container } = render(<RunwayTrajectoryProfilePanel />);

    expect(container.textContent).toContain("x: approach distance from threshold (NM)");

    fireEvent.click(screen.getByRole("button", { name: "m" }));

    expect(container.textContent).toContain("x: approach distance from threshold (m)");
    expect(container.textContent).toContain("m");
  });

  it("keeps the plot domain stable when aircraft tracks enter or move outside the profile frame", () => {
    const { container, rerender } = render(<RunwayTrajectoryProfilePanel />);

    const thresholdX = container
      .querySelector(".runway-profile-threshold-line")
      ?.getAttribute("x1");
    const yAxisLabels = Array.from(container.querySelectorAll(".runway-profile-axis-tick"))
      .map((element) => element.textContent);
    const yMaxLabel = yAxisLabels[yAxisLabels.length - 2];
    expect(container.querySelectorAll(".runway-profile-summary span")).toHaveLength(3);
    expect(container.querySelector(".runway-profile-status")?.textContent).toContain(
      "No aircraft are inside",
    );

    profileMock.state = makeProfileState(
      [closeXRoute],
      [makeAircraftTrack({ xM: 80_000, yM: 0, zM: 15_000 })],
    );
    rerender(<RunwayTrajectoryProfilePanel />);

    expect(container.querySelectorAll(".runway-profile-summary span")).toHaveLength(3);
    expect(container.querySelector(".runway-profile-status")?.textContent).toContain("AAL123:");
    expect(container.querySelector(".runway-profile-threshold-line")?.getAttribute("x1")).toBe(
      thresholdX,
    );
    expect(
      (() => {
        const updatedLabels = Array.from(container.querySelectorAll(".runway-profile-axis-tick"))
          .map((element) => element.textContent);
        return updatedLabels[updatedLabels.length - 2];
      })(),
    ).toBe(yMaxLabel);
    expect(container.querySelector('path[clip-path="url(#runway-profile-plot-clip-side)"]')).toBeTruthy();
    expect(container.querySelector('circle[clip-path="url(#runway-profile-plot-clip-side)"]')).toBeTruthy();
  });

  it("shows segment assessment for the selected aircraft", () => {
    profileMock.state = makeProfileState([closeXRoute], [makeAircraftTrack()]);

    render(<RunwayTrajectoryProfilePanel />);

    expect(screen.getByText(/AAL123:/).textContent).toContain(
      "branch:ABUTTS:profile-segment:2",
    );
    expect(screen.getByText(/AAL123:/).textContent).toContain("station 5.0 NM");
    expect(screen.getByText(/AAL123:/).textContent).toContain("xtrack +0.1 NM");
    expect(screen.getByText(/AAL123:/).textContent).toContain("verr +100 ft");
  });
});
