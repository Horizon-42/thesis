import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ApproachViewState } from "../../hooks/useApproachView";
import ApproachViewPanel from "../ApproachViewPanel";

const appMock = vi.hoisted(() => ({
  approachViewMode: "side-xz" as "side-xz" | "top-xy",
  procedureDisplayLevel: "PROTECTION" as "CORE" | "PROTECTION" | "ESTIMATED" | "DEBUG",
}));

const approachViewMock = vi.hoisted(() => ({
  state: null as ApproachViewState | null,
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    activeAirportCode: "KRDU",
    isApproachViewOpen: true,
    // The profile's runway is the global Landing-Runway selection (bare "23R" → "RW23R").
    selectedRunway: "23R",
    approachViewMode: appMock.approachViewMode,
    setApproachViewOpen: vi.fn(),
    setApproachViewMode: vi.fn(),
    trajectoryDataSource: null,
    optimizedTrajectoryDataSource: null,
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
  plateRoutes: ApproachViewState["plateRoutes"],
  aircraftTracks: ApproachViewState["aircraftTracks"] = [],
): ApproachViewState {
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
    sourceLinked: true,
  };
}

function makeAircraftTrack(
  overrides: Partial<ApproachViewState["aircraftTracks"][number]["current"]> = {},
): ApproachViewState["aircraftTracks"][number] {
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

vi.mock("../../hooks/useApproachView", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../hooks/useApproachView")>();
  return {
    ...actual,
    useApproachView: () => approachViewMock.state,
  };
});

describe("ApproachViewPanel", () => {
  beforeEach(() => {
    appMock.approachViewMode = "side-xz";
    appMock.procedureDisplayLevel = "PROTECTION";
    approachViewMock.state = makeProfileState([closeXRoute]);
  });

  it("draws every side-view route fix even when nearby labels are de-conflicted", () => {
    const { container } = render(<ApproachViewPanel />);

    const routePointCount = closeXRoute.points.length;
    const thresholdCount = 1;
    const referencePoints = container.querySelectorAll(".approach-view-reference-point");
    const referenceLabels = container.querySelectorAll(".approach-view-reference-label");

    expect(referencePoints).toHaveLength(routePointCount + thresholdCount);
    expect(referenceLabels.length).toBeLessThan(referencePoints.length);
    expect(container.textContent).toContain("BUTTS");
    expect(container.textContent).not.toContain("WARMS");
  });

  it("marks source altitude constraints in the vertical profile", () => {
    const { container } = render(<ApproachViewPanel />);

    expect(container.querySelectorAll(".approach-view-altitude-constraint-point")).toHaveLength(2);
    expect(container.querySelectorAll(".approach-view-altitude-constraint-station-line")).toHaveLength(2);
    expect(container.querySelectorAll(".approach-view-altitude-constraint-link")).toHaveLength(2);
    expect(container.querySelectorAll(".approach-view-altitude-constraint-point.is-at-or-above")).toHaveLength(1);
    expect(container.querySelectorAll(".approach-view-altitude-constraint-point.is-at-or-below")).toHaveLength(1);
    expect(container.textContent).toContain("BUTTS >= 5,200 ft");
    expect(container.textContent).toContain("DABKE <= 3,900 ft");
    expect(container.textContent).not.toContain("WARMS 1,500 ft");
  });

  it("removes side-view route fix dots when the active route set changes", () => {
    const { container, rerender } = render(<ApproachViewPanel />);

    expect(container.querySelectorAll('[data-fix-ident="WARMS"]')).toHaveLength(1);

    approachViewMock.state = makeProfileState([shorterRoute]);
    rerender(<ApproachViewPanel />);

    expect(container.querySelectorAll('[data-fix-ident="WARMS"]')).toHaveLength(0);
    expect(container.querySelectorAll('[data-fix-ident="DABKE"]')).toHaveLength(1);
  });

  it("draws every top-view route fix even when nearby labels are de-conflicted", () => {
    appMock.approachViewMode = "top-xy";

    const { container } = render(<ApproachViewPanel />);

    expect(container.querySelectorAll('[data-fix-ident="BUTTS"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-fix-ident="WARMS"]')).toHaveLength(1);
    expect(container.textContent).toContain("BUTTS");
    expect(container.textContent).not.toContain("WARMS");
  });

  it("gates profile protection geometry by procedure display level", () => {
    appMock.approachViewMode = "top-xy";
    appMock.procedureDisplayLevel = "CORE";
    const { container, rerender } = render(<ApproachViewPanel />);

    expect(container.querySelectorAll(".approach-view-route-band")).toHaveLength(0);

    appMock.procedureDisplayLevel = "PROTECTION";
    rerender(<ApproachViewPanel />);

    expect(container.querySelectorAll(".approach-view-route-band")).toHaveLength(1);
    expect(container.querySelector("clipPath#approach-view-plot-clip-top")).toBeTruthy();
    expect(
      container.querySelector(".approach-view-route-band")?.getAttribute("clip-path"),
    ).toBe("url(#approach-view-plot-clip-top)");
  });

  it("gates profile vertical references and segment debug labels by procedure display level", () => {
    approachViewMock.state = makeProfileState([assessedRoute]);
    appMock.procedureDisplayLevel = "PROTECTION";
    const { container, rerender } = render(<ApproachViewPanel />);

    expect(container.querySelectorAll(".approach-view-final-vertical-reference-line")).toHaveLength(0);
    expect(container.querySelectorAll(".approach-view-lnav-vnav-ocs-line")).toHaveLength(0);
    expect(container.querySelectorAll(".approach-view-segment-debug-label")).toHaveLength(0);

    appMock.procedureDisplayLevel = "ESTIMATED";
    rerender(<ApproachViewPanel />);

    expect(container.querySelectorAll(".approach-view-final-vertical-reference-line")).toHaveLength(1);
    expect(container.querySelectorAll(".approach-view-lnav-vnav-ocs-line")).toHaveLength(1);
    expect(container.textContent).toContain("GPA 3.0 deg");
    expect(container.querySelectorAll(".approach-view-segment-debug-label")).toHaveLength(0);

    appMock.procedureDisplayLevel = "DEBUG";
    rerender(<ApproachViewPanel />);

    expect(container.querySelectorAll(".approach-view-final-vertical-reference-line")).toHaveLength(1);
    expect(container.querySelectorAll(".approach-view-segment-debug-label")).toHaveLength(1);
  });

  it("keeps final-route GPA visible in vertical profile when transitions are active", () => {
    appMock.procedureDisplayLevel = "ESTIMATED";
    approachViewMock.state = makeProfileState([closeXRoute, assessedFinalRoute]);

    const { container } = render(<ApproachViewPanel />);

    expect(container.querySelectorAll('[data-route-id="KRDU-R23RY-ABUTTS"]')).toHaveLength(4);
    expect(container.querySelectorAll(".approach-view-final-vertical-reference-line")).toHaveLength(1);
    expect(container.textContent).toContain("GPA 3.0 deg");
  });

  it("switches profile distance axes from nautical miles to metres", () => {
    const { container } = render(<ApproachViewPanel />);

    expect(container.textContent).toContain("x: approach distance from threshold (NM)");

    fireEvent.click(screen.getByRole("button", { name: "m" }));

    expect(container.textContent).toContain("x: approach distance from threshold (m)");
    expect(container.textContent).toContain("m");
  });

  it("grows the plot domain to fit an aircraft track beyond the procedure frame (so it is not clipped)", () => {
    const { container, rerender } = render(<ApproachViewPanel />);

    const procedureOnlyThresholdX = container
      .querySelector(".approach-view-threshold-line")
      ?.getAttribute("x1");
    expect(container.querySelectorAll(".approach-view-summary span")).toHaveLength(3);
    expect(container.querySelector(".approach-view-status")?.textContent).toContain(
      "No aircraft are inside",
    );

    // A track running far beyond the procedure extent must EXPAND the domain to include it —
    // otherwise the whole out-of-corridor stretch this view exists to show is clipped away.
    approachViewMock.state = makeProfileState(
      [closeXRoute],
      [makeAircraftTrack({ xM: 80_000, yM: 0, zM: 15_000 })],
    );
    rerender(<ApproachViewPanel />);

    expect(container.querySelectorAll(".approach-view-summary span")).toHaveLength(3);
    expect(container.querySelector(".approach-view-status")?.textContent).toContain("AAL123:");
    // The domain grew (the x=0 threshold line projects to a new position), so the far track fits.
    expect(container.querySelector(".approach-view-threshold-line")?.getAttribute("x1")).not.toBe(
      procedureOnlyThresholdX,
    );
    expect(container.querySelector('path[clip-path="url(#approach-view-plot-clip-side)"]')).toBeTruthy();
    expect(container.querySelector('circle[clip-path="url(#approach-view-plot-clip-side)"]')).toBeTruthy();
  });

  it("shows segment assessment for the selected aircraft", () => {
    approachViewMock.state = makeProfileState([closeXRoute], [makeAircraftTrack()]);

    render(<ApproachViewPanel />);

    expect(screen.getByText(/AAL123:/).textContent).toContain(
      "branch:ABUTTS:profile-segment:2",
    );
    expect(screen.getByText(/AAL123:/).textContent).toContain("station 5.0 NM");
    expect(screen.getByText(/AAL123:/).textContent).toContain("xtrack +0.1 NM");
    expect(screen.getByText(/AAL123:/).textContent).toContain("verr +100 ft");
  });
});
