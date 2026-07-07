import { describe, expect, it } from "vitest";
import {
  buildProfileAircraftTracks,
  splitTrackByContainment,
  type ProfileAircraftInput,
  type ProfileAircraftSample,
  type SampledRunwayPoint,
} from "./runwayTrajectoryProfileAnalysis";
import type {
  HorizontalPlateContainment,
  HorizontalPlateSegmentAssessment,
} from "../utils/procedureSegmentAssessment";
import type { HorizontalPlateRoute, RunwayFrame } from "../utils/runwayProfileGeometry";

// A straight route down the x-axis (y = 0) with a ±500 m primary corridor; no protection
// surfaces, so classification falls back to the frame projection (lateral distance vs
// halfWidthM). Points near y = 0 are PRIMARY; points far in y are OUTSIDE.
const ROUTE: HorizontalPlateRoute = {
  routeId: "R",
  branchId: "B",
  procedureName: "TEST",
  procedureFamily: "RNAV_GPS",
  procedureIdent: "R05L",
  branchIdent: "BR",
  transitionIdent: null,
  branchType: "transition",
  defaultVisible: true,
  halfWidthM: 500,
  points: [
    { xM: 30000, yM: 0, zM: 1500, fixIdent: "IF", role: "IF", altitudeConstraint: null },
    { xM: 0, yM: 0, zM: 100, fixIdent: "RW", role: "THRESHOLD", altitudeConstraint: null },
  ],
};

const FRAME: RunwayFrame = {
  runwayIdent: "RW05L",
  thresholdLon: 0,
  thresholdLat: 0,
  thresholdAltM: 0,
  approachUnitEast: 1,
  approachUnitNorth: 0,
  leftUnitEast: 0,
  leftUnitNorth: 1,
};

function point(xM: number, yM: number, timeIso: string): SampledRunwayPoint {
  return { xM, yM, zM: 500, geoPosition: { lonDeg: 0, latDeg: 0, altM: 500 }, timeIso };
}

describe("buildProfileAircraftTracks", () => {
  it("keeps the WHOLE track (in- and out-of-corridor) for an aircraft that engages the procedure", () => {
    // Cuts the corner (OUTSIDE) then joins the corridor (PRIMARY) near the threshold, and is
    // currently still OUTSIDE — the whole track must survive, tagged by containment.
    const input: ProfileAircraftInput = {
      flightId: "AAL1",
      current: point(20000, 3000, "t3"), // OUTSIDE right now
      trail: [
        point(28000, 4000, "t0"), // OUTSIDE (corner cut)
        point(24000, 2000, "t1"), // OUTSIDE
        point(8000, 100, "t2"), // PRIMARY (joined the corridor)
        point(20000, 3000, "t3"), // current
      ],
    };

    const [track] = buildProfileAircraftTracks({
      aircraft: [input],
      activePlateRoutes: [ROUTE],
      runwayFrame: FRAME,
      selectedFlightId: null,
    });

    expect(track).toBeTruthy();
    // Every sampled point is retained (none dropped for being out of the corridor).
    expect(track.trail).toHaveLength(4);
    const tiers = track.trail.map((s) => s.segmentAssessment.containment);
    expect(tiers).toContain("PRIMARY");
    expect(tiers).toContain("OUTSIDE");
    // The current point is kept even though it is outside the corridor.
    expect(track.current.segmentAssessment.containment).toBe("OUTSIDE");
  });

  it("drops an aircraft whose whole track never reaches the corridor (unrelated traffic)", () => {
    const input: ProfileAircraftInput = {
      flightId: "FAROFF",
      current: point(15000, 8000, "t1"),
      trail: [point(20000, 9000, "t0"), point(15000, 8000, "t1")],
    };

    const tracks = buildProfileAircraftTracks({
      aircraft: [input],
      activePlateRoutes: [ROUTE],
      runwayFrame: FRAME,
      selectedFlightId: null,
    });

    expect(tracks).toHaveLength(0);
  });
});

describe("splitTrackByContainment", () => {
  const sample = (containment: HorizontalPlateContainment, xM: number): ProfileAircraftSample => ({
    xM,
    yM: 0,
    zM: 0,
    timeIso: `t${xM}`,
    segmentAssessment: { containment } as HorizontalPlateSegmentAssessment,
  });

  it("returns one run when the whole track is one tier", () => {
    const runs = splitTrackByContainment([sample("PRIMARY", 3), sample("PRIMARY", 2)]);
    expect(runs).toHaveLength(1);
    expect(runs[0].containment).toBe("PRIMARY");
    expect(runs[0].points).toHaveLength(2);
  });

  it("keeps SECONDARY as its own tier, distinct from OUTSIDE", () => {
    // The whole point: a still-contained SECONDARY stretch must NOT collapse into OUTSIDE
    // (the panel draws SECONDARY solid/contained, only OUTSIDE dashed).
    const runs = splitTrackByContainment([
      sample("SECONDARY", 3),
      sample("SECONDARY", 2),
      sample("OUTSIDE", 1),
    ]);
    expect(runs.map((r) => r.containment)).toEqual(["SECONDARY", "OUTSIDE"]);
  });

  it("splits at tier crossings and bridges the boundary point so the line has no gap", () => {
    const trail = [
      sample("OUTSIDE", 5),
      sample("OUTSIDE", 4),
      sample("PRIMARY", 3),
      sample("PRIMARY", 2),
      sample("OUTSIDE", 1),
    ];
    const runs = splitTrackByContainment(trail);
    expect(runs.map((r) => r.containment)).toEqual(["OUTSIDE", "PRIMARY", "OUTSIDE"]);
    // Each later run starts with the previous run's last point (shared boundary → gapless).
    expect(runs[1].points[0]).toBe(trail[1]); // bridge into the PRIMARY run
    expect(runs[2].points[0]).toBe(trail[3]); // bridge back out
    expect(runs[1].points).toHaveLength(3); // bridge + the two PRIMARY points
  });

  it("returns nothing for an empty track", () => {
    expect(splitTrackByContainment([])).toEqual([]);
  });
});
