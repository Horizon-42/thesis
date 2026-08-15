import { describe, expect, it } from "vitest";
import {
  isObservedTrajectoryResponse,
  decodeObservedVerdicts,
  planObservedTracks,
  type ObservedTrackInputs,
} from "./observedTracks";

const BACKEND_URL = "http://backend.test";

const base: ObservedTrackInputs = {
  mode: "observe",
  activeAirportCode: "KRDU",
  selectedRunway: null,
  trajectoryComparison: false,
  trajectorySampleCount: 200,
  observedVerdictFilter: "all",
  backendUrl: BACKEND_URL,
};

describe("planObservedTracks", () => {
  it("loads and shows the airport-wide tracks in Observe", () => {
    expect(planObservedTracks(base)).toEqual({
      fileUrl: `${BACKEND_URL}/trajectories?airport=KRDU&limit=200&seed=0`,
      visible: true,
    });
  });

  it("asks the backend to filter the selected runway before loading", () => {
    expect(planObservedTracks({
      ...base,
      selectedRunway: "05L",
      landingsManifest: {
        schemaVersion: "observed-landings-v2-canonical",
        airport: "KRDU",
        combined: "trajectories.czml",
        runways: [
          { runway: "05L", file: "trajectories.czml", count: 12 },
        ],
      },
      landingsStatus: "ready",
    })).toEqual({
      fileUrl:
        `${BACKEND_URL}/trajectories?airport=KRDU&limit=200&seed=0&runway=05L`,
      visible: true,
    });
  });

  it("does not load observed data when the publication contract is invalid", () => {
    expect(planObservedTracks({
      ...base,
      selectedRunway: "05L",
      landingsManifest: null,
      landingsStatus: "error",
    })).toEqual({
      fileUrl: "",
      visible: true,
    });
  });

  it("waits for the publication schema before applying a runway selection", () => {
    expect(planObservedTracks({
      ...base,
      selectedRunway: "05L",
      landingsManifest: null,
      landingsStatus: "loading",
    })).toEqual({
      fileUrl: "",
      visible: true,
    });
  });

  it.each(["fly", "optimize", "compare"] as const)(
    "releases the tracks (no load, hidden) in %s",
    (mode) => {
      expect(planObservedTracks({ ...base, mode })).toEqual({
        fileUrl: "",
        visible: false,
      });
    },
  );

  it.each(["fly", "optimize", "compare"] as const)(
    "does NOT load the observed tracks in %s even with a runway profile open",
    (mode) => {
      // The profile samples observed tracks only in Observe, so no other task needs them
      // loaded. Loading them here previously let the observed layer hijack the shared clock and
      // made the optimized playback vanish — so they must stay released.
      expect(planObservedTracks({ ...base, mode, selectedRunway: "05L" })).toEqual({
        fileUrl: "",
        visible: false,
      });
    },
  );

  it("hides observed tracks in Observe while the 3-colour comparison is on (still loaded)", () => {
    const plan = planObservedTracks({ ...base, trajectoryComparison: true });
    expect(plan.fileUrl).toBe(
      `${BACKEND_URL}/trajectories?airport=KRDU&limit=200&seed=0`,
    );
    expect(plan.visible).toBe(false);
  });

  it("puts sample-count changes in the request URL so the loader refetches", () => {
    expect(planObservedTracks({ ...base, trajectorySampleCount: 75 }).fileUrl).toBe(
      `${BACKEND_URL}/trajectories?airport=KRDU&limit=75&seed=0`,
    );
  });

  it("puts the baseline verdict in the request so the backend filters before sampling", () => {
    expect(planObservedTracks({ ...base, observedVerdictFilter: "fail" }).fileUrl).toBe(
      `${BACKEND_URL}/trajectories?airport=KRDU&limit=200&seed=0&verdict=fail`,
    );
  });

  it("does not apply the baseline verdict to comparison reference loading", () => {
    expect(planObservedTracks({
      ...base,
      trajectoryComparison: true,
      observedVerdictFilter: "fail",
    }).fileUrl).toBe(`${BACKEND_URL}/trajectories?airport=KRDU&limit=200&seed=0`);
  });

  it("loads nothing when no airport is active", () => {
    expect(planObservedTracks({ ...base, activeAirportCode: null })).toEqual({
      fileUrl: "",
      visible: true,
    });
  });
});

describe("observed trajectory response", () => {
  const response = {
    schemaVersion: "observed-trajectories-v1",
    czml: [{ id: "document" }, { id: "flight-fail" }],
    verdicts: {
      counts: { pass: 4, fail: 1, undecided: 2 },
      byFlightId: { "flight-fail": "fail" },
      matched: 6,
      total: 7,
    },
    evaluation: {
      total: 7,
      verdict_counts: { pass: 4, fail: 1, indeterminate: 2 },
      observed: { event_estimated_rate: 0.8 },
      lateral_m: { mean: 12.5 },
      vertical_m: { mean_abs: 4.5 },
    },
  } as const;

  it("validates and converts the bounded backend payload", () => {
    expect(isObservedTrajectoryResponse(response)).toBe(true);
    if (!isObservedTrajectoryResponse(response)) throw new Error("invalid fixture");
    expect([...decodeObservedVerdicts(response.verdicts).byFlightId!.entries()]).toEqual([
      ["flight-fail", "fail"],
    ]);
  });

  it("rejects legacy bare CZML and invalid verdicts", () => {
    expect(isObservedTrajectoryResponse(response.czml)).toBe(false);
    expect(isObservedTrajectoryResponse({
      ...response,
      verdicts: { ...response.verdicts, byFlightId: { bad: "failed" } },
    })).toBe(false);
  });
});
