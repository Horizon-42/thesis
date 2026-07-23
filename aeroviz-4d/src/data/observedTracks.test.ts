import { describe, expect, it } from "vitest";
import { planObservedTracks, type ObservedTrackInputs } from "./observedTracks";
import { airportDataUrl } from "./airportData";

const base: ObservedTrackInputs = {
  mode: "observe",
  activeAirportCode: "KRDU",
  selectedRunway: null,
  trajectoryComparison: false,
};

describe("planObservedTracks", () => {
  it("loads and shows the airport-wide tracks in Observe", () => {
    expect(planObservedTracks(base)).toEqual({
      fileUrl: airportDataUrl("KRDU", "trajectories.czml"),
      visible: true,
      runwayFilter: null,
    });
  });

  it("keeps the canonical file and applies a runway filter when selected", () => {
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
      fileUrl: airportDataUrl("KRDU", "trajectories.czml"),
      visible: true,
      runwayFilter: "05L",
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
      runwayFilter: null,
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
      runwayFilter: null,
    });
  });

  it.each(["fly", "optimize", "compare"] as const)(
    "releases the tracks (no load, hidden) in %s",
    (mode) => {
      expect(planObservedTracks({ ...base, mode })).toEqual({
        fileUrl: "",
        visible: false,
        runwayFilter: null,
      });
    },
  );

  it.each(["fly", "optimize", "compare"] as const)(
    "does NOT load the observed tracks in %s even with a runway profile open",
    (mode) => {
      // The profile samples observed tracks only in Observe, so no other task needs them
      // loaded. Loading them here previously let useCzmlLoader hijack the shared clock and
      // made the optimized playback vanish — so they must stay released.
      expect(planObservedTracks({ ...base, mode, selectedRunway: "05L" })).toEqual({
        fileUrl: "",
        visible: false,
        runwayFilter: "05L",
      });
    },
  );

  it("hides observed tracks in Observe while the 3-colour comparison is on (still loaded)", () => {
    const plan = planObservedTracks({ ...base, trajectoryComparison: true });
    expect(plan.fileUrl).toBe(airportDataUrl("KRDU", "trajectories.czml"));
    expect(plan.visible).toBe(false);
    expect(plan.runwayFilter).toBeNull();
  });

  it("loads nothing when no airport is active", () => {
    expect(planObservedTracks({ ...base, activeAirportCode: null })).toEqual({
      fileUrl: "",
      visible: true,
      runwayFilter: null,
    });
  });
});
