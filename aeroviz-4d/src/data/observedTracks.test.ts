import { describe, expect, it } from "vitest";
import { planObservedTracks, type ObservedTrackInputs } from "./observedTracks";
import { airportDataUrl, airportLandingsRunwayUrl } from "./airportData";

const base: ObservedTrackInputs = {
  mode: "observe",
  activeAirportCode: "KRDU",
  selectedRunway: null,
  isRunwayProfileOpen: false,
  trajectoryComparison: false,
};

describe("planObservedTracks", () => {
  it("loads and shows the airport-wide tracks in Observe", () => {
    expect(planObservedTracks(base)).toEqual({
      fileUrl: airportDataUrl("KRDU", "trajectories.czml"),
      visible: true,
    });
  });

  it("uses the per-runway landings file when a runway is selected", () => {
    expect(planObservedTracks({ ...base, selectedRunway: "05L" })).toEqual({
      fileUrl: airportLandingsRunwayUrl("KRDU", "05L"),
      visible: true,
    });
  });

  it.each(["fly", "optimize", "compare"] as const)(
    "releases the tracks (no load, hidden) in %s with no profile open",
    (mode) => {
      expect(planObservedTracks({ ...base, mode })).toEqual({ fileUrl: "", visible: false });
    },
  );

  it.each(["fly", "optimize", "compare"] as const)(
    "keeps the tracks LOADED but HIDDEN in %s while a runway profile is open (the leak regression)",
    (mode) => {
      const plan = planObservedTracks({
        ...base,
        mode,
        selectedRunway: "05L",
        isRunwayProfileOpen: true,
      });
      // Loaded so the profile chart can still sample it...
      expect(plan.fileUrl).toBe(airportLandingsRunwayUrl("KRDU", "05L"));
      // ...but NOT painted on the globe — tasks stay visually exclusive.
      expect(plan.visible).toBe(false);
    },
  );

  it("hides observed tracks in Observe while the 3-colour comparison is on (still loaded)", () => {
    const plan = planObservedTracks({ ...base, trajectoryComparison: true });
    expect(plan.fileUrl).toBe(airportDataUrl("KRDU", "trajectories.czml"));
    expect(plan.visible).toBe(false);
  });

  it("loads nothing when no airport is active", () => {
    expect(planObservedTracks({ ...base, activeAirportCode: null })).toEqual({
      fileUrl: "",
      visible: true,
    });
  });
});
