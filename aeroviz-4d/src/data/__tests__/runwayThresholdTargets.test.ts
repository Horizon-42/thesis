import { describe, expect, it } from "vitest";
import { buildRunwayThresholdTargets } from "../runwayThresholdTargets";

describe("runwayThresholdTargets", () => {
  it("builds one selectable target per runway threshold", () => {
    const targets = buildRunwayThresholdTargets({
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: {
            type: "Polygon",
            coordinates: [[
              [0, 0.0001],
              [0, -0.0001],
              [0.01, -0.0001],
              [0.01, 0.0001],
              [0, 0.0001],
            ]],
          },
          properties: {
            airport_ident: "TEST",
            runway_ident: "09/27",
            zone_type: "runway_surface",
            le_ident: "09",
            he_ident: "27",
            length_ft: 3000,
            width_ft: 100,
            surface: "ASP",
            lighted: 1,
            le_elevation_ft: 100,
            he_elevation_ft: 120,
          },
        },
      ],
    });

    expect(targets).toHaveLength(2);
    expect(targets[0]).toMatchObject({
      id: "RW09",
      runwayIdent: "RW09",
      runwayPairIdent: "09/27",
      lon: 0,
      lat: 0,
      altM: 30.48,
    });
    expect(targets[0].psiDeg).toBeCloseTo(0);
    expect(targets[1]).toMatchObject({
      id: "RW27",
      runwayIdent: "RW27",
      lon: 0.01,
      lat: 0,
      altM: 36.576,
    });
    expect(targets[1].psiDeg).toBeCloseTo(180);
  });
});
