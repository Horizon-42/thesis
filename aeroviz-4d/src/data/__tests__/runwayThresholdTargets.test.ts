import { describe, expect, it } from "vitest";
import { buildRunwayThresholdTargets } from "../runwayThresholdTargets";
import type { ProcedureDetailsIndexManifest } from "../procedureDetails";

const collection: Parameters<typeof buildRunwayThresholdTargets>[0] = {
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
};

function indexWith(
  runways: ProcedureDetailsIndexManifest["runways"],
): ProcedureDetailsIndexManifest {
  return {
    airport: "TEST",
    airportName: "Test Field",
    sourceCycle: "2603",
    researchUseOnly: true,
    runways,
  };
}

describe("runwayThresholdTargets", () => {
  it("builds one selectable target per runway threshold", () => {
    const targets = buildRunwayThresholdTargets(collection);

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

  it("prefers the CIFP threshold from the procedure-details index over the pavement edge", () => {
    // The geojson runway_surface edge is the PAVEMENT end; on displaced-threshold runways the
    // CIFP landing threshold sits hundreds of metres inward. Position + elevation come from
    // CIFP; the heading stays pavement-derived (the axis direction is accurate).
    const index = indexWith([
      {
        runwayIdent: "RW09",
        chartName: "RNAV (GPS) RWY 09",
        threshold: { lon: 0.002, lat: 0.00001, elevationFt: 105 },
        procedureUids: ["TEST-R09-RW09"],
        procedures: [],
      },
    ]);
    const targets = buildRunwayThresholdTargets(collection, index);

    expect(targets[0]).toMatchObject({
      id: "RW09",
      lon: 0.002,
      lat: 0.00001,
      altM: 105 * 0.3048,
    });
    expect(targets[0].psiDeg).toBeCloseTo(0);      // heading still from the pavement axis
    // RW27 has no procedure -> pavement fallback untouched
    expect(targets[1]).toMatchObject({ id: "RW27", lon: 0.01, lat: 0, altM: 36.576 });
  });

  it("falls back to the pavement elevation when the CIFP threshold has none", () => {
    const index = indexWith([
      {
        runwayIdent: "RW09",
        chartName: "RNAV (GPS) RWY 09",
        threshold: { lon: 0.002, lat: 0, elevationFt: null },
        procedureUids: ["TEST-R09-RW09"],
        procedures: [],
      },
      {
        runwayIdent: "RW27",
        chartName: "RNAV (GPS) RWY 27",
        threshold: null,     // runway coded without a threshold -> full pavement fallback
        procedureUids: ["TEST-R27-RW27"],
        procedures: [],
      },
    ]);
    const targets = buildRunwayThresholdTargets(collection, index);

    expect(targets[0]).toMatchObject({ id: "RW09", lon: 0.002, altM: 30.48 });
    expect(targets[1]).toMatchObject({ id: "RW27", lon: 0.01, altM: 36.576 });
  });
});
