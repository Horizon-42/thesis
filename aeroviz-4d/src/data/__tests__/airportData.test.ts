import { describe, expect, it } from "vitest";

import {
  AIRPORTS_INDEX_URL,
  airportDataUrl,
  airportProcedureDetailUrl,
  airportProcedureDetailsIndexUrl,
  airportLocalTerrainUrl,
  airportChartsIndexUrl,
  isAirportsIndexManifest,
  isComparisonCategoriesManifest,
  normalizeAirportCode,
  sortAirportCatalog,
} from "../airportData";

describe("airportData helpers", () => {
  it("builds airport-scoped data URLs", () => {
    expect(airportDataUrl("krdu", "airport.json")).toBe("/data/airports/KRDU/airport.json");
    expect(airportLocalTerrainUrl("cyvr", "metadata.json")).toBe(
      "/data/airports/CYVR/local-terrain/heightmap/metadata.json",
    );
    expect(airportProcedureDetailsIndexUrl("krdu")).toBe(
      "/data/airports/KRDU/procedure-details/index.json",
    );
    expect(airportProcedureDetailUrl("krdu", "KRDU-R05LY-RW05L")).toBe(
      "/data/airports/KRDU/procedure-details/KRDU-R05LY-RW05L.json",
    );
    expect(airportChartsIndexUrl("krdu")).toBe("/data/airports/KRDU/charts/index.json");
  });

  it("validates and sorts the airport manifest", () => {
    const manifest = {
      defaultAirport: "krdu",
      airports: [
        { code: "CYVR", name: "Vancouver", lat: 49.1, lon: -123.1 },
        { code: "KRDU", name: "Raleigh-Durham", lat: 35.8, lon: -78.7 },
      ],
    };

    expect(AIRPORTS_INDEX_URL).toBe("/data/airports/index.json");
    expect(isAirportsIndexManifest(manifest)).toBe(true);
    expect(normalizeAirportCode(manifest.defaultAirport)).toBe("KRDU");
    expect(sortAirportCatalog(manifest.airports).map((airport) => airport.code)).toEqual([
      "CYVR",
      "KRDU",
    ]);
  });

  it("requires the explicit constrained boolean on every comparison category", () => {
    const entry = { key: "runway_cons", label: "Runway (constrained)", dir: "runway_cons", groups: 3 };
    // Constrained-ness is a manifest FIELD, not a key/dir spelling — an entry
    // without the boolean is rejected so a stale manifest fails loudly.
    expect(isComparisonCategoriesManifest({ categories: [entry] })).toBe(false);
    expect(
      isComparisonCategoriesManifest({ categories: [{ ...entry, constrained: true }] }),
    ).toBe(true);
    expect(
      isComparisonCategoriesManifest({ categories: [{ ...entry, constrained: "yes" }] }),
    ).toBe(false);
  });
});
