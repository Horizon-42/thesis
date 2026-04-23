import { describe, expect, it } from "vitest";

import {
  AIRPORTS_INDEX_URL,
  airportDataUrl,
  airportProcedureDetailUrl,
  airportProcedureDetailsIndexUrl,
  airportDsmHeightmapTerrainUrl,
  airportChartsIndexUrl,
  isAirportsIndexManifest,
  normalizeAirportCode,
  sortAirportCatalog,
} from "../airportData";

describe("airportData helpers", () => {
  it("builds airport-scoped data URLs", () => {
    expect(airportDataUrl("krdu", "airport.json")).toBe("/data/airports/KRDU/airport.json");
    expect(airportDsmHeightmapTerrainUrl("cyvr", "metadata.json")).toBe(
      "/data/airports/CYVR/dsm/heightmap-terrain/metadata.json",
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
});
