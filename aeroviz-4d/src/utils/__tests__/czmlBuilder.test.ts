/**
 * czmlBuilder.test.ts
 * -------------------
 * Unit tests for the CZML construction utilities.
 *
 * These tests don't need Cesium, a browser, or real trajectory data.
 * They verify the JSON structure produced by each builder function.
 *
 * Why test JSON structure?
 *   Cesium silently ignores unknown keys and crashes on missing required ones.
 *   A unit test that checks the exact output structure catches mistakes
 *   (e.g. wrong key name, wrong array layout) before a frustrating runtime bug.
 */

import { describe, it, expect } from "vitest";
import {
  buildDocumentPacket,
  buildSampledPosition,
  buildFlightPacket,
  buildCzml,
  type FlightInput,
  type Waypoint4D,
} from "../czmlBuilder";

// ── Test fixtures ─────────────────────────────────────────────────────────────

const START_ISO = "2026-04-01T08:00:00Z";
const END_ISO   = "2026-04-01T09:00:00Z";

const WAYPOINTS: Waypoint4D[] = [
  { offsetSec: 0,   lon: -119.38, lat: 49.95, altM: 4500 },
  { offsetSec: 120, lon: -119.40, lat: 49.90, altM: 3800 },
  { offsetSec: 240, lon: -119.42, lat: 49.85, altM: 3200 },
];

const FLIGHT: FlightInput = {
  id: "UAL123",
  callsign: "United 123",
  type: "B738",
  waypoints: WAYPOINTS,
  trailColor: [255, 165, 0, 200],
};

// ─────────────────────────────────────────────────────────────────────────────
describe("buildDocumentPacket", () => {
  it("id is always 'document'", () => {
    const doc = buildDocumentPacket(START_ISO, END_ISO);
    expect(doc.id).toBe("document");
  });

  it("clock interval is formatted as 'start/end'", () => {
    const doc = buildDocumentPacket(START_ISO, END_ISO, 60);
    expect(doc.clock?.interval).toBe(`${START_ISO}/${END_ISO}`);
  });

  it("uses the provided multiplier", () => {
    const doc = buildDocumentPacket(START_ISO, END_ISO, 120);
    expect(doc.clock?.multiplier).toBe(120);
  });

  it("defaults multiplier to 60 when not provided", () => {
    const doc = buildDocumentPacket(START_ISO, END_ISO);
    expect(doc.clock?.multiplier).toBe(60);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("buildSampledPosition", () => {
  it("cartographicDegrees length is 4 × number of waypoints", () => {
    const pos = buildSampledPosition(START_ISO, WAYPOINTS);
    expect(pos.cartographicDegrees).toHaveLength(4 * WAYPOINTS.length);
  });

  it("first entry starts with offsetSec 0", () => {
    const pos = buildSampledPosition(START_ISO, WAYPOINTS);
    expect(pos.cartographicDegrees[0]).toBe(0);
  });

  it("values are interleaved as [t, lon, lat, alt, t, lon, lat, alt, ...]", () => {
    const pos = buildSampledPosition(START_ISO, WAYPOINTS);
    const arr = pos.cartographicDegrees;
    expect(arr[0]).toBe(WAYPOINTS[0].offsetSec);
    expect(arr[1]).toBe(WAYPOINTS[0].lon);
    expect(arr[2]).toBe(WAYPOINTS[0].lat);
    expect(arr[3]).toBe(WAYPOINTS[0].altM);
  });

  it("epoch is passed through unchanged", () => {
    const pos = buildSampledPosition(START_ISO, WAYPOINTS);
    expect(pos.epoch).toBe(START_ISO);
  });

  it("returns empty cartographicDegrees for zero waypoints", () => {
    const pos = buildSampledPosition(START_ISO, []);
    expect(pos.cartographicDegrees).toHaveLength(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("buildFlightPacket", () => {
  it("id matches the flight input id", () => {
    const packet = buildFlightPacket(FLIGHT, START_ISO);
    expect(packet.id).toBe("UAL123");
  });

  it("has a model with gltf pointing to aircraft.glb", () => {
    const packet = buildFlightPacket(FLIGHT, START_ISO);
    expect(packet.model?.gltf).toBe("/models/aircraft.glb");
  });

  it("orientation uses a velocityReference to auto-compute heading", () => {
    const packet = buildFlightPacket(FLIGHT, START_ISO);
    expect(packet.orientation?.velocityReference).toBe("#UAL123.position");
  });

  it("path trailTime is 300 seconds", () => {
    const packet = buildFlightPacket(FLIGHT, START_ISO);
    expect(packet.path?.trailTime).toBe(300);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("buildCzml", () => {
  it("first element of returned array has id 'document'", () => {
    const czml = buildCzml([FLIGHT], START_ISO, END_ISO);
    expect(czml[0].id).toBe("document");
  });

  it("array length equals number of flights + 1 (for document packet)", () => {
    const twoFlights = [FLIGHT, { ...FLIGHT, id: "WJA456", callsign: "WestJet 456" }];
    const czml = buildCzml(twoFlights, START_ISO, END_ISO);
    expect(czml).toHaveLength(3);
  });
});
