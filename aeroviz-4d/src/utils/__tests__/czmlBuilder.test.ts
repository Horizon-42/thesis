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
    // TODO — Assert doc.clock?.interval === `${START_ISO}/${END_ISO}`
    expect(doc.clock).toBeDefined(); // replace with specific assertion
  });

  it("uses the provided multiplier", () => {
    const doc = buildDocumentPacket(START_ISO, END_ISO, 120);
    // TODO — Assert doc.clock?.multiplier === 120
    expect(doc.clock?.multiplier).toBeDefined();
  });

  it("defaults multiplier to 60 when not provided", () => {
    const doc = buildDocumentPacket(START_ISO, END_ISO);
    // TODO — Assert doc.clock?.multiplier === 60
    expect(doc.clock?.multiplier).toBeDefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("buildSampledPosition", () => {
  it("cartographicDegrees length is 4 × number of waypoints", () => {
    const pos = buildSampledPosition(START_ISO, WAYPOINTS);
    // Each waypoint contributes 4 numbers: [t, lon, lat, alt]
    // TODO — expect(pos.cartographicDegrees).toHaveLength(4 * WAYPOINTS.length);
    expect(pos.cartographicDegrees).toBeDefined();
  });

  it("first entry starts with offsetSec 0", () => {
    const pos = buildSampledPosition(START_ISO, WAYPOINTS);
    // TODO — expect(pos.cartographicDegrees[0]).toBe(0);
    expect(pos.cartographicDegrees).toBeDefined();
  });

  it("values are interleaved as [t, lon, lat, alt, t, lon, lat, alt, ...]", () => {
    const pos = buildSampledPosition(START_ISO, WAYPOINTS);
    const arr = pos.cartographicDegrees;
    // For the first waypoint: indices 0,1,2,3 = t,lon,lat,alt
    // TODO ① — Assert arr[0] === WAYPOINTS[0].offsetSec
    // TODO ② — Assert arr[1] === WAYPOINTS[0].lon
    // TODO ③ — Assert arr[2] === WAYPOINTS[0].lat
    // TODO ④ — Assert arr[3] === WAYPOINTS[0].altM
    expect(arr).toBeDefined();
  });

  it("epoch is passed through unchanged", () => {
    const pos = buildSampledPosition(START_ISO, WAYPOINTS);
    // TODO — expect(pos.epoch).toBe(START_ISO);
    expect(pos.epoch).toBeDefined();
  });

  it("returns empty cartographicDegrees for zero waypoints", () => {
    const pos = buildSampledPosition(START_ISO, []);
    // TODO — expect(pos.cartographicDegrees).toHaveLength(0);
    expect(pos.cartographicDegrees).toBeDefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("buildFlightPacket", () => {
  it("id matches the flight input id", () => {
    const packet = buildFlightPacket(FLIGHT, START_ISO);
    // TODO — expect(packet.id).toBe("UAL123");
    expect(packet.id).toBeDefined();
  });

  it("has a model with gltf pointing to aircraft.glb", () => {
    const packet = buildFlightPacket(FLIGHT, START_ISO);
    // TODO — expect(packet.model?.gltf).toBe("/models/aircraft.glb");
    expect(packet.model).toBeDefined();
  });

  it("orientation uses a velocityReference to auto-compute heading", () => {
    const packet = buildFlightPacket(FLIGHT, START_ISO);
    // The velocityReference should be `#${id}.position`
    // TODO — expect(packet.orientation?.velocityReference).toBe("#UAL123.position");
    expect(packet.orientation).toBeDefined();
  });

  it("path trailTime is 300 seconds", () => {
    const packet = buildFlightPacket(FLIGHT, START_ISO);
    // TODO — expect(packet.path?.trailTime).toBe(300);
    expect(packet.path).toBeDefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
describe("buildCzml", () => {
  it("first element of returned array has id 'document'", () => {
    const czml = buildCzml([FLIGHT], START_ISO, END_ISO);
    // TODO — expect(czml[0].id).toBe("document");
    expect(czml[0]).toBeDefined();
  });

  it("array length equals number of flights + 1 (for document packet)", () => {
    const twoFlights = [FLIGHT, { ...FLIGHT, id: "WJA456", callsign: "WestJet 456" }];
    const czml = buildCzml(twoFlights, START_ISO, END_ISO);
    // TODO — expect(czml).toHaveLength(3);
    expect(czml.length).toBeGreaterThan(0);
  });
});
