import { describe, expect, it } from "vitest";
import * as Cesium from "cesium";
import type { ComparisonGroup } from "../../data/airportData";
import {
  TRAFFIC_DOCUMENT_PACKET,
  trafficEntityId,
  trafficFlightKey,
  trafficFlightKeys,
  trafficPackets,
  type TimedCzmlPacket,
} from "../comparisonTraffic";

const EPOCH = "2026-04-01T08:00:00Z";
const A = "AAL100_05L_a00001_20260501T000300Z";      // commanded flights
const B = "BAW200_05L_b00002_20260501T020300Z";
const EARLIER = "DAL1312_05L_d4e5f6_20260501T000100Z";
const LATER = "UPS22_05R_a7b8c9_20260501T000500Z";

function group(key: string, recorded: string[], startOffsetsS: number[], withTraffic = true): ComparisonGroup {
  return {
    group: key, flightId: key.split("_")[0], runway: "05L", airport: "KRDU", czml: "c.czml", status: "solved",
    finalTimeS: 300, initialState: null, entities: [`ref-${key}`, `sim-${key}`],
    ...(withTraffic ? { traffic: { outcome: "separated", recorded: recorded.map((k) => `ref-${k}`), startOffsetsS } } : {}),
  };
}

/** An arrival-window packet: t = 0 at the flight's own entry, flying due east a degree per 100 s. */
function packet(id: string): TimedCzmlPacket {
  return {
    id,
    name: id.split("_")[0],
    position: {
      epoch: EPOCH,
      cartographicDegrees: [0, -78.0, 35.0, 1000, 100, -77.0, 35.0, 900, 200, -76.0, 35.0, 800],
    },
    orientation: { epoch: EPOCH, unitQuaternion: [0, 0, 0, 0, 1] },
  };
}

const samplesOf = (p: TimedCzmlPacket) => p.position.cartographicDegrees;

const packets = new Map(
  [EARLIER, LATER].map((key) => [key, packet(key)] as const),
);

describe("trafficFlightKey", () => {
  it("names the flight a ref- reference id stands for, and refuses any other id", () => {
    expect(trafficFlightKey(`ref-${EARLIER}`)).toBe(EARLIER);
    expect(() => trafficFlightKey(EARLIER)).toThrow(/not a ref- reference id/);
    expect(() => trafficFlightKey(`sim-${EARLIER}`)).toThrow(/not a ref- reference id/);
  });
});

describe("trafficFlightKeys", () => {
  it("lists each recorded flight of the shown groups once, and none for a group without traffic", () => {
    const groups = [
      group(A, [EARLIER, LATER], [-9.349, 90.75]),
      group(B, [LATER], [12]),
      group("NONE_05L_c00003_20260501T040300Z", [], [], false),
    ];
    expect(trafficFlightKeys(groups)).toEqual([EARLIER, LATER]);
    expect(trafficFlightKeys([groups[2]])).toEqual([]);
  });
});

describe("trafficPackets", () => {
  it("puts each recorded aircraft at its entry offset on the commanded flight's clock", () => {
    const [earlier, later] = trafficPackets([group(A, [EARLIER, LATER], [-9.349, 90.75])], packets);

    // Entering 90.750 s after the commanded aircraft: its epoch (and its orientation's) moves that far on.
    expect(later.position.epoch).toBe("2026-04-01T08:01:30.750Z");
    expect(later.orientation?.epoch).toBe("2026-04-01T08:01:30.750Z");
    // Already in the scene when the commanded aircraft enters: its epoch is before the clock's zero.
    expect(earlier.position.epoch).toBe("2026-04-01T07:59:50.651Z");
    expect(earlier.id).toBe(trafficEntityId(A, EARLIER));
    // The samples themselves are the backend's, not copied or changed.
    expect(samplesOf(later)).toBe(samplesOf(packets.get(LATER)!));
    expect(later.name).toBe("UPS22");
  });

  it("does not move the source packet", () => {
    trafficPackets([group(A, [LATER], [90.75])], packets);
    expect(packets.get(LATER)!.position.epoch).toBe(EPOCH);
  });

  it("is sampled by Cesium at the shifted time: the aircraft is where it is offset seconds into its own track", async () => {
    const [copy] = trafficPackets([group(A, [LATER], [90.75])], packets);
    const source = await new Cesium.CzmlDataSource("t").load([TRAFFIC_DOCUMENT_PACKET, copy]);
    const entity = source.entities.getById(copy.id)!;
    const at = (seconds: number) => Cesium.JulianDate.addSeconds(
      Cesium.JulianDate.fromIso8601(EPOCH), seconds, new Cesium.JulianDate());
    const lonDegAt = (seconds: number) => {
      const p = Cesium.Cartographic.fromCartesian(entity.position!.getValue(at(seconds))!);
      return Cesium.Math.toDegrees(p.longitude);
    };
    // 100 s into its own track (its second sample) it is at -77.0, which is 190.75 s on the group's clock.
    expect(lonDegAt(190.75)).toBeCloseTo(-77.0, 6);
    expect(lonDegAt(90.75)).toBeCloseTo(-78.0, 6);
    // And it is NOT there at the unshifted time (the bug this shift exists to prevent).
    expect(lonDegAt(100)).not.toBeCloseTo(-77.0, 3);
  });

  it("gives each shown group its own copy of a flight both list, at that group's offset", () => {
    const copies = trafficPackets([group(A, [LATER], [90.75]), group(B, [LATER], [-30])], packets);
    expect(copies.map((copy) => copy.id)).toEqual([trafficEntityId(A, LATER), trafficEntityId(B, LATER)]);
    expect(copies.map((copy) => copy.position.epoch)).toEqual(["2026-04-01T08:01:30.750Z", "2026-04-01T07:59:30.000Z"]);
  });

  it("gives a group without traffic, and no group at all, no copies", () => {
    const plain = group("NONE_05L_c00003_20260501T040300Z", [], [], false);
    expect(trafficPackets([plain], packets)).toEqual([]);
    expect(trafficPackets([], packets)).toEqual([]);
  });

  it("refuses a recorded aircraft the backend did not serve, by group and flight", () => {
    expect(() => trafficPackets([group(A, ["GONE_05L_e00005_20260501T000200Z"], [1])], packets))
      .toThrow(`${A}: the backend served no arrival-window track for recorded aircraft GONE_05L_e00005_20260501T000200Z`);
  });
});
