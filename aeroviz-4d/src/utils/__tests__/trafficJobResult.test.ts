import { describe, expect, it } from "vitest";
import type { ComparisonGroup, ComparisonIndex } from "../../data/airportData";
import type { TrafficArrival } from "../../data/trafficJobs";
import {
  BLOCK_START_HOURS,
  BLOCK_START_MINUTES,
  arrivalsInBlock,
  blockJobLabel,
  blockStartOf,
  defaultBlockHour,
  flightJobLabel,
  summarizeTrafficResult,
  trafficResultRows,
  trafficSummaryText,
} from "../trafficJobResult";

function group(key: string, extra: Partial<ComparisonGroup> = {}): ComparisonGroup {
  return {
    group: key, flightId: key.split("_")[0], runway: "05L", airport: "KRDU", status: "solved", finalTimeS: 300,
    initialState: null, entities: [`ref-${key}`, `sim-${key}`], czml: "a.czml", ...extra,
  };
}
const indexOf = (groups: ComparisonGroup[], scene = false): ComparisonIndex => ({
  schemaVersion: "comparison-v2-generation", generation: "g", epoch: "e", startHidden: true,
  referenceSource: "canonicalObserved", evaluationReport: "r.json", groups,
  ...(scene ? { scene: { startUtc: "2026-05-21T17:47:18.959Z", background: { recorded: [], startOffsetsS: [] } } } : {}),
});
const scene = (outcome: string, delayS: number | null) => ({ scene: { startOffsetS: 0, outcome, delayS } });

describe("trafficResultRows", () => {
  it("lists one row per controlled aircraft, outcomes by their plain names with the raw string in the title", () => {
    const rows = trafficResultRows(indexOf([
      group("AAL1_05L_a_1", scene("separated_at_baseline", 0)),
      group("BAW2_05R_b_2", { runway: "05R", ...scene("unresolved", 80.4) }),
      group("DAL3_05L_c_3", { status: "failed", ...scene("BaselineFailed: ETA solve: x", null) }),
    ], true));
    expect(rows.map((r) => [r.callsign, r.runway, r.outcomeName, r.delayS])).toEqual([
      ["AAL1", "05L", "separated at the first solve (no re-solve)", 0],
      ["BAW2", "05R", "loss left (round limit)", 80.4],
      ["DAL3", "05L", "not optimized: no ETA", null],
    ]);
    expect(rows[1].title).toBe("BAW2_05R_b_2 — loss left (round limit) (unresolved)");
    expect(rows[1].flightKey).toBe("BAW2_05R_b_2");
  });

  it("reads an M1 window's outcome from its traffic block, and names a failed record without one", () => {
    const traffic = { outcome: "separated", recorded: [], startOffsetsS: [] };
    const [solved, failed] = trafficResultRows(indexOf([
      group("AAL1_05L_a_1", { traffic }), group("BAW2_05L_b_2", { status: "failed" })]));
    expect(solved.outcomeName).toBe("separated after re-solve");
    expect(failed.outcome).toBeNull();
    expect(failed.outcomeName).toBe("not optimized");
    expect(failed.delayS).toBeNull();
  });
});

describe("summarizeTrafficResult / trafficSummaryText", () => {
  const rows = trafficResultRows(indexOf([
    group("A_05L_a_1", scene("separated_at_baseline", 0)),
    group("B_05L_a_2", scene("separated_at_baseline", 30)),
    group("C_05L_a_3", scene("unresolved", 100)),
    group("D_05L_a_4", { status: "failed", ...scene("BaselineFailed: slot solve (delay 5.0 s): x", null) }),
  ], true));
  const left = { visual: { answered: 2, not_answered: 1 }, ifr: { answered: 5, not_answered: 4 } };

  it("counts the outcomes by plain name, the delays and the losses left after the block's final check (M2)", () => {
    const summary = summarizeTrafficResult(rows, { flown_aircraft_with_a_loss_left_after_the_block: left }, true);
    expect(summary.aircraft).toBe(4);
    expect(summary.outcomes).toEqual([
      { name: "separated at the first solve (no re-solve)", count: 2 },
      { name: "loss left (round limit)", count: 1 },
      { name: "not optimized: its slot could not be flown", count: 1 },
    ]);
    expect(summary.delays).toEqual({ medianS: 30, maxS: 100, overMinute: 1 });
    expect(summary.lossesLeft).toEqual({ answered: 2, notAnswered: 1 });
    expect(trafficSummaryText(summary)).toBe(
      "4 aircraft controlled: 2 separated at the first solve (no re-solve), 1 loss left (round limit), 1 not optimized: its slot could not be flown" +
        " · delay median 30 s, largest 100 s, 1 over 60 s" +
        " · after the block's final check: 2 aircraft with a loss of visual separation they answer for, 1 with one they do not",
    );
  });

  it("names the arrivals of a block that stayed records for want of a dynamics model", () => {
    const summary = summarizeTrafficResult(rows.slice(0, 2), { skipped_no_dynamics: 3 }, true);
    expect(summary.records).toBe(3);
    expect(trafficSummaryText(summary)).toContain("3 more landed in the block and stayed records (no aircraft dynamics model)");
    expect(trafficSummaryText(summarizeTrafficResult(rows.slice(0, 2), { skipped_no_dynamics: 0 }, true)))
      .not.toContain("stayed records");
    expect(summarizeTrafficResult(rows.slice(0, 1), { skipped_no_dynamics: 3 }, false).records).toBeNull();   // an M1 job
  });

  it("has no delays and no final check for an M1 window", () => {
    const summary = summarizeTrafficResult(rows.slice(0, 1), { flown_aircraft_with_a_loss_left_after_the_block: left }, false);
    expect(summary.delays).toBeNull();
    expect(summary.lossesLeft).toBeNull();
    expect(trafficSummaryText(summary)).toBe("1 aircraft controlled: 1 separated at the first solve (no re-solve)");
  });

  it("shows no delay line when no aircraft has a slot and no final check when the job reports none", () => {
    const summary = summarizeTrafficResult(rows.slice(3), undefined, true);
    expect(summary.delays).toBeNull();
    expect(summary.lossesLeft).toBeNull();
  });
});

describe("the block of an M2 job", () => {
  const arrival = (landingUtc: string): TrafficArrival => ({
    flightKey: landingUtc, callsign: "X", runway: "05L", type: null, entryUtc: landingUtc, landingUtc });
  const day = [arrival("2026-05-21T17:00:00Z"), arrival("2026-05-21T17:29:59Z"), arrival("2026-05-21T17:30:00Z"),
               arrival("2026-05-21T18:10:00Z")];

  it("starts at a 24-hour UTC time of the day: an hour 00–23 and a minute of 00, 15, 30 or 45", () => {
    expect(blockStartOf("2026-05-21", "17", "15")).toBe("2026-05-21T17:15:00Z");
    expect(blockStartOf("2026-05-21", "00", "00")).toBe("2026-05-21T00:00:00Z");
    expect(BLOCK_START_HOURS).toHaveLength(24);
    expect(BLOCK_START_HOURS[0]).toBe("00");
    expect(BLOCK_START_HOURS[23]).toBe("23");
    expect([...BLOCK_START_MINUTES]).toEqual(["00", "15", "30", "45"]);
  });

  it("defaults to the hour of the day's first landing, rounded down", () => {
    expect(defaultBlockHour(day)).toBe("17");
    expect(defaultBlockHour([arrival("2026-05-21T05:59:59Z")])).toBe("05");
    expect(defaultBlockHour([])).toBe("00");
  });

  it("holds the arrivals that land in [start, start + length)", () => {
    expect(arrivalsInBlock(day, "2026-05-21T17:00:00Z", 1800).map((a) => a.landingUtc)).toEqual([
      "2026-05-21T17:00:00Z", "2026-05-21T17:29:59Z"]);                              // 17:30:00 is the next block's
    expect(arrivalsInBlock(day, "2026-05-21T17:00:01Z", 1800).map((a) => a.landingUtc)).toEqual([
      "2026-05-21T17:29:59Z", "2026-05-21T17:30:00Z"]);                                // the start is inclusive, the first landing's is not
    expect(arrivalsInBlock(day, "2026-05-21T17:00:00Z", 3600)).toHaveLength(3);
    expect(arrivalsInBlock(day, "", 3600)).toEqual([]);
  });
});

describe("a job's inputs in words", () => {
  it("names a block by its UTC day and its start and end", () => {
    expect(blockJobLabel("2026-05-21T18:00:00Z", 900)).toBe("Block 2026-05-21 18:00–18:15 UTC");
    expect(blockJobLabel("2026-05-21T17:45:00Z", 3600)).toBe("Block 2026-05-21 17:45–18:45 UTC");
  });

  it("names the second day of a block that runs past midnight", () => {
    expect(blockJobLabel("2026-05-21T23:45:00Z", 1800)).toBe("Block 2026-05-21 23:45–2026-05-22 00:15 UTC");
  });

  it("names a flight without a callsign by the first field of its key", () => {
    const landing = "2026-05-29T12:00:00Z";
    expect(flightJobLabel({ flightKey: "N123AB_05L_a00001_20260529T120000Z", callsign: null, runway: "05L", type: null,
                            entryUtc: landing, landingUtc: landing })).toBe("N123AB, 2026-05-29");
  });

  it("names a flight by its callsign and the day it lands", () => {
    const landing = "2026-07-16T12:00:00Z";
    expect(flightJobLabel({ flightKey: "FFL1206_05L_a_1", callsign: "FFL1206", runway: "05L", type: null,
                            entryUtc: landing, landingUtc: landing })).toBe("FFL1206, 2026-07-16");
  });
});
