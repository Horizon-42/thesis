import { describe, expect, it } from "vitest";
import type { ComparisonGroup, ComparisonIndex } from "../../data/airportData";
import type { EvaluationRow } from "../../data/evaluationReport";
import type { TrafficAircraftChange, TrafficScenarioCatalog } from "../../data/trafficJobs";
import catalogMirror from "../../data/__tests__/fixtures/trafficScenarioCatalog.json";
import * as trafficResultModule from "../trafficJobResult";
import {
  EARLIER_NOTE_OVER_S,
  SLOT_NOTE,
  SOLVER_SENTENCE,
  STAYED_RECORD_REASON,
  VECTORED_NOTE,
  blockJobLabel,
  blockLossShares,
  blockLossesText,
  blockSpan,
  flightJobLabel,
  formatDuration,
  formatElapsed,
  formatMinutesSeconds,
  gateMisses,
  isFlown,
  landingShiftS,
  landsText,
  recordLosses,
  resultLine1,
  resultLine2,
  solverTitle,
  stayedRecordLines,
  summarizeTrafficResult,
  timingText,
  trafficResultRows,
  trafficSummaryText,
  type TrafficResultRow,
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
const change = (over: Partial<TrafficAircraftChange> = {}): TrafficAircraftChange => ({
  type: "B738", firstSolveLosses: 5, finalLosses: 0, landingVsRecordS: -400, delayS: null, blockCheckLosses: null,
  optimizeS: 12.34, optimizeCpuS: 11.8, solves: 3, failedSolves: 1, ...over });
const NOT_FLOWN = { firstSolveLosses: null, finalLosses: null, landingVsRecordS: null };
/** The evaluation report's row of an aircraft off the landing gates (only the fields the gate words read). */
const gateRow = (over: Partial<EvaluationRow> & Record<string, unknown> = {}): EvaluationRow => ({
  id: "x", file: null, flight_key: "N850DP_05R_a_1", subject: "optimized", airport: "KRDU", runway: "05R", benchmark: "lpv",
  solved: true, success: false, verdict: "fail", event_status: "ok", lateral_result: "pass", vertical_result: "pass",
  speed_result: "pass", violations: [], lateral_m: 120, vertical_m: 3, crossing_speed_ms: 58,
  bounds: { lateral_criterion: "runway_half_width_at_threshold", lateral_m: 300, vertical_lower_m: -22, vertical_upper_m: 22,
    speed_lower_ms: 55.2, speed_upper_ms: 60.24 },
  ...over }) as EvaluationRow;
/**
 * Every group of an index has an entry (the hook checks it); an M2 job's flown aircraft carry their slot's delay and the losses
 * after the block (zero unless a test says otherwise), as `traffic_job.per_aircraft` writes them.
 */
const changes = (index: ComparisonIndex, over: Record<string, Partial<TrafficAircraftChange>> = {}) =>
  Object.fromEntries(index.groups.map((g) => {
    const given = change(over[g.group] ?? {});
    const inBlock = index.scene !== undefined && isFlown(given);
    return [g.group, inBlock ? { ...given, delayS: over[g.group]?.delayS ?? 0, blockCheckLosses: over[g.group]?.blockCheckLosses ?? 0 } : given];
  }));
const rowsOf = (index: ComparisonIndex, over: Record<string, Partial<TrafficAircraftChange>> = {},
  reportRows: Record<string, EvaluationRow> = {}) => {
  // an aircraft the index draws off the landing gates has its row in the job's evaluation report
  const offTarget = Object.fromEntries(index.groups.filter((g) => g.status === "offTarget").map((g) => [
    g.group, reportRows[g.group] ?? gateRow({ flight_key: g.group, violations: ["lateral"], lateral_m: 341 })]));
  return trafficResultRows(index, changes(index, over), offTarget);
};

describe("trafficResultRows", () => {
  it("lists one row per controlled aircraft, outcomes by their plain names with the raw string in the title", () => {
    const rows = rowsOf(indexOf([
      group("AAL1_05L_a_1", scene("separated_at_baseline", 0)),
      group("BAW2_05R_b_2", { runway: "05R", ...scene("unresolved", 80.4) }),
      group("DAL3_05L_c_3", { status: "failed", ...scene("BaselineFailed: ETA solve: x", null) }),
    ], true));
    expect(rows.map((r) => [r.callsign, r.runway, r.outcomeName])).toEqual([
      ["AAL1", "05L", "separated at the first solve (no re-solve)"],
      ["BAW2", "05R", "loss left — the best of its solves is shown (fewest losses)"],
      ["DAL3", "05L", "not optimized: no ETA"],
    ]);
    expect(rows[1].title).toBe("BAW2_05R_b_2 — loss left — the best of its solves is shown (fewest losses) (unresolved)");
    expect(rows[1].flightKey).toBe("BAW2_05R_b_2");
  });

  it("reads an M1 window's outcome from its traffic block, and names a failed record without one", () => {
    const traffic = { outcome: "separated", recorded: [], startOffsetsS: [] };
    const [solved, failed] = rowsOf(indexOf([
      group("AAL1_05L_a_1", { traffic }), group("BAW2_05L_b_2", { status: "failed" })]));
    expect(solved.outcomeName).toBe("separated after re-solve");
    expect(failed.outcome).toBeNull();
    expect(failed.outcomeName).toBe("not optimized");
  });

  it("marks the aircraft whose path ended off the landing gates (the group's offTarget status: a yellow path)", () => {
    const rows = rowsOf(indexOf([
      group("A_05L_a_1", { status: "offTarget" }), group("B_05L_a_2", { status: "solved" }),
      group("C_05L_a_3", { status: "failed" })]));
    expect(rows.map((r) => r.offTarget)).toEqual([true, false, false]);
  });

  it("carries what changed for each aircraft, and refuses an index that lists one the job says nothing of", () => {
    const index = indexOf([group("AAL1_05L_a_1", scene("separated", 0)), group("BAW2_05L_b_2", scene("separated", 0))], true);
    const [first] = rowsOf(index, { "AAL1_05L_a_1": { firstSolveLosses: 9 } });
    expect(first.change.firstSolveLosses).toBe(9);
    expect(() => trafficResultRows(index, { "AAL1_05L_a_1": change({ delayS: 0, blockCheckLosses: 0 }) }, {})).toThrow("says nothing of BAW2_05L_b_2");
  });

  it("refuses a flown aircraft of a block that has no slot delay or no loss count after the block, and an M1 one needs neither", () => {
    const block = indexOf([group("AAL1_05L_a_1", scene("separated", 0))], true);
    expect(() => trafficResultRows(block, { "AAL1_05L_a_1": change({ delayS: null, blockCheckLosses: 0 }) }, {})).toThrow("no slot delay");
    expect(() => trafficResultRows(block, { "AAL1_05L_a_1": change({ delayS: 0, blockCheckLosses: null }) }, {})).toThrow("no slot delay");
    const m1 = indexOf([group("AAL1_05L_a_1", scene("separated", 0))]);
    expect(trafficResultRows(m1, { "AAL1_05L_a_1": change() }, {})).toHaveLength(1);
  });

  it("puts the evaluation's own codes in the title of an aircraft drawn off the landing gates", () => {
    const [row] = rowsOf(indexOf([group("A_05L_a_1", { status: "offTarget" })], true), {},
      { A_05L_a_1: gateRow({ flight_key: "A_05L_a_1", violations: ["lateral", "xyz_new"], lateral_m: 341 }) });
    expect(row.title).toBe("A_05L_a_1 — not optimized — the evaluation's own words: lateral, xyz_new");
  });
});

describe("the two lines of an aircraft in the result", () => {
  const rowOf = (over: Partial<TrafficAircraftChange> = {}, groupExtra: Partial<ComparisonGroup> = {}): TrafficResultRow =>
    rowsOf(indexOf([group("N850DP_05R_a_1", { runway: "05R", ...scene("separated_at_baseline", 0), ...groupExtra })], true),
      { N850DP_05R_a_1: over })[0];

  it("names it, its type and its runway, then says how it was flown (the first line)", () => {
    expect(resultLine1(rowOf({ type: "H25B" }))).toBe("N850DP · H25B · 05R — separated at the first solve (no re-solve)");
    expect(resultLine1(rowOf({ type: null }))).toBe("N850DP · — · 05R — separated at the first solve (no re-solve)");
  });

  it("says what changed on the second line of an M1 aircraft: the losses, where it lands, the solver's time, CPU and solves", () => {
    // an M1 job has no slot: no delay, no block, no earliest-arrival solves; one solve is "1 solve"
    expect(resultLine2(rowOf({ delayS: null, solves: 1, failedSolves: 0, optimizeS: 4, optimizeCpuS: 3.6 }), 28, 1, false)).toBe(
      "losses: record 28 → first solve 5 → final 0 · lands 6 min 40 s earlier than it really did " +
      "(shortest published route at minimum time; the real flight was vectored) · optimized in 4.0 s, CPU 3.6 s (1 solve, 0 failed)");
  });

  it("says an M2 aircraft's two numbers apart, each with what it is: when flown, and after the whole block", () => {
    const row = rowOf({ firstSolveLosses: 9, finalLosses: 0, landingVsRecordS: -400, delayS: 0, blockCheckLosses: 3, optimizeS: 12.34,
      optimizeCpuS: 11.8, solves: 3, failedSolves: 1 });
    // an M2 aircraft's solves include its earliest-arrival solves: the line says so
    expect(resultLine2(row, 143, 1, true)).toBe(
      "when flown: record 143 → first solve 9 → final 0 · after the whole block: 3 (from aircraft flown after it) · " +
      "lands 6 min 40 s earlier than it really did (shortest published route at minimum time; the real flight was vectored) · " +
      "slot delay 0 s (after its earliest arrival) · " +
      "optimized in 12.3 s, CPU 11.8 s (3 solves, 1 failed, the earliest-arrival solves included)");
    // none after the block: no "(from aircraft flown after it)" for nothing
    expect(resultLine2(rowOf({ blockCheckLosses: 0, delayS: 12 }), 143, 1, true)).toContain(" · after the whole block: 0 · ");
    expect(resultLine2(rowOf({ blockCheckLosses: 0, delayS: 12 }), 143, 1, true)).toContain("slot delay 12 s (after its earliest arrival)");
  });

  it("says when the optimized flight lands later, and when it lands when it really did — one rounding for every sentence", () => {
    expect(resultLine2(rowOf({ landingVsRecordS: 75 }), 1, 1, false)).toContain("lands 1 min 15 s later than it really did · ");
    expect(resultLine2(rowOf({ landingVsRecordS: 0.4 }), 1, 1, false)).toContain("lands when it really did");
    // -0.5 s and +0.5 s both round away from zero, in the sentence and in the notes beside it
    expect([-0.5, 0.5, 0.4, 59.5, -60.4, -60.5, 0].map(landingShiftS)).toEqual([-1, 1, 0, 60, -60, -61, 0]);
    expect(landsText(-0.5)).toBe("lands 1 s earlier than it really did");
    expect(landsText(0.5)).toBe("lands 1 s later than it really did");
  });

  it("explains an early landing only when it is more than a minute earlier, on the line it belongs to", () => {
    expect(EARLIER_NOTE_OVER_S).toBe(60);
    const text = (landingVsRecordS: number, isScene: boolean) => resultLine2(rowOf({ landingVsRecordS }), 1, 1, isScene);
    for (const isScene of [false, true]) {
      expect(text(-60.4, isScene)).not.toContain(VECTORED_NOTE);                      // 60 s: not more than 60
      expect(text(-60.5, isScene)).toContain(`lands 1 min 1 s earlier than it really did ${VECTORED_NOTE}`);
      expect(text(-400, isScene)).toContain(VECTORED_NOTE);
      expect(text(-30, isScene)).not.toContain(VECTORED_NOTE);
      expect(text(-30, isScene)).not.toContain(SLOT_NOTE);                            // earlier is never "its slot"
    }
    expect(VECTORED_NOTE).toBe("(shortest published route at minimum time; the real flight was vectored)");
  });

  it("says an M2 flight that lands later lands in its slot, and an M1 flight that lands later says nothing of one", () => {
    expect(resultLine2(rowOf({ landingVsRecordS: 30 }), 1, 1, true)).toContain("lands 30 s later than it really did (its slot) · ");
    expect(resultLine2(rowOf({ landingVsRecordS: 0.2 }), 1, 1, true)).not.toContain(SLOT_NOTE);     // when it did
    expect(resultLine2(rowOf({ landingVsRecordS: 30 }), 1, 1, false)).not.toContain(SLOT_NOTE);
    expect(SLOT_NOTE).toBe("(its slot)");
  });

  it("adds, for an aircraft off the landing gates, which gates it missed and by how much", () => {
    const miss = gateRow({ flight_key: "N850DP_05R_a_1", violations: ["lateral", "speed"], lateral_m: 341, crossing_speed_ms: 63.4 });
    const row = rowsOf(indexOf([group("N850DP_05R_a_1", { runway: "05R", status: "offTarget", ...scene("separated", 0) })], true),
      {}, { N850DP_05R_a_1: miss })[0];
    expect(row.gateMisses.map((m) => m.text)).toEqual([
      "lateral 41 m too far (341 m, limit 300 m)", "speed 3.2 m/s too fast (63.4 m/s, window 55.2–60.2 m/s)"]);
    expect(resultLine2(row, 1, 1, false)).toMatch(
      / · optimized in 12\.3 s, CPU 11\.8 s \(3 solves, 1 failed\) · missed the landing gates \(yellow\): lateral 41 m too far \(341 m, limit 300 m\); speed 3\.2 m\/s too fast \(63\.4 m\/s, window 55\.2–60\.2 m\/s\)$/);
    expect(resultLine2(rowOf(), 1, 1, false)).not.toContain("missed");
    expect(rowOf().gateMisses).toEqual([]);
  });

  it("refuses an off-target aircraft the evaluation report has no row of", () => {
    const index = indexOf([group("A_05L_a_1", { status: "offTarget" })], true);
    expect(() => trafficResultRows(index, changes(index), {})).toThrow("has no row of A_05L_a_1");
  });

  it("says each gate in words: lateral, vertical high and low, speed fast and slow", () => {
    const words = (row: EvaluationRow) => gateMisses(row).map((m) => m.text);
    expect(words(gateRow({ violations: ["lateral"], lateral_m: 341 }))).toEqual(["lateral 41 m too far (341 m, limit 300 m)"]);
    expect(words(gateRow({ violations: ["vertical"], vertical_m: 27 }))).toEqual(["vertical 5 m too high (27 m, limits ±22 m)"]);
    expect(words(gateRow({ violations: ["vertical"], vertical_m: -30.4 }))).toEqual(["vertical 8 m too low (-30 m, limits ±22 m)"]);
    expect(words(gateRow({ violations: ["speed"], crossing_speed_ms: 52.0 }))).toEqual(
      ["speed 3.2 m/s too slow (52 m/s, window 55.2–60.2 m/s)"]);
    expect(words(gateRow({ violations: [] }))).toEqual([]);
    expect(words(gateRow({ violations: ["vertical"], vertical_m: 27, bounds: {
      lateral_criterion: "x", lateral_m: 300, vertical_lower_m: -10, vertical_upper_m: 22 } }))).toEqual(
      ["vertical 5 m too high (27 m, limits -10 to 22 m)"]);
    expect(gateMisses(gateRow({ violations: ["lateral"], lateral_m: 341 }))[0].code).toBe("lateral");
  });

  it("says the events that never crossed the threshold in words, with the amount the row holds", () => {
    // `not_reached`: the evaluation's reason holds how far short it stopped
    const notReached = gateRow({ violations: ["not_reached"], event_status: "not_reached", lateral_m: null, vertical_m: null,
      reason: "trajectory ended 87.4 m before the threshold plane" });
    expect(gateMisses(notReached)).toEqual([{ code: "not_reached", text: "did not reach the runway threshold (stopped 87 m short)" }]);
    // `threshold_not_bracketed`: it ended PAST the threshold, its last segment does not cross the plane — no amount to give
    const notBracketed = gateRow({ violations: ["threshold_not_bracketed"], event_status: "threshold_not_bracketed",
      reason: "trajectory ended beyond the threshold but its final segment does not bracket the plane" });
    expect(gateMisses(notBracketed)).toEqual([{ code: "threshold_not_bracketed",
      text: "ended past the runway threshold with no crossing to measure (its last segment does not cross the threshold plane)" }]);
    expect(() => gateMisses(gateRow({ violations: ["not_reached"], reason: "something else" }))).toThrow("without how far short");
    expect(() => gateMisses(gateRow({ violations: ["not_reached"] }))).toThrow("without how far short");
  });

  it("shows a code it does not know as the evaluation names it", () => {
    const unknown = gateRow({ violations: ["no_crossing", "lateral"], lateral_m: 341 });
    expect(gateMisses(unknown)).toEqual([
      { code: "no_crossing", text: "no_crossing" }, { code: "lateral", text: "lateral 41 m too far (341 m, limit 300 m)" }]);
  });

  it("refuses a failed gate whose number the report does not hold", () => {
    expect(() => gateMisses(gateRow({ violations: ["lateral"], lateral_m: null }))).toThrow("fails lateral without its number");
    expect(() => gateMisses(gateRow({ violations: ["speed"], crossing_speed_ms: null }))).toThrow("fails speed without its number");
  });

  it("says how the record's losses were counted when that was not at the job's own check step", () => {
    const row = rowOf();
    expect(resultLine2(row, 143, 1, false)).toContain("losses: record 143 → first solve");                      // the job's step: 1 s
    expect(resultLine2(row, 143, 2, false)).toContain("losses: record 143 at 2 s steps → first solve");
    expect(resultLine2(row, null, 1, false)).toContain("losses: record ? → first solve");                        // a census of part of the roster
    expect(resultLine2(rowOf({ blockCheckLosses: 0, delayS: 0 }), 143, 2, true)).toContain("when flown: record 143 at 2 s steps → first solve");
  });

  it("says an aircraft was not flown, what the schedule gave it, and how long its attempts took", () => {
    const row = rowOf({ ...NOT_FLOWN, delayS: 2458.4, optimizeS: 20, optimizeCpuS: 18.5, solves: 2, failedSolves: 2 });
    expect(isFlown(row.change)).toBe(false);
    expect(resultLine2(row, 143, 1, true)).toBe(
      "not flown — the schedule gave it a slot 2458 s after its earliest arrival · " +
      "tried for 20.0 s, CPU 18.5 s (2 solves, 2 failed, the earliest-arrival solves included)");
    expect(resultLine2(rowOf({ ...NOT_FLOWN, delayS: null }), 143, 1, true)).toBe(
      "not flown — it has no slot (its earliest-arrival solve failed) · tried for 12.3 s, CPU 11.8 s " +
      "(3 solves, 1 failed, the earliest-arrival solves included)");
    expect(resultLine2(rowOf({ ...NOT_FLOWN }), 143, 1, false)).toBe("not flown · tried for 12.3 s, CPU 11.8 s (3 solves, 1 failed)");  // an M1 job
  });

  it("says the time of an aircraft whose failure lost its solves is not recorded, never 0 s", () => {
    const lost = { ...NOT_FLOWN, optimizeS: null, optimizeCpuS: null, solves: null, failedSolves: null };
    expect(resultLine2(rowOf(lost), 143, 1, false)).toBe("not flown · time not recorded (the failure lost it)");
    expect(resultLine2(rowOf({ ...lost, delayS: 12.5 }), 143, 1, true)).toBe(
      "not flown — the schedule gave it a slot 13 s after its earliest arrival · time not recorded (the failure lost it)");
  });

  it("says durations in words, whole seconds", () => {
    expect([0, 40, 59.6, 60, 400, 3600, 3661].map(formatDuration)).toEqual(
      ["0 s", "40 s", "1 min", "1 min", "6 min 40 s", "60 min", "61 min 1 s"]);
    expect(landsText(-400)).toBe("lands 6 min 40 s earlier than it really did");
    expect(landsText(12.4)).toBe("lands 12 s later than it really did");
  });

  it("has no global footnote about optimized flights landing earlier: each line says its own reason", () => {
    expect(landingShiftS).toBeTypeOf("function");
    expect(Object.keys(trafficResultModule)).not.toContain("OPTIMIZED_EARLIER_NOTE");
    expect(Object.keys(trafficResultModule)).not.toContain("landsEarlier");
  });
});

describe("summarizeTrafficResult / trafficSummaryText", () => {
  const rows = rowsOf(indexOf([
    group("A_05L_a_1", scene("separated_at_baseline", 0)),
    group("B_05L_a_2", { status: "offTarget", ...scene("separated_at_baseline", 30) }),
    group("C_05L_a_3", { status: "offTarget", ...scene("unresolved", 100) }),
    group("D_05L_a_4", { status: "failed", ...scene("BaselineFailed: slot solve (delay 5.0 s): x", null) }),
  ], true), {
    A_05L_a_1: { delayS: 0 }, B_05L_a_2: { delayS: 30, blockCheckLosses: 1 }, C_05L_a_3: { delayS: 100, blockCheckLosses: 4 },
    D_05L_a_4: { ...NOT_FLOWN, delayS: 500 },                                          // not flown: its delay is the schedule's
  });
  const left = { visual: { answered: 2, not_answered: 1 }, ifr: { answered: 5, not_answered: 4 } };
  const stayed = { "N1_05L_a_5": { callsign: "N1", type: "C172" }, "N2_05L_a_6": { callsign: null, type: null } };

  it("counts the outcomes by plain name, the delays of the FLOWN aircraft, those off the landing gates and the losses left", () => {
    const summary = summarizeTrafficResult(rows, { flown_aircraft_with_a_loss_left_after_the_block: left }, true, {});
    expect(summary.aircraft).toBe(4);
    expect(summary.outcomes).toEqual([
      { name: "separated at the first solve (no re-solve)", count: 2 },
      { name: "loss left — the best of its solves is shown (fewest losses)", count: 1 },
      { name: "not optimized: its slot could not be flown", count: 1 },
    ]);
    // the delays come from perAircraft, flown aircraft only: 0, 30, 100 — not the 500 s of the aircraft that was not flown
    expect(summary.delays).toEqual({ medianS: 30, maxS: 100, overMinute: 1 });
    expect(summary.offTarget).toEqual(["B", "C"]);
    expect(summary.lossesLeft).toEqual({ answered: 2, notAnswered: 1, names: ["B", "C"] });         // by the block's final check
    expect(trafficSummaryText(summary)).toBe(
      "4 aircraft controlled: 2 separated at the first solve (no re-solve), 1 loss left — the best of its solves is shown (fewest losses), " +
        "1 not optimized: its slot could not be flown" +
        " · 2 missed the landing gates (yellow path): B, C" +
        " · slot delay median 30 s, largest 100 s, 1 over 60 s" +
        " · after the block's final check: 2 aircraft still have a loss they answer for (B, C); " +
        "1 aircraft with a loss others caused",
    );
  });

  it("counts the aircraft with a loss left from the same per-aircraft numbers its cards show, whatever the readout counted", () => {
    // the readout says 5 flown aircraft answer for a loss; the cards (`blockCheckLosses`) show B and C: the header says 2
    const summary = summarizeTrafficResult(rows, { flown_aircraft_with_a_loss_left_after_the_block: {
      visual: { answered: 5, not_answered: 7 }, ifr: { answered: 5, not_answered: 7 } } }, true, {});
    expect(summary.lossesLeft).toEqual({ answered: 2, notAnswered: 7, names: ["B", "C"] });
    expect(trafficSummaryText(summary)).toContain("2 aircraft still have a loss they answer for (B, C); 7 aircraft with a loss others caused");
  });

  it("names no aircraft off the landing gates when there is none, and says so for a single one", () => {
    const text = (rowsToShow: readonly TrafficResultRow[]) => trafficSummaryText(summarizeTrafficResult(rowsToShow, undefined, false, {}));
    expect(text(rows.slice(0, 1))).not.toContain("missed");
    expect(text(rows.slice(1, 2))).toContain("1 missed the landing gates (yellow path): B");
  });

  it("says so when no aircraft has a loss left it answers for, and names the one that has (singular)", () => {
    const quiet = summarizeTrafficResult(rows.slice(0, 1), { flown_aircraft_with_a_loss_left_after_the_block: {
      visual: { answered: 0, not_answered: 3 }, ifr: { answered: 0, not_answered: 0 } } }, true, {});
    expect(trafficSummaryText(quiet)).toContain(
      "after the block's final check: no aircraft has a loss it answers for; 3 aircraft with a loss others caused");
    const one = summarizeTrafficResult(rows.slice(2, 3), { flown_aircraft_with_a_loss_left_after_the_block: {
      visual: { answered: 1, not_answered: 0 }, ifr: { answered: 0, not_answered: 0 } } }, true, {});
    expect(trafficSummaryText(one)).toContain("1 aircraft still has a loss they answer for (C); 0 aircraft with a loss others caused");
    expect(trafficSummaryText(one)).not.toContain("with one they do not");
  });

  it("spells 'loss of separation' nowhere in the result: it is spelled out once, in the list's header", () => {
    const summary = summarizeTrafficResult(rows, { flown_aircraft_with_a_loss_left_after_the_block: left }, true, stayed);
    expect(trafficSummaryText(summary)).not.toContain("separation");
  });

  it("counts the arrivals of a block that flew their records, from the job's stayed records", () => {
    const summary = summarizeTrafficResult(rows.slice(0, 2), {}, true, stayed);
    expect(summary.records).toBe(2);
    expect(trafficSummaryText(summary)).toContain("2 more landed in the block and flew their records (not controllable, named below)");
    expect(trafficSummaryText(summarizeTrafficResult(rows.slice(0, 2), {}, true, { "N1_05L_a_5": stayed["N1_05L_a_5"] })))
      .toContain("1 more landed in the block and flew its record (not controllable, named below)");
    expect(trafficSummaryText(summarizeTrafficResult(rows.slice(0, 2), {}, true, {}))).not.toContain("flew");
    expect(summarizeTrafficResult(rows.slice(0, 1), {}, false, {}).records).toBeNull();                       // an M1 job
  });

  it("names each arrival that stayed its record: its callsign (else the first field of its key), its type and why", () => {
    expect(stayedRecordLines(stayed)).toEqual([
      { flightKey: "N1_05L_a_5", callsign: "N1", type: "C172" },
      { flightKey: "N2_05L_a_6", callsign: "N2", type: null },
    ]);
    expect(stayedRecordLines({})).toEqual([]);
    expect(STAYED_RECORD_REASON).toBe("not controllable (no aircraft dynamics model), flew its record");
  });

  it("has no delays and no final check for an M1 window", () => {
    const summary = summarizeTrafficResult(rows.slice(0, 1), { flown_aircraft_with_a_loss_left_after_the_block: left }, false, {});
    expect(summary.delays).toBeNull();
    expect(summary.lossesLeft).toBeNull();
    expect(trafficSummaryText(summary)).toBe("1 aircraft controlled: 1 separated at the first solve (no re-solve)");
  });

  it("shows no delay line when no flown aircraft has a slot and no final check when the job reports none", () => {
    const summary = summarizeTrafficResult(rows.slice(3), undefined, true, {});       // only the aircraft that was not flown
    expect(summary.delays).toBeNull();
    expect(summary.lossesLeft).toBeNull();
  });
});

describe("a job's inputs in words", () => {
  it("names a block by its UTC day and its start and end", () => {
    expect(blockSpan("2026-05-21T18:00:00Z", 900)).toBe("2026-05-21 18:00–18:15 UTC");
    expect(blockJobLabel("2026-05-21T18:00:00Z", 900)).toBe("Block 2026-05-21 18:00–18:15 UTC");
    expect(blockJobLabel("2026-05-21T17:45:00Z", 3600)).toBe("Block 2026-05-21 17:45–18:45 UTC");
  });

  it("names the second day of a block that runs past midnight", () => {
    expect(blockJobLabel("2026-05-21T23:45:00Z", 1800)).toBe("Block 2026-05-21 23:45–2026-05-22 00:15 UTC");
  });

  it("names a flight without a callsign by the first field of its key", () => {
    expect(flightJobLabel({ flightKey: "N123AB_05L_a00001_20260529T000300Z", callsign: null, runway: "05L", type: null,
      entryUtc: "2026-05-29T00:00:00Z", landingUtc: "2026-05-29T00:03:00Z" })).toBe("N123AB, 2026-05-29");
  });

  it("names a flight by its callsign and the day it lands", () => {
    expect(flightJobLabel({ flightKey: "FFL1206_05L_a_20260716T000000Z", callsign: "FFL1206", runway: "05L", type: "A320",
      entryUtc: "2026-07-16T00:00:00Z", landingUtc: "2026-07-16T00:03:00Z" })).toBe("FFL1206, 2026-07-16");
  });
});

describe("the record's losses from the census", () => {
  const catalog = catalogMirror as TrafficScenarioCatalog;
  const [lightest, swa, nameless] = catalog.m1;           // 40, 28, 6 loss instants in the census

  it("takes an aircraft's record losses from the census list: its row, else none — unless the census judged only part", () => {
    expect(recordLosses(catalog, lightest.flightKey)).toBe(40);
    expect(recordLosses(catalog, swa.flightKey)).toBe(28);
    expect(recordLosses(catalog, nameless.flightKey)).toBe(6);
    expect(recordLosses(catalog, "UAL1_05R_x_1")).toBe(0);                                      // judged, no loss
    expect(recordLosses({ ...catalog, config: { ...catalog.config, limit: 300 } }, "UAL1_05R_x_1")).toBeNull();
  });
});

describe("the job's wall time and its solver settings", () => {
  it("shows the running clock as minutes and seconds, the minutes past 59 for a long job", () => {
    expect([0, 5, 65, 3599, 4000].map(formatElapsed)).toEqual(["00:00", "00:05", "01:05", "59:59", "66:40"]);
  });

  it("shows a stage's time as m:ss and the total with its stages, in the order they happened", () => {
    expect([0, 45, 190, 481.6, 741, 3725].map(formatMinutesSeconds)).toEqual(["0:00", "0:45", "3:10", "8:02", "12:21", "62:05"]);
    expect(timingText({ totalS: 741.2, phases: {
      "reading traffic": 45, "earliest arrivals": 190, schedule: 0.6, optimizing: 481.6, evaluation: 20, "building the scene": 3 } })).toBe(
      "Total 12:21 — reading traffic 0:45 · earliest arrivals 3:10 · schedule 0:01 · optimizing 8:02 · evaluation 0:20 · building the scene 0:03");
    expect(timingText({ totalS: 20, phases: { "reading traffic": 5, optimizing: 15 } })).toBe("Total 0:20 — reading traffic 0:05 · optimizing 0:15");
  });

  it("says the settings in one sentence and keeps the whole list for its title", () => {
    expect(SOLVER_SENTENCE).toBe("Same settings as the batch experiments");
    expect(solverTitle()).toBe("max duration 2000 s · rollout step 0.5 s · IPOPT cap 3000 · check step 1 s · row window 60 s · " +
      "margin 1 % · re-solve rounds 5");
  });
});

describe("which aircraft carry a block's recorded losses", () => {
  const catalog = catalogMirror as TrafficScenarioCatalog;
  const flown = { SWA: change(), DAL: change(), NO: change({ ...NOT_FLOWN }) };
  const perAircraft: Record<string, TrafficAircraftChange> = {
    [catalog.m1[1].flightKey]: flown.SWA,          // SWA3131, 28 s, 2026-05-01 00:06
    [catalog.m1[2].flightKey]: flown.NO,           // N123AB, 6 s, 00:21: not flown
  };

  it("lists the arrivals of the block that have a loss in their record, most seconds first, flown or not", () => {
    const shares = blockLossShares(catalog, "2026-05-01T00:00:00Z", 1800, perAircraft);
    expect(shares.map((s) => [s.callsign, s.seconds, s.flown])).toEqual([["SWA3131", 28, true], ["N123AB", 6, false]]);
    // the sum is the block's own seconds in loss: the catalog's block row adds up the same arrivals
    const block = catalog.m2["1800"].find((b) => b.startUtc === "2026-05-01T00:00:00Z")!;
    expect(shares.reduce((sum, s) => sum + s.seconds, 0)).toBe(block.lossInstants);
  });

  it("holds the arrivals that land in [start, start + length) and no other", () => {
    expect(blockLossShares(catalog, "2026-05-01T00:00:00Z", 900, perAircraft).map((s) => s.callsign)).toEqual(["SWA3131"]);
    expect(blockLossShares(catalog, "2026-05-01T00:15:00Z", 900, perAircraft).map((s) => s.callsign)).toEqual(["N123AB"]);
    expect(blockLossShares(catalog, "2026-05-02T09:00:00Z", 1800, perAircraft)).toEqual([]);
  });

  it("says them in a line the block's number reconciles with, naming those not flown or not in the job", () => {
    const shares = blockLossShares(catalog, "2026-05-01T00:00:00Z", 1800, {});                      // the job says nothing of them
    expect(blockLossesText(shares)).toBe("The block's 34 seconds in loss in the records: SWA3131 28 (not in this job) · N123AB 6 (not in this job)");
    expect(blockLossesText(blockLossShares(catalog, "2026-05-01T00:00:00Z", 1800, perAircraft))).toBe(
      "The block's 34 seconds in loss in the records: SWA3131 28 · N123AB 6 (not flown)");
    expect(blockLossesText([])).toBe("No arrival of the block has a loss in its record.");
  });

  it("counts seconds, not check instants, when the census checked less often than every second", () => {
    const twoSeconds = { ...catalog, config: { ...catalog.config, stepS: 2 } };
    expect(blockLossShares(twoSeconds, "2026-05-01T00:00:00Z", 900, perAircraft)[0].seconds).toBe(56);
  });
});
