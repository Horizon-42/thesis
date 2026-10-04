/**
 * The live executor's answer (`trainingAutopilot.ts`) on the two answers the Python code wrote (`stageA.ts`): a stopped
 * heading segment and the last heading word flown to its outcome. The answer is put on the flight's clock and refused by name
 * unless it is the segment of the word the bar shows.
 */
import { describe, expect, it } from "vitest";
import {
  autopilotColour,
  autopilotFlownAt,
  autopilotOnScreen,
  autopilotPlaybackS,
  autopilotPlaybackSpeedup,
  autopilotWord,
  nextPick,
  parseTrainingAutopilot,
  requestTrainingAutopilot,
  trainingAutopilotRequest,
  TRAINING_AUTOPILOT_SCHEMA,
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotView,
} from "../trainingAutopilot";
import { trainingReadingOf } from "../trainingSample";
import { TRAINING_AUTOPILOT_COLOR, TRAINING_FAILURE_COLOR } from "../../utils/trainingWordColors";
import { requestOf, stageAAnswers, stageASelection } from "./stageA";

const selection = stageASelection();

describe("the answer", () => {
  it("reads a segment stopped at the next word: on the flight's clock, with the commands one fewer", () => {
    const [raw] = stageAAnswers();
    const parsed = parseTrainingAutopilot(raw, requestOf(raw), selection);
    if (!parsed.ok) throw new Error(parsed.problem);
    const { segment, track, crossing, stored } = parsed.value;
    expect(segment).toMatchObject({ column: "heading", row: 52, word: 35, correction: false, startCycle: 104, stopCycle: 112,
      end: TRAINING_AUTOPILOT_SEGMENT_END, endCycle: 112 });
    expect(crossing).toBeNull();
    expect(stored.horizontalM).toBe(0);
    // cycle c counts 1 s from the first predicted step: flight time is the closed loop's start plus c
    const closed = trainingReadingOf(selection.flight, selection.vocabulary.stepS, 2).closed!;
    expect(track.tS[0]).toBe(closed.startS + 104);
    expect(track.tS[track.tS.length - 1]).toBe(closed.startS + 112);
    expect(track.thrustFraction).toHaveLength(track.tS.length - 1);
    expect(track.attitude.headingDeg).toHaveLength(track.tS.length);
    // heights: the live answer gives the ellipsoid height directly
    expect(track.altitudeHaeM[0] - track.altitudeMslM[0]).toBeCloseTo(selection.flight.haeMinusMslM, 1);
    expect(autopilotColour(parsed.value)).toBe(TRAINING_AUTOPILOT_COLOR);
  });

  it("reads the last word flown to its outcome: the crossing and the DA check come back with it", () => {
    const [, raw] = stageAAnswers();
    const parsed = parseTrainingAutopilot(raw, requestOf(raw), selection);
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.segment).toMatchObject({ row: 126, correction: true, stopCycle: null, end: "unstable_at_minimums" });
    expect(parsed.value.crossing!.decision!.passed).toBe(false);
    expect(parsed.value.crossing!.decision!.verticalOk).toBe(false);
    expect(autopilotColour(parsed.value)).toBe(TRAINING_FAILURE_COLOR);
  });

  it("refuses another schema by name", () => {
    const [raw] = stageAAnswers();
    const parsed = parseTrainingAutopilot({ ...raw, schema: "aeroviz-autopilot-segment-v8" }, requestOf(raw), selection);
    expect(parsed).toEqual({
      ok: false, problem: `autopilot answer.schema is "aeroviz-autopilot-segment-v8", not one of ${TRAINING_AUTOPILOT_SCHEMA}`,
    });
  });

  it("refuses an answer for another flight, interval, word or correction mark than the one asked", () => {
    const [raw] = stageAAnswers();
    const request = requestOf(raw);
    expect(parseTrainingAutopilot({ ...raw, flightKey: "other" }, request, selection).ok).toBe(false);
    expect(parseTrainingAutopilot({ ...raw, rowIntervalS: 4 }, request, selection).ok).toBe(false);
    const row = parseTrainingAutopilot(raw, { ...request, row: 53 }, selection);
    expect(row.ok).toBe(false);
    if (!row.ok) expect(row.problem).toContain("Δ row 53 was asked for");
    const word = parseTrainingAutopilot({ ...raw, segment: { ...raw.segment, word: 34 } }, request, selection);
    expect(word.ok).toBe(false);
    if (!word.ok) expect(word.problem).toContain("another sentence of this flight");
    const mark = parseTrainingAutopilot({ ...raw, segment: { ...raw.segment, correction: true } }, request, selection);
    expect(mark.ok).toBe(false);
    const start = parseTrainingAutopilot({ ...raw, segment: { ...raw.segment, startCycle: 100 } }, request, selection);
    expect(start.ok).toBe(false);
  });

  it("refuses a track that does not start at the word, and a crossing for a stopped flight", () => {
    const [raw, final] = stageAAnswers();
    const track = { ...raw.track, cycle: raw.track.cycle.map((c: number) => c + 1) };
    expect(parseTrainingAutopilot({ ...raw, track }, requestOf(raw), selection).ok).toBe(false);
    const crossing = parseTrainingAutopilot({ ...raw, crossing: final.crossing }, requestOf(raw), selection);
    expect(crossing.ok).toBe(false);
    if (!crossing.ok) expect(crossing.problem).toContain("stopped at its segment's end");
    const outcome = parseTrainingAutopilot({ ...raw, segment: { ...raw.segment, end: "went_around" } }, requestOf(raw), selection);
    expect(outcome.ok).toBe(false);
  });
});

describe("the pick and the view", () => {
  it("names a new attempt when the same word is picked again", () => {
    const first = nextPick(null, 2, "heading", 52);
    expect(first).toEqual({ rowIntervalS: 2, column: "heading", row: 52, attempt: 0 });
    expect(nextPick(first, 2, "heading", 52).attempt).toBe(1);
    expect(nextPick(first, 4, "heading", 52).attempt).toBe(0);
    expect(trainingAutopilotRequest(selection, first)).toEqual({
      airport: "KXXX", setId: "fixture_set", flightKey: "KXXX:test_fixture", rowIntervalS: 2, column: "heading", row: 52,
    });
  });

  it("is drawn only over the flight and the Δ it was flown for, and names its word", () => {
    const [raw] = stageAAnswers();
    const request = requestOf(raw);
    const view: TrainingAutopilotView = { status: "flying", request };
    expect(autopilotOnScreen(view, selection, 2)).toBe(view);
    expect(autopilotOnScreen(view, selection, 4)).toBeNull();
    expect(autopilotOnScreen(view, selection, null)).toBeNull();
    expect(autopilotOnScreen(view, { ...selection, setId: "other" }, 2)).toBeNull();
    expect(autopilotWord(request, selection)).toMatch(/^heading [+−]?\d+°$/);
  });

  it("flies a segment out at least 8 times faster than real time, no longer than 20 s", () => {
    expect(autopilotPlaybackSpeedup(40)).toBe(8);
    expect(autopilotPlaybackSpeedup(800)).toBe(40);
    const [raw] = stageAAnswers();
    const parsed = parseTrainingAutopilot(raw, requestOf(raw), selection);
    if (!parsed.ok) throw new Error(parsed.problem);
    const { track } = parsed.value;
    expect(autopilotPlaybackS(track, 1000, 1000)).toBe(0);
    // 8 s simulated at the least speed-up of 8: half a second of the page's time is 4 s flown, and 2 s is all of it
    expect(autopilotPlaybackS(track, 1000, 1000 + 500)).toBe(4);
    expect(autopilotPlaybackS(track, 1000, 1000 + 2000)).toBe(8);
    expect(autopilotFlownAt(track, 0)).toEqual({ index: 0, fraction: 0 });
    expect(autopilotFlownAt(track, 1000).index).toBe(track.tS.length - 1);
    expect(autopilotFlownAt(track, 2.5)).toEqual({ index: 2, fraction: 0.5 });
  });
});

describe("the request", () => {
  it("posts the page's id and a rising number, and names the backend's refusal", async () => {
    const bodies: Array<Record<string, unknown>> = [];
    const answers = [
      { ok: true, status: 200, text: async () => JSON.stringify({ ok: true }) },
      { ok: false, status: 409, text: async () => JSON.stringify({ ok: false, error: "superseded" }) },
    ];
    const fetchMock = async (_url: string, init: RequestInit) => {
      bodies.push(JSON.parse(init.body as string));
      return answers.shift()!;
    };
    const original = globalThis.fetch;
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    try {
      const request = requestOf(stageAAnswers()[0]);
      await requestTrainingAutopilot("http://backend.test/", request);
      await expect(requestTrainingAutopilot("http://backend.test", request)).rejects.toThrow("the backend refused (409): superseded");
      expect(bodies[0]).toMatchObject({ ...request, seq: (bodies[1].seq as number) - 1 });
      expect(typeof bodies[0].clientId).toBe("string");
      expect(bodies[0].column).toBe("heading");
    } finally {
      globalThis.fetch = original;
    }
  });
});
