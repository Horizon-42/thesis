/**
 * The live executor's reader: an answer is accepted only for the segment on screen — the same flight, the run's own end,
 * the vocabulary of the set, and the very words the sentence bar shows for it — and refused whole, by name, otherwise.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseTrainingSample, TRAINING_UNCHANGED, type TrainingSample } from "../trainingSample";
import {
  parseTrainingAutopilot,
  requestTrainingAutopilot,
  segmentWords,
  TRAINING_AUTOPILOT_PATH,
  TRAINING_AUTOPILOT_SCHEMA,
  TRAINING_AUTOPILOT_SEGMENT_END,
} from "../trainingAutopilot";
import { VECTORED_KEY, WORD, mockSample } from "./trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest, mockSelection } from "./trainingAutopilot.fixture";

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

const HEADING_225 = { column: "heading" as const, row: 8 };

function refusal(change: (raw: any) => void, request = HEADING_225): string {
  const set = sample();
  const asked = mockAutopilotRequest(set, VECTORED_KEY, request.column, request.row);
  const raw = mockAutopilotAnswer(set, asked);
  change(raw);
  const result = parseTrainingAutopilot(raw, asked, mockSelection(set, asked));
  if (result.ok) throw new Error("the change was accepted");
  return result.problem;
}

describe("the words a segment is told", () => {
  it("are the six in force at its first step, then every word said before its end, by step then column", () => {
    const flight = sample().flights[0];
    const words = segmentWords(flight, 10, 60);
    expect(words.slice(0, 6).map((word) => word.value)).toEqual([
      WORD.runway09, WORD.notCleared, WORD.heading180, WORD.altitude1110, WORD.level, WORD.speed110,
    ]);
    expect(words.every((word, index) => index < 6 ? word.row === 10 : word.row > 10)).toBe(true);
    expect(words.slice(6).map((word) => [word.row, word.column])).toEqual([[20, 1], [20, 3], [20, 4], [30, 5]]);
    expect(words.some((word) => word.value === TRAINING_UNCHANGED)).toBe(false);
  });
});

describe("parseTrainingAutopilot", () => {
  it("reads a heading word's segment, bound to the flight on screen", () => {
    const set = sample();
    const request = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const parsed = parseTrainingAutopilot(mockAutopilotAnswer(set, request), request, mockSelection(set, request));
    if (!parsed.ok) throw new Error(parsed.problem);
    const segment = parsed.value;
    // a heading word is flown a lead past the next heading word, where its band ends
    expect(segment.segment).toMatchObject({ column: "heading", row: 8, endRow: 10, stopRow: 12, toLanding: false });
    expect(segment.end.reason).toBe(TRAINING_AUTOPILOT_SEGMENT_END);
    expect(segment.word.status).toBe("inside");
    expect(segment.word.heading).toMatchObject({ firstRow: 10, stopRow: 12, inside: [true, true] });
    expect(segment.track.tS[0]).toBe(16);
    expect(segment.judgedTrackDeg).toHaveLength(5);
    // the 2 s step is two of the 1 s cycles: the judged step k is the track's point 2k
    expect(segment.executor.stepCycles).toBe(2);
    // the next heading word, said at step 10, is among the words told
    expect(segment.segment.told.some((word) => word.row === 10 && word.column === 2)).toBe(true);
  });

  it("reads a column's last word, flown to its outcome", () => {
    const set = sample();
    const request = mockAutopilotRequest(set, VECTORED_KEY, "runway", 0);
    const parsed = parseTrainingAutopilot(mockAutopilotAnswer(set, request), request, mockSelection(set, request));
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.segment).toMatchObject({ toLanding: true, endRow: 60, stopRow: 60 });
    expect(parsed.value.end).toMatchObject({ reason: "landed", reachedSegmentEnd: null });
    expect(parsed.value.word.status).toBe("no check");
  });

  it("refuses another schema by name", () => {
    expect(refusal((raw) => { raw.schema = "aeroviz-autopilot-segment-v0"; })).toMatch(
      new RegExp(`schema is "aeroviz-autopilot-segment-v0", expected "${TRAINING_AUTOPILOT_SCHEMA}"`));
  });

  it("refuses an answer for another flight, vocabulary or segment", () => {
    expect(refusal((raw) => { raw.datasetId = "KXXX:other"; })).toMatch(/flew KXXX .* KXXX:other/);
    expect(refusal((raw) => { raw.vocabularySpecSha256 = "0".repeat(64); })).toMatch(/flew vocabulary 000000000000/);
    expect(refusal((raw) => { raw.segment.row = 10; })).toMatch(/is heading from step 10, but heading from step 8/);
    expect(refusal((raw) => { raw.segment.endRow = 12; })).toMatch(/ends at step 12, but the word on screen is in force to step 10/);
    expect(refusal((raw) => { raw.segment.stopRow = 10; })).toMatch(/stops at step 10, but this word's envelope ends at step 12/);
  });

  it("refuses words told that are not the sentence's for the segment", () => {
    expect(refusal((raw) => { raw.segment.told[2].value = WORD.heading270; })).toMatch(
      /told the executor 7 words that are not the 7 the sentence shows/);
    expect(refusal((raw) => { raw.segment.told.push({ row: 9, column: 3, value: WORD.land }); })).toMatch(/told the executor 8 words/);
  });

  it("refuses a verdict its checks do not give, and a band that is not the word's", () => {
    expect(refusal((raw) => { raw.word.status = "outside"; })).toMatch(/is outside, but its checks say/);
    expect(refusal((raw) => { raw.word.heading.firstRow = 9; raw.word.heading.stopRow = 11; })).toMatch(/firstRow is 9/);
    expect(refusal((raw) => { raw.word.heading = null; })).toMatch(/a heading band comes with the track its judge read/);
    expect(refusal((raw) => { raw.word = { status: "not judged", checks: [], reason: null, heading: null }; raw.judgedTrackDeg = null; }))
      .toMatch(/is not judged and says no reason/);
  });

  it("refuses a heading band past the track its judge read, and a verdict its band's rows do not allow", () => {
    // five judged steps from step 8: its rows end by step 13 — on the executor's own steps, which may run past the
    // sentence's stop (12) when it hears the next heading word late
    expect(refusal((raw) => {
      raw.word.heading.stopRow = 14; raw.word.heading.inside = [1, 1, 1, 1];
      raw.word.checks[0] = { ...raw.word.checks[0], inside: 4, rows: 4 };
    })).toMatch(/stopRow is 14, not in 10…13/);
    const set = sample();
    const request = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const lagging = mockAutopilotAnswer(set, request);
    lagging.word.heading.stopRow = 13;
    lagging.word.heading.inside = [1, 1, 1];
    lagging.word.checks[0] = { ...lagging.word.checks[0], inside: 3, rows: 3 };
    const parsed = parseTrainingAutopilot(lagging, request, mockSelection(set, request));
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.word.heading).toMatchObject({ firstRow: 10, stopRow: 13 });
    // inside with no row judged
    expect(refusal((raw) => {
      raw.word.heading.stopRow = 10; raw.word.heading.inside = [];
      raw.word.checks[0] = { ...raw.word.checks[0], inside: 0, rows: 0 };
    })).toMatch(/is inside with no row judged/);
    // "no check" with rows judged
    expect(refusal((raw) => { raw.word.status = "no check"; raw.word.checks = []; raw.word.reason = "none"; }))
      .toMatch(/is no check, but 2 of its rows were judged/);
  });

  it("refuses a word judged, or a band drawn, on a flown track the gate refused", () => {
    expect(refusal((raw) => { raw.end.refused = "too short"; })).toMatch(/is inside, but the gate refused the flown track \(too short\)/);
    expect(refusal((raw) => {
      raw.end.refused = "too short";
      raw.word = { ...raw.word, status: "not judged", checks: [], reason: "the gate refused it" };
    })).toMatch(/carries a heading band, but the gate refused the flown track/);
    const set = sample();
    const request = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const raw = mockAutopilotAnswer(set, request);
    raw.end.refused = "too short";
    raw.word = { status: "not judged", checks: [], reason: "the gate refused it", heading: null };
    raw.judgedTrackDeg = null;
    const parsed = parseTrainingAutopilot(raw, request, mockSelection(set, request));
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.word.status).toBe("not judged");
  });

  it("refuses an end that does not match the segment", () => {
    expect(refusal((raw) => { raw.end.reason = "somewhere"; })).toMatch(/end\.reason is "somewhere", not one of segment_end, landed/);
    expect(refusal((raw) => { raw.end.offsetFromObserved = null; })).toMatch(/offsetFromObserved is given exactly when/);
    expect(refusal((raw) => { raw.end.reachedSegmentEnd = null; })).toMatch(/says nothing of whether the segment's end was reached/);
  });

  it("refuses a track that does not start at the segment's step or run forward", () => {
    expect(refusal((raw) => { raw.track.tS[0] = 0; })).toMatch(/starts at 0 s, not at step 8 \(16 s\)/);
    expect(refusal((raw) => { raw.track.bankRightDeg.push(0); })).toMatch(/bankRightDeg has 9 values, expected 8/);
    expect(refusal((raw) => { raw.timing.cycles = 3; })).toMatch(/3 cycles flown, but the track holds 9 states/);
    expect(refusal((raw) => { delete raw.timing.flyS; })).toMatch(/timing\.flyS is undefined, not a number/);
    expect(refusal((raw) => { raw.track.tS = []; })).toMatch(/tS is empty/);
  });
});

describe("requestTrainingAutopilot", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("posts the request to the backend's segment path", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ ok: true }) }));
    vi.stubGlobal("fetch", fetchMock);
    const request = mockAutopilotRequest(sample(), VECTORED_KEY, "heading", 8);
    await expect(requestTrainingAutopilot("http://backend.test/", request)).resolves.toEqual({ ok: true });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`http://backend.test${TRAINING_AUTOPILOT_PATH}`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual(request);
  });

  it("names the backend's refusal, an answer that is not JSON, and a backend that did not answer", async () => {
    const request = mockAutopilotRequest(sample(), VECTORED_KEY, "heading", 8);
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 502, text: async () => "<html>Bad Gateway</html>" })));
    await expect(requestTrainingAutopilot("http://backend.test", request)).rejects.toThrow(/HTTP 502 with something that is not JSON: <html>/);
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 400, text: async () => JSON.stringify({ ok: false, error: "0 executor specs" }) })));
    await expect(requestTrainingAutopilot("http://backend.test", request)).rejects.toThrow(/refused \(400\): 0 executor specs/);
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    await expect(requestTrainingAutopilot("http://backend.test", request)).rejects.toThrow(/did not answer .*Failed to fetch/);
  });
});
