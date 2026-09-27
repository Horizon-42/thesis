/**
 * The live executor's reader: an answer is accepted only for the segment on screen — the same flight, the run's own end,
 * the vocabulary of the set, and the very words the sentence bar shows for it — and refused whole, by name, otherwise. A
 * model's word is asked with its sentence and answered as its: no observed time, no offset from the observed aircraft.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseTrainingSample, TRAINING_UNCHANGED, type TrainingSample } from "../trainingSample";
import {
  autopilotHasLine,
  autopilotOnScreen,
  autopilotRunAndTail,
  autopilotSampleGap,
  parseTrainingAutopilot,
  requestTrainingAutopilot,
  segmentWords,
  trainingAutopilotRequest,
  TRAINING_AUTOPILOT_CLIENT_ID,
  TRAINING_AUTOPILOT_PATH,
  TRAINING_AUTOPILOT_SCHEMA,
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotRequest,
} from "../trainingAutopilot";
import { VECTORED_KEY, WORD, mockSample } from "./trainingSample.fixture";
import {
  failedAnswer, mockAutopilotAnswer, mockAutopilotRequest, mockModelAutopilotRequest, mockSelection, onSampleLine,
} from "./trainingAutopilot.fixture";
import { BASE_MODEL_ID, MOCK_GENERATION_FIRST_ROW, mockGenerationViews } from "./trainingOverlays.fixture";
import { TRAINING_BELOW_GLIDEPATH, TRAINING_PROCEDURE_ALTITUDES, type TrainingGeneratedSentence } from "../trainingOverlays";

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

const HEADING_225 = { column: "heading" as const, row: 8 };

function refusal(change: (raw: any) => void, request: { column: TrainingAutopilotRequest["column"]; row: number } = HEADING_225): string {
  const set = sample();
  const asked = mockAutopilotRequest(set, VECTORED_KEY, request.column, request.row);
  const raw = mockAutopilotAnswer(set, asked);
  change(raw);
  const result = parseTrainingAutopilot(raw, asked, mockSelection(set, asked));
  if (result.ok) throw new Error("the change was accepted");
  return result.problem;
}

/** The base model's first sample of the vectored flight, and its heading 225° said at step 12, asked for. */
function modelAsk(set: TrainingSample, sampleIndex = 0) {
  const sentence: TrainingGeneratedSentence = mockGenerationViews(set)[0].flight.samples[sampleIndex];
  return { sentence, request: mockModelAutopilotRequest(set, VECTORED_KEY, "heading", 12, BASE_MODEL_ID, sentence) };
}

describe("the words a segment is told", () => {
  it("are the six in force at its first step, then every word said before its end, by step then column", () => {
    const flight = sample().flights[0];
    const words = segmentWords(mockAutopilotRequest(sample(), flight.flightKey, "heading", 10), flight, 60);
    expect(words.slice(0, 6).map((word) => word.value)).toEqual([
      WORD.runway09, WORD.notCleared, WORD.heading180, WORD.altitude1110, WORD.level, WORD.speed110,
    ]);
    expect(words.every((word, index) => index < 6 ? word.row === 10 : word.row > 10)).toBe(true);
    expect(words.slice(6).map((word) => [word.row, word.column])).toEqual([[20, 1], [20, 3], [20, 4], [30, 5]]);
    expect(words.some((word) => word.value === TRAINING_UNCHANGED)).toBe(false);
  });

  it("are a model's whole sentence from its first step to the segment's stop — it is flown again from there", () => {
    const set = sample();
    const { request } = modelAsk(set);
    const words = segmentWords(request, set.flights[0], 18);
    expect(words[0].row).toBe(MOCK_GENERATION_FIRST_ROW);
    expect(words.map((word) => [word.row, word.column])).toEqual([
      [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [12, 2], [16, 2],
    ]);
  });
});

describe("a model's word", () => {
  it("is asked with its sentence — refused without it, or with the truth's pick", () => {
    const set = sample();
    const { sentence } = modelAsk(set);
    const selection = mockSelection(set, mockAutopilotRequest(set, VECTORED_KEY, "heading", 8));
    const source = { overlayId: BASE_MODEL_ID, sample: 0 };
    const masks = [{ name: TRAINING_PROCEDURE_ALTITUDES, dataSha256: "d".repeat(64) }];
    const asked = trainingAutopilotRequest(selection, { source, column: "heading", row: 12, attempt: 0 },
      { sentence, procedureMasks: masks });
    expect(asked.sentence).toMatchObject({ overlayId: BASE_MODEL_ID, sample: 0, firstRow: MOCK_GENERATION_FIRST_ROW, rows: 56 });
    // the masks it was spoken under go with it: the backend cuts the flight where the glidepath lower edge stopped it
    expect(asked.sentence!.procedureMasks).toEqual(masks);
    expect(asked.sentence!.events).toHaveLength(sentence.events.length);
    expect(() => trainingAutopilotRequest(selection, { source, column: "heading", row: 12, attempt: 0 }, null)).toThrow(/with its sentence/);
    expect(() => trainingAutopilotRequest(selection, { source: null, column: "heading", row: 8, attempt: 0 },
      { sentence, procedureMasks: [] })).toThrow();
  });

  it("may end below the glidepath, where its sample was stopped — spoken under the procedure's altitudes only", () => {
    const set = sample();
    const { request: bare } = modelAsk(set);
    const request = { ...bare, sentence: { ...bare.sentence!, procedureMasks: [{ name: TRAINING_PROCEDURE_ALTITUDES, dataSha256: "d".repeat(64) }] } };
    const stopped = mockAutopilotAnswer(set, request);
    stopped.end = { ...stopped.end, reason: TRAINING_BELOW_GLIDEPATH, reachedSegmentEnd: false };
    const parsed = parseTrainingAutopilot(stopped, request, mockSelection(set, request));
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.end.reason).toBe(TRAINING_BELOW_GLIDEPATH);
    // the same answer to a sentence spoken under none
    const unmasked = parseTrainingAutopilot(stopped, bare, mockSelection(set, bare));
    expect(unmasked.ok ? "" : unmasked.problem).toMatch("spoken without the procedure's altitudes");
    // a stop carries no crossing
    const crossed = parseTrainingAutopilot({ ...stopped, end: { ...stopped.end, crossing: { crossM: 1, heightM: 15, atS: 100, runway: 0 } } },
      request, mockSelection(set, request));
    expect(crossed.ok ? "" : crossed.problem).toMatch("ends below_glidepath and carries a crossing");
    const truth = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const refused = mockAutopilotAnswer(set, truth);
    refused.end = { ...refused.end, reason: TRAINING_BELOW_GLIDEPATH, reachedSegmentEnd: false, offsetFromObserved: null };
    const read = parseTrainingAutopilot(refused, truth, mockSelection(set, truth));
    expect(read.ok ? "" : read.problem).toMatch("the truth's flight is never stopped below the glidepath");
  });

  it("reads as the model's: its source echoed, no observed time, no offset from the observed aircraft", () => {
    const set = sample();
    const { request } = modelAsk(set);
    const parsed = parseTrainingAutopilot(mockAutopilotAnswer(set, request), request, mockSelection(set, request));
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(parsed.value.source).toEqual({ kind: "model", overlayId: BASE_MODEL_ID, sample: 0, firstRow: MOCK_GENERATION_FIRST_ROW });
    // heading 225° said at 12, the next heading word at 16, flown a lead past it
    expect(parsed.value.segment).toMatchObject({ row: 12, endRow: 16, stopRow: 18, observedS: null });
    expect(parsed.value.end.offsetFromObserved).toBeNull();
    expect(parsed.value.word.heading!.targetOnTrackDeg).toBe(225);
  });

  it("refuses another sentence flown, an observed time, and an offset from the observed aircraft", () => {
    const set = sample();
    const { request } = modelAsk(set);
    const refused = (change: (raw: any) => void) => {
      const raw = mockAutopilotAnswer(set, request);
      change(raw);
      const result = parseTrainingAutopilot(raw, request, mockSelection(set, request));
      if (result.ok) throw new Error("the change was accepted");
      return result.problem;
    };
    expect(refused((raw) => { raw.source = { kind: "truth" }; })).toMatch(/flew the truth, but generation_base #0 was asked for/);
    expect(refused((raw) => { raw.source.sample = 1; })).toMatch(/flew generation_base #1, but generation_base #0 was asked for/);
    expect(refused((raw) => { raw.segment.observedS = 12; })).toMatch(/gives an observed time for a model's word/);
    expect(refused((raw) => { raw.end.offsetFromObserved = { horizontalM: 1, aboveM: 0, groundSpeedMps: 0 }; }))
      .toMatch(/offsetFromObserved is given exactly when the truth's flight ended at its segment's end/);
    // and the truth's word answered as a model's
    const truth = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const raw = mockAutopilotAnswer(set, truth);
    raw.source = { kind: "model", overlayId: BASE_MODEL_ID, sample: 0, firstRow: 4 };
    const result = parseTrainingAutopilot(raw, truth, mockSelection(set, truth));
    expect(result.ok ? "" : result.problem).toMatch(/flew generation_base #0, but the truth was asked for/);
  });

  it("is drawn only over the sentence it flew", () => {
    const set = sample();
    const { request } = modelAsk(set);
    const selection = mockSelection(set, request);
    const view = { status: "flying" as const, request };
    expect(autopilotOnScreen(view, selection, { overlayId: BASE_MODEL_ID, sample: 0 })).toBe(view);
    expect(autopilotOnScreen(view, selection, { overlayId: BASE_MODEL_ID, sample: 1 })).toBeNull();
    expect(autopilotOnScreen(view, selection, null)).toBeNull();
  });

  it("says how closely the live flight lands on the sample it re-flies, at the times both hold a point", () => {
    const set = sample();
    const { request, sentence } = modelAsk(set);
    const read = (raw: Record<string, any>) => {
      const parsed = parseTrainingAutopilot(raw, request, mockSelection(set, request));
      if (!parsed.ok) throw new Error(parsed.problem);
      return parsed.value;
    };
    // flown 24–36 s every second; the sample holds a point every 2 s: seven shared
    expect(autopilotSampleGap(read(onSampleLine(mockAutopilotAnswer(set, request))), sentence)).toEqual({ gapM: 0, points: 7 });
    const moved = onSampleLine(mockAutopilotAnswer(set, request));
    moved.track.altitudeM[4] += 3;                                     // 28 s, a shared time
    moved.track.altitudeM[5] += 50;                                    // 29 s, the sample has no point there
    expect(autopilotSampleGap(read(moved), sentence)!.gapM).toBeCloseTo(3, 9);
    expect(autopilotSampleGap(read(onSampleLine(mockAutopilotAnswer(set, request))),
      { ...sentence, track: { ...sentence.track, tS: sentence.track.tS.map((at) => at + 0.5) } })).toBeNull();
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
    // ... heard at 20 s: the track's point 4, where the tail (the lead flown into it) begins, shared by both parts
    expect([segment.segment.nextWordHeardS, segment.tailFrom]).toEqual([20, 4]);
    expect(autopilotRunAndTail(segment)).toEqual({ run: [0, 1, 2, 3, 4], tail: [4, 5, 6, 7, 8] });
  });

  it("has no tail where the flight stops before the next word of its column is told", () => {
    const set = sample();
    for (const [column, row] of [["runway", 0], ["altitude", 0]] as const) {
      const request = mockAutopilotRequest(set, VECTORED_KEY, column, row);
      const parsed = parseTrainingAutopilot(mockAutopilotAnswer(set, request), request, mockSelection(set, request));
      if (!parsed.ok) throw new Error(parsed.problem);
      expect([parsed.value.segment.nextWordHeardS, parsed.value.tailFrom]).toEqual([null, null]);
      expect(autopilotRunAndTail(parsed.value).tail).toEqual([]);
    }
  });

  it("refuses a next word heard off the flown track, or by a segment that stops before it is told", () => {
    expect(refusal((raw) => { raw.segment.nextWordHeardS = 20.5; })).toMatch(/heard at 20\.5 s, not a point of the flown track/);
    // nothing flown past it, or heard where the word itself was
    expect(refusal((raw) => { raw.segment.nextWordHeardS = 24; })).toMatch(/heard at 24 s, the flown track's last point/);
    expect(refusal((raw) => { raw.segment.nextWordHeardS = 16; })).toMatch(/heard at 16 s, the flown track's first point/);
    expect(refusal((raw) => { raw.segment.nextWordHeardS = 0; }, { column: "altitude", row: 0 }))
      .toMatch(/heard the next altitude word at 0 s, but its segment stops before that word is told/);
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

  it("reads a dynamics failure: the rows its judge read before it failed, or one state and nothing judged", () => {
    const set = sample();
    const request = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const read = (states: number) => {
      const parsed = parseTrainingAutopilot(failedAnswer(mockAutopilotAnswer(set, request), states), request, mockSelection(set, request));
      if (!parsed.ok) throw new Error(parsed.problem);
      return parsed.value;
    };
    // failed in its 7th cycle: six cycles kept, the judged steps 0–3 (points 0, 2, 4, 6), the band to step 12
    const late = read(7);
    expect(late.end).toMatchObject({ reason: "dynamics_failure", reachedSegmentEnd: false, offsetFromObserved: null });
    expect(late.track.tS).toHaveLength(7);
    expect(late.judgedTrackDeg).toHaveLength(4);
    expect(late.word.heading).toMatchObject({ firstRow: 10, stopRow: 12, inside: [true, true] });
    expect(autopilotHasLine(late)).toBe(true);
    // failed in its first cycle: the state it started from, nothing to draw, the word not judged
    const first = read(1);
    expect(first.track.tS).toEqual([16]);
    expect(first.track.bankRightDeg).toEqual([]);
    expect(first.word).toMatchObject({ status: "not judged", heading: null });
    expect(autopilotHasLine(first)).toBe(false);
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

  it("posts the request to the backend's segment path, named and numbered by this page", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ ok: true }) }));
    vi.stubGlobal("fetch", fetchMock);
    const request = mockAutopilotRequest(sample(), VECTORED_KEY, "heading", 8);
    await expect(requestTrainingAutopilot("http://backend.test/", request)).resolves.toEqual({ ok: true });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`http://backend.test${TRAINING_AUTOPILOT_PATH}`);
    expect(init.method).toBe("POST");
    const sent = JSON.parse(init.body as string);
    expect(sent).toEqual({ ...request, clientId: TRAINING_AUTOPILOT_CLIENT_ID, seq: sent.seq });
    expect(TRAINING_AUTOPILOT_CLIENT_ID).toMatch(/^[0-9a-f]{32}$/);
    // numbered in the page's own order: the backend flies the highest
    await requestTrainingAutopilot("http://backend.test/", request);
    const [, next] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
    expect(JSON.parse(next.body as string).seq).toBe(sent.seq + 1);
  });

  it("names the page where randomUUID is missing — a page opened over plain http from another machine", async () => {
    const real = globalThis.crypto;
    vi.stubGlobal("crypto", { getRandomValues: real.getRandomValues.bind(real) });
    vi.resetModules();
    const reloaded = await import("../trainingAutopilot");
    expect(reloaded.TRAINING_AUTOPILOT_CLIENT_ID).toMatch(/^[0-9a-f]{32}$/);
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
