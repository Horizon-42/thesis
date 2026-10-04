/**
 * trainingAutopilot.ts
 * --------------------
 * THE EXECUTOR, LIVE: one word's segment of the selected Training flight's closed-loop sentence, flown by the backend at
 * the moment it is asked (`POST /autopilot/segment`, the `aeroviz_backend/autopilot_segment/` package; the answer is
 * written by its `payload.py`).
 *
 * A SEGMENT is the word the user CLICKED, flown every control cycle (1 s) from the cycle it is heard (Δ row × Δ, from the
 * sentence's first predicted step) to the cycle the next word of its column is heard — the column's last word on to the
 * flight's outcome, which the judge reads: the crossing and the DA check come back with it. The flight is flown from the
 * first predicted step with the sentence's earlier words, so the aircraft is where those took it; the answer holds the
 * part from the word on. NOTHING IS PRECOMPUTED: every request is flown again, and the answer says how far it lies from
 * the exported flown states on the 2 s rows both have (`stored`).
 *
 * The request names the word by its Δ row in `closedLoop[Δ].words` (the `row` of that Δ's events) and its column's NAME.
 * The answer is refused unless it is the segment of the word the bar shows: its column, row, word and correction mark are
 * the sentence's, and its first cycle is the word's.
 *
 * Time. The answer counts 1 s cycles from the first predicted step; here its times are put on the flight's clock
 * (`startS` of the closed loop added once), like every other track.
 */

import { readAttitude, type TrainingAttitude } from "./trainingAttitude";
import { attempt, Reader, type Parsed } from "./trainingReader";
import {
  lastStateCycle,
  readCrossing,
  rowAtTime,
  sentenceWordAt,
  trainingBandLabel,
  trainingReadingOf,
  unwrapDegrees,
  TRAINING_COLUMNS,
  TRAINING_OUTCOMES,
  TRAINING_RUNWAY_GO_AROUND,
  type TrainingColumn,
  type TrainingCrossing,
  type TrainingOutcome,
  type TrainingSelection,
} from "./trainingSample";
import { TRAINING_AUTOPILOT_COLOR, TRAINING_FAILURE_COLOR } from "../utils/trainingWordColors";

/** MIRROR of `aeroviz_backend/autopilot_segment/payload.py` `SCHEMA`: the backend's answer; anything else is refused by
 *  name. A name changes with the payload's shape, on both sides, in one change. */
export const TRAINING_AUTOPILOT_SCHEMA = "aeroviz-autopilot-segment-v9";
/** MIRROR of `payload.py` `SEGMENT_END`: the flight reached the point where its word's segment stops. */
export const TRAINING_AUTOPILOT_SEGMENT_END = "segment_end";
export const TRAINING_AUTOPILOT_PATH = "/autopilot/segment";
/** This page, as the backend knows it: a request from the same page supersedes an older one still waiting or flying there
 *  (409). Random per page load. */
export const TRAINING_AUTOPILOT_CLIENT_ID = Array.from(globalThis.crypto.getRandomValues(new Uint8Array(16)),
  (byte) => byte.toString(16).padStart(2, "0")).join("");
/** The page's requests, numbered in the order it makes them (`seq`): the backend flies only the page's highest. */
let requestSeq = 0;

/** What the Training view asks for: the clicked word's segment of the selected flight's closed-loop sentence at Δ. */
export interface TrainingAutopilotRequest {
  airport: string;
  setId: string;
  flightKey: string;
  rowIntervalS: number;
  column: TrainingColumn;
  /** The Δ row the word is said at (`closedLoop[Δ].events[].row`). */
  row: number;
}

/** The flown segment every control cycle, on the flight's clock. */
export interface TrainingAutopilotTrack {
  /** The cycles (1 s) from the first predicted step, ascending by one from the word's. */
  cycle: number[];
  /** Flight time (s). */
  tS: number[];
  eM: number[];
  nM: number[];
  lon: number[];
  lat: number[];
  altitudeMslM: number[];
  altitudeHaeM: number[];
  /** Compass degrees in [0, 360). */
  trackDeg: number[];
  /** `trackDeg` continuous, on the branch of the judged track at its first point: what a chart plots. */
  trackPlotDeg: number[];
  groundSpeedMps: number[];
  verticalRateMps: number[];
  /** The commands each cycle flew: one fewer than the states. */
  thrustFraction: number[];
  /** Positive: banked right (turning right). */
  bankRightDeg: number[];
  loadFactor: number[];
  attitude: TrainingAttitude;
}

export interface TrainingAutopilotSegment {
  airport: string;
  setId: string;
  flightKey: string;
  datasetId: string;
  rowIntervalS: number;
  computedUtc: string;
  /** Wall-clock seconds on the backend: the wait, then the parts that add up to `computeS`. */
  timing: { waitS: number; openS: number; flyS: number; cycles: number; judgeS: number; answerS: number; computeS: number };
  executor: { spec: string; specSha256: string; cycleS: number };
  artefact: string;
  segment: {
    column: TrainingColumn;
    row: number;
    word: number;
    correction: boolean;
    startCycle: number;
    /** The cycle the segment stops before; null: flown on to the outcome. */
    stopCycle: number | null;
    /** `TRAINING_AUTOPILOT_SEGMENT_END`, or the judge's outcome. */
    end: typeof TRAINING_AUTOPILOT_SEGMENT_END | TrainingOutcome;
    endCycle: number;
  };
  track: TrainingAutopilotTrack;
  /** Flown to its outcome: the judge's crossing and DA check; null at a segment's end, or when the flight did not cross. */
  crossing: TrainingCrossing | null;
  /** The live flight against the exported flown states on the 2 s rows both have. */
  stored: { rows: number; horizontalM: number; verticalM: number };
}

/** What the panel publishes for the sentence bar, the read-back window and 3D. */
export type TrainingAutopilotView =
  | { status: "flying"; request: TrainingAutopilotRequest }
  | { status: "failed"; request: TrainingAutopilotRequest; problem: string }
  /** `playedAt`: when the answer arrived, and the 3D scene began flying it out. */
  | { status: "ready"; request: TrainingAutopilotRequest; segment: TrainingAutopilotSegment; playedAt: number };

/** THE WORD THE LIVE EXECUTOR FLIES (`trainingPick`) — of the flight on screen at Δ, which the pick belongs to and is reset
 *  with: set only by a click, never by the cursor. */
export interface TrainingPick {
  rowIntervalS: number;
  column: TrainingColumn;
  row: number;
  attempt: number;
}

/** The pick that flies ``column``'s word said at Δ row ``row`` now: a new attempt when it is the word already picked. */
export function nextPick(current: TrainingPick | null, rowIntervalS: number, column: TrainingColumn, row: number): TrainingPick {
  const same = current !== null && current.rowIntervalS === rowIntervalS && current.column === column && current.row === row;
  return { rowIntervalS, column, row, attempt: same ? current.attempt + 1 : 0 };
}

/** The request for a pick of the flight on screen. */
export function trainingAutopilotRequest(selection: TrainingSelection, pick: TrainingPick): TrainingAutopilotRequest {
  return {
    airport: selection.airport, setId: selection.setId, flightKey: selection.flight.flightKey,
    rowIntervalS: pick.rowIntervalS, column: pick.column, row: pick.row,
  };
}

/** The live executor's view if it is of the flight on screen — its airport, set and flight — and of the reading on screen
 *  (``intervalS``: its Δ; null for the open-loop reading, which the backend does not fly). */
export function autopilotOnScreen(
  view: TrainingAutopilotView | null, selection: TrainingSelection | null, intervalS: number | null,
): TrainingAutopilotView | null {
  if (view === null || selection === null || intervalS === null) return null;
  const { request } = view;
  return request.airport === selection.airport && request.setId === selection.setId
    && request.flightKey === selection.flight.flightKey && request.rowIntervalS === intervalS ? view : null;
}

/** The word a request flies, as the sentence reads it: "heading +15°". */
export function autopilotWord(request: TrainingAutopilotRequest, selection: TrainingSelection): string {
  const reading = trainingReadingOf(selection.flight, selection.vocabulary.stepS, request.rowIntervalS);
  const run = sentenceWordAt(reading, request.column, request.row)!;
  return `${request.column} ${trainingBandLabel(run.event.says)}`;
}

/** The flown segment has a line to draw: two states or more (a dynamics failure in its first cycle keeps one). */
export function autopilotHasLine(segment: TrainingAutopilotSegment): boolean {
  return segment.track.tS.length >= 2;
}

/** The colour a flown segment is drawn in, everywhere: blue, or the failure red when the flight was flown on to an outcome
 *  other than a landing. */
export function autopilotColour(segment: TrainingAutopilotSegment): string {
  const { end } = segment.segment;
  return end === TRAINING_AUTOPILOT_SEGMENT_END || end === "landed" ? TRAINING_AUTOPILOT_COLOR : TRAINING_FAILURE_COLOR;
}

// ── flying it out in 3D ──────────────────────────────────────────────────────

/** How much faster than real time the live executor's aircraft flies its segment out in 3D: at least this ... */
export const AUTOPILOT_PLAYBACK_MIN_SPEEDUP = 8;
/** ... and fast enough that no segment takes longer than this, in seconds (the aircraft's label says the speed-up). */
export const AUTOPILOT_PLAYBACK_MAX_S = 20;

export function autopilotPlaybackSpeedup(flownS: number): number {
  return Math.max(AUTOPILOT_PLAYBACK_MIN_SPEEDUP, flownS / AUTOPILOT_PLAYBACK_MAX_S);
}

/** How far into its segment the live executor's aircraft is at ``nowMs``, in simulated seconds: the fly-out's clock —
 *  shared by the 3D aircraft and the sentence bar's cursor, so they stay together. */
export function autopilotPlaybackS(track: TrainingAutopilotTrack, playedAt: number, nowMs: number): number {
  const flownS = track.tS[track.tS.length - 1] - track.tS[0];
  return Math.min(Math.max(nowMs - playedAt, 0) / 1000 * autopilotPlaybackSpeedup(flownS), flownS);
}

/** Where the live executor is ``flownS`` seconds into its segment: the last point at or before it and the fraction of
 *  the way to the next. */
export function autopilotFlownAt(track: TrainingAutopilotTrack, flownS: number): { index: number; fraction: number } {
  const target = track.tS[0] + Math.max(flownS, 0);
  const last = track.tS.length - 1;
  if (target >= track.tS[last]) return { index: last, fraction: 1 };
  let index = 0;
  while (index + 1 < last && track.tS[index + 1] <= target) index += 1;
  return { index, fraction: (target - track.tS[index]) / (track.tS[index + 1] - track.tS[index]) };
}

/** The aircraft's label at a flown point: the simulated time flown so far of the whole, the playback's speed-up, ground
 *  speed, height and bank. */
export function autopilotAircraftLabel(track: TrainingAutopilotTrack, index: number, speedup: number): string {
  const bank = track.bankRightDeg.length === 0 ? 0 : track.bankRightDeg[Math.min(index, track.bankRightDeg.length - 1)];
  const flownS = track.tS[index] - track.tS[0];
  const totalS = track.tS[track.tS.length - 1] - track.tS[0];
  return `autopilot ${flownS.toFixed(0)} / ${totalS.toFixed(0)} s simulated ×${speedup.toFixed(0)} · ` +
    `${track.groundSpeedMps[index].toFixed(0)} m/s · ${track.altitudeMslM[index].toFixed(0)} m · bank ` +
    `${Math.abs(bank).toFixed(0)}°${Math.abs(bank) < 0.5 ? "" : bank > 0 ? " R" : " L"}`;
}

// ── the answer ───────────────────────────────────────────────────────────────

function parseTrack(reader: Reader, startS: number, referenceDeg: number): TrainingAutopilotTrack {
  const cycle = reader.numbers("cycle");
  // one state is an answer too: a dynamics failure in the first cycle keeps only the state it started from
  if (cycle.length < 1) reader.fail("cycle is empty");
  const n = cycle.length;
  cycle.forEach((value, index) => {
    if (!Number.isInteger(value) || (index > 0 && value !== cycle[index - 1] + 1)) {
      reader.fail(`cycle does not run forward one at a time at ${index}`);
    }
  });
  const tS = reader.numbers("tS", n).map((value) => value + startS);
  const trackDeg = reader.numbers("trackDeg", n);
  return {
    cycle, tS, eM: reader.numbers("eM", n), nM: reader.numbers("nM", n), lon: reader.numbers("lonDeg", n),
    lat: reader.numbers("latDeg", n), altitudeMslM: reader.numbers("altitudeMslM", n), altitudeHaeM: reader.numbers("altitudeHaeM", n),
    trackDeg, trackPlotDeg: unwrapDegrees(trackDeg, referenceDeg), groundSpeedMps: reader.numbers("groundSpeedMps", n),
    verticalRateMps: reader.numbers("verticalRateMps", n), thrustFraction: reader.numbers("thrustFraction", n - 1),
    bankRightDeg: reader.numbers("bankRightDeg", n - 1), loadFactor: reader.numbers("loadFactor", n - 1),
    attitude: readAttitude(reader.child("attitude"), n),
  };
}

/** The backend's answer against the request it answers and the sentence on screen: refused by name unless it is the
 *  segment of the word the bar shows. */
export function parseTrainingAutopilot(
  raw: unknown, request: TrainingAutopilotRequest, selection: TrainingSelection,
): Parsed<TrainingAutopilotSegment> {
  return attempt(() => {
    const answer = Reader.of(raw, "autopilot answer");
    answer.oneOf("schema", [TRAINING_AUTOPILOT_SCHEMA]);
    if (!answer.boolean("ok")) answer.fail("ok is false");
    for (const key of ["airport", "setId", "flightKey"] as const) {
      if (answer.string(key) !== request[key]) answer.fail(`${key} is ${answer.raw(key)}, but ${request[key]} was asked for`);
    }
    const rowIntervalS = answer.number("rowIntervalS");
    if (rowIntervalS !== request.rowIntervalS) answer.fail(`rowIntervalS is ${rowIntervalS}, but ${request.rowIntervalS} was asked for`);
    const { flight } = selection;
    const reading = trainingReadingOf(flight, selection.vocabulary.stepS, request.rowIntervalS);
    const closed = reading.closed!;
    const executor = answer.child("executor");
    const cycleS = executor.number("cycleS");
    if (cycleS !== closed.cycleS) executor.fail(`cycleS is ${cycleS} s, but the set's executor cycle is ${closed.cycleS} s`);

    const segment = answer.child("segment");
    const column = TRAINING_COLUMNS[segment.integer("column", 0, TRAINING_COLUMNS.length - 1)];
    const row = segment.count("row");
    if (column !== request.column || row !== request.row) {
      segment.fail(`is ${column} at Δ row ${row}, but ${request.column} at Δ row ${request.row} was asked for`);
    }
    const run = sentenceWordAt(reading, column, row);
    if (run === null || run.row !== row) return segment.fail(`no ${column} word is said at Δ row ${row} of the sentence on screen`);
    const word = segment.integer("word", TRAINING_RUNWAY_GO_AROUND, Number.MAX_SAFE_INTEGER);
    const correction = segment.boolean("correction");
    if (word !== run.event.value || correction !== run.event.correction) {
      segment.fail(`is word ${word}${correction ? " (a correction)" : ""}, but the sentence says word ${run.event.value}` +
        `${run.event.correction ? " (a correction)" : ""} there: the backend flew another sentence of this flight`);
    }
    const startCycle = segment.count("startCycle");
    if (startCycle * cycleS !== row * request.rowIntervalS) {
      segment.fail(`startCycle ${startCycle} (${startCycle * cycleS} s) is not Δ row ${row} (${row * request.rowIntervalS} s)`);
    }
    const stopCycle = segment.nullableCount("stopCycle");
    const end = segment.oneOf("end", [TRAINING_AUTOPILOT_SEGMENT_END, ...TRAINING_OUTCOMES] as const);
    if (end === TRAINING_AUTOPILOT_SEGMENT_END && stopCycle === null) segment.fail("ended at a segment end it has none of");
    const endCycle = segment.count("endCycle");

    const track = parseTrack(answer.child("track"), closed.startS,
      closed.flown.trackPlotDeg[rowAtTime(closed.flown.tS, closed.startS + startCycle * cycleS)]);
    if (track.cycle[0] !== startCycle) answer.fail(`track starts at cycle ${track.cycle[0]}, not at the word's cycle ${startCycle}`);
    // a segment stopped at its end has its last state at `endCycle`; a flight flown to its outcome at the last state the judge
    // keeps (`lastStateCycle`: a dynamics failure's failed state is left out)
    const lastCycle = end === TRAINING_AUTOPILOT_SEGMENT_END ? endCycle : lastStateCycle(end, endCycle);
    if (track.cycle[track.cycle.length - 1] !== lastCycle) {
      answer.fail(`track ends at cycle ${track.cycle[track.cycle.length - 1]}, but a segment that ended at cycle ${endCycle} (${end}) ends at ${lastCycle}`);
    }
    const crossingReader = answer.nullableChild("crossing");
    if ((crossingReader !== null) && end === TRAINING_AUTOPILOT_SEGMENT_END) {
      answer.fail("gives a crossing for a flight stopped at its segment's end");
    }
    const stored = answer.child("stored");
    const timing = answer.child("timing");
    return {
      airport: request.airport, setId: request.setId, flightKey: request.flightKey, datasetId: answer.string("datasetId"),
      rowIntervalS, computedUtc: answer.string("computedUtc"),
      timing: {
        waitS: timing.number("waitS"), openS: timing.number("openS"), flyS: timing.number("flyS"), cycles: timing.count("cycles"),
        judgeS: timing.number("judgeS"), answerS: timing.number("answerS"), computeS: timing.number("computeS"),
      },
      executor: { spec: executor.string("spec"), specSha256: executor.string("specSha256"), cycleS },
      artefact: answer.string("artefact"),
      segment: { column, row, word, correction, startCycle, stopCycle, end, endCycle },
      track,
      crossing: crossingReader === null ? null : readCrossing(crossingReader, selection.candidates.length),
      stored: { rows: stored.count("rows"), horizontalM: stored.number("horizontalM"), verticalM: stored.number("verticalM") },
    };
  });
}

/** Ask the backend to fly the segment. Resolves to its answer's JSON (parse it with `parseTrainingAutopilot`); rejects
 *  with the backend's own refusal, or with the reason nothing answered. */
export async function requestTrainingAutopilot(
  backendUrl: string, request: TrainingAutopilotRequest, signal?: AbortSignal,
): Promise<unknown> {
  const url = `${backendUrl.replace(/\/+$/, "")}${TRAINING_AUTOPILOT_PATH}`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...request, clientId: TRAINING_AUTOPILOT_CLIENT_ID, seq: (requestSeq += 1) }), signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error(`the backend at ${backendUrl} did not answer (is it running? ./start_aeroviz_fullstack.sh): ${
      error instanceof Error ? error.message : String(error)}`);
  }
  const text = await response.text();
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    throw new Error(`the backend at ${backendUrl} answered HTTP ${response.status} with something that is not JSON: ` +
      `${text.slice(0, 120)}`);
  }
  if (!response.ok) {
    const refusal = typeof body === "object" && body !== null && typeof (body as { error?: unknown }).error === "string"
      ? (body as { error: string }).error : `HTTP ${response.status}`;
    throw new Error(`the backend refused (${response.status}): ${refusal}`);
  }
  return body;
}
