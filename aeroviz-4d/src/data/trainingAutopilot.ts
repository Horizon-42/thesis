/**
 * trainingAutopilot.ts
 * --------------------
 * THE EXECUTOR, LIVE: one word's segment of the selected Training flight, flown by the backend at the moment it is
 * asked (`POST /autopilot/segment`, `aeroviz_backend/autopilot_segment.py`). Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §4.7.
 *
 * A SEGMENT is the band the sentence bar draws and the user CLICKED: one column's word from the step it is said to the
 * step the next word of its column is said (`TrainingWordRun`). The executor starts from the observed aircraft's state at
 * that step, is told the six words in force there, then the sentence's words, each where the observed aircraft heard it,
 * and flies to where the word's own envelope ends (`segmentStopRow`: the next word of its column; a heading word's a lead
 * later, the next heading word told on the way as the sentence says) — or, when that is the sentence's end, on to its
 * outcome. Only the selected word is judged, by the executor's judge.
 *
 * NOTHING IS PRECOMPUTED: no replay record, no overlay. Every request is flown again by the executor code the backend
 * runs — its stepper, one control cycle at a time, stopped at the segment's stop — under the executor spec written by
 * that code for the set's vocabulary; the answer says which spec, which code, how many cycles were flown and how long
 * each part took (`timing`), and the view adds the browser's round trip.
 *
 * THE ANSWER IS BOUND TO THE FLIGHT ON SCREEN, or refused whole: the same set, flight and segment (its end is the
 * run's, its stop the word's envelope's), the same vocabulary spec, and the words the executor was told are the words the
 * sentence bar shows for that segment — the six in force at its first step, then every word said before its stop. A verdict drawn on another
 * sentence than the one shown is worse than none.
 *
 * NOTHING HERE IS COMPUTED: the flown track, the verdict and its heading band are the backend's (the judge's own);
 * this file checks bookkeeping and hands numbers to the views.
 *
 * SI units only: metres, m/s, degrees, seconds.
 */

import { asNumber, attempt, Reader, type Parsed } from "./trainingReader";
import {
  readCrossing,
  readJudgedBand,
  readWordVerdict,
  TRAINING_EXECUTOR_OUTCOMES,
  type TrainingCrossing,
  type TrainingExecutorOutcome,
  type TrainingExecutorCheck,
} from "./trainingOverlays";
import { TRAINING_AUTOPILOT_COLOR, TRAINING_AUTOPILOT_OUTSIDE_COLOR } from "../utils/trainingWordColors";
import {
  trainingColumnRuns,
  trainingWordLabel,
  TRAINING_COLUMN_INDEX,
  TRAINING_COLUMNS,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingHeadingBand,
  type TrainingSelection,
  type TrainingVocabulary,
  type TrainingWordRun,
} from "./trainingSample";

/** MIRROR of `aeroviz_backend/autopilot_segment.py` `SCHEMA`: the backend's answer; anything else is refused by name. */
export const TRAINING_AUTOPILOT_SCHEMA = "aeroviz-autopilot-segment-v2";
/** MIRROR of `autopilot_segment.STATUSES`: the selected word's verdict. */
export const TRAINING_AUTOPILOT_STATUSES = ["inside", "outside", "not judged", "no check"] as const;
export type TrainingAutopilotStatus = (typeof TRAINING_AUTOPILOT_STATUSES)[number];
/** MIRROR of `autopilot_segment.SEGMENT_END`: the flight reached the point where the next word of its column was said. */
export const TRAINING_AUTOPILOT_SEGMENT_END = "segment_end" as const;
export const TRAINING_AUTOPILOT_PATH = "/autopilot/segment";

/** What the Training view asks for: the clicked word's segment of the selected flight. */
export interface TrainingAutopilotRequest {
  airport: string;
  setId: string;
  flightKey: string;
  column: TrainingColumn;
  /** The step the selected word is said at: its segment's first step. */
  row: number;
}

/** The flown segment every control cycle, on the flight's own clock and axes (see the backend's `track_payload`). */
export interface TrainingAutopilotTrack {
  /** Seconds from the flight's first step: the segment's first step's time, then one cycle apart. */
  tS: number[];
  eM: number[];
  nM: number[];
  lon: number[];
  lat: number[];
  altitudeM: number[];
  altitudeHaeM: number[];
  groundSpeedMps: number[];
  verticalRateMps: number[];
  /** Unwrapped, on the observed smoothed track's branch at the segment's first step. */
  trackDeg: number[];
  /** The observed flight's distance flown at the segment's first step, plus the executor's own. */
  distanceM: number[];
  /** The commands each cycle flew: one fewer than the states. */
  thrustFraction: number[];
  /** Positive: banked right (turning right). */
  bankRightDeg: number[];
  loadFactor: number[];
}

export interface TrainingAutopilotSegment {
  airport: string;
  setId: string;
  flightKey: string;
  datasetId: string;
  computedUtc: string;
  /** Wall-clock seconds on the backend: the wait, then the parts that add up to `computeS`. */
  timing: {
    /** Waiting for the flight before it (the backend flies one at a time). */
    waitS: number;
    /** The set and the executor spec found. */
    setupS: number;
    /** The flight rebuilt from the data plane — or kept from an earlier request (`flightKept`). */
    openS: number;
    flightKept: boolean;
    /** The segment set up for the executor (its sentence, clock and physics). */
    prepareS: number;
    /** The executor flying its `cycles` control cycles — what was computed, which may run past the judged outcome. */
    flyS: number;
    cycles: number;
    /** The judge. */
    judgeS: number;
    /** The answer written. */
    answerS: number;
    /** The whole request once it began. */
    computeS: number;
  };
  executor: {
    spec: string;
    specSha256: string;
    /** The executor code the spec was written by — the code that flew this (the backend refuses any other). */
    sourceSha256: string;
    wordClock: string;
    cycleS: number;
    /** Control cycles per sentence step: the track's point `k * stepCycles` is the flight's step `segment.row + k`. */
    stepCycles: number;
    timeoutFactor: number;
  };
  artefact: string;
  vocabularySpecSha256: string;
  /** "own dynamics" or "stand-in dynamics" (a stand-in's errors are its aerodynamics, not the executor's). */
  group: string;
  segment: {
    column: TrainingColumn;
    row: number;
    /** Where the next word of its column is said (the sentence's length for the column's last word). */
    endRow: number;
    /** Where the flight stops: `segmentStopRow`. */
    stopRow: number;
    /** `stopRow` is the sentence's end: flown on to the outcome. */
    toLanding: boolean;
    /** The observed aircraft's time over the same steps (to the landing: to its threshold crossing). */
    observedS: number;
    /** The words the executor was told, at the flight's steps, by step then column. */
    told: Array<{ row: number; column: number; value: number }>;
  };
  end: {
    /** `TRAINING_AUTOPILOT_SEGMENT_END`, or the judge's outcome (`TRAINING_EXECUTOR_OUTCOMES`). */
    reason: typeof TRAINING_AUTOPILOT_SEGMENT_END | TrainingExecutorOutcome;
    /** null for a column's last word. */
    reachedSegmentEnd: boolean | null;
    /** At the segment's end: the executor minus the observed aircraft where the next word was said. */
    offsetFromObserved: { horizontalM: number; aboveM: number; groundSpeedMps: number } | null;
    flownS: number;
    crossing: TrainingCrossing | null;
    /** Why the labeller's gate refused the flown segment (nothing is then judged). */
    refused: string | null;
  };
  word: {
    status: TrainingAutopilotStatus;
    checks: TrainingExecutorCheck[];
    reason: string | null;
    /** A heading word the judge judged: its band over the flown rows, at the flight's steps. */
    heading: TrainingHeadingBand | null;
  };
  limits: { cycles: number; bound: Record<string, number> };
  track: TrainingAutopilotTrack;
  /** A heading word's flown segment as the judge read it: its step k is the flight's step `segment.row + k`. */
  judgedTrackDeg: number[] | null;
}

/** What the panel publishes for the sentence bar, the read-back window and the 3D scene. */
export type TrainingAutopilotView =
  | { status: "flying"; request: TrainingAutopilotRequest }
  | { status: "failed"; request: TrainingAutopilotRequest; problem: string }
  /** `playedAt`: when the 3D scene last began flying it out (a replay sets a new one); `roundTripS`: the browser's wait,
   *  from the request sent to the answer read. */
  | { status: "ready"; request: TrainingAutopilotRequest; segment: TrainingAutopilotSegment; playedAt: number;
      roundTripS: number };

/** Where a word's segment stops: where its own envelope ends — the step the next word of its column is said, and for a
 *  heading word a lead later (it is judged from a lead after it is said to a lead after the next heading word is) —
 *  never past the sentence's end. MIRROR of the backend's `segment_of`. */
export function segmentStopRow(flight: TrainingFlight, column: TrainingColumn, endRow: number, headingLeadRows: number): number {
  return Math.min(endRow + (column === "heading" ? headingLeadRows : 0), flight.rows);
}

/** THE WORD THE LIVE EXECUTOR FLIES (`trainingPick`) — of the flight on screen, which the pick belongs to and is reset
 *  with (`AppContext`): the word's column and the step it is said at, and which attempt at it — a new attempt at the
 *  same word flies it again. */
export interface TrainingPick {
  column: TrainingColumn;
  row: number;
  attempt: number;
}

/** The pick that flies ``column``'s word said at ``row`` now: a new attempt when it is the word picked already, else its
 *  first. */
export function nextPick(current: TrainingPick | null, column: TrainingColumn, row: number): TrainingPick {
  const same = current !== null && current.column === column && current.row === row;
  return { column, row, attempt: same ? current.attempt + 1 : 0 };
}

/** The live executor's view if it is of the flight on screen — its airport, set and flight — else null: in the render
 *  after a switch, the view still published is the last flight's, and is not drawn. */
export function autopilotOnScreen(
  view: TrainingAutopilotView | null, selection: TrainingSelection | null,
): TrainingAutopilotView | null {
  if (view === null || selection === null) return null;
  const { request } = view;
  return request.airport === selection.airport && request.setId === selection.setId
    && request.flightKey === selection.flight.flightKey ? view : null;
}

/** The word a request flies, as the sentence reads it: "heading 270°" — the word said at its step, whichever word the
 *  views have selected since. */
export function autopilotWord(request: TrainingAutopilotRequest, selection: TrainingSelection): string {
  const value = selection.flight.words.inForce[TRAINING_COLUMN_INDEX[request.column]][request.row];
  return `${request.column} ${trainingWordLabel(selection.vocabulary, selection.candidates, request.column, value)}`;
}

/** The colour a flown segment is drawn in, everywhere: red when the selected word flew outside its envelope. */
export function autopilotColour(segment: TrainingAutopilotSegment): string {
  return segment.word.status === "outside" ? TRAINING_AUTOPILOT_OUTSIDE_COLOR : TRAINING_AUTOPILOT_COLOR;
}

// ── flying it out in 3D ──────────────────────────────────────────────────────

/** How much faster than real time the live executor's aircraft flies its segment out in 3D: at least this ... */
export const AUTOPILOT_PLAYBACK_MIN_SPEEDUP = 8;
/** ... and fast enough that no segment takes longer than this, in seconds (the aircraft's label says the speed-up). */
export const AUTOPILOT_PLAYBACK_MAX_S = 20;

export function autopilotPlaybackSpeedup(flownS: number): number {
  return Math.max(AUTOPILOT_PLAYBACK_MIN_SPEEDUP, flownS / AUTOPILOT_PLAYBACK_MAX_S);
}

/** Where the live executor is ``flownS`` seconds into its segment: the last point at or before it and the fraction of
 *  the way to the next (1 at and past the segment's end). */
export function autopilotFlownAt(track: TrainingAutopilotTrack, flownS: number): { index: number; fraction: number } {
  const target = track.tS[0] + Math.max(flownS, 0);
  const last = track.tS.length - 1;
  if (target >= track.tS[last]) return { index: last, fraction: 1 };
  let index = 0;
  while (index + 1 < last && track.tS[index + 1] <= target) index += 1;
  return { index, fraction: (target - track.tS[index]) / (track.tS[index + 1] - track.tS[index]) };
}

/** The aircraft's label at a flown point: the simulated time flown so far of the whole, the playback's speed-up, ground
 *  speed, geometric MSL height and bank — the command of the cycle it is in, the last cycle's at the end. */
export function autopilotAircraftLabel(track: TrainingAutopilotTrack, index: number, speedup: number): string {
  const bank = track.bankRightDeg[Math.min(index, track.bankRightDeg.length - 1)];
  const flownS = track.tS[index] - track.tS[0];
  const totalS = track.tS[track.tS.length - 1] - track.tS[0];
  return `autopilot ${flownS.toFixed(0)} / ${totalS.toFixed(0)} s simulated ×${speedup.toFixed(0)} · ` +
    `${track.groundSpeedMps[index].toFixed(0)} m/s · ${track.altitudeM[index].toFixed(0)} m · bank ` +
    `${Math.abs(bank).toFixed(0)}°${Math.abs(bank) < 0.5 ? "" : bank > 0 ? " R" : " L"}`;
}

/** The flown segment at the flight's steps — the points the judge read (every `stepCycles`-th) — from its first step:
 *  their longitudes and latitudes, as many as the judged track (a heading word's), for its rows outside to be drawn on. */
export function autopilotJudgedPoints(segment: TrainingAutopilotSegment): { lon: number[]; lat: number[] } {
  const steps = segment.judgedTrackDeg?.length ?? 0;
  const every = segment.executor.stepCycles;
  return {
    lon: Array.from({ length: steps }, (_, step) => segment.track.lon[step * every]),
    lat: Array.from({ length: steps }, (_, step) => segment.track.lat[step * every]),
  };
}

/** The words the sentence says for a segment, as the backend lists what it told: the six in force at its first step,
 *  then every word said before its end, by step then column. */
export function segmentWords(flight: TrainingFlight, row: number, endRow: number): Array<{ row: number; column: number; value: number }> {
  const opening = TRAINING_COLUMNS.map((_, column) => ({ row, column, value: flight.words.inForce[column][row] }));
  const after = flight.words.events
    .filter((event) => event.row > row && event.row < endRow)
    .map(({ row: step, column, value }) => ({ row: step, column, value }))
    .sort((a, b) => a.row - b.row || a.column - b.column);
  return [...opening, ...after];
}

function parseTrack(reader: Reader, row: number, stepS: number): TrainingAutopilotTrack {
  const tS = reader.numbers("tS");
  // one state is an answer too: a dynamics failure in the first cycle keeps only the state it started from
  if (tS.length < 1) reader.fail("tS is empty");
  if (Math.abs(tS[0] - row * stepS) > 1e-3) reader.fail(`starts at ${tS[0]} s, not at step ${row} (${row * stepS} s)`);
  if (tS.some((value, index) => index > 0 && value <= tS[index - 1])) reader.fail("tS does not run forward");
  const n = tS.length;
  return {
    tS, eM: reader.numbers("eM", n), nM: reader.numbers("nM", n), lon: reader.numbers("lon", n), lat: reader.numbers("lat", n),
    altitudeM: reader.numbers("altitudeM", n), altitudeHaeM: reader.numbers("altitudeHaeM", n),
    groundSpeedMps: reader.numbers("groundSpeedMps", n), verticalRateMps: reader.numbers("verticalRateMps", n),
    trackDeg: reader.numbers("trackDeg", n), distanceM: reader.numbers("distanceM", n),
    thrustFraction: reader.numbers("thrustFraction", n - 1), bankRightDeg: reader.numbers("bankRightDeg", n - 1),
    loadFactor: reader.numbers("loadFactor", n - 1),
  };
}

/** The segment the answer flew: the run on screen that was asked for, stopped where its envelope ends, told the words
 *  the sentence bar shows for it. */
function readSegment(
  segment: Reader, request: TrainingAutopilotRequest, flight: TrainingFlight, vocabulary: TrainingVocabulary,
): TrainingAutopilotSegment["segment"] {
  const column = segment.oneOf("column", TRAINING_COLUMNS);
  const row = segment.integer("row", 0, flight.rows - 1);
  if (column !== request.column || row !== request.row) {
    segment.fail(`is ${column} from step ${row}, but ${request.column} from step ${request.row} was asked for`);
  }
  const run: TrainingWordRun | undefined = trainingColumnRuns(flight, column).find((item) => item.row === row);
  if (run === undefined) segment.fail(`no ${column} word is said at step ${row} of the sentence on screen`);
  const endRow = segment.integer("endRow", row + 1, flight.rows);
  if (endRow !== run.endRow) segment.fail(`ends at step ${endRow}, but the word on screen is in force to step ${run.endRow}`);
  const stopRow = segment.integer("stopRow", row + 1, flight.rows);
  const toLanding = segment.boolean("toLanding");
  const stop = segmentStopRow(flight, column, endRow, vocabulary.headingLeadRows);
  if (stopRow !== stop || toLanding !== (stopRow === flight.rows)) {
    segment.fail(`stops at step ${stopRow}${toLanding ? " (to the landing)" : ""}, but this word's envelope ends at step ${stop}`);
  }
  const told = segment.children("told").map((word) => ({
    row: word.integer("row", 0, flight.rows - 1), column: word.integer("column", 0, TRAINING_COLUMNS.length - 1),
    value: word.count("value"),
  }));
  const shown = segmentWords(flight, row, stopRow);
  const same = told.length === shown.length && told.every((word, index) =>
    word.row === shown[index].row && word.column === shown[index].column && word.value === shown[index].value);
  if (!same) {
    segment.fail(`told the executor ${told.length} words that are not the ${shown.length} the sentence shows for this ` +
      "segment: the backend flew another reading of this flight");
  }
  return { column, row, endRow, stopRow, toLanding, observedS: segment.number("observedS"), told };
}

/** How the flight ended: at its segment's end (with its offset from the observed aircraft there), or the judge's
 *  outcome — to the landing, only the latter. */
function readEnd(end: Reader, toLanding: boolean): TrainingAutopilotSegment["end"] {
  const reason = end.oneOf("reason", [TRAINING_AUTOPILOT_SEGMENT_END, ...TRAINING_EXECUTOR_OUTCOMES] as const);
  const reachedSegmentEnd = end.nullableBoolean("reachedSegmentEnd");
  if ((reachedSegmentEnd === null) !== toLanding) {
    end.fail(toLanding ? "the column's last word is flown to its outcome, not to a segment end"
      : "says nothing of whether the segment's end was reached");
  }
  if (reason === TRAINING_AUTOPILOT_SEGMENT_END && reachedSegmentEnd !== true) end.fail("ended at a segment end it did not reach");
  const offset = end.nullableChild("offsetFromObserved");
  if ((offset !== null) !== (reason === TRAINING_AUTOPILOT_SEGMENT_END)) {
    end.fail("offsetFromObserved is given exactly when the flight ended at its segment's end");
  }
  const crossing = end.nullableChild("crossing");
  return {
    reason, reachedSegmentEnd,
    offsetFromObserved: offset === null ? null : {
      horizontalM: offset.number("horizontalM"), aboveM: offset.number("aboveM"), groundSpeedMps: offset.number("groundSpeedMps"),
    },
    flownS: end.number("flownS"),
    crossing: crossing === null ? null : readCrossing(crossing),
    refused: end.nullableString("refused"),
  };
}

function readTiming(timing: Reader, states: number): TrainingAutopilotSegment["timing"] {
  const cycles = timing.count("cycles");
  if (states - 1 > cycles) timing.fail(`${cycles} cycles flown, but the track holds ${states} states`);
  return {
    waitS: timing.number("waitS"), setupS: timing.number("setupS"), openS: timing.number("openS"),
    flightKept: timing.boolean("flightKept"), prepareS: timing.number("prepareS"), flyS: timing.number("flyS"), cycles,
    judgeS: timing.number("judgeS"), answerS: timing.number("answerS"), computeS: timing.number("computeS"),
  };
}

/** What the Training view asks the backend for: the picked word's segment of the flight on screen. */
export function trainingAutopilotRequest(selection: TrainingSelection, pick: TrainingPick): TrainingAutopilotRequest {
  return { airport: selection.airport, setId: selection.setId, flightKey: selection.flight.flightKey, column: pick.column,
    row: pick.row };
}

/** Parse the backend's answer against the request and the flight on screen — all or nothing. */
export function parseTrainingAutopilot(
  raw: unknown, request: TrainingAutopilotRequest, selection: TrainingSelection,
): Parsed<TrainingAutopilotSegment> {
  return attempt(() => {
    const answer: Reader = Reader.of(raw, "the autopilot's answer");
    if (answer.raw("schema") !== TRAINING_AUTOPILOT_SCHEMA) {
      answer.fail(`schema is ${JSON.stringify(answer.raw("schema"))}, expected ${JSON.stringify(TRAINING_AUTOPILOT_SCHEMA)}`);
    }
    const { vocabulary, flight } = selection;
    if (request.airport !== selection.airport || request.setId !== selection.setId || request.flightKey !== flight.flightKey) {
      answer.fail(`answers ${request.airport} ${request.setId} ${request.flightKey}, but ${selection.airport} ` +
        `${selection.setId} ${flight.flightKey} is on screen`);
    }
    const echo = {
      airport: answer.string("airport"), setId: answer.string("setId"), flightKey: answer.string("flightKey"),
      datasetId: answer.string("datasetId"),
    };
    if (echo.airport !== request.airport || echo.setId !== request.setId || echo.flightKey !== request.flightKey
        || echo.datasetId !== flight.datasetId) {
      answer.fail(`flew ${echo.airport} ${echo.setId} ${echo.datasetId}, but ${request.airport} ${request.setId} ` +
        `${flight.datasetId} was asked for`);
    }
    const vocabularySpecSha256 = answer.string("vocabularySpecSha256");
    if (vocabularySpecSha256 !== vocabulary.specSha256) {
      answer.fail(`flew vocabulary ${vocabularySpecSha256.slice(0, 12)}, the set on screen is ${vocabulary.specSha256.slice(0, 12)}`);
    }
    const segment = readSegment(answer.child("segment"), request, flight, vocabulary);
    const executor = answer.child("executor");
    const cycleS = executor.number("cycleS");
    const stepCycles = vocabulary.stepS / cycleS;
    if (!Number.isInteger(stepCycles) || stepCycles < 1) executor.fail(`a ${cycleS} s cycle does not divide the ${vocabulary.stepS} s step`);
    const track = parseTrack(answer.child("track"), segment.row, vocabulary.stepS);
    const end = readEnd(answer.child("end"), segment.toLanding);

    // ── the selected word's verdict: a heading word's band and the track its judge read come together ──
    const word = answer.child("word");
    const verdict = readWordVerdict(word, TRAINING_AUTOPILOT_STATUSES, end.refused);
    const judgedTrackDeg = answer.nullableNumbers("judgedTrackDeg");
    if ((word.raw("heading") !== null) !== (judgedTrackDeg !== null)) word.fail("a heading band comes with the track its judge read");
    if (judgedTrackDeg !== null && segment.column !== "heading") word.fail(`a ${segment.column} word carries a heading band`);
    if (judgedTrackDeg !== null && end.refused !== null) word.fail("carries a heading band, but the gate refused the flown track");
    let heading: TrainingHeadingBand | null = null;
    if (judgedTrackDeg !== null) {
      const misplaced = judgedTrackDeg.findIndex((_, step) =>
        !(Math.abs(track.tS[step * stepCycles] - (segment.row + step) * vocabulary.stepS) <= 1e-3));
      if (misplaced >= 0) answer.fail(`judgedTrackDeg's step ${misplaced} is not a step of the flown track`);
      // Its rows are FLOWN steps, and end by the track its judge read: where the executor heard the next heading word
      // (plus the lead) is its own step, not the sentence's — lagging the observed aircraft on the track clock, it hears
      // it later — so the segment's stop, a sentence step, does not bound them.
      const targetDeg = vocabulary.headingTargetsDeg[flight.words.inForce[TRAINING_COLUMN_INDEX.heading][segment.row]];
      heading = readJudgedBand(word, verdict, segment.row, targetDeg, vocabulary, segment.row + judgedTrackDeg.length);
    }
    const limits = answer.child("limits");
    return {
      ...echo,
      computedUtc: answer.string("computedUtc"),
      timing: readTiming(answer.child("timing"), track.tS.length),
      executor: {
        spec: executor.string("spec"), specSha256: executor.string("specSha256"), sourceSha256: executor.string("sourceSha256"),
        wordClock: executor.string("wordClock"), cycleS, stepCycles, timeoutFactor: executor.number("timeoutFactor"),
      },
      artefact: answer.string("artefact"),
      vocabularySpecSha256,
      group: answer.string("group"),
      segment,
      end,
      word: { status: verdict.status, checks: verdict.checks, reason: verdict.reason, heading },
      limits: { cycles: limits.count("cycles"), bound: limits.record("bound", asNumber) },
      track,
      judgedTrackDeg,
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
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request), signal,
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
