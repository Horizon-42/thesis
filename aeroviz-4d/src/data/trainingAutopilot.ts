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
 * runs, under the executor spec written by that code for the set's vocabulary; the answer says which spec, which code,
 * and how long it took.
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

import {
  Reader,
  Refusal,
  asNumber,
  attempt,
  type TrainingExecutorCheck,
} from "./trainingOverlays";
import {
  headingBandProblem,
  trainingColumnRuns,
  TRAINING_COLUMNS,
  type Parsed,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingHeadingBand,
  type TrainingSample,
} from "./trainingSample";

/** MIRROR of `aeroviz_backend/autopilot_segment.py` `SCHEMA`: the backend's answer; anything else is refused by name. */
export const TRAINING_AUTOPILOT_SCHEMA = "aeroviz-autopilot-segment-v1";
/** MIRROR of `autopilot_segment.STATUSES`: the selected word's verdict. */
export const TRAINING_AUTOPILOT_STATUSES = ["inside", "outside", "not judged", "no check"] as const;
export type TrainingAutopilotStatus = (typeof TRAINING_AUTOPILOT_STATUSES)[number];
/** MIRROR of `autopilot_segment.SEGMENT_END`: the flight reached the point where the next word of its column was said. */
export const TRAINING_AUTOPILOT_SEGMENT_END = "segment_end";
/** MIRROR of `ts_transformer.autopilot.judge.OUTCOMES`: how a flight that did not end at its segment's end ended. */
export const TRAINING_AUTOPILOT_OUTCOMES = [
  "landed", "crossed_without_capture", "crossed_off_runway", "ground_contact", "timeout", "dynamics_failure",
] as const;
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
  /** How long the backend took to answer, once it began: the flight rebuilt (unless cached), flown and judged. */
  computeS: number;
  /** How long it waited first for the flight before it (the backend flies one at a time). */
  waitS: number;
  executor: {
    spec: string;
    specSha256: string;
    /** The executor code the spec was written by — the code that flew this (the backend refuses any other). */
    sourceSha256: string;
    wordClock: string;
    cycleS: number;
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
    /** `TRAINING_AUTOPILOT_SEGMENT_END`, or the judge's outcome. */
    reason: string;
    /** null for a column's last word. */
    reachedSegmentEnd: boolean | null;
    /** At the segment's end: the executor minus the observed aircraft where the next word was said. */
    offsetFromObserved: { horizontalM: number; aboveM: number; groundSpeedMps: number } | null;
    flownS: number;
    crossing: { crossM: number; heightM: number; atS: number } | null;
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
  /** `playedAt`: when the 3D scene last began flying it out (a replay sets a new one). */
  | { status: "ready"; request: TrainingAutopilotRequest; segment: TrainingAutopilotSegment; playedAt: number };

/** Where a word's segment stops: where its own envelope ends — the step the next word of its column is said, and for a
 *  heading word a lead later (it is judged from a lead after it is said to a lead after the next heading word is) —
 *  never past the sentence's end. MIRROR of the backend's `segment_of`. */
export function segmentStopRow(flight: TrainingFlight, column: TrainingColumn, endRow: number, headingLeadRows: number): number {
  return Math.min(endRow + (column === "heading" ? headingLeadRows : 0), flight.rows);
}

/** The same request: the same set, flight and segment. */
export function sameAutopilotRequest(a: TrainingAutopilotRequest, b: TrainingAutopilotRequest): boolean {
  return a.airport === b.airport && a.setId === b.setId && a.flightKey === b.flightKey && a.column === b.column
    && a.row === b.row;
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
  if (tS.length < 1) reader.fail("tS is empty");
  if (Math.abs(tS[0] - row * stepS) > 1e-3) reader.fail(`starts at ${tS[0]} s, not at step ${row} (${row * stepS} s)`);
  if (tS.some((value, index) => index > 0 && value <= tS[index - 1])) reader.fail("tS does not run forward");
  const n = tS.length;
  const cycles = Math.max(n - 1, 0);
  return {
    tS, eM: reader.numbers("eM", n), nM: reader.numbers("nM", n), lon: reader.numbers("lon", n), lat: reader.numbers("lat", n),
    altitudeM: reader.numbers("altitudeM", n), altitudeHaeM: reader.numbers("altitudeHaeM", n),
    groundSpeedMps: reader.numbers("groundSpeedMps", n), verticalRateMps: reader.numbers("verticalRateMps", n),
    trackDeg: reader.numbers("trackDeg", n), distanceM: reader.numbers("distanceM", n),
    thrustFraction: reader.numbers("thrustFraction", cycles), bankRightDeg: reader.numbers("bankRightDeg", cycles),
    loadFactor: reader.numbers("loadFactor", cycles),
  };
}

function parseChecks(word: Reader): TrainingExecutorCheck[] {
  return word.list("checks").map((check, position) => {
    const item = Reader.of(check, word.at(`checks[${position}]`));
    return {
      name: item.string("name"), ok: item.boolean("ok"),
      inside: item.nullableInteger("inside", 0, Number.MAX_SAFE_INTEGER), rows: item.nullableInteger("rows", 0, Number.MAX_SAFE_INTEGER),
    };
  });
}

/** Parse the backend's answer against the request and the flight on screen — all or nothing. */
export function parseTrainingAutopilot(
  raw: unknown, request: TrainingAutopilotRequest, sample: TrainingSample,
): Parsed<TrainingAutopilotSegment> {
  return attempt(() => {
    const answer = Reader.of(raw, "the autopilot's answer");
    if (answer.raw("schema") !== TRAINING_AUTOPILOT_SCHEMA) {
      answer.fail(`schema is ${JSON.stringify(answer.raw("schema"))}, expected ${JSON.stringify(TRAINING_AUTOPILOT_SCHEMA)}`);
    }
    const flight = sample.flights.find((item) => item.flightKey === request.flightKey);
    if (flight === undefined) throw new Refusal(`the set ${sample.setId} has no flight ${request.flightKey}`);
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
    if (vocabularySpecSha256 !== sample.vocabulary.specSha256) {
      answer.fail(`flew vocabulary ${vocabularySpecSha256.slice(0, 12)}, the set on screen is ${sample.vocabulary.specSha256.slice(0, 12)}`);
    }

    // ── the segment: the run on screen, and the words it shows ──
    const segmentReader = answer.child("segment");
    const column = segmentReader.string("column");
    const row = segmentReader.integer("row", 0, flight.rows - 1);
    if (column !== request.column || row !== request.row) {
      segmentReader.fail(`is ${column} from step ${row}, but ${request.column} from step ${request.row} was asked for`);
    }
    const run = trainingColumnRuns(flight, request.column).find((item) => item.row === row);
    if (run === undefined) segmentReader.fail(`no ${column} word is said at step ${row} of the sentence on screen`);
    const endRow = segmentReader.integer("endRow", row + 1, flight.rows);
    if (endRow !== run!.endRow) segmentReader.fail(`ends at step ${endRow}, but the word on screen is in force to step ${run!.endRow}`);
    const stopRow = segmentReader.integer("stopRow", row + 1, flight.rows);
    const toLanding = segmentReader.boolean("toLanding");
    const stop = segmentStopRow(flight, request.column, endRow, sample.vocabulary.headingLeadRows);
    if (stopRow !== stop || toLanding !== (stopRow === flight.rows)) {
      segmentReader.fail(`stops at step ${stopRow}${toLanding ? " (to the landing)" : ""}, but this word's envelope ends at ` +
        `step ${stop}`);
    }
    const told = segmentReader.list("told").map((item, position) => {
      const word = Reader.of(item, segmentReader.at(`told[${position}]`));
      return { row: word.integer("row", 0, flight.rows - 1), column: word.integer("column", 0, 5), value: word.integer("value", 0, Number.MAX_SAFE_INTEGER) };
    });
    const shown = segmentWords(flight, row, stopRow);
    const same = told.length === shown.length && told.every((word, index) =>
      word.row === shown[index].row && word.column === shown[index].column && word.value === shown[index].value);
    if (!same) {
      segmentReader.fail(`told the executor ${told.length} words that are not the ${shown.length} the sentence shows for this ` +
        "segment: the backend flew another reading of this flight");
    }

    // ── the flight ──
    const executor = answer.child("executor");
    const cycleS = executor.number("cycleS");
    const stepRows = sample.vocabulary.stepS / cycleS;
    if (!Number.isInteger(stepRows) || stepRows < 1) executor.fail(`a ${cycleS} s cycle does not divide the ${sample.vocabulary.stepS} s step`);
    const track = parseTrack(answer.child("track"), row, sample.vocabulary.stepS);
    const end = answer.child("end");
    const reason = end.string("reason");
    if (reason !== TRAINING_AUTOPILOT_SEGMENT_END && !(TRAINING_AUTOPILOT_OUTCOMES as readonly string[]).includes(reason)) {
      end.fail(`reason is ${reason}, not ${TRAINING_AUTOPILOT_SEGMENT_END} nor one of ${TRAINING_AUTOPILOT_OUTCOMES.join(", ")}`);
    }
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
    const refused = end.nullableString("refused");

    // ── the selected word's verdict ──
    const word = answer.child("word");
    const status = word.string("status");
    if (!(TRAINING_AUTOPILOT_STATUSES as readonly string[]).includes(status)) {
      word.fail(`status is ${status}, not one of ${TRAINING_AUTOPILOT_STATUSES.join(", ")}`);
    }
    const checks = parseChecks(word);
    const judged = status === "inside" || status === "outside";
    if (judged && (checks.length === 0 || (status === "inside") !== checks.every((check) => check.ok))) {
      word.fail(`is ${status}, but its checks say ${checks.map((check) => `${check.name} ${check.ok}`).join(", ") || "nothing"}`);
    }
    const wordReason = word.nullableString("reason");
    if (!judged && wordReason === null) word.fail(`is ${status} and says no reason`);
    const judgedTrackDeg = answer.nullableNumbers("judgedTrackDeg");
    // a heading word's band and the track its judge read come together, and only for a heading word
    if ((word.raw("heading") !== null) !== (judgedTrackDeg !== null)) word.fail("a heading band comes with the track its judge read");
    if (judgedTrackDeg !== null && request.column !== "heading") word.fail(`a ${request.column} word carries a heading band`);
    let heading: TrainingHeadingBand | null = null;
    if (judgedTrackDeg !== null) {
      const misplaced = judgedTrackDeg.findIndex((_, step) =>
        !(Math.abs(track.tS[step * stepRows] - (row + step) * sample.vocabulary.stepS) <= 1e-3));
      if (misplaced >= 0) answer.fail(`judgedTrackDeg's step ${misplaced} is not a step of the flown track`);
      const band = word.child("heading");
      const firstRow = band.integer("firstRow", 0, Number.MAX_SAFE_INTEGER);
      const stopRow = band.integer("stopRow", firstRow, Number.MAX_SAFE_INTEGER);
      const bandDeg = band.numbers("bandDeg", 2);
      heading = {
        firstRow, stopRow, targetOnTrackDeg: band.number("targetOnTrackDeg"), bandDeg: [bandDeg[0], bandDeg[1]],
        inside: band.flags("inside", stopRow - firstRow),
      };
      const envelope = flight.envelopes.heading.find((item) => item.row === row);
      if (envelope === undefined) band.fail(`the sentence on screen has no heading word at step ${row}`);
      const problem = headingBandProblem(heading, row, envelope!.targetDeg, sample.vocabulary, row + judgedTrackDeg.length);
      if (problem !== null) band.fail(problem);
      const rows = heading.inside.length;
      const counted = heading.inside.filter(Boolean).length;
      if (rows > 0 && !checks.some((check) => check.rows === rows && check.inside === counted)) {
        word.fail(`its band counts ${counted} of ${rows} rows inside, and no check says so`);
      }
    }
    const limits = answer.child("limits");
    return {
      ...echo,
      computedUtc: answer.string("computedUtc"),
      computeS: answer.number("computeS"),
      waitS: answer.number("waitS"),
      executor: {
        spec: executor.string("spec"), specSha256: executor.string("specSha256"), sourceSha256: executor.string("sourceSha256"),
        wordClock: executor.string("wordClock"), cycleS, timeoutFactor: executor.number("timeoutFactor"),
      },
      artefact: answer.string("artefact"),
      vocabularySpecSha256,
      group: answer.string("group"),
      segment: { column: request.column, row, endRow, stopRow, toLanding, observedS: segmentReader.number("observedS"), told },
      end: {
        reason, reachedSegmentEnd,
        offsetFromObserved: offset === null ? null : {
          horizontalM: offset.number("horizontalM"), aboveM: offset.number("aboveM"), groundSpeedMps: offset.number("groundSpeedMps"),
        },
        flownS: end.number("flownS"),
        crossing: crossing === null ? null : { crossM: crossing.number("crossM"), heightM: crossing.number("heightM"), atS: crossing.number("atS") },
        refused,
      },
      word: { status: status as TrainingAutopilotStatus, checks, reason: wordReason, heading },
      limits: { cycles: limits.integer("cycles", 0, Number.MAX_SAFE_INTEGER), bound: limits.record("bound", asNumber) },
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
