/**
 * trainingSlider.ts
 * -----------------
 * The cursor slider's model (frontend §6.1, D155): the rows it stops at, the row in force at a time, and the marks each
 * stage's session puts on its track. Pure: the component is `components/training/CursorSlider.tsx`.
 *
 * THE ROWS. The slider's track is the sentence bar's axis — the flight's time from 0 to where the sentence (or its judged
 * track) ends (`readingAxisEndS`). It stops at the sentence's rows: before the sentence opens (a closed-loop sentence opens
 * at the first predicted step), at the observed track's 2 s rows; from there on, at the sentence's own rows, Δ apart
 * (2 s apart on a labelled sentence), to its last row. A key moves one row; the value the slider reports is the row's place
 * in that list.
 *
 * THE MARKS are in flight time, given by the session of the stage on screen; the slider draws them and computes none.
 */

import { readingRowTimeS, type TrainingReading } from "./trainingSample";
import {
  TRAINING_CORRECTION_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_FAILURE_COLOR,
  TRAINING_TRAFFIC_COLOR,
} from "../utils/trainingWordColors";

/** A mark on the slider's track: a tick at one time, a line of values in [0, 1] over times, or a band over a span of
 *  times (stage D: an aircraft's silence to its end, frontend §6.1). */
export type TrainingSliderMark =
  | { kind: "tick"; key: string; atS: number; colour: string; title: string }
  | { kind: "line"; key: string; points: Array<[number, number]>; colour: string; title: string }
  | { kind: "band"; key: string; fromS: number; toS: number; colour: string; title: string };

/** The times the slider stops at (see the module's note), ascending. */
export function sliderStops(reading: TrainingReading, stepS: number): number[] {
  const stops: number[] = [];
  for (let k = 0; k * stepS < reading.originS - 1e-9; k += 1) stops.push(k * stepS);
  for (let row = 0; row < reading.rows; row += 1) stops.push(readingRowTimeS(reading, row));
  return stops;
}

/** The place of the row in force at ``atS``: the last stop at or before it (the first before the first). */
export function stopAt(stops: number[], atS: number): number {
  let place = 0;
  while (place + 1 < stops.length && stops[place + 1] <= atS + 1e-9) place += 1;
  return place;
}

/** The place of the stop nearest to ``atS``. */
export function nearestStop(stops: number[], atS: number): number {
  let best = 0;
  for (let place = 1; place < stops.length; place += 1) {
    if (Math.abs(stops[place] - atS) < Math.abs(stops[best] - atS)) best = place;
  }
  return best;
}

/** The place a key moves the slider to from ``place``, or null for a key the slider does not take: ← and → one row, with
 *  Shift ten, Home and End the first and the last. */
export function stopForKey(stops: number[], place: number, key: string, shift: boolean): number | null {
  const last = stops.length - 1;
  const step = shift ? 10 : 1;
  if (key === "ArrowLeft") return Math.max(place - step, 0);
  if (key === "ArrowRight") return Math.min(place + step, last);
  if (key === "Home") return 0;
  if (key === "End") return last;
  return null;
}

// ── each stage's marks ───────────────────────────────────────────────────────

/** Every stage: a tick at the first predicted step. */
export function firstStepMark(atS: number): TrainingSliderMark {
  return { kind: "tick", key: "first-step", atS, colour: TRAINING_EXECUTOR_COLOR, title: `the first predicted step, ${atS.toFixed(0)} s` };
}

/** Stage A: a tick at each row of a closed-loop sentence with a word the reading added (a correction). */
export function correctionMarks(reading: TrainingReading): TrainingSliderMark[] {
  const rows = [...new Set(reading.events.filter((event) => event.correction).map((event) => event.row))].sort((a, b) => a - b);
  return rows.map((row) => ({
    kind: "tick", key: `correction-${row}`, atS: readingRowTimeS(reading, row), colour: TRAINING_CORRECTION_COLOR,
    title: `a word the closed-loop reading added, row ${row} (${readingRowTimeS(reading, row).toFixed(0)} s)`,
  }));
}

/** Stage B, on a sample's tab: the probability the prior gave "go-around" at each row of the sentence, as a line. */
export function goAroundLine(reading: TrainingReading, probability: number[]): TrainingSliderMark {
  return {
    kind: "line", key: "go-around", colour: TRAINING_CORRECTION_COLOR, title: "the probability of go-around at each row",
    points: probability.map((value, row) => [readingRowTimeS(reading, row), value]),
  };
}

/** Stages C and D: a red tick at a loss of separation that the aircraft on screen is in. */
export function lossMark(atS: number, other: string): TrainingSliderMark {
  return {
    kind: "tick", key: `loss-${other}-${atS}`, atS, colour: TRAINING_FAILURE_COLOR,
    title: `a loss of separation with ${other}, ${atS.toFixed(0)} s`,
  };
}

/** Stage D: a slate band from the row the aircraft on screen is silent on to its end (frontend §6.1, D144). */
export function silenceMark(fromS: number, toS: number): TrainingSliderMark {
  return {
    kind: "band", key: "silent", fromS, toS, colour: TRAINING_TRAFFIC_COLOR,
    title: `silent from ${fromS.toFixed(0)} s: it answered for a loss of separation and flies on its words in force`,
  };
}
