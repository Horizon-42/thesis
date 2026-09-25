/**
 * readbackModel.ts
 * ----------------
 * Everything the read-back window's four charts share, computed once per render: which lines are drawn (the executor's
 * replay and the live segment, when they have two points), the selected word and what recedes, and every scale — the
 * plan view's one scale on both axes, the time and distance axes extended to hold the executor's lines, each chart's y
 * extent over what it draws. No envelope is computed: the numbers scaled are the exporter's and the backend's.
 */

import type { TrainingLayers } from "../../context/AppContext";
import { TRAINING_AUTOPILOT_COLOR } from "../../utils/trainingWordColors";
import {
  formatSeconds,
  outsideSpans,
  rowAtTime,
  trainingWordAt,
  trainingWordLabel,
  TRAINING_COLUMN_INDEX,
  type TrainingCandidate,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingHeadingBand,
  type TrainingVocabulary,
} from "../../data/trainingSample";
import type { TrainingExecutorFlight } from "../../data/trainingOverlays";
import { autopilotColour, autopilotHasLine, type TrainingAutopilotSegment } from "../../data/trainingAutopilot";

export const GUTTER = 64;
export const PAD_R = 16;
export const PLAN_H = 300;
export const CHART_H = 132;
/** The plot inside a chart: from its top to the axis, leaving room below for the tick labels and the axis caption. */
export const PLOT_TOP = 6;
export const PLOT_H = CHART_H - 36;
/** Every envelope that is not the selected word's is drawn at this opacity while a word is selected. */
export const FADED = 0.3;

/** [low, high] of the values, padded by 8 %, never zero-wide. */
export function extent(values: number[]): [number, number] {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    if (value < low) low = value;
    if (value > high) high = value;
  }
  if (high === low) return [low - 1, high + 1];
  const pad = (high - low) * 0.08;
  return [low - pad, high + pad];
}

/** A band has rows to draw. */
export const judged = (band: TrainingHeadingBand) => band.stopRow > band.firstRow;

export interface ReadbackInputs {
  flight: TrainingFlight;
  vocabulary: TrainingVocabulary;
  candidates: TrainingCandidate[];
  layers: TrainingLayers;
  cursorS: number;
  column: TrainingColumn | null;
  executor: TrainingExecutorFlight | null;
  autopilot: TrainingAutopilotSegment | null;
  width: number;
}

export function readbackModel({ flight, vocabulary, candidates, layers, cursorS, column, executor, autopilot, width }: ReadbackInputs) {
  const { signals, envelopes } = flight;
  const { tS } = signals;
  const last = flight.rows - 1;
  const cursorRow = rowAtTime(tS, cursorS);
  const designated = candidates[flight.runwayIndex];
  const label = (name: TrainingColumn, value: number) => trainingWordLabel(vocabulary, candidates, name, value);
  const inForce = (name: TrainingColumn) => flight.words.inForce[TRAINING_COLUMN_INDEX[name]][cursorRow];

  // ── the executor's replay: drawn on its own clock, and not at all without two points ──
  const flown = executor?.flown && executor.track.tS.length >= 2 ? executor : null;
  const flownTrack = flown?.track ?? null;
  // its heading words' bands, on the flown track as its judge read it
  const judgedTrack = flown?.judgedTrackDeg ?? null;
  const flownBands = flown && judgedTrack ? flown.words.flatMap((word) => (word.heading === null ? [] : [word.heading])) : [];

  // ── the live segment (not drawn without two points: a dynamics failure in its first cycle keeps one); its judged
  // step k is the flight's step `row + k`, the flown track's point k × `stepCycles` ──
  const live = autopilot !== null && autopilotHasLine(autopilot) ? autopilot : null;
  const liveColour = live === null ? TRAINING_AUTOPILOT_COLOR : autopilotColour(live);
  const liveBand = live?.word.heading ?? null;
  const liveJudged = live?.judgedTrackDeg ?? null;
  /** Its rows outside its heading band, as [first, last] judged steps. */
  const liveOutside = live && liveBand && liveJudged && layers.headingBands
    ? outsideSpans(liveBand.inside, liveBand.firstRow, live.segment.row + liveJudged.length - 1)
      .map(([first, lastRow]) => [first - live.segment.row, lastRow - live.segment.row] as [number, number])
    : [];

  // ── the selected word: one column's, never the step's ──
  const focus = column === null ? null : trainingWordAt(flight, column, cursorRow);
  /** Is this the selected word — the `index`-th word (and envelope) of column `name`? */
  const focused = (name: TrainingColumn, index: number) => focus !== null && column === name && focus.index === index;
  // The clearance owns the approach's envelopes: the capture turn and the corridor.
  const approachFocused = column === "approach" && focus !== null && focus.event.kind === "clear";
  /** An envelope's opacity: full for the selected word's, or for every word when none is selected. */
  const recede = (mine: boolean) => (column === null || mine ? 1 : FADED);
  // The rows the selected word is in force, on to the next word's issue so that it meets it; only a word issued on the
  // last row has no stretch.
  const focusRows = focus === null
    ? []
    : Array.from({ length: Math.min(focus.endRow, last) - focus.row + 1 }, (_, offset) => focus.row + offset);
  const capture = envelopes.approach.captureTurn;
  const captureOk = capture !== null && capture.check.progressOk && capture.check.rateOk;

  // ── the time charts share one x: the observed flight's, or longer when an executor flew longer ──
  const plotW = width - GUTTER - PAD_R;
  const endS = Math.max(tS[last], flownTrack ? flownTrack.tS[flownTrack.tS.length - 1] : 0,
    live ? live.track.tS[live.track.tS.length - 1] : 0);
  const xTime = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  // The cursor is the observed flight's time: past its end (where only the executor's lines run) it stays at the end.
  const timeAtX = (x: number) => Math.min(Math.max(((x - GUTTER) / plotW) * endS, 0), tS[last]);
  /** A row as a time edge; the step after the last row is the last row. */
  const edge = (row: number) => tS[Math.min(row, last)];
  const rowX = (row: number) => xTime(tS[row]);
  /** A flown step's time; the step after the flown track's last is its last. */
  const flownEdge = (step: number) => flownTrack!.tS[Math.min(step, flownTrack!.tS.length - 1)];
  const distanceEnd = Math.max(signals.smoothed.distanceM[last],
    flownTrack ? flownTrack.distanceM[flownTrack.distanceM.length - 1] : 0,
    live ? live.track.distanceM[live.track.distanceM.length - 1] : 0);
  const xDistance = (metres: number) => GUTTER + (metres / distanceEnd) * plotW;
  const distanceAtX = (x: number) => Math.min(Math.max(((x - GUTTER) / plotW) * distanceEnd, 0), distanceEnd);
  const rowXDistance = (row: number) => xDistance(signals.smoothed.distanceM[row]);

  // ── the plan view: the track and the threshold decide the frame, one scale on both axes ──
  const km = (metres: number) => metres / 1000;
  const [eLow, eHigh] = extent([...signals.eM, designated.thresholdEM, ...envelopes.approach.corridor.axis.eM,
    ...(flownTrack?.eM ?? []), ...(live?.track.eM ?? [])].map(km));
  const [nLow, nHigh] = extent([...signals.nM, designated.thresholdNM, ...envelopes.approach.corridor.axis.nM,
    ...(flownTrack?.nM ?? []), ...(live?.track.nM ?? [])].map(km));
  const planScale = Math.min((plotW - 12) / (eHigh - eLow), (PLAN_H - 16) / (nHigh - nLow));
  // centred in whichever direction has room to spare
  const planLeft = GUTTER + 6 + (plotW - 12 - (eHigh - eLow) * planScale) / 2;
  const planTop = 8 + (PLAN_H - 16 - (nHigh - nLow) * planScale) / 2;
  const px = (eM: number) => planLeft + (km(eM) - eLow) * planScale;
  const py = (nM: number) => planTop + (nHigh - km(nM)) * planScale;
  /** Points in the airport frame, in plan — all of them, or those at ``indices``. */
  const planPoints = (line: { eM: number[]; nM: number[] }, indices?: number[]) =>
    (indices ?? line.eM.map((_, index) => index)).map((index) => `${px(line.eM[index])},${py(line.nM[index])}`).join(" ");
  /** Rows first..last (inclusive). */
  const rows = (first: number, lastRow: number) => Array.from({ length: lastRow - first + 1 }, (_, offset) => first + offset);
  const at = (row: number) => ({ x: px(signals.eM[row]), y: py(signals.nM[row]) });

  // ── each chart's y, over what it draws ──
  const yOf = (low: number, high: number) => (value: number) => PLOT_TOP + ((high - value) / (high - low)) * PLOT_H;
  const [hLow, hHigh] = extent([
    ...signals.smoothed.trackDeg, ...signals.raw.trackDeg,
    ...(layers.headingBands ? envelopes.heading.flatMap((item) => (judged(item) ? item.bandDeg : [])) : []),
    ...(layers.corridor ? [...envelopes.approach.courseBandDeg, ...(capture ? [capture.courseOnTrackDeg] : [])] : []),
    ...(flownTrack?.trackDeg ?? []), ...(judgedTrack ?? []),
    ...(layers.headingBands ? flownBands.flatMap((band) => (judged(band) ? band.bandDeg : [])) : []),
    ...(live?.track.trackDeg ?? []), ...(liveJudged ?? []),
    ...(layers.headingBands && liveBand && judged(liveBand) ? liveBand.bandDeg : []),
  ]);
  const [aLow, aHigh] = extent([
    ...signals.smoothed.altitudeM, ...signals.raw.altitudeM, designated.elevationM,
    ...(layers.vertical ? envelopes.altitude.flatMap((tube) => [...tube.lowerM, ...tube.upperM]) : []),
    ...(flownTrack?.altitudeM ?? []), ...(live?.track.altitudeM ?? []),
  ]);
  const [sLow, sHigh] = extent([
    ...signals.smoothed.groundSpeedMps, ...signals.raw.groundSpeedMps,
    ...(layers.vertical ? envelopes.speed.flatMap((span) =>
      [...(span.bandMps ?? []), ...(span.transitionLowerMps ?? []), ...(span.transitionUpperMps ?? [])]) : []),
    ...(flownTrack?.groundSpeedMps ?? []), ...(live?.track.groundSpeedMps ?? []),
  ]);

  return {
    flight, vocabulary, candidates, layers, cursorS, column, last, cursorRow, designated, label, inForce,
    flown, flownTrack, judgedTrack, flownBands,
    live, liveColour, liveBand, liveJudged, liveOutside,
    focus, focused, approachFocused, recede, focusRows, capture, captureOk,
    width, plotW, endS, xTime, timeAtX, edge, rowX, flownEdge, distanceEnd, xDistance, distanceAtX, rowXDistance,
    px, py, planPoints, rows, at,
    yHeading: yOf(hLow, hHigh), yAltitude: yOf(aLow, aHigh), ySpeed: yOf(sLow, sHigh),
  };
}

export type ReadbackModel = ReturnType<typeof readbackModel>;

const FRACTIONS = [0, 0.25, 0.5, 0.75, 1];

/** The time axis's ticks: seconds at the quarters. */
export function timeTicks(m: ReadbackModel): Array<{ x: number; text: string }> {
  return FRACTIONS.map((fraction) => ({ x: m.xTime(fraction * m.endS), text: formatSeconds(Math.round(fraction * m.endS)) }));
}

/** The distance axis's ticks: kilometres at the quarters. */
export function distanceTicks(m: ReadbackModel): Array<{ x: number; text: string }> {
  return FRACTIONS.map((fraction) => ({ x: m.xDistance(fraction * m.distanceEnd), text: ((fraction * m.distanceEnd) / 1000).toFixed(1) }));
}
