/**
 * readbackModel.ts
 * ----------------
 * Everything the read-back window's four charts share, computed once per render: which lines are drawn (the observed
 * track, the flown path of a closed-loop reading, the live segment), the selected word and what recedes, and every
 * scale — the plan view's one scale on both axes, the one time axis, each chart's y extent over what it draws. No
 * envelope is computed: the numbers scaled are the exporter's and the backend's.
 *
 * ONE CLOCK: flight time. The sentence's rows, the envelopes' rows and every track are put on it here
 * (`readingRowTimeS`, `envelopeTimeS`); a chart draws what it is told at the time it is told.
 */

import { flownSentenceColour } from "../../data/trainingSentenceKind";
import type { TrainingLayers } from "../../context/AppContext";
import { TRAINING_AUTOPILOT_COLOR } from "../../utils/trainingWordColors";
import {
  formatSeconds,
  nearestBranch,
  readingRowAt,
  readingRowTimeS,
  sentenceColumnRuns,
  sentenceWordAt,
  trainingBandLabel,
  trainingEnvelopeIndex,
  type TrainingCandidate,
  type TrainingColumn,
  type TrainingHeadingBand,
  type TrainingReading,
  type TrainingSelection,
  type TrainingWordRun,
} from "../../data/trainingSample";
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
  selection: TrainingSelection;
  reading: TrainingReading;
  layers: TrainingLayers;
  cursorS: number;
  column: TrainingColumn | null;
  /** The picked word's segment, flown live (`trainingAutopilot`, ready); null otherwise. */
  autopilot: TrainingAutopilotSegment | null;
  width: number;
}

export function readbackModel({ selection, reading, layers, cursorS, column, autopilot, width }: ReadbackInputs) {
  const { flight, vocabulary, candidates } = selection;
  const stepS = vocabulary.stepS;
  const { observed, judged: judgedTrack, closed } = reading;
  // the labelled sentence judges the observed track: its envelopes go with that track's switch
  const envelopes = closed === null && !layers.observed ? null : reading.envelopes;
  const flown = closed === null ? null : closed.flown;
  const designated = candidates[flight.runwayIndex];
  const cursorRow = readingRowAt(reading, cursorS);

  /** Flight time of an envelope row. */
  const envelopeTimeS = (row: number) => reading.envelopeOriginS + row * stepS;
  const rowTimeS = (row: number) => readingRowTimeS(reading, row);

  // ── the live segment (not drawn without two points: a dynamics failure in its first cycle keeps one) ──
  const live = autopilot !== null && autopilotHasLine(autopilot) ? autopilot : null;
  const liveColour = live === null ? TRAINING_AUTOPILOT_COLOR : autopilotColour(live);

  // ── the selected word: one column's, never the step's ──
  const focus: TrainingWordRun | null = column === null ? null : sentenceWordAt(reading, column, cursorRow);
  const focusIndex = column === null || focus === null ? null : trainingEnvelopeIndex(reading, stepS, column, focus);
  /** Is this the selected word — the `index`-th envelope of column `name`? */
  const focused = (name: TrainingColumn, index: number) => focusIndex !== null && column === name && focusIndex === index;
  /** An envelope's opacity: full for the selected word's, or for every word when none is selected. */
  const recede = (mine: boolean) => (column === null || mine ? 1 : FADED);
  /** The span of flight time the selected word is in force. */
  const focusSpanS: [number, number] | null = focus === null ? null : [rowTimeS(focus.row), rowTimeS(focus.endRow)];
  /** The points of ``tS`` inside the selected word's span, on to the next one so that it meets the next word's stretch. */
  // the judged track hidden (the labelled sentence's, with the observed track): no word is marked on it
  const judgedShown = closed !== null || layers.observed;
  /** The rows of the JUDGED track the selected word is in force. */
  const focusPoints = (tS: number[]): number[] => {
    if (focusSpanS === null || !judgedShown) return [];
    const out: number[] = [];
    tS.forEach((t, index) => {
      if (t >= focusSpanS[0] - 1e-9 && t <= focusSpanS[1] + 1e-9) out.push(index);
    });
    return out;
  };

  // ── the time axis: the flight's clock from 0, to the longest line drawn ──
  const plotW = width - GUTTER - PAD_R;
  const lastOf = (tS: number[]) => tS[tS.length - 1];
  const endS = Math.max(lastOf(observed.tS), flown === null ? 0 : lastOf(flown.tS), live === null ? 0 : lastOf(live.track.tS));
  const xTime = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  const timeAtX = (x: number) => Math.min(Math.max(((x - GUTTER) / plotW) * endS, 0), endS);

  // ── the plan view: the tracks and the threshold decide the frame, one scale on both axes ──
  const km = (metres: number) => metres / 1000;
  const reach = Math.max(1000, ...[observed, ...(flown ? [flown] : [])].flatMap((track) =>
    track.eM.map((e, i) => Math.hypot(e - designated.thresholdEM, track.nM[i] - designated.thresholdNM))));
  const courseRad = (designated.courseDeg * Math.PI) / 180;
  /** The approach centreline of a candidate, from ``reach`` out on the approach side to its threshold, and its runway. */
  const runwayLine = (candidate: TrainingCandidate) => {
    const east = Math.sin((candidate.courseDeg * Math.PI) / 180);
    const north = Math.cos((candidate.courseDeg * Math.PI) / 180);
    return {
      runway: [[candidate.thresholdEM, candidate.thresholdNM],
        [candidate.thresholdEM + candidate.lengthM * east, candidate.thresholdNM + candidate.lengthM * north]],
      centreline: [[candidate.thresholdEM - reach * east, candidate.thresholdNM - reach * north],
        [candidate.thresholdEM, candidate.thresholdNM]],
    };
  };
  const decision = closed?.replay.crossing?.decision ?? null;
  const [eLow, eHigh] = extent([...observed.eM, ...(flown?.eM ?? []), ...(live?.track.eM ?? []), designated.thresholdEM,
    designated.thresholdEM - reach * Math.sin(courseRad) * 0.2].map(km));
  const [nLow, nHigh] = extent([...observed.nM, ...(flown?.nM ?? []), ...(live?.track.nM ?? []), designated.thresholdNM,
    designated.thresholdNM - reach * Math.cos(courseRad) * 0.2].map(km));
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
  const rows = (first: number, last: number) => Array.from({ length: last - first + 1 }, (_, offset) => first + offset);
  /** Where the judged track was at flight time ``t`` (its row at or before it). */
  const indexAt = (tS: number[], t: number) => Math.min(Math.max(Math.floor((t - tS[0]) / stepS + 1e-9), 0), tS.length - 1);

  // a heading band's centre, on the branch of the judged track where the band begins
  const bandCentre = (band: TrainingHeadingBand) =>
    nearestBranch(band.targetDeg, judgedTrack.trackPlotDeg[Math.min(band.firstRow, judgedTrack.trackPlotDeg.length - 1)]);

  // ── each chart's y, over what it draws ──
  const yOf = (low: number, high: number) => (value: number) => PLOT_TOP + ((high - value) / (high - low)) * PLOT_H;
  const bands = envelopes === null || !layers.headingBands ? [] : envelopes.heading.filter(judged);
  const [hLow, hHigh] = extent([
    ...observed.trackPlotDeg, ...(flown?.trackPlotDeg ?? []), ...(live?.track.trackPlotDeg ?? []),
    ...bands.flatMap((band) => [bandCentre(band) - band.toleranceDeg, bandCentre(band) + band.toleranceDeg]),
  ]);
  const tubes = envelopes === null || !layers.vertical ? [] : envelopes.altitude;
  const [aLow, aHigh] = extent([
    ...observed.altitudeMslM, ...(flown?.altitudeMslM ?? []), ...(live?.track.altitudeMslM ?? []), designated.elevationM,
    ...tubes.flatMap((tube) => [...tube.lowMslM, ...tube.highMslM]),
  ]);
  const spans = envelopes === null || !layers.vertical ? [] : envelopes.speed;
  const [sLow, sHigh] = extent([
    ...observed.groundSpeedMps, ...(flown?.groundSpeedMps ?? []), ...(live?.track.groundSpeedMps ?? []),
    ...spans.flatMap((span) => [span.targetMps - span.toleranceMps, span.targetMps + span.toleranceMps]),
  ]);

  /** The words of ``columns`` the closed-loop reading added, at their flight time. */
  const corrections = (...columns: TrainingColumn[]) => reading.events
    .filter((event) => event.correction && columns.includes(event.says.column))
    .map((event) => ({ event, atS: rowTimeS(event.row) }));
  /** The unspecified-speed runs: where the speed word leaves the speed to the pilot. */
  const unspecifiedRuns = sentenceColumnRuns(reading, "speed").filter((run) => run.event.says.column === "speed" && run.event.says.speedMps === null);

  return {
    selection, flight, vocabulary, candidates, layers, reading, cursorS, cursorRow, column, designated, stepS,
    observed, flown, flownColour: flownSentenceColour(flight), judged: judgedTrack, envelopes, closed, decision,
    live, liveColour, focus, focusIndex, focused, recede, focusSpanS, focusPoints, corrections, unspecifiedRuns,
    width, plotW, endS, xTime, timeAtX, envelopeTimeS, rowTimeS,
    px, py, planPoints, rows, indexAt, runwayLine, reach, bandCentre, bands, tubes, spans,
    yHeading: yOf(hLow, hHigh), yAltitude: yOf(aLow, aHigh), ySpeed: yOf(sLow, sHigh),
    /** The word of ``name`` in force at the cursor, short. */
    label: (name: TrainingColumn) => {
      const run = sentenceWordAt(reading, name, cursorRow);
      return run === null ? "—" : trainingBandLabel(run.event.says);
    },
  };
}

export type ReadbackModel = ReturnType<typeof readbackModel>;

const FRACTIONS = [0, 0.25, 0.5, 0.75, 1];

/** The time axis's ticks: seconds at the quarters. */
export function timeTicks(m: ReadbackModel): Array<{ x: number; text: string }> {
  return FRACTIONS.map((fraction) => ({ x: m.xTime(fraction * m.endS), text: formatSeconds(Math.round(fraction * m.endS)) }));
}
