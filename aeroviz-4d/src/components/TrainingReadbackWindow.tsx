/**
 * TrainingReadbackWindow.tsx
 * --------------------------
 * The read-back check: one flight's sentence against its track, envelope by envelope. Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 *  • PLAN VIEW (the airport frame, one scale on both axes): every candidate runway and its
 *    extended centreline, the capture corridor, the capture turn's rows, the track, where each
 *    heading word was issued, the clearance, the capture and the end of the sentence; the rows a
 *    heading word's band judged outside, in red.
 *  • HEADING against time: each heading word's BAND (instruction-v3, vocabulary design §10.1) — its
 *    target ± the heading tolerance over the rows it is judged on, from a lead after it is said to the
 *    next heading word's, never past the clearance — with the rows outside in red; the capture turn's
 *    span onto the course; the course band after the capture.
 *  • ALTITUDE against the horizontal distance flown — the axis the tubes are defined on: each
 *    altitude word's tube, the angle words that re-anchor it, the runway's elevation.
 *  • SPEED against time: each word's transition and band, the "unspecified" spans, the clearance.
 *
 * WHAT IS SELECTED STANDS OUT, THE REST RECEDES: every other word's envelope is drawn faded.
 *
 * WHAT IS YELLOW is the selected word alone (`column`, the class chosen in the sentence bar or by
 * clicking a chart, and its word in force at the cursor): its own envelope's edge, and the rows it is
 * in force drawn over the track and over the one chart that plots its signal. Hovering moves the
 * cursor only; a click on a chart selects that chart's column.
 *
 * EVERY SHAPE IS THE EXPORTER'S. Bands, row verdicts and tubes arrive as numbers computed in Python
 * from the vocabulary's own functions, and every verdict is the labeller's; this window draws them
 * and computes none. The smoothed signal is the bright line (it is what the labeller read), the
 * raw rows are behind it, and red marks rows the labeller counted outside their envelope.
 *
 * THE EXECUTOR'S REPLAY, when it is on (`executor`): its flown track in the plan view and, dashed teal, its
 * heading, altitude and ground speed on its OWN clock and its own distance flown — it flies at its own pace, so
 * the axes are extended to hold both, and nothing is aligned. Its heading words' bands are drawn on its own clock
 * too (teal outlines, from where IT was told each word plus the lead), with its rows outside in red on the flown
 * track as its judge read it; its other words are judged on envelopes re-drawn from where it was told them (the
 * sentence bar's dots).
 *
 * THE EXECUTOR, LIVE, when the backend has flown the selected word's segment (`autopilot`): solid blue, from where the
 * word was said, on the flight's own clock and distance axis — it starts where the observed aircraft was, so its lines
 * begin on the observed ones and part from them as it flies at its own pace. A selected heading word's band is drawn as
 * its judge read it (a blue outline, red when a row is outside), with those rows red on its flown track.
 *
 * It renders through a PORTAL into `document.body` (AV7): `.flight-ops-panel` carries a
 * `backdrop-filter`, which would make a `position: fixed` descendant position against it.
 */

import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { TrainingLayers } from "../context/AppContext";
import {
  TRAINING_AUTOPILOT_COLOR,
  TRAINING_CANDIDATE_COLOR,
  TRAINING_CAPTURE_TURN_COLOR,
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_SPEED_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  formatSeconds,
  rowAtTime,
  trainingBandOutsideSpans,
  trainingKindLabel,
  trainingWordAt,
  trainingWordLabel,
  TRAINING_COLUMN_INDEX,
  type TrainingCandidate,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingHeadingBand,
  type TrainingPlanLine,
  type TrainingVocabulary,
} from "../data/trainingSample";
import type { TrainingExecutorFlight } from "../data/trainingOverlays";
import { autopilotColour, type TrainingAutopilotSegment } from "../data/trainingAutopilot";

const GUTTER = 64;
const PAD_R = 16;
const PLAN_H = 300;
const CHART_H = 150;
const DEFAULT_W = 980;
const MIN_W = 420;
/** Every envelope that is not the selected word's is drawn at this opacity while a word is selected. */
const FADED = 0.3;

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

/** The contiguous runs of `true` in a per-row flag, as inclusive [first, last] pairs. */
export function runsOf(flags: boolean[]): Array<[number, number]> {
  const runs: Array<[number, number]> = [];
  let start: number | null = null;
  flags.forEach((flag, index) => {
    if (flag && start === null) start = index;
    if (!flag && start !== null) {
      runs.push([start, index - 1]);
      start = null;
    }
  });
  if (start !== null) runs.push([start, flags.length - 1]);
  return runs;
}

/** The row whose horizontal distance flown is nearest at or before `metres`. */
export function rowAtDistance(distanceM: number[], metres: number): number {
  return rowAtTime(distanceM, metres);
}

const tick = (ok: boolean) => (ok ? "✓" : "✗");

export interface TrainingReadbackWindowProps {
  flight: TrainingFlight;
  vocabulary: TrainingVocabulary;
  candidates: TrainingCandidate[];
  layers: TrainingLayers;
  cursorS: number;
  onCursorChange: (seconds: number) => void;
  /** The selected word class; its word in force at the cursor is the one drawn yellow. */
  column: TrainingColumn | null;
  onColumnChange: (column: TrainingColumn) => void;
  onClose: () => void;
  /** The executor's replay of this flight, when its overlay is on; null otherwise. */
  executor?: TrainingExecutorFlight | null;
  /** The selected word's segment, flown live (`trainingAutopilot`, ready); null otherwise. */
  autopilot?: TrainingAutopilotSegment | null;
}

export default function TrainingReadbackWindow({
  flight, vocabulary, candidates, layers, cursorS, onCursorChange, column, onColumnChange, onClose, executor = null,
  autopilot = null,
}: TrainingReadbackWindowProps) {
  const frameRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number>(DEFAULT_W);

  useLayoutEffect(() => {
    const node = frameRef.current;
    if (!node) return;
    const measure = () => setWidth(Math.max(node.clientWidth, MIN_W));
    measure();
    if (typeof ResizeObserver === "undefined") return;   // jsdom has no layout to observe
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const { signals, envelopes } = flight;
  const { tS } = signals;
  const last = flight.rows - 1;
  // the executor's flown track, drawn on its own clock (and not at all without two points)
  const flownTrack = executor?.track && executor.track.tS.length >= 2 ? executor.track : null;
  // its heading words' bands, on the flown track as its judge read it
  const flownBands = flownTrack && executor?.judgedTrackDeg
    ? executor.words.flatMap((word) => (word.heading === null ? [] : [word.heading]))
    : [];
  const judgedTrack = flownTrack ? executor?.judgedTrackDeg ?? null : null;
  // the live executor's segment (and not at all without two points); its judged track's step k is the flight's
  // step `row + k`, the flown track's point k × (step / cycle)
  const live = autopilot && autopilot.track.tS.length >= 2 ? autopilot : null;
  // blue, or red when the selected word flew outside its envelope
  const liveColour = live === null ? TRAINING_AUTOPILOT_COLOR : autopilotColour(live);
  const liveTrack = live?.track ?? null;
  const liveBand = live?.word.heading ?? null;
  const liveJudged = live?.judgedTrackDeg ?? null;
  const liveStepRows = live ? Math.round(vocabulary.stepS / live.executor.cycleS) : 1;
  /** The live segment's rows outside its heading band, as [first, last] judged steps. */
  const liveOutside = live && liveBand && liveJudged && layers.headingBands
    ? trainingBandOutsideSpans(liveBand, live.segment.row + liveJudged.length - 1)
      .map(([first, lastRow]) => [first - live.segment.row, lastRow - live.segment.row] as [number, number])
    : [];
  const plotW = width - GUTTER - PAD_R;
  const cursorRow = rowAtTime(tS, cursorS);
  const designated = candidates[flight.runwayIndex];
  const label = (column: "heading" | "altitude" | "angle" | "speed", value: number) =>
    trainingWordLabel(vocabulary, candidates, column, value);
  const inForce = (name: "heading" | "altitude" | "angle" | "speed") =>
    flight.words.inForce[TRAINING_COLUMN_INDEX[name]][cursorRow];
  // ── the selected word: one column's, never the step's ────────────────────
  const focus = column === null ? null : trainingWordAt(flight, column, cursorRow);
  /** Is this the selected word — the `index`-th word (and envelope) of column `name`? */
  const focused = (name: TrainingColumn, index: number) => column === name && focus!.index === index;
  // The clearance owns the approach's envelopes: the capture turn and the corridor.
  const approachFocused = column === "approach" && focus!.event.kind === "clear";
  const capture = envelopes.approach.captureTurn;
  const captureOk = capture !== null && capture.check.progressOk && capture.check.rateOk;
  /** An envelope's opacity: full for the selected word's, or for every word when none is selected. */
  const recede = (mine: boolean) => (column === null || mine ? 1 : FADED);
  /** A band has rows to draw. */
  const judged = (band: TrainingHeadingBand) => band.stopRow > band.firstRow;

  // ── time charts share one x: the observed flight's, or longer when the executor flew longer ──
  const endS = Math.max(tS[last], flownTrack ? flownTrack.tS[flownTrack.tS.length - 1] : 0,
    liveTrack ? liveTrack.tS[liveTrack.tS.length - 1] : 0);
  const xTime = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  // The cursor is the observed flight's time: past its end (where only the executor's lines run) it stays at the end.
  const timeAtX = (x: number) => Math.min(Math.max(((x - GUTTER) / plotW) * endS, 0), tS[last]);
  /** A row as a time edge; the step after the last row is the last row. */
  const edge = (row: number) => tS[Math.min(row, last)];

  // ── the plan view: the track and the threshold decide the frame ──
  const km = (metres: number) => metres / 1000;
  const frameE = [...signals.eM, designated.thresholdEM, ...envelopes.approach.corridor.axis.eM, ...(flownTrack?.eM ?? []),
    ...(liveTrack?.eM ?? [])].map(km);
  const frameN = [...signals.nM, designated.thresholdNM, ...envelopes.approach.corridor.axis.nM, ...(flownTrack?.nM ?? []),
    ...(liveTrack?.nM ?? [])].map(km);
  const [eLow, eHigh] = extent(frameE);
  const [nLow, nHigh] = extent(frameN);
  const planScale = Math.min((plotW - 12) / (eHigh - eLow), (PLAN_H - 30) / (nHigh - nLow));
  // one scale on both axes, so the frame is centred in whichever direction has room to spare
  const planLeft = GUTTER + 6 + (plotW - 12 - (eHigh - eLow) * planScale) / 2;
  const planTop = 20 + (PLAN_H - 30 - (nHigh - nLow) * planScale) / 2;
  const px = (eKm: number) => planLeft + (eKm - eLow) * planScale;
  const py = (nKm: number) => planTop + (nHigh - nKm) * planScale;
  const points = (line: TrainingPlanLine) =>
    line.eM.map((e, index) => `${px(km(e))},${py(km(line.nM[index]))}`).join(" ");
  const at = (row: number) => ({ x: px(km(signals.eM[row])), y: py(km(signals.nM[row])) });
  /** Rows first..last (inclusive) of the observed track, in plan. */
  const planRows = (first: number, lastRow: number) =>
    Array.from({ length: lastRow - first + 1 }, (_, offset) => `${at(first + offset).x},${at(first + offset).y}`).join(" ");
  /** Rows first..last (inclusive) of the executor's flown track, in plan. */
  const flownPlanRows = (first: number, lastRow: number) =>
    Array.from({ length: lastRow - first + 1 }, (_, offset) =>
      `${px(km(flownTrack!.eM[first + offset]))},${py(km(flownTrack!.nM[first + offset]))}`).join(" ");

  // ── the charts' y ────────────────────────────────────────────────────────
  const headingValues = [
    ...signals.smoothed.trackDeg, ...signals.raw.trackDeg,
    ...(layers.headingBands ? envelopes.heading.flatMap((item) => (judged(item) ? item.bandDeg : [])) : []),
    ...(layers.corridor ? [...envelopes.approach.courseBandDeg, ...(capture ? [capture.courseOnTrackDeg] : [])] : []),
    ...(flownTrack?.trackDeg ?? []), ...(judgedTrack ?? []),
    ...(layers.headingBands ? flownBands.flatMap((band) => (judged(band) ? band.bandDeg : [])) : []),
    ...(liveTrack?.trackDeg ?? []), ...(liveJudged ?? []),
    ...(layers.headingBands && liveBand && judged(liveBand) ? liveBand.bandDeg : []),
  ];
  const [hLow, hHigh] = extent(headingValues);
  const altitudeValues = [
    ...signals.smoothed.altitudeM, ...signals.raw.altitudeM, designated.elevationM,
    ...(layers.vertical ? envelopes.altitude.flatMap((tube) => [...tube.lowerM, ...tube.upperM]) : []),
    ...(flownTrack?.altitudeM ?? []), ...(liveTrack?.altitudeM ?? []),
  ];
  const [aLow, aHigh] = extent(altitudeValues);
  const speedValues = [
    ...signals.smoothed.groundSpeedMps, ...signals.raw.groundSpeedMps,
    ...(layers.vertical ? envelopes.speed.flatMap((span) => [...(span.bandMps ?? []), ...(span.transitionLowerMps ?? []), ...(span.transitionUpperMps ?? [])]) : []),
    ...(flownTrack?.groundSpeedMps ?? []), ...(liveTrack?.groundSpeedMps ?? []),
  ];
  const [sLow, sHigh] = extent(speedValues);
  const plotTop = 18;
  // room below the plot for the tick labels and the axis caption
  const plotH = CHART_H - 48;
  const yOf = (low: number, high: number) => (value: number) => plotTop + ((high - value) / (high - low)) * plotH;
  const yHeading = yOf(hLow, hHigh);
  const yAltitude = yOf(aLow, aHigh);
  const ySpeed = yOf(sLow, sHigh);
  const distanceEnd = Math.max(signals.smoothed.distanceM[last],
    flownTrack ? flownTrack.distanceM[flownTrack.distanceM.length - 1] : 0,
    liveTrack ? liveTrack.distanceM[liveTrack.distanceM.length - 1] : 0);
  const xDistance = (metres: number) => GUTTER + (metres / distanceEnd) * plotW;
  const distanceAtX = (x: number) => Math.min(Math.max(((x - GUTTER) / plotW) * distanceEnd, 0), distanceEnd);

  // The chart titles read out the words in force at the cursor, whichever column is selected.
  const heading = envelopes.heading[trainingWordAt(flight, "heading", cursorRow).index];
  const tube = envelopes.altitude[trainingWordAt(flight, "altitude", cursorRow).index];
  const inCaptureTurn = capture !== null && capture.startRow <= cursorRow && cursorRow <= capture.endRow;

  const timeAxis = (
    <g>
      <line x1={GUTTER} x2={GUTTER + plotW} y1={plotTop + plotH} y2={plotTop + plotH} className="training-readback-axis" />
      {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
        <text key={fraction} x={xTime(fraction * endS)} y={plotTop + plotH + 14} textAnchor="middle" className="training-readback-tick">
          {formatSeconds(Math.round(fraction * endS))}
        </text>
      ))}
      <text x={GUTTER + plotW} y={plotTop + plotH + 26} textAnchor="end" className="training-readback-tick">s from the first step</text>
    </g>
  );
  const trace = (values: number[], x: (row: number) => number, y: (value: number) => number, colour: string, faint: boolean) => (
    <polyline
      points={values.map((value, row) => `${x(row)},${y(value)}`).join(" ")}
      fill="none"
      stroke={colour}
      strokeOpacity={faint ? 0.4 : 1}
      strokeWidth={faint ? 1 : 1.4}
      className={faint ? "training-readback-raw" : "training-readback-trace"}
    />
  );
  const rowX = (row: number) => xTime(tS[row]);
  const rowXDistance = (row: number) => xDistance(signals.smoothed.distanceM[row]);
  /** The executor's flown values against its own time (or distance): one dashed teal line. */
  const flownTrace = (along: number[], values: number[], xOf: (value: number) => number, y: (value: number) => number, name: string) => (
    <polyline points={values.map((value, index) => `${xOf(along[index])},${y(value)}`).join(" ")} fill="none"
      stroke={TRAINING_EXECUTOR_COLOR} strokeWidth={1.4} strokeDasharray="5 3" className="training-readback-executor">
      <title>{name}</title>
    </polyline>
  );
  /** The live segment's values against the flight's time (or distance): one solid line in its verdict's colour. */
  const liveTrace = (along: number[], values: number[], xOf: (value: number) => number, y: (value: number) => number, name: string) => (
    <polyline points={values.map((value, index) => `${xOf(along[index])},${y(value)}`).join(" ")} fill="none"
      stroke={liveColour} strokeWidth={1.8} className="training-readback-autopilot">
      <title>{name}</title>
    </polyline>
  );
  /** A flown step's time; the step after the flown track's last is its last. */
  const flownEdge = (step: number) => flownTrack!.tS[Math.min(step, flownTrack!.tS.length - 1)];
  // The rows the selected word is in force; the stretch runs on to the next word's issue so that it
  // meets it. Only a word issued on the last row has no stretch.
  const focusRows = focus === null
    ? []
    : Array.from({ length: Math.min(focus.endRow, last) - focus.row + 1 }, (_, offset) => focus.row + offset);
  const focusTrace = (values: number[], x: (row: number) => number, y: (value: number) => number) =>
    focusRows.length < 2 ? null : (
      <polyline
        points={focusRows.map((row) => `${x(row)},${y(values[row])}`).join(" ")}
        fill="none" stroke={TRAINING_WORD_COLOR} strokeWidth={2.6} strokeOpacity={0.9} strokeLinecap="round"
        className="training-readback-focus"
      />
    );
  /** The title's reading of the heading word in force: its rows and their verdict. */
  const headingReadout = judged(heading)
    ? ` · its band: steps ${heading.firstRow}–${heading.stopRow - 1} (${vocabulary.headingLeadS} s after it was said), ` +
      `${heading.check.inside}/${heading.check.rows} rows within ±${vocabulary.headingToleranceDeg}°`
    : ` · no row of its own: the ${vocabulary.headingLeadS} s lead carries it to the clearance`;

  return createPortal(
    <div className="training-readback-backdrop">
      <div
        className="training-readback-window"
        role="dialog"
        aria-label="Read-back check"
        aria-modal="false"
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") onClose();
        }}
      >
        <header className="training-readback-head">
          <strong>Read-back check</strong>
          <span>{flight.callsign}</span>
          <span>runway {flight.runway}</span>
          <span>{flight.stratum}</span>
          <span className="training-readback-cursor">t = {formatSeconds(cursorS)} s · step {cursorRow}</span>
          <button type="button" onClick={onClose} aria-label="Close the read-back check">×</button>
        </header>

        <div className="training-readback-frame" ref={frameRef}>
          {/* ── the plan view ─────────────────────────────────────────────── */}
          <svg className="training-readback-svg" width={width} height={PLAN_H} viewBox={`0 0 ${width} ${PLAN_H}`}
            aria-label="Plan view">
            <defs>
              <clipPath id="training-plan-clip">
                <rect x={GUTTER} y={14} width={plotW} height={PLAN_H - 18} />
              </clipPath>
            </defs>
            <text x={GUTTER} y={11} className="training-readback-title">
              plan view · airport frame, km east × km north of the reference point · framed on the track
            </text>
            <g clipPath="url(#training-plan-clip)">
              {candidates.map((candidate) => {
                const pointed = candidate.index === flight.runwayIndex;
                if (!pointed && !layers.candidates) return null;
                const colour = pointed ? TRAINING_DESIGNATED_COLOR : TRAINING_CANDIDATE_COLOR;
                const stroke = column === "runway" && focus!.value === candidate.index ? TRAINING_WORD_COLOR : colour;
                const name = `${pointed ? "the designated runway" : "candidate"} ${candidate.ident}: course ` +
                  `${candidate.courseDeg.toFixed(1)}°, threshold ${candidate.elevationM.toFixed(1)} m MSL`;
                return (
                  <g key={`candidate-${candidate.ident}`} className="training-readback-candidate" aria-label={name}>
                    <title>{name}</title>
                    <polyline points={points(candidate.centreline)} fill="none" stroke={stroke} strokeDasharray="5 4"
                      strokeOpacity={pointed ? 0.9 : 0.5} strokeWidth={1} />
                    <polyline points={points(candidate.runway)} fill="none" stroke={stroke} strokeWidth={4}
                      strokeOpacity={pointed ? 1 : 0.6} />
                    <text x={px(km(candidate.thresholdEM)) + 4} y={py(km(candidate.thresholdNM)) - 4}
                      className="training-readback-candidate-label" fill={colour}>
                      {candidate.ident}
                    </text>
                  </g>
                );
              })}

              {layers.corridor ? (
                  <polygon
                    className="training-readback-corridor"
                    points={points(envelopes.approach.corridor.outline)}
                    fill={TRAINING_CORRIDOR_COLOR}
                    fillOpacity={approachFocused ? 0.4 : 0.2}
                    stroke={approachFocused ? TRAINING_WORD_COLOR : TRAINING_CORRIDOR_COLOR}
                    strokeWidth={approachFocused ? 1.4 : 0.8}
                    opacity={recede(approachFocused)}
                  >
                    <title>
                      the capture corridor: {envelopes.approach.corridor.halfWidthAtCaptureM.toFixed(0)} m half width at the
                      capture ({(envelopes.approach.corridor.beforeThresholdM / 1000).toFixed(1)} km out),{" "}
                      {envelopes.approach.corridor.halfWidthAtThresholdM.toFixed(0)} m at the threshold, course ±
                      {vocabulary.corridorCourseToleranceDeg}° — every one of its {envelopes.approach.corridor.rows} rows inside
                    </title>
                  </polygon>
              ) : null}

              <polyline points={signals.eM.map((e, row) => `${px(km(e))},${py(km(signals.nM[row]))}`).join(" ")}
                fill="none" stroke={TRAINING_TRACE_COLOR} strokeWidth={1.4} className="training-readback-trace" />
              {/* the capture turn: its rows on the track, from the clearance to the capture */}
              {layers.corridor && capture !== null && capture.endRow > capture.startRow ? (
                <polyline
                  className="training-readback-capture-turn"
                  points={planRows(capture.startRow, capture.endRow)}
                  fill="none"
                  stroke={approachFocused ? TRAINING_WORD_COLOR : captureOk ? TRAINING_CAPTURE_TURN_COLOR : TRAINING_OUTSIDE_COLOR}
                  strokeWidth={approachFocused ? 3 : 2.2}
                  strokeDasharray="4 3"
                  opacity={recede(approachFocused)}
                >
                  <title>the capture turn, steps {capture.startRow}–{capture.endRow}, onto the course {designated.courseDeg.toFixed(1)}°</title>
                </polyline>
              ) : null}
              {/* the rows a heading word's band judged outside */}
              {layers.headingBands ? envelopes.heading.flatMap((item, index) => trainingBandOutsideSpans(item, last).map(([first, lastRow]) => (
                <polyline key={`plan-heading-out-${index}-${first}`} className="training-readback-outside"
                  points={planRows(first, lastRow)} fill="none" stroke={TRAINING_OUTSIDE_COLOR}
                  strokeWidth={2.4} strokeLinecap="round" />
              ))) : null}
              {flownTrack ? (
                <g aria-label="the executor's flown track">
                  <polyline points={flownTrack.eM.map((e, index) => `${px(km(e))},${py(km(flownTrack.nM[index]))}`).join(" ")}
                    fill="none" stroke={TRAINING_EXECUTOR_COLOR} strokeWidth={1.4} strokeDasharray="5 3"
                    className="training-readback-executor">
                    <title>the executor's flown track — the truth sentence flown from row 0</title>
                  </polyline>
                  {layers.headingBands && judgedTrack ? flownBands.flatMap((band, index) =>
                    trainingBandOutsideSpans(band, judgedTrack.length - 1).map(([first, lastStep]) => (
                      <polyline key={`plan-executor-out-${index}-${first}`} className="training-readback-executor-outside"
                        points={flownPlanRows(first, lastStep)} fill="none"
                        stroke={TRAINING_OUTSIDE_COLOR} strokeWidth={2.4} strokeLinecap="round" />
                    ))) : null}
                  <rect x={px(km(flownTrack.eM[flownTrack.eM.length - 1])) - 3.5} y={py(km(flownTrack.nM[flownTrack.nM.length - 1])) - 3.5}
                    width={7} height={7} fill={TRAINING_EXECUTOR_COLOR} stroke="black" strokeWidth={0.6} />
                  <text x={px(km(flownTrack.eM[flownTrack.eM.length - 1])) + 6} y={py(km(flownTrack.nM[flownTrack.nM.length - 1])) + 12}
                    className="training-readback-path-label" fill={TRAINING_EXECUTOR_COLOR}>
                    executor: {(executor!.outcome ?? "").replace(/_/g, " ")}
                  </text>
                </g>
              ) : null}
              {focusRows.length >= 2 ? (
                <polyline points={focusRows.map((row) => `${at(row).x},${at(row).y}`).join(" ")} fill="none"
                  stroke={TRAINING_WORD_COLOR} strokeWidth={3} strokeOpacity={0.9} strokeLinecap="round"
                  className="training-readback-focus" />
              ) : null}
              {liveTrack ? (
                <g aria-label="the autopilot's flown segment">
                  <polyline points={liveTrack.eM.map((e, index) => `${px(km(e))},${py(km(liveTrack.nM[index]))}`).join(" ")}
                    fill="none" stroke={liveColour} strokeWidth={1.8} className="training-readback-autopilot">
                    <title>the autopilot's flown segment — from where the selected word was said, flown live</title>
                  </polyline>
                  {liveOutside.map(([first, lastStep]) => (
                    <polyline key={`plan-autopilot-out-${first}`} className="training-readback-autopilot-outside"
                      points={Array.from({ length: (lastStep - first) * liveStepRows + 1 }, (_, offset) => first * liveStepRows + offset)
                        .map((index) => `${px(km(liveTrack.eM[index]))},${py(km(liveTrack.nM[index]))}`).join(" ")}
                      fill="none" stroke={TRAINING_OUTSIDE_COLOR} strokeWidth={2.4} strokeLinecap="round" />
                  ))}
                  <circle cx={px(km(liveTrack.eM[liveTrack.eM.length - 1]))} cy={py(km(liveTrack.nM[liveTrack.nM.length - 1]))} r={4}
                    fill={liveColour} stroke="black" strokeWidth={0.6} />
                </g>
              ) : null}
              {envelopes.heading.map((item, index) => {
                const point = at(item.row);
                const name = `heading ${label("heading", item.value)} issued at step ${item.row} — ${trainingKindLabel(item.kind)}`;
                return (
                  <g key={`issue-${index}`} className="training-readback-issue" aria-label={name}>
                    <title>{name}</title>
                    <circle cx={point.x} cy={point.y} r={2.6} fill="none" stroke={TRAINING_COLUMN_COLOR.heading} strokeWidth={1.3} />
                  </g>
                );
              })}
              <g aria-label={`cleared at step ${flight.joinRow}`}>
                <title>cleared to join the final at step {flight.joinRow}</title>
                <rect x={at(flight.joinRow).x - 4} y={at(flight.joinRow).y - 4} width={8} height={8}
                  transform={`rotate(45 ${at(flight.joinRow).x} ${at(flight.joinRow).y})`}
                  fill="none" stroke={TRAINING_COLUMN_COLOR.approach} strokeWidth={1.6} />
              </g>
              <g aria-label={`captured at step ${flight.captureRow}`}>
                <title>
                  the final captured at step {flight.captureRow}, {(flight.captureBeforeThresholdM / 1000).toFixed(1)} km before
                  the threshold
                </title>
                <rect x={at(flight.captureRow).x - 4} y={at(flight.captureRow).y - 4} width={8} height={8}
                  fill={TRAINING_CORRIDOR_COLOR} stroke="black" strokeWidth={0.6} />
              </g>
              <g aria-label="the end of the sentence">
                <title>
                  the sentence ends here, {(envelopes.approach.landing.lastRowBeforeThresholdM / 1000).toFixed(2)} km before the
                  threshold{envelopes.approach.landing.cutAtCrossing ? " (cut before the last passage of the threshold)" : ""}
                </title>
                <polygon
                  points={`${at(last).x},${at(last).y - 5} ${at(last).x - 4},${at(last).y + 3} ${at(last).x + 4},${at(last).y + 3}`}
                  fill={TRAINING_TRACE_COLOR}
                />
              </g>
              <circle cx={at(cursorRow).x} cy={at(cursorRow).y} r={4.5} className="training-readback-cursor-dot" />
            </g>
          </svg>

          {/* ── heading ───────────────────────────────────────────────────── */}
          <svg className="training-readback-svg" width={width} height={CHART_H} viewBox={`0 0 ${width} ${CHART_H}`}
            aria-label="Heading chart"
            onMouseMove={(event) => onCursorChange(timeAtX(event.nativeEvent.offsetX))}
            onClick={(event) => {
              onCursorChange(timeAtX(event.nativeEvent.offsetX));
              onColumnChange("heading");
            }}>
            <text x={GUTTER} y={11} className="training-readback-title">
              heading — ground track, ° true, unwrapped · in force: {label("heading", inForce("heading"))}
              {headingReadout}
              {inCaptureTurn
                ? ` · the capture turn: ${tick(capture.check.progressOk)} monotone, ${tick(capture.check.rateOk)} rate ` +
                  `(mean ${capture.check.meanRateDegS.toFixed(2)}°/s, max ${capture.check.maxRateDegS.toFixed(2)}°/s, ` +
                  `max bank ${capture.check.maxBankDeg.toFixed(1)}°` +
                  `${capture.check.rateMinApplies ? "" : `; under ${vocabulary.turnRateMinFromDeg}°, the lowest rate not judged`})`
                : ""}
              {cursorRow >= flight.captureRow ? " · captured: the corridor holds" : ""}
            </text>
            <defs>
              <clipPath id="training-heading-clip">
                <rect x={GUTTER} y={plotTop} width={plotW} height={plotH} />
              </clipPath>
            </defs>
            <g clipPath="url(#training-heading-clip)">
              {layers.headingBands ? envelopes.heading.map((item, index) => {
                if (!judged(item)) return null;
                const selected = focused("heading", index);
                return (
                  <rect key={`heading-band-${index}`} className="training-readback-heading-band"
                    x={rowX(item.firstRow)} width={Math.max(xTime(edge(item.stopRow)) - rowX(item.firstRow), 1)}
                    y={yHeading(item.bandDeg[1])} height={yHeading(item.bandDeg[0]) - yHeading(item.bandDeg[1])}
                    fill={TRAINING_HEADING_BAND_COLOR} fillOpacity={selected ? 0.34 : 0.16}
                    stroke={selected ? TRAINING_WORD_COLOR : item.check.inside < item.check.rows ? TRAINING_OUTSIDE_COLOR : TRAINING_HEADING_BAND_COLOR}
                    strokeWidth={selected ? 1.4 : 0.6} opacity={recede(selected)}>
                    <title>
                      {label("heading", item.value)} ±{vocabulary.headingToleranceDeg}°, said at step {item.row} and judged at
                      steps {item.firstRow}–{item.stopRow - 1} ({vocabulary.headingLeadS} s later, to the next heading word's):{" "}
                      {item.check.inside} of {item.check.rows} rows inside
                    </title>
                  </rect>
                );
              }) : null}
              {layers.corridor && capture !== null ? (
                <g aria-label="the capture turn, against time" opacity={recede(approachFocused)}>
                  <rect className="training-readback-capture-band"
                    x={rowX(capture.startRow)} width={Math.max(rowX(capture.endRow) - rowX(capture.startRow), 1)}
                    y={plotTop} height={plotH}
                    fill={TRAINING_CAPTURE_TURN_COLOR} fillOpacity={approachFocused ? 0.18 : 0.08}
                    stroke={approachFocused ? TRAINING_WORD_COLOR : captureOk ? TRAINING_CAPTURE_TURN_COLOR : TRAINING_OUTSIDE_COLOR}
                    strokeDasharray="4 3" strokeWidth={0.9}>
                    <title>
                      the capture turn, steps {capture.startRow}–{capture.endRow}: from the clearance onto the course, monotone{" "}
                      {tick(capture.check.progressOk)}, rate and bank {tick(capture.check.rateOk)}
                    </title>
                  </rect>
                  <line className="training-readback-capture-course"
                    x1={rowX(capture.startRow)} x2={Math.max(rowX(capture.endRow), rowX(capture.startRow) + 1)}
                    y1={yHeading(capture.courseOnTrackDeg)} y2={yHeading(capture.courseOnTrackDeg)}
                    stroke={approachFocused ? TRAINING_WORD_COLOR : TRAINING_CAPTURE_TURN_COLOR} strokeDasharray="2 2" />
                </g>
              ) : null}
              {layers.corridor ? (
                <rect
                  className="training-readback-course-band"
                  x={rowX(flight.captureRow)} width={Math.max(rowX(last) - rowX(flight.captureRow), 1)}
                  y={yHeading(envelopes.approach.courseBandDeg[1])}
                  height={yHeading(envelopes.approach.courseBandDeg[0]) - yHeading(envelopes.approach.courseBandDeg[1])}
                  fill={TRAINING_CORRIDOR_COLOR} fillOpacity={approachFocused ? 0.36 : 0.22}
                  stroke={approachFocused ? TRAINING_WORD_COLOR : "none"} strokeWidth={0.7}
                  opacity={recede(approachFocused)}
                >
                  <title>after the capture: the course ±{vocabulary.corridorCourseToleranceDeg}°</title>
                </rect>
              ) : null}
              {/* the executor's heading bands, on its own clock, from where IT was told each word */}
              {layers.headingBands ? flownBands.map((band, index) => judged(band) ? (
                <rect key={`executor-band-${index}`} className="training-readback-executor-band"
                  x={xTime(flownEdge(band.firstRow))} width={Math.max(xTime(flownEdge(band.stopRow)) - xTime(flownEdge(band.firstRow)), 1)}
                  y={yHeading(band.bandDeg[1])} height={yHeading(band.bandDeg[0]) - yHeading(band.bandDeg[1])}
                  fill="none" stroke={band.inside.every(Boolean) ? TRAINING_EXECUTOR_COLOR : TRAINING_OUTSIDE_COLOR} strokeWidth={0.8} />
              ) : null) : null}
              {/* the live segment's heading band, as its judge read it */}
              {layers.headingBands && liveBand && judged(liveBand) ? (
                <rect className="training-readback-autopilot-band"
                  x={xTime(liveBand.firstRow * vocabulary.stepS)}
                  width={Math.max(xTime(liveBand.stopRow * vocabulary.stepS) - xTime(liveBand.firstRow * vocabulary.stepS), 1)}
                  y={yHeading(liveBand.bandDeg[1])} height={yHeading(liveBand.bandDeg[0]) - yHeading(liveBand.bandDeg[1])}
                  fill="none" stroke={liveBand.inside.every(Boolean) ? TRAINING_AUTOPILOT_COLOR : TRAINING_OUTSIDE_COLOR}
                  strokeWidth={1.1}>
                  <title>
                    the autopilot's band for the selected heading word, judged at steps {liveBand.firstRow}–{liveBand.stopRow - 1}
                    {" "}of its flown segment: {liveBand.inside.filter(Boolean).length} of {liveBand.inside.length} rows inside
                  </title>
                </rect>
              ) : null}
            </g>
            {trace(signals.raw.trackDeg, rowX, yHeading, TRAINING_RAW_COLOR, true)}
            {trace(signals.smoothed.trackDeg, rowX, yHeading, TRAINING_TRACE_COLOR, false)}
            {flownTrack ? flownTrace(flownTrack.tS, flownTrack.trackDeg, xTime, yHeading, "the executor's track, on its own clock") : null}
            {live && liveJudged ? liveOutside.map(([first, lastStep]) => (
              <polyline key={`autopilot-out-${first}`} className="training-readback-autopilot-outside"
                points={Array.from({ length: lastStep - first + 1 }, (_, offset) => first + offset)
                  .map((step) => `${xTime((live.segment.row + step) * vocabulary.stepS)},${yHeading(liveJudged[step])}`).join(" ")}
                fill="none" stroke={TRAINING_OUTSIDE_COLOR} strokeWidth={2} strokeDasharray="3 2" />
            )) : null}
            {column === "heading" ? focusTrace(signals.smoothed.trackDeg, rowX, yHeading) : null}
            {liveTrack ? liveTrace(liveTrack.tS, liveTrack.trackDeg, xTime, yHeading, "the autopilot's track over the selected segment") : null}
            {/* the rows each band judged outside: the observed track's, and the executor's as its judge read it */}
            {layers.headingBands ? envelopes.heading.flatMap((item, index) => trainingBandOutsideSpans(item, last).map(([first, lastRow]) => (
              <polyline key={`heading-out-${index}-${first}`} className="training-readback-outside"
                points={Array.from({ length: lastRow - first + 1 }, (_, offset) => first + offset)
                  .map((row) => `${rowX(row)},${yHeading(signals.smoothed.trackDeg[row])}`).join(" ")}
                fill="none" stroke={TRAINING_OUTSIDE_COLOR} strokeWidth={2} />
            ))) : null}
            {layers.headingBands && judgedTrack ? flownBands.flatMap((band, index) =>
              trainingBandOutsideSpans(band, judgedTrack.length - 1).map(([first, lastStep]) => (
                <polyline key={`executor-out-${index}-${first}`} className="training-readback-executor-outside"
                  points={Array.from({ length: lastStep - first + 1 }, (_, offset) => first + offset)
                    .map((step) => `${xTime(flownTrack!.tS[step])},${yHeading(judgedTrack[step])}`).join(" ")}
                  fill="none" stroke={TRAINING_OUTSIDE_COLOR} strokeWidth={2} strokeDasharray="3 2" />
              ))) : null}
            {envelopes.heading.map((item, index) => (
              <circle key={`heading-issue-${index}`} cx={rowX(item.row)} cy={yHeading(signals.smoothed.trackDeg[item.row])} r={2.6}
                fill="none" stroke={TRAINING_COLUMN_COLOR.heading} strokeWidth={1.3} />
            ))}
            <line x1={rowX(flight.joinRow)} x2={rowX(flight.joinRow)} y1={plotTop} y2={plotTop + plotH}
              stroke={TRAINING_COLUMN_COLOR.approach} strokeDasharray="3 3" />
            <line x1={xTime(cursorS)} x2={xTime(cursorS)} y1={plotTop} y2={plotTop + plotH} className="training-readback-cursor-line" />
            {timeAxis}
          </svg>

          {/* ── altitude against distance flown ───────────────────────────── */}
          <svg className="training-readback-svg" width={width} height={CHART_H} viewBox={`0 0 ${width} ${CHART_H}`}
            aria-label="Altitude chart"
            onMouseMove={(event) => onCursorChange(tS[rowAtDistance(signals.smoothed.distanceM, distanceAtX(event.nativeEvent.offsetX))])}
            onClick={(event) => {
              onCursorChange(tS[rowAtDistance(signals.smoothed.distanceM, distanceAtX(event.nativeEvent.offsetX))]);
              onColumnChange("altitude");
            }}>
            <text x={GUTTER} y={11} className="training-readback-title">
              altitude — geometric MSL (m) against distance flown · in force: {label("altitude", inForce("altitude"))},{" "}
              {label("angle", inForce("angle"))}
              {` · this tube: ${tube.check.inside}/${tube.check.rows} rows inside ${tick(tube.check.contained)}`}
            </text>
            {layers.vertical ? envelopes.altitude.map((item, index) => {
              const rows = item.lowerM.map((_, offset) => item.row + offset);
              const selected = focused("altitude", index);
              return (
                <polygon
                  key={`tube-${index}`}
                  className="training-readback-tube"
                  points={[
                    ...rows.map((row, offset) => `${rowXDistance(row)},${yAltitude(item.upperM[offset])}`),
                    ...rows.map((row, offset) => `${rowXDistance(row)},${yAltitude(item.lowerM[offset])}`).reverse(),
                  ].join(" ")}
                  fill={TRAINING_TUBE_COLOR}
                  fillOpacity={selected ? 0.34 : 0.16}
                  stroke={selected ? TRAINING_WORD_COLOR : item.check.contained ? TRAINING_TUBE_COLOR : TRAINING_OUTSIDE_COLOR}
                  strokeWidth={selected ? 1.2 : 0.7}
                  opacity={recede(selected)}
                >
                  <title>
                    {label("altitude", item.value)} from step {item.row}: the tube ±{vocabulary.altitudeToleranceM} m, re-anchored at
                    each angle word, {item.check.tubeWidthEndM.toFixed(0)} m wide at its end — {item.check.inside} of {item.check.rows} rows
                    inside
                  </title>
                </polygon>
              );
            }) : null}
            <line x1={GUTTER} x2={GUTTER + plotW} y1={yAltitude(designated.elevationM)} y2={yAltitude(designated.elevationM)}
              stroke={TRAINING_DESIGNATED_COLOR} strokeDasharray="2 3" className="training-readback-elevation" />
            <text x={GUTTER + plotW - 2} y={yAltitude(designated.elevationM) - 3} textAnchor="end" className="training-readback-tick"
              fill={TRAINING_DESIGNATED_COLOR}>
              threshold {designated.ident} {designated.elevationM.toFixed(1)} m
            </text>
            {envelopes.angle.map((item, index) => (
              <g key={`angle-${index}`} aria-label={`angle word ${label("angle", item.value)} at step ${item.row}`}
                opacity={recede(focused("angle", index))}>
                <line x1={rowXDistance(item.row)} x2={rowXDistance(item.row)} y1={plotTop} y2={plotTop + plotH}
                  stroke={focused("angle", index) ? TRAINING_WORD_COLOR : TRAINING_COLUMN_COLOR.angle}
                  strokeOpacity={focused("angle", index) ? 1 : 0.6} strokeDasharray="2 2" />
                <text x={rowXDistance(item.row) + 2} y={plotTop + 9} className="training-readback-tick"
                  fill={focused("angle", index) ? TRAINING_WORD_COLOR : TRAINING_COLUMN_COLOR.angle}>
                  {vocabulary.angleClasses[item.value].name}
                  {item.measuredDeg === null ? "" : ` ${item.measuredDeg.toFixed(2)}°`}
                </text>
              </g>
            ))}
            {trace(signals.raw.altitudeM, rowXDistance, yAltitude, TRAINING_RAW_COLOR, true)}
            {trace(signals.smoothed.altitudeM, rowXDistance, yAltitude, TRAINING_TRACE_COLOR, false)}
            {flownTrack ? flownTrace(flownTrack.distanceM, flownTrack.altitudeM, xDistance, yAltitude, "the executor's altitude, against its own distance flown") : null}
            {column === "altitude" || column === "angle" ? focusTrace(signals.smoothed.altitudeM, rowXDistance, yAltitude) : null}
            {liveTrack ? liveTrace(liveTrack.distanceM, liveTrack.altitudeM, xDistance, yAltitude,
              "the autopilot's altitude over the selected segment, against the distance flown") : null}
            {envelopes.altitude.flatMap((item, index) =>
              runsOf(item.inside.map((ok) => !ok)).map(([first, lastOut]) => (
                <polyline
                  key={`altitude-out-${index}-${first}`}
                  className="training-readback-outside"
                  points={signals.smoothed.altitudeM.slice(item.row + first, item.row + lastOut + 1)
                    .map((value, offset) => `${rowXDistance(item.row + first + offset)},${yAltitude(value)}`).join(" ")}
                  fill="none" stroke={TRAINING_OUTSIDE_COLOR} strokeWidth={2}
                />
              )))}
            <line x1={rowXDistance(cursorRow)} x2={rowXDistance(cursorRow)} y1={plotTop} y2={plotTop + plotH}
              className="training-readback-cursor-line" />
            <line x1={GUTTER} x2={GUTTER + plotW} y1={plotTop + plotH} y2={plotTop + plotH} className="training-readback-axis" />
            {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
              <text key={fraction} x={xDistance(fraction * distanceEnd)} y={plotTop + plotH + 14} textAnchor="middle"
                className="training-readback-tick">
                {((fraction * distanceEnd) / 1000).toFixed(1)}
              </text>
            ))}
            <text x={GUTTER + plotW} y={plotTop + plotH + 26} textAnchor="end" className="training-readback-tick">
              km flown (the smoothed ground speed, integrated)
            </text>
          </svg>

          {/* ── speed ─────────────────────────────────────────────────────── */}
          <svg className="training-readback-svg" width={width} height={CHART_H} viewBox={`0 0 ${width} ${CHART_H}`}
            aria-label="Speed chart"
            onMouseMove={(event) => onCursorChange(timeAtX(event.nativeEvent.offsetX))}
            onClick={(event) => {
              onCursorChange(timeAtX(event.nativeEvent.offsetX));
              onColumnChange("speed");
            }}>
            <text x={GUTTER} y={11} className="training-readback-title">
              speed — ground speed (m/s) · in force: {label("speed", inForce("speed"))}
            </text>
            {layers.vertical ? envelopes.speed.map((span, index) => {
              if (span.targetMps === null) {
                return (
                  <rect key={`speed-${index}`} className="training-readback-unspecified" opacity={recede(focused("speed", index))}
                    x={rowX(span.row)} width={Math.max(xTime(edge(span.endRow)) - rowX(span.row), 1)}
                    y={plotTop} height={plotH} fill={TRAINING_RAW_COLOR} fillOpacity={focused("speed", index) ? 0.22 : 0.12}
                    stroke={focused("speed", index) ? TRAINING_WORD_COLOR : "none"} strokeWidth={1.4}>
                    <title>unspecified from step {span.row}: the pilot's own speed — only the range {span.rangeMps![0]}–{span.rangeMps![1]} m/s holds</title>
                  </rect>
                );
              }
              const transitionRows = span.transitionLowerMps!.map((_, offset) => span.row + offset);
              return (
                <g key={`speed-${index}`} aria-label={`speed ${label("speed", span.value)} from step ${span.row}`}
                  opacity={recede(focused("speed", index))}>
                  <polygon className="training-readback-transition"
                    points={[
                      ...transitionRows.map((row, offset) => `${rowX(row)},${ySpeed(span.transitionUpperMps![offset])}`),
                      ...transitionRows.map((row, offset) => `${rowX(row)},${ySpeed(span.transitionLowerMps![offset])}`).reverse(),
                    ].join(" ")}
                    fill={TRAINING_SPEED_COLOR} fillOpacity={focused("speed", index) ? 0.2 : 0.1}
                    stroke={focused("speed", index) ? TRAINING_WORD_COLOR
                      : span.check!.transitionOk && span.check!.accelOk ? TRAINING_SPEED_COLOR : TRAINING_OUTSIDE_COLOR}
                    strokeOpacity={focused("speed", index) ? 1 : 0.6} strokeWidth={focused("speed", index) ? 1.4 : 0.6}>
                    <title>
                      transition to {label("speed", span.value)}: monotone {tick(span.check!.transitionOk)}, at most{" "}
                      {vocabulary.speedAccelMaxMps2} m/s² {tick(span.check!.accelOk)}
                    </title>
                  </polygon>
                  {span.arrivalRow !== null ? (
                    <rect className="training-readback-speed-band"
                      x={rowX(span.arrivalRow)} width={Math.max(xTime(edge(span.endRow)) - rowX(span.arrivalRow), 1)}
                      y={ySpeed(span.bandMps![1])} height={ySpeed(span.bandMps![0]) - ySpeed(span.bandMps![1])}
                      fill={TRAINING_SPEED_COLOR} fillOpacity={focused("speed", index) ? 0.34 : 0.2}
                      stroke={focused("speed", index) ? TRAINING_WORD_COLOR
                        : span.check!.contained ? TRAINING_SPEED_COLOR : TRAINING_OUTSIDE_COLOR}
                      strokeWidth={focused("speed", index) ? 1.4 : 0.7}>
                      <title>
                        {label("speed", span.value)} ±{vocabulary.speedToleranceMps} m/s: {span.check!.bandInside} of{" "}
                        {span.check!.bandRows} rows inside
                      </title>
                    </rect>
                  ) : null}
                </g>
              );
            }) : null}
            {trace(signals.raw.groundSpeedMps, rowX, ySpeed, TRAINING_RAW_COLOR, true)}
            {trace(signals.smoothed.groundSpeedMps, rowX, ySpeed, TRAINING_TRACE_COLOR, false)}
            {flownTrack ? flownTrace(flownTrack.tS, flownTrack.groundSpeedMps, xTime, ySpeed, "the executor's ground speed, on its own clock") : null}
            {column === "speed" ? focusTrace(signals.smoothed.groundSpeedMps, rowX, ySpeed) : null}
            {liveTrack ? liveTrace(liveTrack.tS, liveTrack.groundSpeedMps, xTime, ySpeed, "the autopilot's ground speed over the selected segment") : null}
            {envelopes.speed.flatMap((span, index) =>
              span.arrivalRow === null || span.bandInside === null ? [] :
                runsOf(span.bandInside.map((ok) => !ok)).map(([first, lastOut]) => (
                  <polyline
                    key={`speed-out-${index}-${first}`}
                    className="training-readback-outside"
                    points={signals.smoothed.groundSpeedMps.slice(span.arrivalRow! + first, span.arrivalRow! + lastOut + 1)
                      .map((value, offset) => `${rowX(span.arrivalRow! + first + offset)},${ySpeed(value)}`).join(" ")}
                    fill="none" stroke={TRAINING_OUTSIDE_COLOR} strokeWidth={2}
                  />
                )))}
            <line x1={rowX(flight.joinRow)} x2={rowX(flight.joinRow)} y1={plotTop} y2={plotTop + plotH}
              stroke={TRAINING_COLUMN_COLOR.approach} strokeDasharray="3 3">
              <title>cleared at step {flight.joinRow}</title>
            </line>
            <line x1={xTime(cursorS)} x2={xTime(cursorS)} y1={plotTop} y2={plotTop + plotH} className="training-readback-cursor-line" />
            {timeAxis}
          </svg>

          {/* ── the executor's replay ──────────────────────────────────────── */}
          <div className="training-readback-slots">
            <div className="training-readback-slot" aria-label="Executor replay">
              <strong style={{ color: TRAINING_EXECUTOR_COLOR }}>Executor replay</strong> —{" "}
              {executor === null
                ? "off, or not published for this set (the panel's switch says which)."
                : !executor.flown
                  ? `not flown: ${executor.group}.`
                  : flownTrack === null
                    ? `${(executor.outcome ?? "").replace(/_/g, " ")} on ${executor.group} within its first step: no flown ` +
                      "track to draw."
                  : `${(executor.outcome ?? "").replace(/_/g, " ")} on ${executor.group}; dashed teal: its flown track, ` +
                    "and its heading, altitude and ground speed on its own clock and its own distance flown. It flies at its " +
                    "own pace from row 0, each word said where the observed aircraft heard it, so its lines do not line up " +
                    "with the observed ones in time. Its heading words' bands are the teal outlines on the heading chart, " +
                    "from where IT was told each word plus the lead, its rows outside them red (dashed: on the flown track " +
                    "as its judge read it); its other words are judged on envelopes re-drawn from where it was told them — " +
                    "the sentence bar's dots — not on the observed flight's drawn here."}
            </div>
            <div className="training-readback-slot" aria-label="The autopilot, live">
              <strong style={{ color: TRAINING_AUTOPILOT_COLOR }}>The autopilot, live</strong> —{" "}
              {live === null
                ? "select a word (with the panel's switch on) and the executor flies its segment now; its lines appear here."
                : `solid ${live.word.status === "outside" ? "red — outside its envelope" : "blue"}: the selected ` +
                  `${live.segment.column} word's segment, flown by the executor when it was picked, ` +
                  `from the observed state at step ${live.segment.row} — its track, altitude (against the distance flown, from ` +
                  "the observed aircraft's there) and ground speed on the flight's own clock, so they start on the observed " +
                  "lines and part from them as it flies at its own pace." +
                  (liveBand ? " Its heading band is the blue outline, as its judge read the flown segment, its rows outside red." : "")}
            </div>
          </div>
        </div>

        <footer className="training-readback-legend">
          <span>
            Move the pointer across a chart to read it out; the plan view's dot, the sentence bar and the 3D scene follow
            the same cursor. Yellow is the SELECTED word only — the class chosen in the sentence bar, or by clicking a
            chart — at the cursor: its envelope's edge, and the rows it is in force on the track and on its own chart.
          </span>
          <span>
            <b style={{ color: TRAINING_HEADING_BAND_COLOR }}>▩</b> a heading word's band — it says where the track is{" "}
            {vocabulary.headingLeadS} s after it is said, so from then to the next heading word's the track must stay within
            ±{vocabulary.headingToleranceDeg}° of its target; the band stops at the clearance, and a word the lead carries
            there has no row of its own · <b style={{ color: TRAINING_CAPTURE_TURN_COLOR }}>▩</b> the capture turn, from the
            clearance onto the course: monotone, {vocabulary.turnRateMinDegS}–{vocabulary.turnRateMaxDegS}°/s (the lowest
            rate only from {vocabulary.turnRateMinFromDeg}° of turn), at most {vocabulary.turnBankMaxDeg}° of bank ·{" "}
            <b style={{ color: TRAINING_CORRIDOR_COLOR }}>▩</b> the capture corridor — {vocabulary.corridorHalfWidthM} m at the
            threshold, widening {vocabulary.corridorWideningDeg}° outward, course ±{vocabulary.corridorCourseToleranceDeg}°.
          </span>
          <span>
            <b style={{ color: TRAINING_TUBE_COLOR }}>▩</b> an altitude word's tube, from where it was issued: between the lines
            of its angle class's two edges, stopped at the target, ±{vocabulary.altitudeToleranceM} m, and re-anchored at every
            angle word · <b style={{ color: TRAINING_SPEED_COLOR }}>▩</b> a speed word: the transition (monotone toward the
            target, at most {vocabulary.speedAccelMaxMps2} m/s²), then the band ±{vocabulary.speedToleranceMps} m/s · grey:
            "unspecified", the pilot's own speed.
          </span>
          <span>
            <b style={{ color: TRAINING_TRACE_COLOR }}>——</b> the smoothed signal the labeller read (track{" "}
            {vocabulary.smoothingS.track} s, altitude {vocabulary.smoothingS.altitude} s, speed {vocabulary.smoothingS.speed} s) ·
            faint: the raw rows · <b style={{ color: TRAINING_OUTSIDE_COLOR }}>——</b> rows the labeller counted outside, or
            an envelope whose check failed. Every band and verdict is the exporter's and the labeller's; nothing here is
            recomputed.
          </span>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
