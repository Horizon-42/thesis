/**
 * TrainingReadbackWindow.tsx
 * --------------------------
 * The read-back check: one flight's sentence against its track, envelope by envelope. Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 *  • PLAN VIEW (the airport frame, one scale on both axes): every candidate runway and its
 *    extended centreline, the capture corridor, each heading word's turn region and hold funnel,
 *    the capture turn, the track, where each heading word was issued, the clearance, the capture
 *    and the end of the sentence.
 *  • HEADING against time: each word's turn band and hold band, the capture turn, the course
 *    band after the capture.
 *  • ALTITUDE against the horizontal distance flown — the axis the tubes are defined on: each
 *    altitude word's tube, the angle words that re-anchor it, the runway's elevation.
 *  • SPEED against time: each word's transition and band, the "unspecified" spans, the clearance.
 *
 * EVERY SHAPE IS THE EXPORTER'S. Regions, bands and tubes arrive as numbers computed in Python
 * from the vocabulary's own functions, and every verdict is the labeller's; this window draws them
 * and computes none. The smoothed signal is the bright line (it is what the labeller read), the
 * raw rows are behind it, and red marks rows the labeller counted outside their envelope.
 *
 * It renders through a PORTAL into `document.body` (AV7): `.flight-ops-panel` carries a
 * `backdrop-filter`, which would make a `position: fixed` descendant position against it.
 */

import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { TrainingLayers } from "../context/AppContext";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_FUNNEL_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_SPEED_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_TURN_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  altitudeTubeAt,
  formatSeconds,
  headingEnvelopeAt,
  rowAtTime,
  trainingKindLabel,
  trainingWordLabel,
  TRAINING_COLUMN_INDEX,
  type TrainingCandidate,
  type TrainingFlight,
  type TrainingPlanLine,
  type TrainingVocabulary,
} from "../data/trainingSample";

const GUTTER = 64;
const PAD_R = 16;
const PLAN_H = 300;
const CHART_H = 150;
const DEFAULT_W = 980;
const MIN_W = 420;

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
  onClose: () => void;
}

export default function TrainingReadbackWindow({
  flight, vocabulary, candidates, layers, cursorS, onCursorChange, onClose,
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
  const plotW = width - GUTTER - PAD_R;
  const cursorRow = rowAtTime(tS, cursorS);
  const headingInForce = headingEnvelopeAt(flight, cursorRow);
  const tubeInForce = altitudeTubeAt(flight, cursorRow);
  const designated = candidates[flight.runwayIndex];
  const label = (column: "heading" | "altitude" | "angle" | "speed", value: number) =>
    trainingWordLabel(vocabulary, candidates, column, value);
  const inForce = (column: "heading" | "altitude" | "angle" | "speed") =>
    flight.words.inForce[TRAINING_COLUMN_INDEX[column]][cursorRow];

  // ── time charts share one x ───────────────────────────────────────────────
  const endS = tS[last];
  const xTime = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  const timeAtX = (x: number) => Math.min(Math.max(((x - GUTTER) / plotW) * endS, 0), endS);
  /** A row as a time edge; the step after the last row is the last row. */
  const edge = (row: number) => tS[Math.min(row, last)];

  // ── the plan view: the track and the thresholds decide the frame ─────────
  const km = (metres: number) => metres / 1000;
  const frameE = [...signals.eM, designated.thresholdEM, ...envelopes.approach.corridor.axis.eM].map(km);
  const frameN = [...signals.nM, designated.thresholdNM, ...envelopes.approach.corridor.axis.nM].map(km);
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

  // ── the charts' y ────────────────────────────────────────────────────────
  const headingValues = [
    ...signals.smoothed.trackDeg, ...signals.raw.trackDeg,
    ...(layers.lateral ? envelopes.heading.flatMap((item) => [...(item.turnBandDeg ?? []), ...(item.holdBandDeg ?? [])]) : []),
    ...(layers.lateral ? envelopes.approach.courseBandDeg : []),
  ];
  const [hLow, hHigh] = extent(headingValues);
  const altitudeValues = [
    ...signals.smoothed.altitudeM, ...signals.raw.altitudeM, designated.elevationM,
    ...(layers.vertical ? envelopes.altitude.flatMap((tube) => [...tube.lowerM, ...tube.upperM]) : []),
  ];
  const [aLow, aHigh] = extent(altitudeValues);
  const speedValues = [
    ...signals.smoothed.groundSpeedMps, ...signals.raw.groundSpeedMps,
    ...(layers.vertical ? envelopes.speed.flatMap((span) => [...(span.bandMps ?? []), ...(span.transitionLowerMps ?? []), ...(span.transitionUpperMps ?? [])]) : []),
  ];
  const [sLow, sHigh] = extent(speedValues);
  const plotTop = 18;
  // room below the plot for the tick labels and the axis caption
  const plotH = CHART_H - 48;
  const yOf = (low: number, high: number) => (value: number) => plotTop + ((high - value) / (high - low)) * plotH;
  const yHeading = yOf(hLow, hHigh);
  const yAltitude = yOf(aLow, aHigh);
  const ySpeed = yOf(sLow, sHigh);
  const distanceEnd = signals.smoothed.distanceM[last];
  const xDistance = (metres: number) => GUTTER + (metres / distanceEnd) * plotW;
  const distanceAtX = (x: number) => Math.min(Math.max(((x - GUTTER) / plotW) * distanceEnd, 0), distanceEnd);

  const heading = headingInForce >= 0 ? envelopes.heading[headingInForce] : null;
  const tube = envelopes.altitude[tubeInForce];
  const capture = envelopes.approach.captureTurn;

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
              plan view · airport frame, km east × km north of the reference point · clipped to the track
            </text>
            <g clipPath="url(#training-plan-clip)">
              {candidates.map((candidate) => {
                const pointed = candidate.index === flight.runwayIndex;
                if (!pointed && !layers.candidates) return null;
                const colour = pointed ? TRAINING_DESIGNATED_COLOR : TRAINING_CANDIDATE_COLOR;
                const name = `${pointed ? "the designated runway" : "candidate"} ${candidate.ident}: course ` +
                  `${candidate.courseDeg.toFixed(1)}°, threshold ${candidate.elevationM.toFixed(1)} m MSL`;
                return (
                  <g key={`candidate-${candidate.ident}`} className="training-readback-candidate" aria-label={name}>
                    <title>{name}</title>
                    <polyline points={points(candidate.centreline)} fill="none" stroke={colour} strokeDasharray="5 4"
                      strokeOpacity={pointed ? 0.9 : 0.5} strokeWidth={1} />
                    <polyline points={points(candidate.runway)} fill="none" stroke={colour} strokeWidth={4}
                      strokeOpacity={pointed ? 1 : 0.6} />
                    <text x={px(km(candidate.thresholdEM)) + 4} y={py(km(candidate.thresholdNM)) - 4}
                      className="training-readback-candidate-label" fill={colour}>
                      {candidate.ident}
                    </text>
                  </g>
                );
              })}

              {layers.lateral ? (
                <>
                  <polygon
                    className="training-readback-corridor"
                    points={points(envelopes.approach.corridor.outline)}
                    fill={TRAINING_CORRIDOR_COLOR}
                    fillOpacity={headingInForce < 0 ? 0.4 : 0.2}
                    stroke={headingInForce < 0 ? TRAINING_WORD_COLOR : TRAINING_CORRIDOR_COLOR}
                    strokeWidth={0.8}
                  >
                    <title>
                      the capture corridor: {envelopes.approach.corridor.halfWidthAtCaptureM.toFixed(0)} m half width at the
                      capture ({(envelopes.approach.corridor.beforeThresholdM / 1000).toFixed(1)} km out),{" "}
                      {envelopes.approach.corridor.halfWidthAtThresholdM.toFixed(0)} m at the threshold, course ±
                      {vocabulary.corridorCourseToleranceDeg}° — every one of its {envelopes.approach.corridor.rows} rows inside
                    </title>
                  </polygon>
                  {envelopes.heading.map((item, index) => {
                    const selected = index === headingInForce;
                    const failed = item.check !== null && !(item.check.progressOk && item.check.bankOk);
                    return (
                      <g key={`plan-heading-${index}`} aria-label={`heading word ${index + 1} envelope`}>
                        {item.turn ? (
                          <polygon
                            className="training-readback-turn"
                            points={points(item.turn.region)}
                            fill={TRAINING_TURN_COLOR}
                            fillOpacity={selected ? 0.3 : 0.1}
                            stroke={selected ? TRAINING_WORD_COLOR : failed ? TRAINING_OUTSIDE_COLOR : TRAINING_TURN_COLOR}
                            strokeWidth={selected ? 1.4 : 0.7}
                          >
                            <title>
                              turn to {label("heading", item.value)} from {item.turn.fromTrackDeg.toFixed(1)}° (
                              {item.turn.turnDeg > 0 ? "right" : "left"} {Math.abs(item.turn.turnDeg).toFixed(1)}°) at{" "}
                              {item.turn.groundSpeedMps.toFixed(0)} m/s: radius {(item.turn.radiusMinM / 1000).toFixed(2)}–
                              {(item.turn.radiusMaxM / 1000).toFixed(2)} km for bank {vocabulary.turnBankMaxDeg}–
                              {vocabulary.turnBankMinDeg}°
                            </title>
                          </polygon>
                        ) : null}
                        {item.funnel ? (
                          <polygon
                            className="training-readback-funnel"
                            points={points(item.funnel.outline)}
                            fill={TRAINING_FUNNEL_COLOR}
                            fillOpacity={selected ? 0.3 : 0.1}
                            stroke={selected ? TRAINING_WORD_COLOR : TRAINING_FUNNEL_COLOR}
                            strokeWidth={selected ? 1.4 : 0.7}
                          >
                            <title>
                              hold {label("heading", item.value)} ±{vocabulary.headingToleranceDeg}°:{" "}
                              {(item.funnel.lengthM / 1000).toFixed(1)} km, half width{" "}
                              {item.funnel.startHalfWidthM.toFixed(0)} → {item.funnel.endHalfWidthM.toFixed(0)} m
                            </title>
                          </polygon>
                        ) : null}
                      </g>
                    );
                  })}
                  {capture ? (
                    <polygon
                      className="training-readback-capture-turn"
                      points={points(capture.turn.region)}
                      fill={TRAINING_TURN_COLOR}
                      fillOpacity={0.06}
                      stroke={capture.check.progressOk && capture.check.bankOk ? TRAINING_TURN_COLOR : TRAINING_OUTSIDE_COLOR}
                      strokeDasharray="4 3"
                      strokeWidth={0.9}
                    >
                      <title>the capture turn onto the course {designated.courseDeg.toFixed(1)}°</title>
                    </polygon>
                  ) : null}
                </>
              ) : null}

              <polyline points={signals.eM.map((e, row) => `${px(km(e))},${py(km(signals.nM[row]))}`).join(" ")}
                fill="none" stroke={TRAINING_TRACE_COLOR} strokeWidth={1.4} className="training-readback-trace" />
              {envelopes.heading.map((item, index) => {
                const point = at(item.row);
                const name = `heading ${label("heading", item.value)} issued at step ${item.row} — ${trainingKindLabel(item.kind, item.split)}`;
                return (
                  <g key={`issue-${index}`} className="training-readback-issue" aria-label={name}>
                    <title>{name}</title>
                    <circle cx={point.x} cy={point.y} r={3.4} fill="none" stroke={TRAINING_COLUMN_COLOR.heading} strokeWidth={1.6} />
                    <text x={point.x + 5} y={point.y - 5} className="training-readback-issue-label" fill={TRAINING_COLUMN_COLOR.heading}>
                      {label("heading", item.value)}
                    </text>
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
            onClick={(event) => onCursorChange(timeAtX(event.nativeEvent.offsetX))}>
            <text x={GUTTER} y={11} className="training-readback-title">
              heading — ground track, ° true, unwrapped · in force: {label("heading", inForce("heading"))}
              {heading?.check
                ? ` · its turn: ${tick(heading.check.progressOk)} monotone, ${tick(heading.check.bankOk)} bank ` +
                  `(mean ${heading.check.meanBankDeg.toFixed(1)}°, max ${heading.check.maxBankDeg.toFixed(1)}°` +
                  `${heading.check.bankMinApplies ? "" : `; under ${vocabulary.turnBankMinFromDeg}°, the lowest bank not judged`})`
                : heading === null ? " · captured: the corridor holds" : " · in force at entry, no turn"}
            </text>
            {layers.lateral ? (
              <>
                {envelopes.heading.map((item, index) => (
                  <g key={`heading-band-${index}`}>
                    {item.turnBandDeg && item.turnEndRow !== null ? (
                      <rect
                        className="training-readback-turn-band"
                        x={rowX(item.row)} width={Math.max(xTime(edge(item.turnEndRow)) - rowX(item.row), 1)}
                        y={yHeading(item.turnBandDeg[1])} height={yHeading(item.turnBandDeg[0]) - yHeading(item.turnBandDeg[1])}
                        fill={TRAINING_TURN_COLOR} fillOpacity={0.12}
                        stroke={item.check && !item.check.progressOk ? TRAINING_OUTSIDE_COLOR : TRAINING_TURN_COLOR}
                        strokeOpacity={0.6} strokeWidth={0.7}
                      >
                        <title>turn to {label("heading", item.value)}: from the track at issue to the target, ±{vocabulary.headingToleranceDeg}°</title>
                      </rect>
                    ) : null}
                    {item.holdBandDeg && item.holdStartRow !== null ? (
                      <rect
                        className="training-readback-hold-band"
                        x={rowX(item.holdStartRow)} width={Math.max(xTime(edge(item.holdEndRow)) - rowX(item.holdStartRow), 1)}
                        y={yHeading(item.holdBandDeg[1])} height={yHeading(item.holdBandDeg[0]) - yHeading(item.holdBandDeg[1])}
                        fill={TRAINING_FUNNEL_COLOR} fillOpacity={index === headingInForce ? 0.32 : 0.16}
                        stroke={index === headingInForce ? TRAINING_WORD_COLOR : TRAINING_FUNNEL_COLOR} strokeWidth={0.7}
                      >
                        <title>hold {label("heading", item.value)} ±{vocabulary.headingToleranceDeg}°</title>
                      </rect>
                    ) : null}
                  </g>
                ))}
                {capture ? (
                  <rect
                    className="training-readback-capture-band"
                    x={rowX(capture.startRow)} width={Math.max(rowX(flight.captureRow) - rowX(capture.startRow), 1)}
                    y={yHeading(capture.bandDeg[1])} height={yHeading(capture.bandDeg[0]) - yHeading(capture.bandDeg[1])}
                    fill="none" stroke={TRAINING_TURN_COLOR} strokeDasharray="4 3" strokeWidth={0.9}
                  >
                    <title>the capture turn onto the course</title>
                  </rect>
                ) : null}
                <rect
                  className="training-readback-course-band"
                  x={rowX(flight.captureRow)} width={Math.max(rowX(last) - rowX(flight.captureRow), 1)}
                  y={yHeading(envelopes.approach.courseBandDeg[1])}
                  height={yHeading(envelopes.approach.courseBandDeg[0]) - yHeading(envelopes.approach.courseBandDeg[1])}
                  fill={TRAINING_CORRIDOR_COLOR} fillOpacity={0.22}
                >
                  <title>after the capture: the course ±{vocabulary.corridorCourseToleranceDeg}°</title>
                </rect>
              </>
            ) : null}
            {trace(signals.raw.trackDeg, rowX, yHeading, TRAINING_RAW_COLOR, true)}
            {trace(signals.smoothed.trackDeg, rowX, yHeading, TRAINING_TRACE_COLOR, false)}
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
            onClick={(event) => onCursorChange(tS[rowAtDistance(signals.smoothed.distanceM, distanceAtX(event.nativeEvent.offsetX))])}>
            <text x={GUTTER} y={11} className="training-readback-title">
              altitude — geometric MSL (m) against distance flown · in force: {label("altitude", inForce("altitude"))},{" "}
              {label("angle", inForce("angle"))}
              {tube ? ` · this tube: ${tube.check.inside}/${tube.check.rows} rows inside ${tick(tube.check.contained)}` : ""}
            </text>
            {layers.vertical ? envelopes.altitude.map((item, index) => {
              const rows = item.lowerM.map((_, offset) => item.row + offset);
              const selected = index === tubeInForce;
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
              <g key={`angle-${index}`} aria-label={`angle word ${label("angle", item.value)} at step ${item.row}`}>
                <line x1={rowXDistance(item.row)} x2={rowXDistance(item.row)} y1={plotTop} y2={plotTop + plotH}
                  stroke={TRAINING_COLUMN_COLOR.angle} strokeOpacity={0.6} strokeDasharray="2 2" />
                <text x={rowXDistance(item.row) + 2} y={plotTop + 9} className="training-readback-tick" fill={TRAINING_COLUMN_COLOR.angle}>
                  {vocabulary.angleClasses[item.value].name}
                  {item.measuredDeg === null ? "" : ` ${item.measuredDeg.toFixed(2)}°`}
                </text>
              </g>
            ))}
            {trace(signals.raw.altitudeM, rowXDistance, yAltitude, TRAINING_RAW_COLOR, true)}
            {trace(signals.smoothed.altitudeM, rowXDistance, yAltitude, TRAINING_TRACE_COLOR, false)}
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
            onClick={(event) => onCursorChange(timeAtX(event.nativeEvent.offsetX))}>
            <text x={GUTTER} y={11} className="training-readback-title">
              speed — ground speed (m/s) · in force: {label("speed", inForce("speed"))}
            </text>
            {layers.vertical ? envelopes.speed.map((span, index) => {
              if (span.targetMps === null) {
                return (
                  <rect key={`speed-${index}`} className="training-readback-unspecified"
                    x={rowX(span.row)} width={Math.max(xTime(edge(span.endRow)) - rowX(span.row), 1)}
                    y={plotTop} height={plotH} fill={TRAINING_RAW_COLOR} fillOpacity={0.12}>
                    <title>unspecified from step {span.row}: the pilot's own speed — only the range {span.rangeMps![0]}–{span.rangeMps![1]} m/s holds</title>
                  </rect>
                );
              }
              const transitionRows = span.transitionLowerMps!.map((_, offset) => span.row + offset);
              return (
                <g key={`speed-${index}`} aria-label={`speed ${label("speed", span.value)} from step ${span.row}`}>
                  <polygon className="training-readback-transition"
                    points={[
                      ...transitionRows.map((row, offset) => `${rowX(row)},${ySpeed(span.transitionUpperMps![offset])}`),
                      ...transitionRows.map((row, offset) => `${rowX(row)},${ySpeed(span.transitionLowerMps![offset])}`).reverse(),
                    ].join(" ")}
                    fill={TRAINING_SPEED_COLOR} fillOpacity={0.1}
                    stroke={span.check!.transitionOk && span.check!.accelOk ? TRAINING_SPEED_COLOR : TRAINING_OUTSIDE_COLOR}
                    strokeOpacity={0.6} strokeWidth={0.6}>
                    <title>
                      transition to {label("speed", span.value)}: monotone {tick(span.check!.transitionOk)}, at most{" "}
                      {vocabulary.speedAccelMaxMps2} m/s² {tick(span.check!.accelOk)}
                    </title>
                  </polygon>
                  {span.arrivalRow !== null ? (
                    <rect className="training-readback-speed-band"
                      x={rowX(span.arrivalRow)} width={Math.max(xTime(edge(span.endRow)) - rowX(span.arrivalRow), 1)}
                      y={ySpeed(span.bandMps![1])} height={ySpeed(span.bandMps![0]) - ySpeed(span.bandMps![1])}
                      fill={TRAINING_SPEED_COLOR} fillOpacity={0.2}
                      stroke={span.check!.contained ? TRAINING_SPEED_COLOR : TRAINING_OUTSIDE_COLOR} strokeWidth={0.7}>
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

          {/* ── the two slots that are empty on purpose ───────────────────── */}
          <div className="training-readback-slots">
            <div className="training-readback-slot" aria-label="Executor replay slot">
              <strong>Executor replay</strong> — not built yet. The executor that flies a sentence by dynamics alone
              is being designed (stage 3); its replay of this sentence will be drawn here beside the track.
            </div>
            <div className="training-readback-slot" aria-label="Prior sentence slot">
              <strong>Prior-generated sentence</strong> — no prior is trained on this vocabulary yet (stage 5); what a
              prior says for this flight will be drawn here against the labelled sentence.
            </div>
          </div>
        </div>

        <footer className="training-readback-legend">
          <span>
            Move the pointer across a chart to read it out; the plan view's dot, the sentence bar and the 3D scene follow
            the same cursor. The word in force is drawn in yellow.
          </span>
          <span>
            <b style={{ color: TRAINING_TURN_COLOR }}>▩</b> a heading word's turn region — the arcs of every bank from{" "}
            {vocabulary.turnBankMinDeg}° to {vocabulary.turnBankMaxDeg}° at the issue ground speed, turning the shorter way;
            under {vocabulary.turnBankMinFromDeg}° of turn the lowest bank is not judged, so its outer arc does not bind ·{" "}
            <b style={{ color: TRAINING_FUNNEL_COLOR }}>▩</b> its hold funnel — along the target from where the turn ends,
            starting as wide as the turn's end and widening by the distance × tan {vocabulary.headingToleranceDeg}° ·{" "}
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
            an envelope whose check failed. Every region and verdict is the exporter's and the labeller's; nothing here is
            recomputed.
          </span>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
