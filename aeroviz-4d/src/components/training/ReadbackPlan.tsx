/**
 * ReadbackPlan.tsx
 * ----------------
 * The read-back window's plan view, in the airport frame at one scale on both axes: every candidate runway and its
 * extended centreline, the capture corridor, the capture turn's rows, the track, where each heading word was issued,
 * the clearance, the capture and the end of the sentence; the rows a heading word's band judged outside, in red; the
 * executor's replay (dashed teal) and the live segment (solid, in its verdict's colour), each with its rows outside.
 */

import { useId } from "react";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_CAPTURE_TURN_COLOR,
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import { outsideSpans, trainingKindLabel } from "../../data/trainingSample";
import { TRAINING_OUTCOME_TAG } from "../../data/trainingText";
import { ChartFrame } from "./chartKit";
import { GUTTER, PLAN_H, type ReadbackModel } from "./readbackModel";

export default function ReadbackPlan({ m }: { m: ReadbackModel }) {
  const clip = useId();
  const { flight, vocabulary, candidates, layers, column, focus, designated, last, at, px, py, planPoints, rows } = m;
  const { envelopes, signals } = flight;
  const { capture } = m;
  const corridor = envelopes.approach.corridor;
  const outside = (key: string, points: string, className: string) => (
    <polyline key={key} className={className} points={points} fill="none" stroke={TRAINING_OUTSIDE_COLOR} strokeWidth={2.4}
      strokeLinecap="round" />
  );
  const endOf = (line: { eM: number[]; nM: number[] }) => ({ x: px(line.eM[line.eM.length - 1]), y: py(line.nM[line.nM.length - 1]) });

  return (
    <ChartFrame captionIndent={GUTTER} label="Plan view" width={m.width} height={PLAN_H}
      caption="plan view · km east × km north of the airport's reference point, framed on the track">
      <defs>
        <clipPath id={clip}>
          <rect x={GUTTER} y={2} width={m.plotW} height={PLAN_H - 4} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clip})`}>
        {candidates.map((candidate) => {
          const pointed = candidate.index === flight.runwayIndex;
          if (!pointed && !layers.candidates) return null;
          const colour = pointed ? TRAINING_DESIGNATED_COLOR : TRAINING_CANDIDATE_COLOR;
          const stroke = column === "runway" && focus?.value === candidate.index ? TRAINING_WORD_COLOR : colour;
          const name = `${pointed ? "the designated runway" : "candidate"} ${candidate.ident}: course ` +
            `${candidate.courseDeg.toFixed(1)}°, threshold ${candidate.elevationM.toFixed(1)} m MSL`;
          return (
            <g key={`candidate-${candidate.ident}`} className="training-readback-candidate" aria-label={name}>
              <title>{name}</title>
              <polyline points={planPoints(candidate.centreline)} fill="none" stroke={stroke} strokeDasharray="5 4"
                strokeOpacity={pointed ? 0.9 : 0.5} strokeWidth={1} />
              <polyline points={planPoints(candidate.runway)} fill="none" stroke={stroke} strokeWidth={4}
                strokeOpacity={pointed ? 1 : 0.6} />
              <text x={px(candidate.thresholdEM) + 4} y={py(candidate.thresholdNM) - 4}
                className="training-readback-candidate-label" fill={colour}>
                {candidate.ident}
              </text>
            </g>
          );
        })}

        {layers.corridor ? (
          <polygon className="training-readback-corridor" points={planPoints(corridor.outline)}
            fill={TRAINING_CORRIDOR_COLOR} fillOpacity={m.approachFocused ? 0.4 : 0.2}
            stroke={m.approachFocused ? TRAINING_WORD_COLOR : TRAINING_CORRIDOR_COLOR} strokeWidth={m.approachFocused ? 1.4 : 0.8}
            opacity={m.recede(m.approachFocused)}>
            <title>
              the capture corridor: {corridor.halfWidthAtCaptureM.toFixed(0)} m half width at the capture
              ({(corridor.beforeThresholdM / 1000).toFixed(1)} km out), {corridor.halfWidthAtThresholdM.toFixed(0)} m at the
              threshold, course ±{vocabulary.corridorCourseToleranceDeg}° — every one of its {corridor.rows} rows inside
            </title>
          </polygon>
        ) : null}

        <polyline points={planPoints(signals)} fill="none" stroke={TRAINING_TRACE_COLOR} strokeWidth={1.4}
          className="training-readback-trace" />
        {/* the capture turn: its rows on the track, from the clearance to the capture */}
        {layers.corridor && capture !== null && capture.endRow > capture.startRow ? (
          <polyline className="training-readback-capture-turn" points={planPoints(signals, rows(capture.startRow, capture.endRow))}
            fill="none" strokeDasharray="4 3" strokeWidth={m.approachFocused ? 3 : 2.2} opacity={m.recede(m.approachFocused)}
            stroke={m.approachFocused ? TRAINING_WORD_COLOR : m.captureOk ? TRAINING_CAPTURE_TURN_COLOR : TRAINING_OUTSIDE_COLOR}>
            <title>the capture turn, steps {capture.startRow}–{capture.endRow}, onto the course {designated.courseDeg.toFixed(1)}°</title>
          </polyline>
        ) : null}
        {/* the rows a heading word's band judged outside */}
        {layers.headingBands ? envelopes.heading.flatMap((item, index) => outsideSpans(item.inside, item.firstRow, last)
          .map(([first, lastRow]) => outside(`plan-heading-out-${index}-${first}`, planPoints(signals, rows(first, lastRow)),
            "training-readback-outside"))) : null}

        {m.flownTrack ? (
          <g aria-label="the executor's flown track">
            <polyline points={planPoints(m.flownTrack)} fill="none" stroke={TRAINING_EXECUTOR_COLOR} strokeWidth={1.4}
              strokeDasharray="5 3" className="training-readback-executor">
              <title>the executor's flown track — the truth sentence flown from row 0</title>
            </polyline>
            {layers.headingBands && m.judgedTrack ? m.flownBands.flatMap((band, index) =>
              outsideSpans(band.inside, band.firstRow, m.judgedTrack!.length - 1).map(([first, lastStep]) =>
                outside(`plan-executor-out-${index}-${first}`, planPoints(m.flownTrack!, rows(first, lastStep)),
                  "training-readback-executor-outside"))) : null}
            <rect x={endOf(m.flownTrack).x - 3.5} y={endOf(m.flownTrack).y - 3.5} width={7} height={7}
              fill={TRAINING_EXECUTOR_COLOR} stroke="black" strokeWidth={0.6} />
            <text x={endOf(m.flownTrack).x + 6} y={endOf(m.flownTrack).y + 12} className="training-readback-path-label"
              fill={TRAINING_EXECUTOR_COLOR}>
              executor: {TRAINING_OUTCOME_TAG[m.flown!.outcome]}
            </text>
          </g>
        ) : null}
        {m.focusRows.length >= 2 ? (
          <polyline points={planPoints(signals, m.focusRows)} fill="none" stroke={TRAINING_WORD_COLOR} strokeWidth={3}
            strokeOpacity={0.9} strokeLinecap="round" className="training-readback-focus" />
        ) : null}
        {m.live ? (
          <g aria-label="the autopilot's flown segment">
            <polyline points={planPoints(m.live.track)} fill="none" stroke={m.liveColour} strokeWidth={1.8}
              className="training-readback-autopilot">
              <title>the autopilot's flown segment, from where its word was said</title>
            </polyline>
            {m.liveOutside.map(([first, lastStep]) => outside(`plan-autopilot-out-${first}`,
              planPoints(m.live!.track, rows(first * m.live!.executor.stepCycles, lastStep * m.live!.executor.stepCycles)),
              "training-readback-autopilot-outside"))}
            <circle cx={endOf(m.live.track).x} cy={endOf(m.live.track).y} r={4} fill={m.liveColour} stroke="black" strokeWidth={0.6} />
          </g>
        ) : null}

        {envelopes.heading.map((item, index) => {
          const name = `heading ${m.label("heading", item.value)} issued at step ${item.row} — ${trainingKindLabel(item.kind)}`;
          return (
            <g key={`issue-${index}`} className="training-readback-issue" aria-label={name}>
              <title>{name}</title>
              <circle cx={at(item.row).x} cy={at(item.row).y} r={2.6} fill="none" stroke={TRAINING_COLUMN_COLOR.heading}
                strokeWidth={1.3} />
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
          <title>the final captured at step {flight.captureRow}, {(flight.captureBeforeThresholdM / 1000).toFixed(1)} km before the threshold</title>
          <rect x={at(flight.captureRow).x - 4} y={at(flight.captureRow).y - 4} width={8} height={8}
            fill={TRAINING_CORRIDOR_COLOR} stroke="black" strokeWidth={0.6} />
        </g>
        <g aria-label="the end of the sentence">
          <title>
            the sentence ends here, {(envelopes.approach.landing.lastRowBeforeThresholdM / 1000).toFixed(2)} km before the
            threshold{envelopes.approach.landing.cutAtCrossing ? " (cut before the last passage of the threshold)" : ""}
          </title>
          <polygon fill={TRAINING_TRACE_COLOR}
            points={`${at(last).x},${at(last).y - 5} ${at(last).x - 4},${at(last).y + 3} ${at(last).x + 4},${at(last).y + 3}`} />
        </g>
        <circle cx={at(m.cursorRow).x} cy={at(m.cursorRow).y} r={4.5} className="training-readback-cursor-dot" />
      </g>
    </ChartFrame>
  );
}
