/**
 * ReadbackPlan.tsx
 * ----------------
 * The read-back window's plan view, in the airport frame at one scale on both axes: every candidate runway (its runway
 * and its approach centreline), the observed track, the flown path of a closed-loop reading beside it, the words the
 * closed-loop reading added, the DA point of the crossing and the live segment (solid, in its colour); the rows a heading
 * band judged outside, in red.
 */

import { useId } from "react";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_CORRECTION_COLOR,
  TRAINING_DECISION_FAIL_COLOR,
  TRAINING_DECISION_PASS_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import { outsideSpans, trainingBandLabel } from "../../data/trainingSample";
import { decisionText, TRAINING_OUTCOME_TAG } from "../../data/trainingText";
import AutopilotLine from "./AutopilotLine";
import { ChartFrame } from "./chartKit";
import { GUTTER, PLAN_H, type ReadbackModel } from "./readbackModel";

export default function ReadbackPlan({ m }: { m: ReadbackModel }) {
  const clip = useId();
  const { flight, candidates, layers, column, focus, designated, observed, flown, px, py, planPoints, rows } = m;
  const outside = (key: string, points: string, className: string) => (
    <polyline key={key} className={className} points={points} fill="none" stroke={TRAINING_OUTSIDE_COLOR} strokeWidth={2.4}
      strokeLinecap="round" />
  );
  const endOf = (line: { eM: number[]; nM: number[] }) => ({ x: px(line.eM[line.eM.length - 1]), y: py(line.nM[line.nM.length - 1]) });
  const focusRows = m.focusPoints(m.judged.tS);
  const decision = m.decision;

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
          const stroke = column === "runway" && focus?.event.says.column === "runway" && !focus.event.says.goAround
            && focus.event.says.runwayIndex === candidate.index ? TRAINING_WORD_COLOR : colour;
          const line = m.runwayLine(candidate);
          const name = `${pointed ? "the runway the flight lands on" : "candidate"} ${candidate.ident}: course ` +
            `${candidate.courseDeg.toFixed(1)}°, threshold ${candidate.elevationM.toFixed(1)} m MSL`;
          return (
            <g key={`candidate-${candidate.ident}`} className="training-readback-candidate" aria-label={name}>
              <title>{name}</title>
              <polyline points={line.centreline.map(([e, n]) => `${px(e)},${py(n)}`).join(" ")} fill="none" stroke={stroke}
                strokeDasharray="5 4" strokeOpacity={pointed ? 0.9 : 0.5} strokeWidth={1} />
              <polyline points={line.runway.map(([e, n]) => `${px(e)},${py(n)}`).join(" ")} fill="none" stroke={stroke}
                strokeWidth={4} strokeOpacity={pointed ? 1 : 0.6} />
              <text x={px(candidate.thresholdEM) + 4} y={py(candidate.thresholdNM) - 4}
                className="training-readback-candidate-label" fill={colour}>
                {candidate.ident}
              </text>
            </g>
          );
        })}

        <polyline points={planPoints(observed)} fill="none" stroke={TRAINING_TRACE_COLOR} strokeWidth={1.4}
          className="training-readback-trace">
          <title>the observed track</title>
        </polyline>
        {/* the rows a heading word's band judged outside, on the judged track */}
        {layers.headingBands && m.envelopes !== null ? m.envelopes.heading.flatMap((band, index) =>
          outsideSpans(band.inside, band.firstRow, m.judged.tS.length - 1).map(([first, last]) =>
            outside(`plan-heading-out-${index}-${first}`, planPoints(m.judged, rows(first, last)), "training-readback-outside"))) : null}

        {flown ? (
          <g aria-label="the flown path">
            <polyline points={planPoints(flown)} fill="none" stroke={TRAINING_EXECUTOR_COLOR} strokeWidth={1.6}
              className="training-readback-executor">
              <title>the flown path: the closed-loop sentence flown by the executor from the first predicted step</title>
            </polyline>
            <rect x={endOf(flown).x - 3.5} y={endOf(flown).y - 3.5} width={7} height={7} fill={TRAINING_EXECUTOR_COLOR}
              stroke="black" strokeWidth={0.6} />
            <text x={endOf(flown).x + 6} y={endOf(flown).y + 12} className="training-readback-path-label" fill={TRAINING_EXECUTOR_COLOR}>
              flown: {TRAINING_OUTCOME_TAG[m.closed!.replay.outcome]}
            </text>
          </g>
        ) : null}
        {column !== null && column !== "runway" && focusRows.length >= 2 ? (
          <polyline points={planPoints(m.judged, focusRows)} fill="none" stroke={TRAINING_WORD_COLOR} strokeWidth={3}
            strokeOpacity={0.9} strokeLinecap="round" className="training-readback-focus" />
        ) : null}
        {/* the words the closed-loop reading added, where they were said on the flown path */}
        {flown ? m.corrections("runway", "heading", "altitude", "angle", "speed").map(({ event, atS }, index) => {
          const at = m.indexAt(flown.tS, atS);
          return (
            <g key={`correction-${index}`} className="training-readback-correction"
              aria-label={`correction: ${event.says.column} ${trainingBandLabel(event.says)}`}>
              <title>correction: {event.says.column} {trainingBandLabel(event.says)}, added by the closed-loop reading at {atS} s</title>
              <circle cx={px(flown.eM[at])} cy={py(flown.nM[at])} r={3} fill={TRAINING_CORRECTION_COLOR} stroke="black" strokeWidth={0.5} />
            </g>
          );
        }) : null}
        {decision !== null ? (
          <g aria-label="the decision-altitude point">
            <title>{decisionText(decision)}</title>
            <circle cx={px(decision.eM)} cy={py(decision.nM)} r={5.5} fill="none" strokeWidth={2}
              stroke={decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR} />
            <text x={px(decision.eM) + 8} y={py(decision.nM) - 6} className="training-readback-path-label"
              fill={decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR}>
              DA {decision.passed ? "✓" : "✗"}
            </text>
          </g>
        ) : null}
        {m.live ? (
          <g aria-label="the autopilot's flown segment">
            <AutopilotLine m={m} x={(index) => px(m.live!.track.eM[index])} y={(index) => py(m.live!.track.nM[index])}
              title="the autopilot's flown segment, from where its word was said" />
            <circle cx={endOf(m.live.track).x} cy={endOf(m.live.track).y} r={4} fill={m.liveColour} stroke="black" strokeWidth={0.6} />
          </g>
        ) : null}
        {m.cursorS >= observed.tS[0] ? (
          <circle cx={px(observed.eM[m.indexAt(observed.tS, m.cursorS)])} cy={py(observed.nM[m.indexAt(observed.tS, m.cursorS)])}
            r={4.5} className="training-readback-cursor-dot" />
        ) : null}
        {flown && m.cursorS >= flown.tS[0] ? (
          <circle cx={px(flown.eM[m.indexAt(flown.tS, m.cursorS)])} cy={py(flown.nM[m.indexAt(flown.tS, m.cursorS)])} r={4.5}
            fill={TRAINING_EXECUTOR_COLOR} stroke="black" strokeWidth={0.6} />
        ) : null}
      </g>
      <text x={GUTTER + 4} y={PLAN_H - 6} className="training-readback-tick">
        runway {designated.ident} · course {designated.courseDeg.toFixed(0)}°
      </text>
    </ChartFrame>
  );
}
