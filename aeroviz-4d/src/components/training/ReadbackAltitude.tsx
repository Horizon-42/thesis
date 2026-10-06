/**
 * ReadbackAltitude.tsx
 * --------------------
 * The read-back window's altitude chart, against flight time: each altitude word's tube (a level above the airport
 * elevation E, in MSL, or "no level-off") over the rows it covers, the angle words, the threshold's elevation, the rows
 * outside in red — on the judged track; the observed track and the flown path beside each other; the DA point; the live
 * segment.
 */

import {
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRECTION_COLOR,
  TRAINING_DECISION_FAIL_COLOR,
  TRAINING_DECISION_PASS_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import { closedCycleTimeS, sentenceColumnRuns, trainingBandLabel } from "../../data/trainingSample";
import { checkMark, decisionText } from "../../data/trainingText";
import AutopilotLine from "./AutopilotLine";
import { Axis, ChartFrame, Line, VLine } from "./chartKit";
import { CorrectionTicks, OutcomeLine } from "./chartMarks";
import { CHART_H, GUTTER, PLOT_H, PLOT_TOP, timeTicks, type ReadbackModel } from "./readbackModel";

export default function ReadbackAltitude({ m, onCursorChange, onColumnChange }: {
  m: ReadbackModel; onCursorChange: (seconds: number) => void; onColumnChange: (column: "altitude") => void;
}) {
  const { observed, flown, judged, yAltitude, designated, decision } = m;
  const bottom = PLOT_TOP + PLOT_H;
  const tube = m.envelopes?.altitude.find((item) => m.envelopeTimeS(item.row) <= m.cursorS && m.cursorS < m.envelopeTimeS(item.endRow)) ?? null;
  const pick = (x: number) => onCursorChange(m.timeAtX(x));
  const angleRuns = sentenceColumnRuns(m.reading, "angle");
  // an angle label only where it clears the last one
  let lastLabelX = -Infinity;
  const focusActive = (m.column === "altitude" || m.column === "angle") && m.focusPoints(judged.tS).length >= 2;

  return (
    <ChartFrame captionIndent={GUTTER} label="Altitude chart" width={m.width} height={CHART_H} onPointer={pick}
      onPick={(x) => {
        pick(x);
        onColumnChange("altitude");
      }}
      caption={`altitude — geometric MSL (m) · in force: ${m.label("altitude")} above E, angle ${m.label("angle")}` +
        (tube === null ? "" : ` · this tube: ${tube.inside}/${tube.rows} rows inside ${checkMark(tube.contained)}`)}>
      {m.tubes.map((item, index) => {
        const times = item.lowMslM.map((_, offset) => m.xTime(m.envelopeTimeS(item.row + offset)));
        const selected = m.focused("altitude", index);
        return (
          <polygon key={`tube-${index}`} className="training-readback-tube"
            points={[
              ...times.map((x, offset) => `${x},${yAltitude(item.highMslM[offset])}`),
              ...times.map((x, offset) => `${x},${yAltitude(item.lowMslM[offset])}`).reverse(),
            ].join(" ")}
            fill={TRAINING_TUBE_COLOR} fillOpacity={selected ? 0.34 : 0.16}
            stroke={selected ? TRAINING_WORD_COLOR : item.contained ? TRAINING_TUBE_COLOR : TRAINING_OUTSIDE_COLOR}
            strokeWidth={selected ? 1.2 : 0.7} opacity={m.recede(selected)}>
            <title>
              {item.levelM === null ? "no level-off" : `${item.levelM.toFixed(0)} m above E`} from {m.envelopeTimeS(item.row)} s: the tube, re-anchored
              at each angle word — {item.inside} of {item.rows} rows inside
            </title>
          </polygon>
        );
      })}
      <line x1={GUTTER} x2={GUTTER + m.plotW} y1={yAltitude(designated.elevationM)} y2={yAltitude(designated.elevationM)}
        stroke={TRAINING_DESIGNATED_COLOR} strokeDasharray="2 3" className="training-readback-elevation" />
      <text x={GUTTER + m.plotW - 2} y={yAltitude(designated.elevationM) - 3} textAnchor="end" className="training-readback-tick"
        fill={TRAINING_DESIGNATED_COLOR}>
        threshold {designated.ident} {designated.elevationM.toFixed(1)} m
      </text>
      {angleRuns.map((run) => {
        const x = m.xTime(m.rowTimeS(run.row));
        const selected = m.column === "angle" && m.focus?.row === run.row;
        const colour = selected ? TRAINING_WORD_COLOR : run.event.correction ? TRAINING_CORRECTION_COLOR : TRAINING_COLUMN_COLOR.angle;
        const labelled = x - lastLabelX >= 44;
        if (labelled) lastLabelX = x;
        return (
          <g key={`angle-${run.row}`} aria-label={`angle word ${trainingBandLabel(run.event.says)} at ${m.rowTimeS(run.row)} s`}
            opacity={m.recede(selected)}>
            <line x1={x} x2={x} y1={PLOT_TOP} y2={bottom} stroke={colour} strokeOpacity={selected ? 1 : 0.6} strokeDasharray="2 2" />
            {labelled ? (
              <text x={x + 2} y={PLOT_TOP + 9} className="training-readback-tick" fill={colour}>{trainingBandLabel(run.event.says)}</text>
            ) : null}
          </g>
        );
      })}
      <Line xs={observed.tS.map(m.xTime)} ys={observed.altitudeMslM.map(yAltitude)} stroke={TRAINING_TRACE_COLOR} width={1.4}
        className="training-readback-trace" title="the observed track" />
      {flown ? (
        <Line xs={flown.tS.map(m.xTime)} ys={flown.altitudeMslM.map(yAltitude)} stroke={m.flownColour} width={1.6}
          className="training-readback-executor" title="the flown path's altitude" />
      ) : null}
      {focusActive ? (
        <Line xs={m.focusPoints(judged.tS).map((i) => m.xTime(judged.tS[i]))}
          ys={m.focusPoints(judged.tS).map((i) => yAltitude(judged.altitudeMslM[i]))}
          stroke={TRAINING_WORD_COLOR} width={2.6} opacity={0.9} round className="training-readback-focus" />
      ) : null}
      {m.live ? (
        <AutopilotLine m={m} x={(index) => m.xTime(m.live!.track.tS[index])} y={(index) => yAltitude(m.live!.track.altitudeMslM[index])}
          title="the autopilot's altitude over its segment" />
      ) : null}
      {decision !== null && m.closed !== null ? (
        <circle cx={m.xTime(closedCycleTimeS(m.closed, decision.cycle))} cy={yAltitude(decision.heightMslM)} r={5} fill="none" strokeWidth={2}
          stroke={decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR}>
          <title>{decisionText(decision)}</title>
        </circle>
      ) : null}
      <CorrectionTicks m={m} columns={["altitude", "angle"]} bottom={bottom} colour={TRAINING_CORRECTION_COLOR}
        label={(event) => `${event.says.column} ${trainingBandLabel(event.says)}`} />
      <OutcomeLine m={m} top={PLOT_TOP} bottom={bottom} />
      <VLine x={m.xTime(Math.min(m.cursorS, m.endS))} top={PLOT_TOP} bottom={bottom} className="training-readback-cursor-line" />
      <Axis left={GUTTER} right={GUTTER + m.plotW} y={bottom} ticks={timeTicks(m)} caption="s from the track's first row" />
    </ChartFrame>
  );
}
