/**
 * ReadbackSpeed.tsx
 * -----------------
 * The read-back window's speed chart, against flight time: each speed word's band (its ground speed ± the tolerance from
 * the row the speed arrived), the transition to it, and the spans where the speed is left to the pilot; the observed
 * track and the flown path beside each other; the live segment.
 */

import {
  TRAINING_CORRECTION_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_SPEED_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import { trainingBandLabel } from "../../data/trainingSample";
import { checkMark } from "../../data/trainingText";
import AutopilotLine from "./AutopilotLine";
import { Axis, bandRect, ChartFrame, Line, VLine } from "./chartKit";
import { CorrectionTicks, OutcomeLine } from "./chartMarks";
import { CHART_H, GUTTER, PLOT_H, PLOT_TOP, timeTicks, type ReadbackModel } from "./readbackModel";

export default function ReadbackSpeed({ m, onCursorChange, onColumnChange }: {
  m: ReadbackModel; onCursorChange: (seconds: number) => void; onColumnChange: (column: "speed") => void;
}) {
  const { observed, flown, judged, ySpeed } = m;
  const bottom = PLOT_TOP + PLOT_H;
  const pick = (x: number) => onCursorChange(m.timeAtX(x));

  return (
    <ChartFrame captionIndent={GUTTER} label="Speed chart" width={m.width} height={CHART_H} onPointer={pick}
      onPick={(x) => {
        pick(x);
        onColumnChange("speed");
      }}
      caption={`speed — ground speed (m/s) · in force: ${m.label("speed")}`}>
      {m.layers.vertical ? m.unspecifiedRuns.map((run) => {
        const selected = m.column === "speed" && m.focus?.row === run.row;
        return (
          <rect key={`unspecified-${run.row}`} className="training-readback-unspecified" opacity={m.recede(selected)}
            {...bandRect(m.xTime(m.rowTimeS(run.row)), m.xTime(m.rowTimeS(run.endRow)), PLOT_TOP, bottom)}
            fill={TRAINING_RAW_COLOR} fillOpacity={selected ? 0.22 : 0.12} stroke={selected ? TRAINING_WORD_COLOR : "none"} strokeWidth={1.4}>
            <title>unspecified from {m.rowTimeS(run.row)} s: the pilot's own speed</title>
          </rect>
        );
      }) : null}
      {m.spans.map((span, index) => {
        const selected = m.focused("speed", index);
        const ok = span.transitionOk && span.accelOk;
        return (
          <g key={`speed-${index}`} aria-label={`speed ${span.targetMps} m/s from ${m.envelopeTimeS(span.row)} s`} opacity={m.recede(selected)}>
            <line className="training-readback-transition" x1={m.xTime(m.envelopeTimeS(span.row))} x2={m.xTime(m.envelopeTimeS(span.arrivalRow))}
              y1={ySpeed(span.targetMps)} y2={ySpeed(span.targetMps)} stroke={selected ? TRAINING_WORD_COLOR : ok ? TRAINING_SPEED_COLOR : TRAINING_OUTSIDE_COLOR}
              strokeDasharray="3 2" strokeWidth={1}>
              <title>
                transition to {span.targetMps} m/s: monotone and at most the acceleration limit {checkMark(ok)}
              </title>
            </line>
            <rect className="training-readback-speed-band"
              {...bandRect(m.xTime(m.envelopeTimeS(span.arrivalRow)), m.xTime(m.envelopeTimeS(span.endRow)),
                ySpeed(span.targetMps + span.toleranceMps), ySpeed(span.targetMps - span.toleranceMps))}
              fill={TRAINING_SPEED_COLOR} fillOpacity={selected ? 0.34 : 0.2}
              stroke={selected ? TRAINING_WORD_COLOR : span.contained ? TRAINING_SPEED_COLOR : TRAINING_OUTSIDE_COLOR}
              strokeWidth={selected ? 1.4 : 0.7}>
              <title>
                {span.targetMps} ±{span.toleranceMps} m/s from {m.envelopeTimeS(span.arrivalRow)} s to {m.envelopeTimeS(span.endRow)} s:
                {span.contained ? " held" : " not held"} {checkMark(span.contained)}
              </title>
            </rect>
          </g>
        );
      })}
      <Line xs={observed.tS.map(m.xTime)} ys={observed.groundSpeedMps.map(ySpeed)} stroke={TRAINING_TRACE_COLOR} width={1.4}
        className="training-readback-trace" title="the observed track" />
      {flown ? (
        <Line xs={flown.tS.map(m.xTime)} ys={flown.groundSpeedMps.map(ySpeed)} stroke={TRAINING_EXECUTOR_COLOR} width={1.6}
          className="training-readback-executor" title="the flown path's ground speed" />
      ) : null}
      {m.column === "speed" && m.focusPoints(judged.tS).length >= 2 ? (
        <Line xs={m.focusPoints(judged.tS).map((i) => m.xTime(judged.tS[i]))}
          ys={m.focusPoints(judged.tS).map((i) => ySpeed(judged.groundSpeedMps[i]))}
          stroke={TRAINING_WORD_COLOR} width={2.6} opacity={0.9} round className="training-readback-focus" />
      ) : null}
      {m.live ? (
        <AutopilotLine m={m} x={(index) => m.xTime(m.live!.track.tS[index])} y={(index) => ySpeed(m.live!.track.groundSpeedMps[index])}
          title="the autopilot's ground speed over its segment" />
      ) : null}
      <CorrectionTicks m={m} columns={["speed"]} bottom={bottom} colour={TRAINING_CORRECTION_COLOR}
        label={(event) => `speed ${trainingBandLabel(event.says)}`} />
      <OutcomeLine m={m} top={PLOT_TOP} bottom={bottom} />
      <VLine x={m.xTime(Math.min(m.cursorS, m.endS))} top={PLOT_TOP} bottom={bottom} className="training-readback-cursor-line" />
      <Axis left={GUTTER} right={GUTTER + m.plotW} y={bottom} ticks={timeTicks(m)} caption="s from the track's first row" />
    </ChartFrame>
  );
}
