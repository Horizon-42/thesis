/**
 * ReadbackSpeed.tsx
 * -----------------
 * The read-back window's speed chart, against time: each speed word's transition and band, the "unspecified" spans,
 * the clearance, the rows outside in red; the executor's ground speed on its own clock and the live segment's.
 */

import {
  TRAINING_COLUMN_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_SPEED_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import { outsideSpans } from "../../data/trainingSample";
import { checkMark } from "../../data/trainingText";
import { Axis, bandRect, ChartFrame, Line, VLine } from "./chartKit";
import { CHART_H, GUTTER, PLOT_H, PLOT_TOP, timeTicks, type ReadbackModel } from "./readbackModel";

export default function ReadbackSpeed({ m, onCursorChange, onColumnChange }: {
  m: ReadbackModel; onCursorChange: (seconds: number) => void; onColumnChange: (column: "speed") => void;
}) {
  const { flight, vocabulary, layers, last, rowX, xTime, ySpeed } = m;
  const { envelopes, signals } = flight;
  const bottom = PLOT_TOP + PLOT_H;
  const pick = (x: number) => onCursorChange(m.timeAtX(x));

  return (
    <ChartFrame captionIndent={GUTTER} label="Speed chart" width={m.width} height={CHART_H} onPointer={pick}
      onPick={(x) => {
        pick(x);
        onColumnChange("speed");
      }}
      caption={`speed — ground speed (m/s) · in force: ${m.label("speed", m.inForce("speed"))}`}>
      {layers.vertical ? envelopes.speed.map((span, index) => {
        const selected = m.focused("speed", index);
        if (span.targetMps === null) {
          return (
            <rect key={`speed-${index}`} className="training-readback-unspecified" opacity={m.recede(selected)}
              {...bandRect(rowX(span.row), xTime(m.edge(span.endRow)), PLOT_TOP, bottom)}
              fill={TRAINING_RAW_COLOR} fillOpacity={selected ? 0.22 : 0.12}
              stroke={selected ? TRAINING_WORD_COLOR : "none"} strokeWidth={1.4}>
              <title>unspecified from step {span.row}: the pilot's own speed — only the range {span.rangeMps![0]}–{span.rangeMps![1]} m/s holds</title>
            </rect>
          );
        }
        const check = span.check!;
        const transition = m.rows(span.row, span.row + span.transitionLowerMps!.length - 1);
        return (
          <g key={`speed-${index}`} aria-label={`speed ${m.label("speed", span.value)} from step ${span.row}`}
            opacity={m.recede(selected)}>
            <polygon className="training-readback-transition"
              points={[
                ...transition.map((row, offset) => `${rowX(row)},${ySpeed(span.transitionUpperMps![offset])}`),
                ...transition.map((row, offset) => `${rowX(row)},${ySpeed(span.transitionLowerMps![offset])}`).reverse(),
              ].join(" ")}
              fill={TRAINING_SPEED_COLOR} fillOpacity={selected ? 0.2 : 0.1}
              stroke={selected ? TRAINING_WORD_COLOR : check.transitionOk && check.accelOk ? TRAINING_SPEED_COLOR : TRAINING_OUTSIDE_COLOR}
              strokeOpacity={selected ? 1 : 0.6} strokeWidth={selected ? 1.4 : 0.6}>
              <title>
                transition to {m.label("speed", span.value)}: monotone {checkMark(check.transitionOk)}, at most{" "}
                {vocabulary.speedAccelMaxMps2} m/s² {checkMark(check.accelOk)}
              </title>
            </polygon>
            {span.arrivalRow !== null ? (
              <rect className="training-readback-speed-band"
                {...bandRect(rowX(span.arrivalRow), xTime(m.edge(span.endRow)), ySpeed(span.bandMps![1]), ySpeed(span.bandMps![0]))}
                fill={TRAINING_SPEED_COLOR} fillOpacity={selected ? 0.34 : 0.2}
                stroke={selected ? TRAINING_WORD_COLOR : check.contained ? TRAINING_SPEED_COLOR : TRAINING_OUTSIDE_COLOR}
                strokeWidth={selected ? 1.4 : 0.7}>
                <title>
                  {m.label("speed", span.value)} ±{vocabulary.speedToleranceMps} m/s: {check.bandInside} of {check.bandRows} rows inside
                </title>
              </rect>
            ) : null}
          </g>
        );
      }) : null}
      <Line xs={signals.tS.map((_, row) => rowX(row))} ys={signals.raw.groundSpeedMps.map(ySpeed)} stroke={TRAINING_RAW_COLOR}
        width={1} opacity={0.4} className="training-readback-raw" />
      <Line xs={signals.tS.map((_, row) => rowX(row))} ys={signals.smoothed.groundSpeedMps.map(ySpeed)} stroke={TRAINING_TRACE_COLOR}
        width={1.4} className="training-readback-trace" />
      {m.flownTrack ? (
        <Line xs={m.flownTrack.tS.map(xTime)} ys={m.flownTrack.groundSpeedMps.map(ySpeed)} stroke={TRAINING_EXECUTOR_COLOR}
          width={1.4} dash="5 3" className="training-readback-executor" title="the executor's ground speed, on its own clock" />
      ) : null}
      {m.column === "speed" && m.focusRows.length >= 2 ? (
        <Line xs={m.focusRows.map(rowX)} ys={m.focusRows.map((row) => ySpeed(signals.smoothed.groundSpeedMps[row]))}
          stroke={TRAINING_WORD_COLOR} width={2.6} opacity={0.9} round className="training-readback-focus" />
      ) : null}
      {m.live ? (
        <Line xs={m.live.track.tS.map(xTime)} ys={m.live.track.groundSpeedMps.map(ySpeed)} stroke={m.liveColour} width={1.8}
          className="training-readback-autopilot" title="the autopilot's ground speed over its segment" />
      ) : null}
      {layers.vertical ? envelopes.speed.flatMap((span, index) => (span.arrivalRow === null || span.bandInside === null ? []
        : outsideSpans(span.bandInside, span.arrivalRow, last).map(([first, lastRow]) => (
          <Line key={`speed-out-${index}-${first}`} className="training-readback-outside" stroke={TRAINING_OUTSIDE_COLOR} width={2}
            xs={m.rows(first, lastRow).map(rowX)} ys={m.rows(first, lastRow).map((row) => ySpeed(signals.smoothed.groundSpeedMps[row]))} />
        )))) : null}
      <VLine x={rowX(flight.joinRow)} top={PLOT_TOP} bottom={bottom} stroke={TRAINING_COLUMN_COLOR.approach} dash="3 3"
        title={`cleared at step ${flight.joinRow}`} />
      <VLine x={xTime(m.cursorS)} top={PLOT_TOP} bottom={bottom} className="training-readback-cursor-line" />
      <Axis left={GUTTER} right={GUTTER + m.plotW} y={bottom} ticks={timeTicks(m)} caption="s from the first step" />
    </ChartFrame>
  );
}
