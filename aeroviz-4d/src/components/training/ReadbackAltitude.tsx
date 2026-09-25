/**
 * ReadbackAltitude.tsx
 * --------------------
 * The read-back window's altitude chart, against the horizontal distance flown — the axis the tubes are defined on:
 * each altitude word's tube, the angle words that re-anchor it, the runway's elevation, the rows outside in red; the
 * executor's altitude and the live segment's against their own distance flown.
 */

import {
  TRAINING_COLUMN_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import { outsideSpans, rowAtTime, trainingWordAt } from "../../data/trainingSample";
import { checkMark } from "../../data/trainingText";
import { Axis, ChartFrame, Line, VLine } from "./chartKit";
import { CHART_H, distanceTicks, GUTTER, PLOT_H, PLOT_TOP, type ReadbackModel } from "./readbackModel";

export default function ReadbackAltitude({ m, onCursorChange, onColumnChange }: {
  m: ReadbackModel; onCursorChange: (seconds: number) => void; onColumnChange: (column: "altitude") => void;
}) {
  const { flight, vocabulary, layers, last, rowXDistance, xDistance, yAltitude, designated } = m;
  const { envelopes, signals } = flight;
  const bottom = PLOT_TOP + PLOT_H;
  const tube = envelopes.altitude[trainingWordAt(flight, "altitude", m.cursorRow).index];
  const pick = (x: number) => onCursorChange(signals.tS[rowAtTime(signals.smoothed.distanceM, m.distanceAtX(x))]);
  const rowsOf = (first: number, count: number) => m.rows(first, first + count - 1);

  return (
    <ChartFrame label="Altitude chart" width={m.width} height={CHART_H} onPointer={pick}
      onPick={(x) => {
        pick(x);
        onColumnChange("altitude");
      }}
      caption={`altitude — geometric MSL (m) against distance flown · in force: ${m.label("altitude", m.inForce("altitude"))}, ` +
        `${m.label("angle", m.inForce("angle"))} · this tube: ${tube.check.inside}/${tube.check.rows} rows inside ` +
        `${checkMark(tube.check.contained)}`}>
      {layers.vertical ? envelopes.altitude.map((item, index) => {
        const rows = rowsOf(item.row, item.lowerM.length);
        const selected = m.focused("altitude", index);
        return (
          <polygon key={`tube-${index}`} className="training-readback-tube"
            points={[
              ...rows.map((row, offset) => `${rowXDistance(row)},${yAltitude(item.upperM[offset])}`),
              ...rows.map((row, offset) => `${rowXDistance(row)},${yAltitude(item.lowerM[offset])}`).reverse(),
            ].join(" ")}
            fill={TRAINING_TUBE_COLOR} fillOpacity={selected ? 0.34 : 0.16}
            stroke={selected ? TRAINING_WORD_COLOR : item.check.contained ? TRAINING_TUBE_COLOR : TRAINING_OUTSIDE_COLOR}
            strokeWidth={selected ? 1.2 : 0.7} opacity={m.recede(selected)}>
            <title>
              {m.label("altitude", item.value)} from step {item.row}: the tube ±{vocabulary.altitudeToleranceM} m, re-anchored at
              each angle word, {item.check.tubeWidthEndM.toFixed(0)} m wide at its end — {item.check.inside} of {item.check.rows} rows
              inside
            </title>
          </polygon>
        );
      }) : null}
      <line x1={GUTTER} x2={GUTTER + m.plotW} y1={yAltitude(designated.elevationM)} y2={yAltitude(designated.elevationM)}
        stroke={TRAINING_DESIGNATED_COLOR} strokeDasharray="2 3" className="training-readback-elevation" />
      <text x={GUTTER + m.plotW - 2} y={yAltitude(designated.elevationM) - 3} textAnchor="end" className="training-readback-tick"
        fill={TRAINING_DESIGNATED_COLOR}>
        threshold {designated.ident} {designated.elevationM.toFixed(1)} m
      </text>
      {envelopes.angle.map((item, index) => {
        const selected = m.focused("angle", index);
        const colour = selected ? TRAINING_WORD_COLOR : TRAINING_COLUMN_COLOR.angle;
        return (
          <g key={`angle-${index}`} aria-label={`angle word ${m.label("angle", item.value)} at step ${item.row}`}
            opacity={m.recede(selected)}>
            <line x1={rowXDistance(item.row)} x2={rowXDistance(item.row)} y1={PLOT_TOP} y2={bottom} stroke={colour}
              strokeOpacity={selected ? 1 : 0.6} strokeDasharray="2 2" />
            <text x={rowXDistance(item.row) + 2} y={PLOT_TOP + 9} className="training-readback-tick" fill={colour}>
              {vocabulary.angleClasses[item.value].name}{item.measuredDeg === null ? "" : ` ${item.measuredDeg.toFixed(2)}°`}
            </text>
          </g>
        );
      })}
      <Line xs={signals.tS.map((_, row) => rowXDistance(row))} ys={signals.raw.altitudeM.map(yAltitude)} stroke={TRAINING_RAW_COLOR}
        width={1} opacity={0.4} className="training-readback-raw" />
      <Line xs={signals.tS.map((_, row) => rowXDistance(row))} ys={signals.smoothed.altitudeM.map(yAltitude)}
        stroke={TRAINING_TRACE_COLOR} width={1.4} className="training-readback-trace" />
      {m.flownTrack ? (
        <Line xs={m.flownTrack.distanceM.map(xDistance)} ys={m.flownTrack.altitudeM.map(yAltitude)} stroke={TRAINING_EXECUTOR_COLOR}
          width={1.4} dash="5 3" className="training-readback-executor" title="the executor's altitude, against its own distance flown" />
      ) : null}
      {(m.column === "altitude" || m.column === "angle") && m.focusRows.length >= 2 ? (
        <Line xs={m.focusRows.map(rowXDistance)} ys={m.focusRows.map((row) => yAltitude(signals.smoothed.altitudeM[row]))}
          stroke={TRAINING_WORD_COLOR} width={2.6} opacity={0.9} className="training-readback-focus" />
      ) : null}
      {m.live ? (
        <Line xs={m.live.track.distanceM.map(xDistance)} ys={m.live.track.altitudeM.map(yAltitude)} stroke={m.liveColour}
          width={1.8} className="training-readback-autopilot" title="the autopilot's altitude over its segment, against the distance flown" />
      ) : null}
      {layers.vertical ? envelopes.altitude.flatMap((item, index) => outsideSpans(item.inside, item.row, last)
        .map(([first, lastRow]) => (
          <Line key={`altitude-out-${index}-${first}`} className="training-readback-outside" stroke={TRAINING_OUTSIDE_COLOR} width={2}
            xs={m.rows(first, lastRow).map(rowXDistance)} ys={m.rows(first, lastRow).map((row) => yAltitude(signals.smoothed.altitudeM[row]))} />
        ))) : null}
      <VLine x={rowXDistance(m.cursorRow)} top={PLOT_TOP} bottom={bottom} className="training-readback-cursor-line" />
      <Axis left={GUTTER} right={GUTTER + m.plotW} y={bottom} ticks={distanceTicks(m)}
        caption="km flown (the smoothed ground speed, integrated)" />
    </ChartFrame>
  );
}
