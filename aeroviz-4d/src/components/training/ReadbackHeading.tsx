/**
 * ReadbackHeading.tsx
 * -------------------
 * The read-back window's heading chart, against time: each heading word's BAND (instruction-v3, vocabulary design §10.1)
 * — its target ± the heading tolerance over the rows it is judged on, from a lead after it is said to the next heading
 * word's, never past the clearance — with the rows outside in red; the capture turn's span onto the course; the course
 * band after the capture; the executor's heading bands on its own clock (teal outlines), and the live segment's.
 */

import { useId } from "react";
import {
  TRAINING_AUTOPILOT_COLOR,
  TRAINING_CAPTURE_TURN_COLOR,
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import { outsideSpans, trainingWordAt } from "../../data/trainingSample";
import { checkMark } from "../../data/trainingText";
import { Axis, bandRect, ChartFrame, Line, VLine } from "./chartKit";
import { CHART_H, GUTTER, judged, PLOT_H, PLOT_TOP, timeTicks, type ReadbackModel } from "./readbackModel";

export default function ReadbackHeading({ m, onCursorChange, onColumnChange }: {
  m: ReadbackModel; onCursorChange: (seconds: number) => void; onColumnChange: (column: "heading") => void;
}) {
  const clip = useId();
  const { flight, vocabulary, layers, capture, last, rowX, xTime, yHeading } = m;
  const { envelopes, signals } = flight;
  const bottom = PLOT_TOP + PLOT_H;
  const heading = envelopes.heading[trainingWordAt(flight, "heading", m.cursorRow).index];
  const inCaptureTurn = capture !== null && capture.startRow <= m.cursorRow && m.cursorRow <= capture.endRow;
  const approachStroke = m.approachFocused ? TRAINING_WORD_COLOR : m.captureOk ? TRAINING_CAPTURE_TURN_COLOR : TRAINING_OUTSIDE_COLOR;
  const caption = (
    <>
      heading — ground track, ° true, unwrapped · in force: {m.label("heading", m.inForce("heading"))}
      {judged(heading)
        ? ` · its band: steps ${heading.firstRow}–${heading.stopRow - 1} (${vocabulary.headingLeadS} s after it was said), ` +
          `${heading.check.inside}/${heading.check.rows} rows within ±${vocabulary.headingToleranceDeg}°`
        : ` · no row of its own: the ${vocabulary.headingLeadS} s lead carries it to the clearance`}
      {inCaptureTurn
        ? ` · the capture turn: ${checkMark(capture.check.progressOk)} monotone, ${checkMark(capture.check.rateOk)} rate ` +
          `(mean ${capture.check.meanRateDegS.toFixed(2)}°/s, max ${capture.check.maxRateDegS.toFixed(2)}°/s, ` +
          `max bank ${capture.check.maxBankDeg.toFixed(1)}°` +
          `${capture.check.rateMinApplies ? "" : `; under ${vocabulary.turnRateMinFromDeg}°, the lowest rate not judged`})`
        : ""}
      {m.cursorRow >= flight.captureRow ? " · captured: the corridor holds" : ""}
    </>
  );
  const pick = (x: number) => onCursorChange(m.timeAtX(x));

  return (
    <ChartFrame label="Heading chart" caption={caption} width={m.width} height={CHART_H} onPointer={pick}
      onPick={(x) => {
        pick(x);
        onColumnChange("heading");
      }}>
      <defs>
        <clipPath id={clip}>
          <rect x={GUTTER} y={PLOT_TOP} width={m.plotW} height={PLOT_H} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clip})`}>
        {layers.headingBands ? envelopes.heading.map((item, index) => {
          if (!judged(item)) return null;
          const selected = m.focused("heading", index);
          return (
            <rect key={`heading-band-${index}`} className="training-readback-heading-band"
              {...bandRect(rowX(item.firstRow), xTime(m.edge(item.stopRow)), yHeading(item.bandDeg[1]), yHeading(item.bandDeg[0]))}
              fill={TRAINING_HEADING_BAND_COLOR} fillOpacity={selected ? 0.34 : 0.16}
              stroke={selected ? TRAINING_WORD_COLOR : item.check.inside < item.check.rows ? TRAINING_OUTSIDE_COLOR : TRAINING_HEADING_BAND_COLOR}
              strokeWidth={selected ? 1.4 : 0.6} opacity={m.recede(selected)}>
              <title>
                {m.label("heading", item.value)} ±{vocabulary.headingToleranceDeg}°, said at step {item.row} and judged at
                steps {item.firstRow}–{item.stopRow - 1} ({vocabulary.headingLeadS} s later, to the next heading word's):{" "}
                {item.check.inside} of {item.check.rows} rows inside
              </title>
            </rect>
          );
        }) : null}
        {layers.corridor && capture !== null ? (
          <g aria-label="the capture turn, against time" opacity={m.recede(m.approachFocused)}>
            <rect className="training-readback-capture-band" {...bandRect(rowX(capture.startRow), rowX(capture.endRow), PLOT_TOP, bottom)}
              fill={TRAINING_CAPTURE_TURN_COLOR} fillOpacity={m.approachFocused ? 0.18 : 0.08} stroke={approachStroke}
              strokeDasharray="4 3" strokeWidth={0.9}>
              <title>
                the capture turn, steps {capture.startRow}–{capture.endRow}: from the clearance onto the course, monotone{" "}
                {checkMark(capture.check.progressOk)}, rate and bank {checkMark(capture.check.rateOk)}
              </title>
            </rect>
            <line className="training-readback-capture-course"
              x1={rowX(capture.startRow)} x2={Math.max(rowX(capture.endRow), rowX(capture.startRow) + 1)}
              y1={yHeading(capture.courseOnTrackDeg)} y2={yHeading(capture.courseOnTrackDeg)}
              stroke={m.approachFocused ? TRAINING_WORD_COLOR : TRAINING_CAPTURE_TURN_COLOR} strokeDasharray="2 2" />
          </g>
        ) : null}
        {layers.corridor ? (
          <rect className="training-readback-course-band"
            {...bandRect(rowX(flight.captureRow), rowX(last), yHeading(envelopes.approach.courseBandDeg[1]),
              yHeading(envelopes.approach.courseBandDeg[0]))}
            fill={TRAINING_CORRIDOR_COLOR} fillOpacity={m.approachFocused ? 0.36 : 0.22}
            stroke={m.approachFocused ? TRAINING_WORD_COLOR : "none"} strokeWidth={0.7} opacity={m.recede(m.approachFocused)}>
            <title>after the capture: the course ±{vocabulary.corridorCourseToleranceDeg}°</title>
          </rect>
        ) : null}
        {/* the executor's heading bands, on its own clock, from where IT was told each word */}
        {layers.headingBands ? m.flownBands.map((band, index) => (judged(band) ? (
          <rect key={`executor-band-${index}`} className="training-readback-executor-band"
            {...bandRect(xTime(m.flownEdge(band.firstRow)), xTime(m.flownEdge(band.stopRow)), yHeading(band.bandDeg[1]),
              yHeading(band.bandDeg[0]))}
            fill="none" stroke={band.inside.every(Boolean) ? TRAINING_EXECUTOR_COLOR : TRAINING_OUTSIDE_COLOR} strokeWidth={0.8} />
        ) : null)) : null}
        {/* the live segment's heading band, as its judge read it */}
        {layers.headingBands && m.liveBand && judged(m.liveBand) ? (
          <rect className="training-readback-autopilot-band"
            {...bandRect(xTime(m.liveBand.firstRow * vocabulary.stepS), xTime(m.liveBand.stopRow * vocabulary.stepS),
              yHeading(m.liveBand.bandDeg[1]), yHeading(m.liveBand.bandDeg[0]))}
            fill="none" stroke={m.liveBand.inside.every(Boolean) ? TRAINING_AUTOPILOT_COLOR : TRAINING_OUTSIDE_COLOR} strokeWidth={1.1}>
            <title>
              the autopilot's band for its heading word, judged at steps {m.liveBand.firstRow}–{m.liveBand.stopRow - 1} of its
              flown segment: {m.liveBand.inside.filter(Boolean).length} of {m.liveBand.inside.length} rows inside
            </title>
          </rect>
        ) : null}
      </g>
      <Line xs={signals.tS.map((_, row) => rowX(row))} ys={signals.raw.trackDeg.map(yHeading)} stroke={TRAINING_RAW_COLOR}
        width={1} opacity={0.4} className="training-readback-raw" />
      <Line xs={signals.tS.map((_, row) => rowX(row))} ys={signals.smoothed.trackDeg.map(yHeading)} stroke={TRAINING_TRACE_COLOR}
        width={1.4} className="training-readback-trace" />
      {m.flownTrack ? (
        <Line xs={m.flownTrack.tS.map(xTime)} ys={m.flownTrack.trackDeg.map(yHeading)} stroke={TRAINING_EXECUTOR_COLOR} width={1.4}
          dash="5 3" className="training-readback-executor" title="the executor's track, on its own clock" />
      ) : null}
      {m.live && m.liveJudged ? m.liveOutside.map(([first, lastStep]) => (
        <Line key={`autopilot-out-${first}`} className="training-readback-autopilot-outside" stroke={TRAINING_OUTSIDE_COLOR}
          width={2} dash="3 2" xs={m.rows(first, lastStep).map((step) => xTime((m.live!.segment.row + step) * vocabulary.stepS))}
          ys={m.rows(first, lastStep).map((step) => yHeading(m.liveJudged![step]))} />
      )) : null}
      {m.column === "heading" && m.focusRows.length >= 2 ? (
        <Line xs={m.focusRows.map(rowX)} ys={m.focusRows.map((row) => yHeading(signals.smoothed.trackDeg[row]))}
          stroke={TRAINING_WORD_COLOR} width={2.6} opacity={0.9} className="training-readback-focus" />
      ) : null}
      {m.live ? (
        <Line xs={m.live.track.tS.map(xTime)} ys={m.live.track.trackDeg.map(yHeading)} stroke={m.liveColour} width={1.8}
          className="training-readback-autopilot" title="the autopilot's track over its segment" />
      ) : null}
      {/* the rows each band judged outside: the observed track's, and the executor's as its judge read it */}
      {layers.headingBands ? envelopes.heading.flatMap((item, index) => outsideSpans(item.inside, item.firstRow, last)
        .map(([first, lastRow]) => (
          <Line key={`heading-out-${index}-${first}`} className="training-readback-outside" stroke={TRAINING_OUTSIDE_COLOR} width={2}
            xs={m.rows(first, lastRow).map(rowX)} ys={m.rows(first, lastRow).map((row) => yHeading(signals.smoothed.trackDeg[row]))} />
        ))) : null}
      {layers.headingBands && m.judgedTrack ? m.flownBands.flatMap((band, index) =>
        outsideSpans(band.inside, band.firstRow, m.judgedTrack!.length - 1).map(([first, lastStep]) => (
          <Line key={`executor-out-${index}-${first}`} className="training-readback-executor-outside" stroke={TRAINING_OUTSIDE_COLOR}
            width={2} dash="3 2" xs={m.rows(first, lastStep).map((step) => xTime(m.flownTrack!.tS[step]))}
            ys={m.rows(first, lastStep).map((step) => yHeading(m.judgedTrack![step]))} />
        ))) : null}
      {envelopes.heading.map((item, index) => (
        <circle key={`heading-issue-${index}`} cx={rowX(item.row)} cy={yHeading(signals.smoothed.trackDeg[item.row])} r={2.6}
          fill="none" stroke={TRAINING_COLUMN_COLOR.heading} strokeWidth={1.3} />
      ))}
      <VLine x={rowX(flight.joinRow)} top={PLOT_TOP} bottom={bottom} stroke={TRAINING_COLUMN_COLOR.approach} dash="3 3"
        title={`cleared at step ${flight.joinRow}`} />
      <VLine x={xTime(m.cursorS)} top={PLOT_TOP} bottom={bottom} className="training-readback-cursor-line" />
      <Axis left={GUTTER} right={GUTTER + m.plotW} y={bottom} ticks={timeTicks(m)} caption="s from the first step" />
    </ChartFrame>
  );
}
