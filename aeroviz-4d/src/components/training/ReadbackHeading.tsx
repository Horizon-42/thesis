/**
 * ReadbackHeading.tsx
 * -------------------
 * The read-back window's heading chart, against flight time: each heading word's BAND — its target ± the heading
 * tolerance over the rows it is judged on, from a lead after it is said — with the rows outside in red, on the judged
 * track (the observed one for the labelled sentence, the flown path for a closed-loop one); the observed and the flown
 * track beside each other; the words the closed-loop reading added as orange ticks on the axis; where the flown flight
 * ended; the live segment.
 */

import { useId } from "react";
import {
  TRAINING_AUTOPILOT_COLOR,
  TRAINING_CORRECTION_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
} from "../../utils/trainingWordColors";
import { outsideSpans, trainingBandLabel } from "../../data/trainingSample";
import AutopilotLine from "./AutopilotLine";
import { Axis, bandRect, ChartFrame, Line, VLine } from "./chartKit";
import { CorrectionTicks, OutcomeLine } from "./chartMarks";
import { CHART_H, GUTTER, PLOT_H, PLOT_TOP, timeTicks, type ReadbackModel } from "./readbackModel";

export default function ReadbackHeading({ m, onCursorChange, onColumnChange }: {
  m: ReadbackModel; onCursorChange: (seconds: number) => void; onColumnChange: (column: "heading") => void;
}) {
  const clip = useId();
  const { vocabulary, layers, observed, flown, judged, yHeading } = m;
  const bottom = PLOT_TOP + PLOT_H;
  const run = m.focus !== null && m.column === "heading" ? m.focus : null;
  const inForce = m.label("heading");
  const here = m.envelopes?.heading.find((band) => m.envelopeTimeS(band.row) <= m.cursorS
    && m.cursorS < m.envelopeTimeS(band.stopRow)) ?? null;
  const caption = (
    <>
      heading — ground track, ° true, continuous · in force: {inForce}{inForce === "—" ? "" : " from the course"}
      {here !== null && here.stopRow > here.firstRow
        ? ` · its band: ${here.inside.filter(Boolean).length}/${here.inside.length} rows within ±${here.toleranceDeg}° of ` +
          `${here.targetDeg.toFixed(0)}° (${vocabulary.headingLeadS} s after it was said)`
        : ""}
      {run !== null && run.event.correction ? " · the selected word is a correction" : ""}
    </>
  );
  const pick = (x: number) => onCursorChange(m.timeAtX(x));

  return (
    <ChartFrame captionIndent={GUTTER} label="Heading chart" caption={caption} width={m.width} height={CHART_H} onPointer={pick}
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
        {m.envelopes !== null && layers.headingBands ? m.envelopes.heading.map((band, index) => {
          if (band.stopRow <= band.firstRow) return null;
          const selected = m.focused("heading", index);
          const centre = m.bandCentre(band);
          return (
            <rect key={`heading-band-${index}`} className="training-readback-heading-band"
              {...bandRect(m.xTime(m.envelopeTimeS(band.firstRow)), m.xTime(m.envelopeTimeS(band.stopRow)),
                yHeading(centre + band.toleranceDeg), yHeading(centre - band.toleranceDeg))}
              fill={TRAINING_HEADING_BAND_COLOR} fillOpacity={selected ? 0.34 : 0.16}
              stroke={selected ? TRAINING_WORD_COLOR : band.inside.every(Boolean) ? TRAINING_HEADING_BAND_COLOR : TRAINING_OUTSIDE_COLOR}
              strokeWidth={selected ? 1.4 : 0.6} opacity={m.recede(selected)}>
              <title>
                track {band.targetDeg.toFixed(0)}° ±{band.toleranceDeg}°, said at {m.envelopeTimeS(band.row)} s and judged from{" "}
                {m.envelopeTimeS(band.firstRow)} s to {m.envelopeTimeS(band.stopRow)} s: {band.inside.filter(Boolean).length} of{" "}
                {band.inside.length} rows inside
              </title>
            </rect>
          );
        }) : null}
      </g>
      <Line xs={observed.tS.map(m.xTime)} ys={observed.trackPlotDeg.map(yHeading)} stroke={TRAINING_TRACE_COLOR} width={1.4}
        className="training-readback-trace" title="the observed track" />
      {flown ? (
        <Line xs={flown.tS.map(m.xTime)} ys={flown.trackPlotDeg.map(yHeading)} stroke={m.flownColour} width={1.6}
          className="training-readback-executor" title="the flown path: the sentence on screen flown by the executor" />
      ) : null}
      {m.column === "heading" && m.focusPoints(judged.tS).length >= 2 ? (
        <Line xs={m.focusPoints(judged.tS).map((i) => m.xTime(judged.tS[i]))}
          ys={m.focusPoints(judged.tS).map((i) => yHeading(judged.trackPlotDeg[i]))}
          stroke={TRAINING_WORD_COLOR} width={2.6} opacity={0.9} round className="training-readback-focus" />
      ) : null}
      {m.live ? (
        <AutopilotLine m={m} x={(index) => m.xTime(m.live!.track.tS[index])} y={(index) => yHeading(m.live!.track.trackPlotDeg[index])}
          title="the autopilot's track over its segment" />
      ) : null}
      {/* the rows each band judged outside, on the judged track */}
      {layers.headingBands && m.envelopes !== null ? m.envelopes.heading.flatMap((band, index) =>
        outsideSpans(band.inside, band.firstRow, judged.tS.length - 1).map(([first, last]) => (
          <Line key={`heading-out-${index}-${first}`} className="training-readback-outside" stroke={TRAINING_OUTSIDE_COLOR} width={2}
            xs={m.rows(first, last).map((row) => m.xTime(judged.tS[row]))}
            ys={m.rows(first, last).map((row) => yHeading(judged.trackPlotDeg[row]))} />
        ))) : null}
      <CorrectionTicks m={m} columns={["heading"]} bottom={bottom} colour={TRAINING_CORRECTION_COLOR}
        label={(event) => `heading ${trainingBandLabel(event.says)}`} />
      <OutcomeLine m={m} top={PLOT_TOP} bottom={bottom} />
      {m.live ? <circle cx={m.xTime(m.live.track.tS[0])} cy={yHeading(m.live.track.trackPlotDeg[0])} r={3} fill={TRAINING_AUTOPILOT_COLOR} /> : null}
      <VLine x={m.xTime(Math.min(m.cursorS, m.endS))} top={PLOT_TOP} bottom={bottom} className="training-readback-cursor-line" />
      <Axis left={GUTTER} right={GUTTER + m.plotW} y={bottom} ticks={timeTicks(m)} caption="s from the track's first row" />
    </ChartFrame>
  );
}
