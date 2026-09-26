/**
 * AutopilotLine.tsx
 * -----------------
 * The live segment on a read-back chart: its RUN solid in the segment's colour, and its TAIL — past where the executor
 * heard the next word of the column, flown only because a heading word is judged to a lead after it — faded and dashed
 * (`autopilotRunAndTail`, the same split the 3D scene draws).
 */

import { AUTOPILOT_TAIL_DASH, AUTOPILOT_TAIL_OPACITY } from "../../data/trainingAutopilot";
import { Line } from "./chartKit";
import type { ReadbackModel } from "./readbackModel";

/** ``x`` / ``y``: a flown point's position on the chart, by its index in the flown track. */
export default function AutopilotLine({ m, x, y, title }: {
  m: ReadbackModel; x: (index: number) => number; y: (index: number) => number; title: string;
}) {
  return (
    <>
      <Line xs={m.liveRun.map(x)} ys={m.liveRun.map(y)} stroke={m.liveColour} width={1.8} className="training-readback-autopilot"
        title={title} />
      {m.liveTail.length >= 2 ? (
        <Line xs={m.liveTail.map(x)} ys={m.liveTail.map(y)} stroke={m.liveColour} width={1.8} dash={AUTOPILOT_TAIL_DASH}
          opacity={AUTOPILOT_TAIL_OPACITY} className="training-readback-autopilot-tail"
          title={`past where it heard the next ${m.live!.segment.column} word: already flying that word, still judged for this one`} />
      ) : null}
    </>
  );
}
