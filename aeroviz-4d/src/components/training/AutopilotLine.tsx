/**
 * AutopilotLine.tsx
 * -----------------
 * The live segment on a read-back chart: its line in the segment's colour (blue, or the failure red when the flight was
 * flown on to an outcome other than a landing, `autopilotColour`).
 */

import { Line } from "./chartKit";
import type { ReadbackModel } from "./readbackModel";

/** ``x`` / ``y``: a flown point's position on the chart, by its index in the flown track. */
export default function AutopilotLine({ m, x, y, title }: {
  m: ReadbackModel; x: (index: number) => number; y: (index: number) => number; title: string;
}) {
  const points = m.live!.track.tS.map((_, index) => index);
  return (
    <Line xs={points.map(x)} ys={points.map(y)} stroke={m.liveColour} width={1.8} className="training-readback-autopilot"
      title={title} />
  );
}
