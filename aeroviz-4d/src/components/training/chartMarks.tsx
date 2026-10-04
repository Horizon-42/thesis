/**
 * chartMarks.tsx
 * --------------
 * The marks of the closed-loop reading every time chart shares: an orange tick on the axis at each word the reading ADDED
 * (a correction) in the chart's columns, and a vertical line where the flown flight ended, in the outcome's colour.
 */

import { TRAINING_CORRECTION_COLOR, trainingOutcomeColour } from "../../utils/trainingWordColors";
import { closedCycleTimeS, formatSeconds, type TrainingColumn, type TrainingEvent } from "../../data/trainingSample";
import { replayText, TRAINING_OUTCOME_TAG } from "../../data/trainingText";
import { VLine } from "./chartKit";
import type { ReadbackModel } from "./readbackModel";

/** A triangle on the axis at the time of each correction word of ``columns``, with what it says in its title. */
export function CorrectionTicks({ m, columns, bottom, colour = TRAINING_CORRECTION_COLOR, label }: {
  m: ReadbackModel; columns: TrainingColumn[]; bottom: number; colour?: string; label: (event: TrainingEvent) => string;
}) {
  return (
    <>
      {m.corrections(...columns).map(({ event, atS }, index) => {
        const x = m.xTime(atS);
        return (
          <polygon key={`correction-${event.column}-${event.row}-${index}`} className="training-readback-correction"
            points={`${x},${bottom - 1} ${x - 3.5},${bottom - 8} ${x + 3.5},${bottom - 8}`} fill={colour}>
            <title>correction: {label(event)}, added by the closed-loop reading at {formatSeconds(atS)} s</title>
          </polygon>
        );
      })}
    </>
  );
}

/** Where the flown flight ended — the judge's outcome — as a vertical line in the outcome's colour (none for the labelled
 *  sentence, which was not flown). */
export function OutcomeLine({ m, top, bottom }: { m: ReadbackModel; top: number; bottom: number }) {
  if (m.closed === null) return null;
  const { replay } = m.closed;
  const atS = closedCycleTimeS(m.closed, replay.endCycle);
  return (
    <VLine x={m.xTime(Math.min(atS, m.endS))} top={top} bottom={bottom} stroke={trainingOutcomeColour(replay.outcome)} dash="2 3"
      title={`the flown flight ended at ${formatSeconds(atS)} s: ${TRAINING_OUTCOME_TAG[replay.outcome]} — ${replayText(replay)}`} />
  );
}
