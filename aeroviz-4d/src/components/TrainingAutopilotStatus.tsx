/**
 * TrainingAutopilotStatus.tsx
 * ---------------------------
 * The live executor's answer for the picked word (`trainingAutopilot`, `data/trainingAutopilot.ts`), in the sentence bar:
 * one short line, sharing the header's row with its buttons — the verdict, "N s flown · computed M ms", and how the
 * flight ended (a tag) only when it ended badly; the word only once the selection has moved off it (else it is the
 * selected band); the word and the full reading are its tooltip.
 *
 * Every number is the backend's. The word named is the one flown (`autopilotWord`), whichever the views have selected
 * since.
 */

import { TRAINING_OUTSIDE_COLOR } from "../utils/trainingWordColors";
import {
  autopilotColour,
  autopilotWord,
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotSegment,
  type TrainingAutopilotView,
} from "../data/trainingAutopilot";
import type { TrainingFreeOutcome } from "../data/trainingOverlays";
import type { TrainingSelection } from "../data/trainingSample";
import {
  crossingText,
  formatElapsed,
  TRAINING_OUTCOME_TAG,
  TRAINING_OUTCOME_TEXT,
  TRAINING_VERDICT_TEXT,
} from "../data/trainingText";

/** How the flight ended when it ended short of what it was flown to (neither at its segment's stop nor landed); null
 *  when it did not. */
function badEnd(segment: TrainingAutopilotSegment): TrainingFreeOutcome | null {
  const { reason } = segment.end;
  return reason === TRAINING_AUTOPILOT_SEGMENT_END || reason === "landed" ? null : reason;
}

/** How it ended, in words. */
function endText(segment: TrainingAutopilotSegment): string {
  const { end } = segment;
  if (end.reason === TRAINING_AUTOPILOT_SEGMENT_END) {
    return segment.segment.stopRow === segment.segment.endRow
      ? `reached the point where the next ${segment.segment.column} word was said`
      : `reached the point a lead after the next ${segment.segment.column} word was said, where this word's band ends`;
  }
  return `${TRAINING_OUTCOME_TEXT[end.reason]}${end.crossing === null ? "" : `, ${crossingText(end.crossing)}`}`;
}

/** The sentence bar's line, short enough to share the header's row with its buttons: the verdict in the segment's
 *  colour, the two times — and, only when the flight ended badly, how, in a tag. The word flown is named in the line only
 *  when it is not the selected band (`named`: the selection has moved on since); always in the tooltip, with the full
 *  reading. */
export function TrainingAutopilotStatus({ view, selection, named }: {
  view: TrainingAutopilotView; selection: TrainingSelection; named: boolean;
}) {
  const word = autopilotWord(view.request, selection);
  const head = named ? `Autopilot · ${word}` : "Autopilot";
  if (view.status === "flying") {
    return <span className="training-sentence-autopilot" role="status" title={`flying ${word}`}>{head} · flying …</span>;
  }
  if (view.status === "failed") {
    return (
      <span className="training-sentence-autopilot" role="alert" title={`${word} not flown — ${view.problem}`}
        style={{ color: TRAINING_OUTSIDE_COLOR }}>
        {head} · not flown
      </span>
    );
  }
  const { segment } = view;
  const bad = badEnd(segment);
  const flown = formatElapsed(segment.end.flownS);
  const computed = formatElapsed(segment.timing.computeS);
  return (
    <span className="training-sentence-autopilot"
      title={`${word}: ${TRAINING_VERDICT_TEXT[segment.word.status]}; the flight ${endText(segment)}; ${flown} flown, ` +
        `computed ${computed}`}>
      {head} ·{" "}
      <strong style={{ color: autopilotColour(segment) }}>{TRAINING_VERDICT_TEXT[segment.word.status]}</strong>
      {bad === null ? null : <span style={{ color: TRAINING_OUTSIDE_COLOR }}> · {TRAINING_OUTCOME_TAG[bad]}</span>}
      {" "}· {flown} flown · computed {computed}
    </span>
  );
}
