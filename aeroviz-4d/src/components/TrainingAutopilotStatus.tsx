/**
 * TrainingAutopilotStatus.tsx
 * ---------------------------
 * The live executor's answer for the picked word (`trainingAutopilot`, `data/trainingAutopilot.ts`), in the sentence bar:
 * one short line, sharing the header's row with its buttons — how the segment ended (a segment end, or the judge's outcome
 * when the word was flown on to the flight's end), "N s flown · computed M ms"; the word only once the selection has moved
 * off it (else it is the selected band); the word and the full reading are its tooltip.
 *
 * Every number is the backend's. The word named is the one flown (`autopilotWord`), whichever the views have selected
 * since.
 *
 * AN ANSWER THAT TAKES LONG says why: the backend opens a set (its flights drawn and read again) the first time a set is
 * asked for, and after a start it does so for every set in turn (vocabulary A43); after `OPENING_NOTE_AFTER_MS` of "flying"
 * the line says that the backend is opening the set. No timeout is added: the request waits as before.
 */

import { useEffect, useState } from "react";

import { TRAINING_OUTSIDE_COLOR } from "../utils/trainingWordColors";
import { autopilotColour, autopilotWord, type TrainingAutopilotView } from "../data/trainingAutopilot";
import type { TrainingSelection } from "../data/trainingSample";
import { autopilotStatusText, formatElapsed, segmentEndText } from "../data/trainingText";

/** How long a request flies before the line says that the backend is opening the set. */
export const OPENING_NOTE_AFTER_MS = 2000;

/** True once the request ``request`` has been "flying" for `OPENING_NOTE_AFTER_MS`; false for another request (no stale
 *  note on the first commit of a new one) and once there is an answer. */
function useWaitedLong(flying: boolean, request: unknown): boolean {
  const [longFor, setLongFor] = useState<unknown>(null);
  useEffect(() => {
    if (!flying) return undefined;
    const timer = setTimeout(() => setLongFor(request), OPENING_NOTE_AFTER_MS);
    return () => clearTimeout(timer);
  }, [flying, request]);
  return flying && longFor === request;
}

/** The sentence bar's line, short enough to share the header's row with its buttons. The word flown is named in the line
 *  only when it is not the selected band (`named`: the selection has moved on since); always in the tooltip, with the
 *  full reading. */
export function TrainingAutopilotStatus({ view, selection, named }: {
  view: TrainingAutopilotView; selection: TrainingSelection; named: boolean;
}) {
  const word = autopilotWord(view.request, selection);
  const head = named ? `Autopilot · ${word}` : "Autopilot";
  const long = useWaitedLong(view.status === "flying", view.request);
  if (view.status === "flying") {
    return (
      <span className="training-sentence-autopilot" role="status"
        title={long ? `flying ${word} — the backend is opening the set (the first click of a set after a start takes up to a minute)` : `flying ${word}`}>
        {head} · flying …{long ? " the backend is opening the set" : ""}
      </span>
    );
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
  const flown = formatElapsed(segment.track.tS[segment.track.tS.length - 1] - segment.track.tS[0]);
  const computed = formatElapsed(segment.timing.computeS);
  return (
    <span className="training-sentence-autopilot"
      title={`${word}${segment.segment.correction ? " (a correction)" : ""}: the flight ${segmentEndText(segment)}; ${flown} flown, ` +
        `computed ${computed}; ${segment.stored.horizontalM.toFixed(3)} m horizontally and ${segment.stored.verticalM.toFixed(3)} m ` +
        "vertically from the exported flown states"}>
      {head} · <strong style={{ color: autopilotColour(segment) }}>{autopilotStatusText(view)}</strong> · {flown} flown · computed {computed}
    </span>
  );
}
