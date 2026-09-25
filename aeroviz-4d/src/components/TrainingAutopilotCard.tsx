/**
 * TrainingAutopilotCard.tsx
 * -------------------------
 * The live executor's answer for the selected word (`trainingAutopilot`, `data/trainingAutopilot.ts`), in the Training
 * panel: which segment was flown, how it ended, the word's verdict with its checks, the limits that bound, and which
 * executor spec and code flew it and how long that took — with "Fly again" (a new request) and "Replay in 3D" (the scene
 * flies the same answer out again). `autopilotSummary` is its one-line form, for the sentence bar.
 *
 * Every number is the backend's; this card writes them out.
 */

import { useApp } from "../context/AppContext";
import { TRAINING_AUTOPILOT_COLOR, TRAINING_OUTSIDE_COLOR } from "../utils/trainingWordColors";
import {
  TRAINING_AUTOPILOT_OUTCOMES,
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotSegment,
  type TrainingAutopilotView,
} from "../data/trainingAutopilot";
import {
  formatSeconds,
  trainingWordLabel,
  TRAINING_COLUMN_INDEX,
  type TrainingCandidate,
  type TrainingFlight,
  type TrainingVocabulary,
} from "../data/trainingSample";

const SHA_SHOWN = 12;

/** How a flight that did not end at its segment's end ended (the judge's outcomes). */
const OUTCOME_TEXT: Record<(typeof TRAINING_AUTOPILOT_OUTCOMES)[number], string> = {
  landed: "landed",
  crossed_without_capture: "crossed the threshold without capturing the final",
  crossed_off_runway: "crossed the threshold off the runway",
  ground_contact: "reached the threshold's elevation before the threshold",
  timeout: "did not get there within its time limit",
  dynamics_failure: "left the dynamics (a non-finite state, no airspeed or a stall)",
};

/** The selected word, as the sentence reads it: "heading 270°". */
export function autopilotWordText(
  view: TrainingAutopilotView, flight: TrainingFlight, vocabulary: TrainingVocabulary, candidates: TrainingCandidate[],
): string {
  const { column, row } = view.request;
  const value = flight.words.inForce[TRAINING_COLUMN_INDEX[column]][row];
  return `${column} ${trainingWordLabel(vocabulary, candidates, column, value)}`;
}

/** Where the segment runs: "steps 12–30", "steps 12–30, flown on to step 32" (a heading word: a lead past the next
 *  heading word) or "step 12 to the landing". */
export function autopilotSpanText(segment: TrainingAutopilotSegment["segment"]): string {
  if (segment.toLanding) return `step ${segment.row} to the landing`;
  const on = segment.stopRow === segment.endRow ? "" : `, flown on to step ${segment.stopRow}`;
  return `steps ${segment.row}–${segment.endRow}${on}`;
}

/** How it ended, in words. */
export function autopilotEndText(segment: TrainingAutopilotSegment): string {
  const { end } = segment;
  if (end.reason === TRAINING_AUTOPILOT_SEGMENT_END) {
    return segment.segment.stopRow === segment.segment.endRow
      ? `reached the point where the next ${segment.segment.column} word was said`
      : `reached the point a lead after the next ${segment.segment.column} word was said, where this word's band ends`;
  }
  const crossing = end.crossing === null ? "" : `, ${Math.abs(end.crossing.crossM).toFixed(1)} m ${end.crossing.crossM >= 0
    ? "right" : "left"} of the centreline and ${end.crossing.heightM.toFixed(1)} m above the threshold`;
  return `${OUTCOME_TEXT[end.reason as (typeof TRAINING_AUTOPILOT_OUTCOMES)[number]]}${crossing}`;
}

/** The sentence bar's line: what is being flown, or what was. */
export function autopilotSummary(
  view: TrainingAutopilotView, flight: TrainingFlight, vocabulary: TrainingVocabulary, candidates: TrainingCandidate[],
): string {
  const word = autopilotWordText(view, flight, vocabulary, candidates);
  if (view.status === "flying") return `the autopilot (live): flying ${word} from step ${view.request.row} …`;
  if (view.status === "failed") return `the autopilot (live): ${word} from step ${view.request.row} not flown — ${view.problem}`;
  const { segment } = view;
  return `the autopilot (live): ${word}, ${autopilotSpanText(segment.segment)} — ${autopilotEndText(segment)} after ` +
    `${formatSeconds(segment.end.flownS)} s (observed ${formatSeconds(segment.segment.observedS)} s); the word: ` +
    `${segment.word.status} · computed in ${segment.computeS.toFixed(2)} s`;
}

export default function TrainingAutopilotCard({ flight, vocabulary, candidates, selected, flyAgain }: {
  flight: TrainingFlight | null; vocabulary: TrainingVocabulary; candidates: TrainingCandidate[]; selected: boolean;
  flyAgain: () => void;
}) {
  const { trainingAutopilot: view, setTrainingAutopilot } = useApp();
  if (!selected || view === null || flight === null || view.request.flightKey !== flight.flightKey) {
    return (
      <p className="training-autopilot-note">
        Click a word's band in the sentence bar and the executor flies that word's segment now, from where the observed
        aircraft was when the word was said.
      </p>
    );
  }
  const word = autopilotWordText(view, flight, vocabulary, candidates);
  if (view.status === "flying") {
    return <p className="training-autopilot-note" role="status">Flying {word} from step {view.request.row} …</p>;
  }
  if (view.status === "failed") {
    return (
      <div className="training-problem" role="alert">
        <p className="training-empty-title">The autopilot did not fly {word} from step {view.request.row}.</p>
        <p className="training-problem-detail">{view.problem}</p>
        <button type="button" className="training-autopilot-button" onClick={flyAgain}>Fly again</button>
      </div>
    );
  }
  const { segment } = view;
  const { end } = segment;
  const verdictColour = segment.word.status === "outside" ? TRAINING_OUTSIDE_COLOR : TRAINING_AUTOPILOT_COLOR;
  const bound = Object.entries(segment.limits.bound).filter(([, cycles]) => cycles > 0);
  return (
    <section className="training-autopilot-card" aria-label="The autopilot, live">
      <p className="training-autopilot-title" style={{ color: TRAINING_AUTOPILOT_COLOR }}>
        {word}, {autopilotSpanText(segment.segment)}
      </p>
      <p>
        From the observed state at step {segment.segment.row}, told the six words in force there and then the sentence's
        words as the observed aircraft heard them: {autopilotEndText(segment)} after {formatSeconds(end.flownS)} s (the observed
        aircraft: {formatSeconds(segment.segment.observedS)} s).
        {end.offsetFromObserved === null ? "" : ` There it was ${end.offsetFromObserved.horizontalM.toFixed(0)} m from the ` +
          `observed aircraft, ${Math.abs(end.offsetFromObserved.aboveM).toFixed(0)} m ${end.offsetFromObserved.aboveM >= 0
            ? "above" : "below"} it, ${Math.abs(end.offsetFromObserved.groundSpeedMps).toFixed(1)} m/s ` +
          `${end.offsetFromObserved.groundSpeedMps >= 0 ? "faster" : "slower"}.`}
        {end.refused === null ? "" : ` The labeller's gate refused the flown segment: ${end.refused}.`}
      </p>
      <p>
        The word: <strong style={{ color: verdictColour }}>{segment.word.status}</strong>
        {segment.word.reason === null ? "" : ` — ${segment.word.reason}`}
      </p>
      {segment.word.checks.length ? (
        <ul className="training-autopilot-checks">
          {segment.word.checks.map((check) => (
            <li key={check.name} style={{ color: check.ok ? undefined : TRAINING_OUTSIDE_COLOR }}>
              {check.ok ? "✓" : "✗"} {check.name}{check.rows === null ? "" : ` (${check.inside}/${check.rows} rows)`}
            </li>
          ))}
        </ul>
      ) : null}
      <p className="training-autopilot-meta">
        {segment.limits.cycles} cycles of {segment.executor.cycleS} s
        {bound.length ? `; limits bound: ${bound.map(([name, cycles]) => `${name.replace(/_/g, " ")} ${cycles}`).join(", ")}` : "; no limit bound"}
      </p>
      <p className="training-autopilot-meta">
        {segment.group} · spec {segment.executor.specSha256.slice(0, SHA_SHOWN)} ({segment.executor.spec}) · executor code{" "}
        {segment.executor.sourceSha256.slice(0, SHA_SHOWN)} · words said on the {segment.executor.wordClock} clock · computed
        in {segment.computeS.toFixed(2)} s{segment.waitS >= 0.05 ? ` after waiting ${segment.waitS.toFixed(2)} s for the flight before it` : ""}{" "}
        at {segment.computedUtc}
      </p>
      <div className="training-autopilot-buttons">
        <button type="button" className="training-autopilot-button" onClick={flyAgain}>Fly again</button>
        <button type="button" className="training-autopilot-button"
          onClick={() => setTrainingAutopilot({ ...view, playedAt: Date.now() })}>
          Replay in 3D
        </button>
      </div>
    </section>
  );
}
