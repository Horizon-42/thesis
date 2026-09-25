/**
 * TrainingAutopilotCard.tsx
 * -------------------------
 * The live executor's answer for the picked word (`trainingAutopilot`, `data/trainingAutopilot.ts`), kept short:
 *
 *  • in the Training panel (`TrainingAutopilotCard`): the word and its steps; THE VERDICT — did the flown segment stay
 *    inside the word's envelope — in the segment's own colour (red when outside, `autopilotColour`); the two times side
 *    by side, the SIMULATED flight (beside the observed aircraft's) and the COMPUTATION (the backend's, beside the
 *    browser's round trip); the checks behind the verdict; everything else — how it ended, the limits that bound, the
 *    computation part by part, the spec and code — folded into Details. "Fly again" asks anew, "Replay in 3D" flies the
 *    same answer out again;
 *  • in the sentence bar (`TrainingAutopilotStatus`): one line — the word, the verdict, "N s flown in M ms", and how the
 *    flight ended only when it ended badly.
 *
 * Every number is the backend's; this card writes them out.
 */

import { useApp } from "../context/AppContext";
import { TRAINING_OUTSIDE_COLOR } from "../utils/trainingWordColors";
import {
  autopilotColour,
  nextPick,
  TRAINING_AUTOPILOT_OUTCOMES,
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotSegment,
  type TrainingAutopilotStatus,
  type TrainingAutopilotView,
} from "../data/trainingAutopilot";
import {
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

/** The verdict, as it is read first. */
export const VERDICT_TEXT: Record<TrainingAutopilotStatus, string> = {
  inside: "✓ inside its envelope",
  outside: "✗ outside its envelope",
  "not judged": "not judged",
  "no check": "no envelope of its own",
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

/** The flight ended short of what it was flown to: neither at its segment's stop nor landed. */
export function autopilotEndedBadly(segment: TrainingAutopilotSegment): boolean {
  return segment.end.reason !== TRAINING_AUTOPILOT_SEGMENT_END && segment.end.reason !== "landed";
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

/** A duration as the times read: milliseconds under a second, then seconds (two decimals under ten, one above) —
 *  the unit chosen after rounding, so 0.9996 s reads "1.00 s", never "1000 ms". */
export function formatDuration(seconds: number): string {
  const ms = Math.round(seconds * 1000);
  if (ms < 1000) return `${ms} ms`;
  return Math.round(seconds * 100) < 1000 ? `${seconds.toFixed(2)} s` : `${seconds.toFixed(1)} s`;
}

/** The computation, part by part — the backend's parts add up to its total — then the wait before it and the round trip. */
export function autopilotTimingText(segment: TrainingAutopilotSegment, roundTripS: number): string {
  const { timing } = segment;
  return [
    `executor ${formatDuration(timing.flyS)} (${timing.cycles} cycles of ${segment.executor.cycleS} s computed)`,
    `judge ${formatDuration(timing.judgeS)}`,
    timing.flightKept ? `flight kept from an earlier request (${formatDuration(timing.openS)})`
      : `flight rebuilt ${formatDuration(timing.openS)}`,
    `set and spec ${formatDuration(timing.setupS)}`,
    `segment set up ${formatDuration(timing.prepareS)}`,
    `answer ${formatDuration(timing.answerS)}`,
    ...(timing.waitS >= 0.05 ? [`waited ${formatDuration(timing.waitS)} for the flight before it`] : []),
    `round trip ${formatDuration(roundTripS)}`,
  ].join(" · ");
}

/** The sentence bar's line: the word, the verdict in the segment's colour, the two times — and how the flight ended only
 *  when it ended badly. */
export function TrainingAutopilotStatus({ view, flight, vocabulary, candidates }: {
  view: TrainingAutopilotView; flight: TrainingFlight; vocabulary: TrainingVocabulary; candidates: TrainingCandidate[];
}) {
  const word = autopilotWordText(view, flight, vocabulary, candidates);
  if (view.status === "flying") {
    return <span className="training-sentence-autopilot" role="status">Autopilot · flying {word} …</span>;
  }
  if (view.status === "failed") {
    return (
      <span className="training-sentence-autopilot training-sentence-autopilot-failed" title={view.problem}
        style={{ color: TRAINING_OUTSIDE_COLOR }}>
        Autopilot · {word} not flown — {view.problem}
      </span>
    );
  }
  const { segment } = view;
  return (
    <span className="training-sentence-autopilot">
      Autopilot · {word} ·{" "}
      <strong style={{ color: autopilotColour(segment) }}>{VERDICT_TEXT[segment.word.status]}</strong>
      {autopilotEndedBadly(segment) ? (
        <span style={{ color: TRAINING_OUTSIDE_COLOR }}> · {autopilotEndText(segment)}</span>
      ) : null}
      {" "}· {formatDuration(segment.end.flownS)} flown in {formatDuration(segment.timing.computeS)}
    </span>
  );
}

export default function TrainingAutopilotCard({ flight, vocabulary, candidates }: {
  flight: TrainingFlight | null; vocabulary: TrainingVocabulary; candidates: TrainingCandidate[];
}) {
  const { trainingAutopilot: view, setTrainingAutopilot, trainingPick, setTrainingPick } = useApp();
  if (view === null || flight === null || view.request.flightKey !== flight.flightKey) {
    return (
      <p className="training-autopilot-note">
        Select a word in the sentence bar (click its band) and press <b>▶ Fly this segment</b> above it: the executor flies
        that word's segment now, from where the observed aircraft was when the word was said.
      </p>
    );
  }
  const flyAgain = () => setTrainingPick(nextPick(trainingPick, view.request.flightKey, view.request.column, view.request.row));
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
  const colour = autopilotColour(segment);
  const bound = Object.entries(segment.limits.bound).filter(([, cycles]) => cycles > 0);
  return (
    <section className="training-autopilot-card" aria-label="The autopilot, live" style={{ borderLeftColor: colour }}>
      <p className="training-autopilot-title">{word} · {autopilotSpanText(segment.segment)}</p>
      <p className="training-autopilot-verdict" style={{ color: colour }}>
        {VERDICT_TEXT[segment.word.status]}
        {segment.word.reason === null ? "" : <span className="training-autopilot-reason"> — {segment.word.reason}</span>}
      </p>
      {autopilotEndedBadly(segment) ? (
        <p className="training-autopilot-ended" style={{ color: TRAINING_OUTSIDE_COLOR }}>The flight {autopilotEndText(segment)}.</p>
      ) : null}
      <div className="training-autopilot-times">
        <div className="training-autopilot-time" aria-label="Simulated flight time">
          <span className="training-autopilot-time-label">Simulated flight</span>
          <span className="training-autopilot-time-value">{formatDuration(end.flownS)}</span>
          <span className="training-autopilot-time-note">observed {formatDuration(segment.segment.observedS)}</span>
        </div>
        <div className="training-autopilot-time" aria-label="Computation time">
          <span className="training-autopilot-time-label">Computed in</span>
          <span className="training-autopilot-time-value">{formatDuration(segment.timing.computeS)}</span>
          <span className="training-autopilot-time-note">round trip {formatDuration(view.roundTripS)}</span>
        </div>
      </div>
      {segment.word.checks.length ? (
        <ul className="training-autopilot-checks">
          {segment.word.checks.map((check) => (
            <li key={check.name} style={{ color: check.ok ? undefined : TRAINING_OUTSIDE_COLOR }}>
              {check.ok ? "✓" : "✗"} {check.name}{check.rows === null ? "" : ` (${check.inside}/${check.rows} rows)`}
            </li>
          ))}
        </ul>
      ) : null}
      <details className="training-autopilot-details">
        <summary>Details</summary>
        <p>
          From the observed state at step {segment.segment.row}, told the six words in force there and then the sentence's
          words as the observed aircraft heard them: {autopilotEndText(segment)}.
          {end.offsetFromObserved === null ? "" : ` There it was ${end.offsetFromObserved.horizontalM.toFixed(0)} m from the ` +
            `observed aircraft, ${Math.abs(end.offsetFromObserved.aboveM).toFixed(0)} m ${end.offsetFromObserved.aboveM >= 0
              ? "above" : "below"} it, ${Math.abs(end.offsetFromObserved.groundSpeedMps).toFixed(1)} m/s ` +
            `${end.offsetFromObserved.groundSpeedMps >= 0 ? "faster" : "slower"}.`}
          {end.refused === null ? "" : ` The labeller's gate refused the flown segment: ${end.refused}.`}
        </p>
        <p>
          {segment.limits.cycles} cycles of {segment.executor.cycleS} s judged
          {bound.length ? `; limits bound: ${bound.map(([name, cycles]) => `${name.replace(/_/g, " ")} ${cycles}`).join(", ")}` : "; no limit bound"}.
        </p>
        <p>Computed: {autopilotTimingText(segment, view.roundTripS)}.</p>
        <p>
          {segment.group} · spec {segment.executor.specSha256.slice(0, SHA_SHOWN)} ({segment.executor.spec}) · executor
          code {segment.executor.sourceSha256.slice(0, SHA_SHOWN)} · words said on the {segment.executor.wordClock} clock ·
          computed at {segment.computedUtc}
        </p>
      </details>
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
