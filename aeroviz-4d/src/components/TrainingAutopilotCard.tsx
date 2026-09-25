/**
 * TrainingAutopilotCard.tsx
 * -------------------------
 * The live executor's answer for the picked word (`trainingAutopilot`, `data/trainingAutopilot.ts`), kept short:
 *
 *  • in the Training panel (`TrainingAutopilotCard`): the word and its steps; THE VERDICT — did the flown segment stay
 *    inside the word's envelope — in the segment's own colour (red when outside, `autopilotColour`); the two times side
 *    by side, the SIMULATED flight (beside the observed aircraft's) and the COMPUTATION (the backend's, beside the
 *    browser's round trip); the checks behind the verdict; everything else — how it ended, the limits that bound, the
 *    computation part by part, the spec and code — folded into Details. "Replay in 3D" flies the same answer out again;
 *    flying it anew is the sentence bar's button;
 *  • in the sentence bar (`TrainingAutopilotStatus`): one line — the word, the verdict, "N s flown in M ms", and how the
 *    flight ended only when it ended badly.
 *
 * Every number is the backend's; this card writes them out. The word named is the one flown (`autopilotWord`), whichever
 * the views have selected since.
 */

import { useApp } from "../context/AppContext";
import { TRAINING_OUTSIDE_COLOR } from "../utils/trainingWordColors";
import {
  autopilotColour,
  autopilotOnScreen,
  autopilotWord,
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotSegment,
  type TrainingAutopilotView,
} from "../data/trainingAutopilot";
import type { TrainingSelection } from "../data/trainingSample";
import {
  checkText,
  crossingText,
  formatElapsed,
  shortSha,
  TRAINING_OUTCOME_TEXT,
  TRAINING_VERDICT_TEXT,
} from "../data/trainingText";
import ProblemBox from "./training/ProblemBox";

/** Where the segment runs: "steps 12–30", "steps 12–30, flown on to step 32" (a heading word: a lead past the next
 *  heading word) or "step 12 to the landing". */
function spanText(segment: TrainingAutopilotSegment["segment"]): string {
  if (segment.toLanding) return `step ${segment.row} to the landing`;
  const on = segment.stopRow === segment.endRow ? "" : `, flown on to step ${segment.stopRow}`;
  return `steps ${segment.row}–${segment.endRow}${on}`;
}

/** The flight ended short of what it was flown to: neither at its segment's stop nor landed. */
function endedBadly(segment: TrainingAutopilotSegment): boolean {
  return segment.end.reason !== TRAINING_AUTOPILOT_SEGMENT_END && segment.end.reason !== "landed";
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

/** The computation, part by part — the backend's parts add up to its total — then the wait before it and the round trip. */
function timingText(segment: TrainingAutopilotSegment, roundTripS: number): string {
  const { timing } = segment;
  return [
    `executor ${formatElapsed(timing.flyS)} (${timing.cycles} cycles of ${segment.executor.cycleS} s computed)`,
    `judge ${formatElapsed(timing.judgeS)}`,
    timing.flightKept ? `flight kept from an earlier request (${formatElapsed(timing.openS)})`
      : `flight rebuilt ${formatElapsed(timing.openS)}`,
    `set and spec ${formatElapsed(timing.setupS)}`,
    `segment set up ${formatElapsed(timing.prepareS)}`,
    `answer ${formatElapsed(timing.answerS)}`,
    ...(timing.waitS >= 0.05 ? [`waited ${formatElapsed(timing.waitS)} for the flight before it`] : []),
    `round trip ${formatElapsed(roundTripS)}`,
  ].join(" · ");
}

/** The sentence bar's line: the word, the verdict in the segment's colour, the two times — and how the flight ended only
 *  when it ended badly. */
export function TrainingAutopilotStatus({ view, selection }: { view: TrainingAutopilotView; selection: TrainingSelection }) {
  const word = autopilotWord(view.request, selection);
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
      <strong style={{ color: autopilotColour(segment) }}>{TRAINING_VERDICT_TEXT[segment.word.status]}</strong>
      {endedBadly(segment) ? <span style={{ color: TRAINING_OUTSIDE_COLOR }}> · {endText(segment)}</span> : null}
      {" "}· {formatElapsed(segment.end.flownS)} flown in {formatElapsed(segment.timing.computeS)}
    </span>
  );
}

export default function TrainingAutopilotCard() {
  const { trainingSelection: selection, trainingAutopilot, replayTrainingAutopilot } = useApp();
  const view = autopilotOnScreen(trainingAutopilot, selection);
  if (view === null || selection === null) return null;
  const word = autopilotWord(view.request, selection);
  if (view.status === "flying") {
    return <p className="training-autopilot-note" role="status">Flying {word} from step {view.request.row} …</p>;
  }
  if (view.status === "failed") {
    return <ProblemBox title={`The autopilot did not fly ${word} from step ${view.request.row}.`} detail={view.problem} />;
  }
  const { segment } = view;
  const { end } = segment;
  const colour = autopilotColour(segment);
  const bound = Object.entries(segment.limits.bound).filter(([, cycles]) => cycles > 0);
  return (
    <section className="training-autopilot-card" aria-label="The autopilot, live" style={{ borderLeftColor: colour }}>
      <p className="training-autopilot-title">{word} · {spanText(segment.segment)}</p>
      <p className="training-autopilot-verdict" style={{ color: colour }}>
        {TRAINING_VERDICT_TEXT[segment.word.status]}
        {segment.word.reason === null ? "" : <span className="training-autopilot-reason"> — {segment.word.reason}</span>}
      </p>
      {endedBadly(segment) ? (
        <p className="training-autopilot-ended" style={{ color: TRAINING_OUTSIDE_COLOR }}>The flight {endText(segment)}.</p>
      ) : null}
      <div className="training-autopilot-times">
        <div className="training-autopilot-time" aria-label="Simulated flight time">
          <span className="training-autopilot-time-label">Simulated flight</span>
          <span className="training-autopilot-time-value">{formatElapsed(end.flownS)}</span>
          <span className="training-autopilot-time-note">observed {formatElapsed(segment.segment.observedS)}</span>
        </div>
        <div className="training-autopilot-time" aria-label="Computation time">
          <span className="training-autopilot-time-label">Computed in</span>
          <span className="training-autopilot-time-value">{formatElapsed(segment.timing.computeS)}</span>
          <span className="training-autopilot-time-note">round trip {formatElapsed(view.roundTripS)}</span>
        </div>
      </div>
      {segment.word.checks.length ? (
        <ul className="training-autopilot-checks">
          {segment.word.checks.map((check) => (
            <li key={check.name} style={{ color: check.ok ? undefined : TRAINING_OUTSIDE_COLOR }}>{checkText(check)}</li>
          ))}
        </ul>
      ) : null}
      <details className="training-autopilot-details">
        <summary>Details</summary>
        <p>
          From the observed state at step {segment.segment.row}, told the six words in force there and then the sentence's
          words as the observed aircraft heard them: {endText(segment)}.
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
        <p>Computed: {timingText(segment, view.roundTripS)}.</p>
        <p>
          {segment.group} · spec {shortSha(segment.executor.specSha256)} ({segment.executor.spec}) · executor
          code {shortSha(segment.executor.sourceSha256)} · words said on the {segment.executor.wordClock} clock ·
          computed at {segment.computedUtc}
        </p>
      </details>
      <button type="button" className="training-autopilot-button" onClick={replayTrainingAutopilot}>Replay in 3D</button>
    </section>
  );
}
