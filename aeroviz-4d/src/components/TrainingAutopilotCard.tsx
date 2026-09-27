/**
 * TrainingAutopilotCard.tsx
 * -------------------------
 * The live executor's answer for the picked word (`trainingAutopilot`, `data/trainingAutopilot.ts`), kept short:
 *
 *  • in the Training panel (`TrainingAutopilotCard`): the word and its steps; THE VERDICT — did the flown segment stay
 *    inside the word's envelope — in the segment's own colour (red when outside, `autopilotColour`); what its faded,
 *    dashed tail is, when it has one (`tailText`); the two times side
 *    by side, the SIMULATED flight (beside the observed aircraft's) and the COMPUTATION (the backend's, beside the
 *    browser's round trip); the checks behind the verdict; everything else — how it ended, the limits that bound, the
 *    computation part by part, the spec and code — folded into Details. "Replay in 3D" flies the same answer out again;
 *    flying it anew is the sentence bar's button — and a refusal's "Fly again", since the word selected may have moved;
 *  • in the sentence bar (`TrainingAutopilotStatus`): one short line, sharing the header's row with its buttons — the
 *    verdict, "N s flown · computed M ms", and how the flight ended (a tag) only when it ended badly; the word only once
 *    the selection has moved off it (else it is the selected band); the word and the full reading are its tooltip.
 *
 * Every number is the backend's; this card writes them out. The word named is the one flown (`autopilotWord`), whichever
 * the views have selected since.
 */

import { useApp } from "../context/AppContext";
import { TRAINING_OUTSIDE_COLOR } from "../utils/trainingWordColors";
import {
  autopilotColour,
  autopilotHasLine,
  autopilotOnScreen,
  autopilotSampleGap,
  autopilotWord,
  nextPick,
  requestSource,
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotSegment,
  type TrainingAutopilotView,
} from "../data/trainingAutopilot";
import { sourceOnScreen, type TrainingFreeOutcome } from "../data/trainingOverlays";
import { formatSeconds, type TrainingSelection } from "../data/trainingSample";
import {
  checkText,
  crossingText,
  formatElapsed,
  formatUtc,
  shortSha,
  TRAINING_OUTCOME_TAG,
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

/** What the tail drawn faded and dashed is (`autopilotRunAndTail`): the flight past where the executor heard the next
 *  word of the column — a heading word's lead into the next heading word, flown because this word is judged to its
 *  end. null without a tail. */
function tailText(segment: TrainingAutopilotSegment): string | null {
  const heardS = segment.segment.nextWordHeardS;
  if (heardS === null) return null;
  const { tS } = segment.track;
  const tailS = Math.round((tS[tS.length - 1] - heardS) * 1000) / 1000;
  return `Faded, dashed: the last ${formatSeconds(tailS)} s, past where it heard the next ` +
    `${segment.segment.column} word (at ${formatSeconds(heardS)} s) — already turning to that word, still judged for this ` +
    "one, whose band ends a lead after it.";
}

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
      <span className="training-sentence-autopilot" title={`${word} not flown — ${view.problem}`}
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

export default function TrainingAutopilotCard() {
  const {
    trainingSelection: selection, trainingAutopilot, replayTrainingAutopilot, trainingPick, setTrainingPick, trainingGenerations,
    trainingSource,
  } = useApp();
  // the answer of the sentence read — and, for a model's word, the sample it re-flies
  const read = sourceOnScreen(trainingGenerations, trainingSource, selection);
  const view = autopilotOnScreen(trainingAutopilot, selection, read.source);
  if (view === null || selection === null) return null;
  const word = autopilotWord(view.request, selection);
  if (view.status === "flying") {
    return <p className="training-autopilot-note" role="status">Flying {word} from step {view.request.row} …</p>;
  }
  if (view.status === "failed") {
    return (
      <ProblemBox title={`The autopilot did not fly ${word} from step ${view.request.row}.`} detail={view.problem}>
        <button type="button" className="training-autopilot-button"
          onClick={() => setTrainingPick(nextPick(trainingPick, requestSource(view.request), view.request.column, view.request.row))}>
          Fly again
        </button>
      </ProblemBox>
    );
  }
  const { segment } = view;
  const { end } = segment;
  const colour = autopilotColour(segment);
  // a model's word re-flies its sample: how closely the live flight lands on it
  const gap = read.sentence === null ? null : autopilotSampleGap(segment, read.sentence);
  const bound = Object.entries(segment.limits.bound).filter(([, cycles]) => cycles > 0);
  return (
    <section className="training-autopilot-card" aria-label="The autopilot, live" style={{ borderLeftColor: colour }}>
      <p className="training-autopilot-title">{word} · {spanText(segment.segment)}</p>
      <p className="training-autopilot-verdict" style={{ color: colour }}>
        {TRAINING_VERDICT_TEXT[segment.word.status]}
        {segment.word.reason === null ? "" : <span className="training-autopilot-reason"> — {segment.word.reason}</span>}
      </p>
      {badEnd(segment) !== null ? (
        <p className="training-autopilot-ended" style={{ color: TRAINING_OUTSIDE_COLOR }}>The flight {endText(segment)}.</p>
      ) : null}
      {tailText(segment) === null ? null : <p className="training-autopilot-tail">{tailText(segment)}</p>}
      <div className="training-autopilot-times">
        <div className="training-autopilot-time" aria-label="Simulated flight time">
          <span className="training-autopilot-time-label">Simulated flight</span>
          <span className="training-autopilot-time-value">{formatElapsed(end.flownS)}</span>
          <span className="training-autopilot-time-note">
            {segment.segment.observedS === null ? "a model's word" : `observed ${formatElapsed(segment.segment.observedS)}`}
          </span>
        </div>
        <div className="training-autopilot-time" aria-label="Computation time">
          <span className="training-autopilot-time-label">Computed in</span>
          <span className="training-autopilot-time-value">{formatElapsed(segment.timing.computeS)}</span>
          <span className="training-autopilot-time-note">round trip {formatElapsed(view.roundTripS)}</span>
        </div>
      </div>
      {gap !== null ? (
        <p className="training-autopilot-sample" title={`the live flight against the exported sample ${read.sentence!.sample + 1}, ` +
          `at the ${gap.points} times both hold a point: the executor is deterministic, so they are one flight`}>
          {gap.gapM < 0.5 ? "✓ the sample's own flight" : "✗ not the sample's flight"} — {gap.gapM.toFixed(gap.gapM < 10 ? 2 : 0)} m
          at most from sample #{read.sentence!.sample + 1} over {gap.points} points
        </p>
      ) : null}
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
          {segment.source.kind === "truth"
            ? `From the observed state at step ${segment.segment.row}, told the six words in force there and then the sentence's ` +
              "words as the observed aircraft heard them"
            : `The model's sentence flown again from its first step (${segment.source.firstRow}), from the observed state there, ` +
              `each word heard at the step it was said, as its free generation flew it; shown from step ${segment.segment.row}`}
          : {endText(segment)}.
          {end.offsetFromObserved === null ? "" : ` There it was ${end.offsetFromObserved.horizontalM.toFixed(0)} m from the ` +
            `observed aircraft, ${Math.abs(end.offsetFromObserved.aboveM).toFixed(0)} m ${end.offsetFromObserved.aboveM >= 0
              ? "above" : "below"} it, ${Math.abs(end.offsetFromObserved.groundSpeedMps).toFixed(1)} m/s ` +
            `${end.offsetFromObserved.groundSpeedMps >= 0 ? "faster" : "slower"}.`}
          {end.refused === null ? "" : ` The labeller's gate refused the flown segment: ${end.refused}.`}
        </p>
        <p>
          {segment.limits.cycles} cycles of {segment.executor.cycleS} s judged
          {segment.source.kind === "truth" ? "" : ` — the model's whole flight from its first step (${segment.source.firstRow}), ` +
            "not only this word's segment"}
          {bound.length ? `; limits bound: ${bound.map(([name, cycles]) => `${name.replace(/_/g, " ")} ${cycles}`).join(", ")}` : "; no limit bound"}.
        </p>
        <p>Computed: {timingText(segment, view.roundTripS)}.</p>
        <p>
          {segment.group} · spec {shortSha(segment.executor.specSha256)} ({segment.executor.spec}) · executor
          code {shortSha(segment.executor.sourceSha256)} · words said on the {segment.executor.wordClock} clock ·
          computed at {formatUtc(segment.computedUtc)}
        </p>
      </details>
      {autopilotHasLine(segment) ? (
        <button type="button" className="training-autopilot-button" onClick={replayTrainingAutopilot}>Replay in 3D</button>
      ) : null}
    </section>
  );
}
