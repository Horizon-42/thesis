/**
 * TrainingSentenceBar.tsx
 * -----------------------
 * The sentence as a picture: the six columns as six rows — runway pointer, approach, heading, altitude, angle, speed, in
 * the vocabulary's order — against the flight's own time. Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 * A BAND IS A WORD IN FORCE, from the step it was issued to the step the next word of its column replaces it. Step 0
 * carries all six columns, so every row opens with a band at 0; after that a column is "unchanged" until its next word,
 * and most steps say nothing at all. The tick at a band's left edge is the issue itself; the numbers along the top are
 * the steps that say anything. The three moments that are not words — the clearance, the capture of the final and the
 * speed becoming "unspecified" — are dashed lines across the rows.
 *
 * WHICH SENTENCE (`trainingSource`): the tabs at the head of the bar — "Truth", then one per model whose own sentences
 * are published for the set (`trainingGenerations`, each in its colour) — choose what the rows draw. The TRUTH is the
 * labelled sentence, as above. A MODEL's is one of its samples (the numbered buttons, each marked by how the flight
 * ended): its bands HATCHED and dash-edged, so a model's word never passes for a labelled one; the rows it only observed
 * (before its first predicted step) shaded; under each row a white tick wherever the truth says a word of that column,
 * so where the two sentences part is read at a glance; a dashed line where the model cleared the flight, a solid one in
 * its colour where its flight ended, a dashed white one where the observed flight's sentence ends. Words a model said
 * after its flight ended (the executor flies on after two outcomes its judge reads earlier) sit under a grey shade. The
 * time axis is the observed flight's, stretched to the sentence read when that one runs longer (a model's flight that
 * timed out runs on to 1.5× the observed time): the truth is never squeezed by a sample it is not showing. Choosing a
 * model starts at its first sample. A model that does not fly the flight on screen leaves the truth drawn, with its whole
 * header, and says so. The Read-back and Prior windows read the truth, so they are offered only while it is read.
 *
 * THE HEADER IS SHORT: the tabs, the callsign (its type, stratum and counts in its tooltip), the runway, one chip of the
 * labeller's own verdicts (a model's: how its sample ended), one of the executor's replay when it is on, the live
 * executor's line, the cursor, and the buttons — Fly, Read-back and Prior (the truth's), and ⓘ for the notes. Every chip
 * carries its full reading in its tooltip.
 *
 * The cursor is in flight time and is moved by clicking a band or a step number: what it reports is the artefact's own
 * step, never a rounded pixel. It does NOT drive `viewer.clock`: Training loads no CZML, and the clock belongs to
 * Observe's playback.
 *
 * ONE COLUMN IS HIGHLIGHTED, NEVER A STEP. Clicking a band selects its word class (`trainingColumn`) and puts the cursor
 * at its issue; every view then highlights that column's word in force at the cursor, and only it — of the sentence read.
 * Clicking the selected band again clears it.
 *
 * THE OVERLAYS, when the panel publishes them: the EXECUTOR's verdict on each word as a dot at the band's left (teal
 * inside its envelope, red outside, hollow grey not judged, not reached or superseded; none for a word with no check of
 * its own); the PRIOR's window behind its button (`TrainingPriorWindow`). One window is open at a time.
 *
 * THE EXECUTOR, LIVE (`trainingAutopilot`): the Fly button PICKS the selected word for the live executor (`trainingPick`;
 * "↻ Fly again" once it has flown), and so does clicking a band while the panel's switch is on (`trainingAutopilotAuto`;
 * clicking the selected band again clears the pick) — never the cursor. Its line (`TrainingAutopilotStatus`) names the
 * word it flew, whether it stayed inside its envelope and the two times. It flies truth words only.
 *
 * THE BAR MEASURES ITSELF: its height is published on the workbench as `--training-bar-height`, so the docks end above it
 * (`index.css`) instead of under it.
 */

import { useLayoutEffect, useState } from "react";
import { useApp, useTrainingCursor } from "../context/AppContext";
import TrainingLegend from "./TrainingLegend";
import TrainingPriorWindow from "./TrainingPriorWindow";
import TrainingReadbackWindow from "./TrainingReadbackWindow";
import { TrainingAutopilotStatus } from "./TrainingAutopilotCard";
import useMeasuredWidth from "../hooks/useMeasuredWidth";
import {
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_SURFACE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
  trainingModelColour,
} from "../utils/trainingWordColors";
import { autopilotColour, autopilotHasLine, autopilotOnScreen, nextPick } from "../data/trainingAutopilot";
import {
  executorWordAt,
  executorWordCounts,
  generatedRowAt,
  generationOnScreen,
  overlayOnScreen,
  TRAINING_CROSSING_OUTCOMES,
  type TrainingExecutorFlight,
  type TrainingExecutorWord,
  type TrainingGeneratedSentence,
  type TrainingGenerationView,
} from "../data/trainingOverlays";
import {
  formatSeconds,
  rowAtTime,
  sentenceColumnRuns,
  trainingClearedValue,
  trainingKindLabel,
  trainingTruthSentence,
  trainingVerdicts,
  trainingWordAt,
  trainingWordLabel,
  TRAINING_COLUMN_INDEX,
  TRAINING_COLUMNS,
  type TrainingFlight,
  type TrainingSentence,
  type TrainingSentenceEvent,
  type TrainingWordEvent,
} from "../data/trainingSample";
import {
  checkMark, checkText, crossingText, TRAINING_COLUMN_LABEL, TRAINING_OUTCOME_TAG, TRAINING_OUTCOME_TEXT,
} from "../data/trainingText";

// One SVG unit is one pixel: the bar is as wide as the dock and always VIEW_H tall.
const GUTTER = 96;
const PAD_R = 22;
const HEAD_H = 22;
const ROW_H = 20;
const AXIS_H = 22;
const VIEW_H = HEAD_H + TRAINING_COLUMNS.length * ROW_H + AXIS_H;
/** Used until the element has been measured, and in jsdom, which has no layout. */
const DEFAULT_PLOT_W = 1074;
const MIN_PLOT_W = 320;
/** A band carries its word only above this many pixels; the tooltip always does. */
const LABEL_MIN_W = 40;
const STEP_LABEL_GAP = 14;
const TICK_LABEL_GAP = 34;
/** The observed-only span says so in words above this many pixels. */
const OBSERVED_LABEL_MIN_W = 52;
/** The fills a model's sentence is drawn with (`<defs>`): its bands' hatch, the span it only observed. */
const HATCH_ID = "training-model-hatch";
const OBSERVED_ID = "training-observed-hatch";

/**
 * Which of these ascending x positions may carry a text label: greedy from the left, keeping one only when it clears the
 * last kept by `minGap`. With `keepLast` the last position is always kept, evicting the previous keeper if they would
 * collide. The label is dropped, never the step: its hit area and tooltip stay.
 */
export function spacedLabels(xs: number[], minGap: number, keepLast = false): boolean[] {
  const keep = xs.map(() => false);
  let lastKept = -Infinity;
  xs.forEach((x, index) => {
    if (x - lastKept < minGap) return;
    keep[index] = true;
    lastKept = x;
  });
  if (keepLast && xs.length) {
    const end = xs.length - 1;
    if (!keep[end]) {
      for (let i = end - 1; i >= 0; i -= 1) {
        if (keep[i]) {
          if (xs[end] - xs[i] < minGap) keep[i] = false;
          break;
        }
      }
      keep[end] = true;
    }
  }
  return keep;
}

/** A word's executor verdict as the band's tooltip reads it. */
function executorVerdictText(word: TrainingExecutorWord): string {
  const checks = word.checks.map(checkText).join(", ");
  const told = word.flownRow === null ? "" : `, told at its step ${word.flownRow}`;
  return `the executor: ${word.status}${told}${checks ? ` — ${checks}` : ""}${word.reason ? ` — ${word.reason}` : ""}`;
}

/** The replay's chip — its outcome and its words inside of those judged — and its full reading, for the tooltip. */
function replayChip(flight: TrainingExecutorFlight): { text: string; ok: boolean; title: string } {
  if (!flight.flown) return { text: "replay: not flown", ok: false, title: `the executor's replay does not fly it: ${flight.group}` };
  const counts = executorWordCounts(flight);
  // the judge's own tally, as the replay gate counts it (the word left to intercept the final on its own counts twice)
  const { wordsInside, wordsJudged } = flight.counts;
  const title = `The executor (${flight.group}) flew this sentence from row 0, each word said where the observed aircraft ` +
    `heard it: ${TRAINING_OUTCOME_TAG[flight.outcome]}` +
    (flight.crossing === null ? "" : `, ${crossingText(flight.crossing)} at ${formatSeconds(flight.crossing.atS)} s`) +
    ` · ${wordsInside}/${wordsJudged} words inside their envelopes, each re-drawn from where the executor was told it` +
    (counts.notJudged ? `, ${counts.notJudged} not judged` : "") + (counts.notReached ? `, ${counts.notReached} not reached` : "") +
    (counts.superseded ? `, ${counts.superseded} superseded` : "") +
    ` · evaluation ${flight.evaluation.replay} (observed ${flight.evaluation.observed})` +
    (flight.refused === null ? "" : ` · its track refused by the labeller's gate: ${flight.refused}`);
  return {
    text: `replay: ${TRAINING_OUTCOME_TAG[flight.outcome]} · ${wordsInside}/${wordsJudged}${flight.refused === null ? "" : " · refused"}`,
    ok: flight.outcome === "landed" && wordsInside === wordsJudged && flight.refused === null, title,
  };
}

/** The labeller's verdicts in one chip, and what each count leaves out, for the tooltip. */
function verdictChip(flight: TrainingFlight): { text: string; title: string } {
  const verdicts = trainingVerdicts(flight);
  const capture = verdicts.captureTurn;
  const captureText = capture === null ? "—"
    : capture.progressOk && capture.rateOk ? "✓"
      : `✗ ${[capture.progressOk ? null : "monotone", capture.rateOk ? null : "rate"].filter(Boolean).join(", ")}`;
  return {
    text: `heading ${verdicts.headingContained}/${verdicts.headingJudged} · capture ${captureText} · altitude ` +
      `${verdicts.altitudeContained}/${verdicts.altitudeWords} · speed ${verdicts.speedContained}/${verdicts.speedWords}`,
    title: "The labeller's own checks of this flight's envelopes (Reading.checks): heading words inside their bands" +
      (verdicts.headingNotJudged ? ` (${verdicts.headingNotJudged} more with no row of their own: the lead reaches the clearance)` : "") +
      `; the capture turn ${capture === null ? "— none, on the final at entry" : `monotone ${checkMark(capture.progressOk)}, rate ` +
        `and bank ${checkMark(capture.rateOk)}`}; altitude tubes held; speed words held.`,
  };
}

/** The dot a word's executor verdict draws: filled for a verdict, hollow for none; none for a word with no check. */
function verdictMark(status: TrainingExecutorWord["status"]): { fill: string; stroke: string } | null {
  switch (status) {
    case "inside": return { fill: TRAINING_EXECUTOR_COLOR, stroke: TRAINING_EXECUTOR_COLOR };
    case "outside": return { fill: TRAINING_OUTSIDE_COLOR, stroke: TRAINING_OUTSIDE_COLOR };
    case "no check": return null;
    default: return { fill: "none", stroke: TRAINING_RAW_COLOR };
  }
}

/** A model's sample as its chip reads it — how the flight ended, where and when — and its full reading, for the tooltip. */
function sampleChip(view: TrainingGenerationView, sentence: TrainingGeneratedSentence, flight: TrainingFlight,
  runwayName: (index: number) => string, stepS: number): { text: string; title: string } {
  const { model, generation } = view.overlay;
  const later = sentence.events.filter((event) => event.row > sentence.firstRow);
  const said = later.filter((event) => event.row * stepS < sentence.endS).length;
  const where = TRAINING_CROSSING_OUTCOMES.includes(sentence.outcome)
    ? ` on ${runwayName(sentence.lastRunway)}` : `, pointing at ${runwayName(sentence.lastRunway)}`;
  return {
    text: `#${sentence.sample + 1}: ${TRAINING_OUTCOME_TAG[sentence.outcome]}${where} at ${formatSeconds(sentence.endS)} s ` +
      `(observed ${formatSeconds(flight.rows * stepS)} s)`,
    title: `${model.label}, sample ${sentence.sample + 1} of ${generation.samples}: it spoke from step ` +
      `${sentence.firstRow}, ${said} words after it before the flight ended` +
      (later.length > said ? ` (and ${later.length - said} after the end, to where the executor stopped)` : "") +
      `; the executor flew each step as it was said and the flight ` +
      `${TRAINING_OUTCOME_TEXT[sentence.outcome]}${sentence.crossing === null ? "" : ` — ${crossingText(sentence.crossing)}`} at ` +
      `${formatSeconds(sentence.endS)} s. Runway ${runwayName(sentence.firstRunway)} at its first step` +
      (sentence.lastRunway === sentence.firstRunway ? "" : `, ${runwayName(sentence.lastRunway)} at its end`) +
      ` (observed ${flight.runway})` + (sentence.runwayChanges ? `, ${sentence.runwayChanges} runway changes` : "") +
      (sentence.goArounds ? `, ${sentence.goArounds} go-around words` : "") +
      `; ${sentence.clearedAtEnd ? "cleared" : "not cleared"} at its end. The masks removed a mean ` +
      Object.entries(sentence.forbiddenMass).map(([column, mass]) => `${column} ${mass.toFixed(4)}`).join(", ") +
      " of its probability a step.",
  };
}

/** The bar's own height on the workbench (`--training-bar-height`): the docks end above it. */
function useBarHeight(): (node: HTMLElement | null) => void {
  const [node, setNode] = useState<HTMLElement | null>(null);
  useLayoutEffect(() => {
    const shell = node?.closest<HTMLElement>(".workbench") ?? null;
    if (node === null || shell === null) return;
    const publish = () => shell.style.setProperty("--training-bar-height", `${node.offsetHeight}px`);
    publish();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(publish);
    observer?.observe(node);
    return () => {
      observer?.disconnect();
      shell.style.removeProperty("--training-bar-height");
    };
  }, [node]);
  return setNode;
}

export default function TrainingSentenceBar() {
  const {
    mode, trainingSelection: selection, trainingLayers,
    trainingColumn: focusColumn, setTrainingColumn: setFocusColumn,
    trainingExecutor, trainingPrior, trainingAutopilot, trainingPick, setTrainingPick, trainingAutopilotAuto,
    trainingGenerations, trainingSource, setTrainingSource,
  } = useApp();
  const { trainingCursorS: cursorS, setTrainingCursorS: setCursorS } = useTrainingCursor();
  const [frame, frameW] = useMeasuredWidth(MIN_PLOT_W + GUTTER + PAD_R, DEFAULT_PLOT_W + GUTTER + PAD_R);
  const bar = useBarHeight();
  const [openWindow, setOpenWindow] = useState<"readback" | "prior" | null>(null);
  const [notesOpen, setNotesOpen] = useState<boolean>(false);

  // the Training session outlives a task switch (the panel stays mounted): the bar draws only in Training
  if (!selection || mode !== "training") return null;
  const { flight, vocabulary, candidates } = selection;
  const stepS = vocabulary.stepS;
  const plotW = frameW - GUTTER - PAD_R;
  const { tS } = flight.signals;
  // THE SENTENCE READ: the truth, or the chosen model's sample of this flight (none: a flight its readout does not fly)
  const models = trainingGenerations.flatMap((view) => overlayOnScreen(view, selection) ?? []);
  const read = generationOnScreen(trainingGenerations, trainingSource, selection);
  const model = read?.view ?? null;
  const generated = read?.sentence ?? null;
  const modelColour = model === null ? null : trainingModelColour(model.overlay.model);
  const sentence: TrainingSentence<TrainingSentenceEvent> = generated ?? trainingTruthSentence(flight);
  // Step r covers [r·step, (r+1)·step): the axis ends where the last step does — of the flight, or of the sentence read
  // when that runs longer.
  const endS = Math.max(flight.rows * stepS, generated === null ? 0 : Math.max(generated.rows * stepS, generated.endS));
  const timeOf = (row: number) => row * stepS;
  const xFor = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  const verdicts = trainingVerdicts(flight);
  // a model's rows run on past the observed ones when its flight lasts longer
  const cursorRow = generated === null ? rowAtTime(tS, cursorS) : generatedRowAt(stepS, cursorS);
  const toggle = (name: "readback" | "prior") => setOpenWindow((open) => (open === name ? null : name));
  const runwayName = (index: number) => candidates[index].ident;

  // The steps that SAY something, numbered; the sentence's first step is its opening and is always complete.
  const issueRows = [...new Set(sentence.events.map((event) => event.row))].sort((a, b) => a - b);
  const stepShown = spacedLabels(issueRows.map((row) => xFor(timeOf(row))), STEP_LABEL_GAP);
  // the axis ends where the longer sentence does, and says so
  const tickRows = [...issueRows, Math.round(endS / stepS)];
  const tickShown = spacedLabels(tickRows.map((row) => xFor(timeOf(row))), TICK_LABEL_GAP, true);
  const cleared = trainingClearedValue(vocabulary);
  const modelClearance = generated === null ? null
    : generated.events.find((event) => event.column === TRAINING_COLUMN_INDEX.approach && event.value === cleared && event.row > generated.firstRow) ?? null;
  const markers = generated === null ? [
    { at: timeOf(flight.joinRow), key: "cleared", colour: TRAINING_COLUMN_COLOR.approach, dash: "3 3",
      title: `cleared to join the final at ${formatSeconds(timeOf(flight.joinRow))} s` },
    { at: timeOf(flight.captureRow), key: "captured", colour: TRAINING_CORRIDOR_COLOR, dash: "3 3",
      title: `the final captured at ${formatSeconds(timeOf(flight.captureRow))} s, ` +
        `${(flight.captureBeforeThresholdM / 1000).toFixed(1)} km before the threshold` },
    { at: timeOf(flight.unspecifiedRow), key: "unspecified", colour: TRAINING_COLUMN_COLOR.speed, dash: "3 3",
      title: `speed left to the pilot from ${formatSeconds(timeOf(flight.unspecifiedRow))} s` },
  ] : [
    ...(modelClearance === null ? [] : [{ at: timeOf(modelClearance.row), key: "model-cleared", colour: TRAINING_COLUMN_COLOR.approach,
      dash: "3 3", title: `${model!.overlay.model.label} cleared the flight to join the final at ${formatSeconds(timeOf(modelClearance.row))} s` }]),
    { at: flight.rows * stepS, key: "observed-end", colour: TRAINING_TRACE_COLOR, dash: "2 3",
      title: `the observed flight's sentence ends at ${formatSeconds(flight.rows * stepS)} s` },
    { at: generated.endS, key: "model-end", colour: modelColour!, dash: undefined,
      title: `${model!.overlay.model.label}'s flight ${TRAINING_OUTCOME_TEXT[generated.outcome]} at ${formatSeconds(generated.endS)} s` },
  ];
  const landing = flight.envelopes.approach.landing;
  // the overlays and the live executor, when they are of the flight on screen (not the last one's, in flight)
  const executor = overlayOnScreen(trainingExecutor, selection)?.flight ?? null;
  const prior = overlayOnScreen(trainingPrior, selection);
  const autopilot = autopilotOnScreen(trainingAutopilot, selection);
  const replay = executor === null ? null : replayChip(executor);
  const verdictsChip = verdictChip(flight);
  const flightFacts = `${flight.typecode ?? "type unknown"} · ${flight.stratum} · ${flight.rows} steps · ` +
    `${verdicts.instructionsAfterStep0} words after step 0 · ${verdicts.silentSteps} of ${flight.rows - 1} later steps silent`;
  // the Fly button: the selected TRUTH word's segment, and what the live executor is doing with it
  const focusRun = focusColumn === null || generated !== null ? null : trainingWordAt(flight, focusColumn, cursorRow);
  const pickedHere = focusRun !== null && trainingPick !== null && trainingPick.column === focusColumn
    && trainingPick.row === focusRun.row;
  const flyingHere = pickedHere && autopilot?.status === "flying";
  const flyLabel = flyingHere ? "Flying …" : pickedHere && autopilot !== null ? "↻ Fly again" : "▶ Fly";
  const chip = model !== null && generated !== null ? sampleChip(model, generated, flight, runwayName, stepS) : null;
  const truthEvents = flight.words.events;
  const observedW = generated === null ? 0 : xFor(timeOf(generated.firstRow)) - GUTTER;
  // the rows a model spoke on after its flight ended, to where the executor stopped — only when it said at least one
  // whole step more (every sentence's last step runs on past its end by up to a step)
  const afterEndW = generated === null || timeOf(generated.rows) - generated.endS <= stepS ? 0
    : xFor(timeOf(generated.rows)) - xFor(generated.endS);
  // a model chosen that does not fly this flight: the truth is drawn, and said so
  const notFlown = model !== null && generated === null ? model : null;
  /** Choose a model's sentence: its first sample. */
  const chooseModel = (overlayId: string) => setTrainingSource({ overlayId, sample: 0 });

  return (
    <section className="training-sentence-bar" aria-label="Sentence bar" ref={bar}
      style={generated === null ? undefined : { borderColor: modelColour! }}>
      <TrainingLegend layers={trainingLayers} vocabulary={vocabulary} executorTrack={executor?.flown === true}
        autopilotColour={autopilot?.status === "ready" && autopilotHasLine(autopilot.segment) ? autopilotColour(autopilot.segment) : null}
        model={model === null || generated === null ? null : { label: model.overlay.model.label, colour: modelColour!, samples: model.flight.samples.length }} />
      <header className="training-sentence-head">
        {models.length > 0 ? (
          <span className="training-source-tabs" role="group" aria-label="Which sentence is read">
            <button type="button" className="training-source-tab" aria-pressed={model === null}
              title="The labelled sentence of the observed flight" onClick={() => setTrainingSource(null)}>
              Truth
            </button>
            {models.map((view) => {
              const colour = trainingModelColour(view.overlay.model);
              const on = model?.overlay.overlayId === view.overlay.overlayId;
              return (
                <button key={view.overlay.overlayId} type="button" className="training-source-tab" aria-pressed={on}
                  style={on ? { borderColor: colour, color: colour } : undefined} title={view.overlay.model.fineTuning === null
                    ? `${view.overlay.model.label}: the prior trained on data alone, speaking its own sentences`
                    : `${view.overlay.model.label}: post-trained (${view.overlay.model.fineTuning.schema}, round ` +
                      `${view.overlay.model.fineTuning.round}), speaking its own sentences`}
                  onClick={() => chooseModel(view.overlay.overlayId)}>
                  <span className="training-source-dot" style={{ background: colour }} />
                  {view.overlay.model.label}
                </button>
              );
            })}
          </span>
        ) : null}
        {model !== null && model.flight.flown ? (
          <span className="training-sample-buttons" role="group" aria-label={`${model.overlay.model.label}'s samples`}>
            {model.flight.samples.map((item) => (
              <button key={item.sample} type="button" aria-pressed={generated?.sample === item.sample}
                className={`training-sample-button${item.outcome === "landed" ? " landed" : ""}`}
                style={generated?.sample === item.sample ? { borderColor: modelColour!, background: `${modelColour}33` } : undefined}
                title={`sample ${item.sample + 1}: ${TRAINING_OUTCOME_TAG[item.outcome]} at ${formatSeconds(item.endS)} s`}
                onClick={() => setTrainingSource({ overlayId: model.overlay.overlayId, sample: item.sample })}>
                {item.sample + 1}{item.outcome === "landed" ? "" : " ✗"}
              </button>
            ))}
          </span>
        ) : null}
        <strong title={flightFacts}>{flight.callsign}</strong>
        <span className="training-chip">runway {flight.runway}</span>
        {notFlown !== null ? (
          <span className="training-chip training-sample-chip" title={`${notFlown.overlay.model.label}'s sentences fly only the ` +
            "flights on their own aircraft dynamics, as its readout does"}>
            {notFlown.overlay.model.label}: not flown ({notFlown.flight.group}) — the truth is shown
          </span>
        ) : null}
        {generated === null ? (
          <>
            <span className="training-chip" title={verdictsChip.title}>{verdictsChip.text}</span>
            {replay !== null ? (
              <span className="training-chip training-sentence-executor" title={replay.title}
                style={{ color: replay.ok ? TRAINING_EXECUTOR_COLOR : executor!.flown ? TRAINING_OUTSIDE_COLOR : TRAINING_RAW_COLOR }}>
                {replay.text}
              </span>
            ) : null}
            {autopilot ? <TrainingAutopilotStatus view={autopilot} selection={selection} /> : null}
          </>
        ) : (
          <span className="training-chip training-sample-chip" title={chip!.title}
            style={{ color: generated.outcome === "landed" ? modelColour! : TRAINING_OUTSIDE_COLOR }}>
            {chip!.text}
          </span>
        )}
        <span className="training-sentence-cursor-readout">t = {formatSeconds(cursorS)} s · step {cursorRow}</span>
        {generated === null ? (
          <button type="button" className="training-autopilot-fly" disabled={focusRun === null || flyingHere}
            title={focusRun === null
              ? "Select a word (click its band): the executor then flies that word's segment from where it was said."
              : `The executor flies ${focusColumn} from step ${focusRun.row} now, on the backend, from the observed state there.`}
            onClick={() => setTrainingPick(nextPick(trainingPick, focusColumn!, focusRun!.row))}>
            {flyLabel}
          </button>
        ) : null}
        {generated === null ? (
          <button type="button" className="training-sentence-readback-button" aria-pressed={openWindow === "readback"}
            title="The truth sentence against its track, envelope by envelope" onClick={() => toggle("readback")}>
            Read-back
          </button>
        ) : null}
        {prior && generated === null ? (
          <button type="button" className="training-sentence-readback-button" aria-pressed={openWindow === "prior"}
            title={`What the prior gives each column at each step (teacher-forced) — this flight ` +
              `${prior.flight.nllPerStep.toFixed(3)} nats per step, ${prior.overlay.readout.split} ` +
              `${prior.overlay.readout.model.nllPerStep.toFixed(4)}`}
            onClick={() => toggle("prior")}>
            Prior
          </button>
        ) : null}
        <button type="button" className="training-sentence-notes-toggle" aria-expanded={notesOpen}
          aria-label={notesOpen ? "Hide the notes" : "How to read the bar"} onClick={() => setNotesOpen((open) => !open)}>
          ⓘ
        </button>
      </header>

      <div className="training-sentence-frame" ref={frame}>
        <svg className="training-sentence-svg" width={GUTTER + plotW + PAD_R} height={VIEW_H}
          viewBox={`0 0 ${GUTTER + plotW + PAD_R} ${VIEW_H}`} role="group"
          aria-label={generated === null ? `The sentence of ${flight.callsign} on runway ${flight.runway}`
            : `${model!.overlay.model.label}'s sentence ${generated.sample + 1} for ${flight.callsign}`}>
          <defs>
            <pattern id={HATCH_ID} patternUnits="userSpaceOnUse" width={5} height={5} patternTransform="rotate(45)">
              <line x1={0} y1={0} x2={0} y2={5} stroke="#ffffff" strokeOpacity={0.28} strokeWidth={1.6} />
            </pattern>
            <pattern id={OBSERVED_ID} patternUnits="userSpaceOnUse" width={6} height={6} patternTransform="rotate(-45)">
              <line x1={0} y1={0} x2={0} y2={6} stroke={TRAINING_RAW_COLOR} strokeOpacity={0.35} strokeWidth={1} />
            </pattern>
          </defs>
          {/* the steps that say something: numbers along the top, each a button */}
          {issueRows.map((row, index) => {
            const left = index === 0 ? GUTTER : xFor((timeOf(issueRows[index - 1]) + timeOf(row)) / 2);
            const right = index === issueRows.length - 1 ? GUTTER + plotW : xFor((timeOf(row) + timeOf(issueRows[index + 1])) / 2);
            const words = sentence.events.filter((event) => event.row === row);
            const name = `Step ${row} at ${formatSeconds(timeOf(row))} s: ` + words.map((event) =>
              `${TRAINING_COLUMNS[event.column]} ${trainingWordLabel(vocabulary, candidates, TRAINING_COLUMNS[event.column], event.value)}`).join(", ");
            return (
              <g key={`step-${row}`} role="button" tabIndex={0} aria-label={name} className="training-sentence-event"
                onClick={() => setCursorS(timeOf(row))}
                onKeyDown={(keyEvent) => {
                  if (keyEvent.key === "Enter" || keyEvent.key === " ") setCursorS(timeOf(row));
                }}>
                <title>{name}</title>
                <rect x={left} y={2} width={Math.max(right - left, 1)} height={HEAD_H - 6} fill="transparent" />
                <line x1={xFor(timeOf(row))} x2={xFor(timeOf(row))} y1={HEAD_H - 6} y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H}
                  className="training-sentence-event-line" />
                {stepShown[index] ? (
                  <text x={xFor(timeOf(row))} y={HEAD_H - 9} textAnchor="middle" className="training-sentence-event-number">{row}</text>
                ) : null}
              </g>
            );
          })}

          {/* a model's sentence: the steps it only observed, before it speaks */}
          {generated !== null ? (
            <g aria-label={`steps 0–${generated.firstRow - 1}: observed only, the model speaks from step ${generated.firstRow}`}
              className="training-sentence-observed">
              <title>steps 0–{generated.firstRow - 1}: observed only — the model speaks from step {generated.firstRow}</title>
              <rect x={GUTTER} y={HEAD_H} width={Math.max(observedW, 0)} height={TRAINING_COLUMNS.length * ROW_H}
                fill={`url(#${OBSERVED_ID})`} />
              {observedW >= OBSERVED_LABEL_MIN_W ? (
                <text x={GUTTER + observedW / 2} y={HEAD_H + (TRAINING_COLUMNS.length * ROW_H) / 2 + 4} textAnchor="middle"
                  className="training-sentence-observed-label">observed</text>
              ) : null}
            </g>
          ) : null}

          {TRAINING_COLUMNS.map((column, position) => {
            const y = HEAD_H + position * ROW_H;
            const colour = TRAINING_COLUMN_COLOR[column];
            const selectedColumn = column === focusColumn;
            return (
              <g key={column} aria-label={`${column} row`}>
                <rect x={GUTTER} y={y} width={plotW} height={ROW_H} className={`training-sentence-row-bg${position % 2 ? " odd" : ""}`} />
                <text x={GUTTER - 8} y={y + ROW_H / 2 + 4} textAnchor="end" className="training-sentence-row-label"
                  style={selectedColumn ? { fill: TRAINING_WORD_COLOR, fontWeight: 600 } : undefined}>
                  {TRAINING_COLUMN_LABEL[column]}
                </text>
                {sentenceColumnRuns(sentence, column).map((run) => {
                  const x = xFor(timeOf(run.row));
                  const width = xFor(timeOf(run.endRow)) - x;
                  const label = trainingWordLabel(vocabulary, candidates, column, run.value);
                  const verdict = executor === null || generated !== null ? null : executorWordAt(executor, run.row, column);
                  const mark = verdict === null ? null : verdictMark(verdict.status);
                  const title = generated === null
                    ? `${column} ${label} — ${trainingKindLabel((run.event as TrainingWordEvent).kind)}, issued at step ${run.row} ` +
                      `(${formatSeconds(timeOf(run.row))} s), in force to ${formatSeconds(timeOf(run.endRow))} s` +
                      (verdict === null ? "" : `\n${executorVerdictText(verdict)}`)
                    : `${column} ${label} — said by ${model!.overlay.model.label} at step ${run.row} ` +
                      `(${formatSeconds(timeOf(run.row))} s), in force to ${formatSeconds(timeOf(run.endRow))} s`;
                  const selected = selectedColumn && run.row <= cursorRow && cursorRow < run.endRow;
                  const choose = () => {
                    if (selected) {
                      setFocusColumn(null);
                      if (generated === null) setTrainingPick(null);
                      return;
                    }
                    setFocusColumn(column);
                    setCursorS(timeOf(run.row));
                    if (generated === null && trainingAutopilotAuto) setTrainingPick(nextPick(trainingPick, column, run.row));
                  };
                  return (
                    <g key={`${column}-${run.row}`} role="button" tabIndex={0} aria-label={title} aria-pressed={selected}
                      className={`training-sentence-band${generated === null ? "" : " model"}`} onClick={choose}
                      onKeyDown={(keyEvent) => {
                        if (keyEvent.key === "Enter" || keyEvent.key === " ") choose();
                      }}>
                      <title>{title}</title>
                      <rect x={x + 1} y={y + 4} width={Math.max(width - 2, 1)} height={ROW_H - 8} rx={3} fill={colour}
                        fillOpacity={selected ? 0.4 : 0.16} stroke={selected ? TRAINING_WORD_COLOR : colour}
                        strokeOpacity={selected ? 1 : 0.6} strokeWidth={selected ? 2 : 1}
                        strokeDasharray={generated === null || selected ? undefined : "3 2"} className="training-sentence-band-fill" />
                      {/* a model's word: hatched, so it never passes for a labelled one */}
                      {generated !== null ? (
                        <rect x={x + 1} y={y + 4} width={Math.max(width - 2, 1)} height={ROW_H - 8} rx={3}
                          fill={`url(#${HATCH_ID})`} pointerEvents="none" className="training-sentence-hatch" />
                      ) : null}
                      {/* the issue itself */}
                      <rect x={x} y={y + 2} width={2} height={ROW_H - 4} fill={colour} className="training-sentence-issue" />
                      {/* the executor's verdict on this word */}
                      {mark !== null ? (
                        <circle cx={x + 7} cy={y + ROW_H / 2} r={3.2} fill={mark.fill} stroke={mark.stroke} strokeWidth={1.2}
                          className={`training-sentence-verdict training-sentence-verdict-${verdict!.status.replace(/ /g, "-")}`} />
                      ) : null}
                      {width >= LABEL_MIN_W ? (
                        <text x={x + width / 2} y={y + ROW_H / 2 + 4} textAnchor="middle" className="training-sentence-word" fill={colour}>
                          {label}
                        </text>
                      ) : null}
                    </g>
                  );
                })}
                {/* under a model's row: where the truth says a word of this column */}
                {generated !== null ? truthEvents.filter((event) => event.row > 0 && event.column === position).map((event) => (
                  <line key={`truth-${event.row}`} x1={xFor(timeOf(event.row))} x2={xFor(timeOf(event.row))} y1={y + ROW_H - 5}
                    y2={y + ROW_H} stroke={TRAINING_TRACE_COLOR} strokeWidth={1.5} className="training-sentence-truth-tick" />
                )) : null}
              </g>
            );
          })}

          {/* a model's sentence: the rows it spoke on after its flight ended — over its bands, which it darkens */}
          {generated !== null && afterEndW > 0 ? (
            <g aria-label={`after ${formatSeconds(generated.endS)} s: the flight had ended; the model spoke on to where the executor stopped`}
              className="training-sentence-observed">
              <title>after {formatSeconds(generated.endS)} s the flight had ended; the model spoke on to where the executor stopped</title>
              <rect x={xFor(generated.endS)} y={HEAD_H} width={afterEndW} height={TRAINING_COLUMNS.length * ROW_H}
                fill={TRAINING_SURFACE_COLOR} fillOpacity={0.55} />
              {afterEndW >= OBSERVED_LABEL_MIN_W ? (
                <text x={xFor(generated.endS) + afterEndW / 2} y={HEAD_H + (TRAINING_COLUMNS.length * ROW_H) / 2 + 4}
                  textAnchor="middle" className="training-sentence-observed-label">after the end</text>
              ) : null}
            </g>
          ) : null}

          {/* the moments that are not words */}
          {markers.map((marker) => (
            <g key={marker.key} aria-label={marker.title}>
              <title>{marker.title}</title>
              <line x1={xFor(marker.at)} x2={xFor(marker.at)} y1={HEAD_H} y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H}
                stroke={marker.colour} strokeDasharray={marker.dash} strokeWidth={marker.dash === undefined ? 2 : 1}
                className="training-sentence-marker" />
            </g>
          ))}

          <line x1={GUTTER} x2={GUTTER + plotW} y1={HEAD_H + TRAINING_COLUMNS.length * ROW_H}
            y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H} className="training-sentence-axis" />
          {tickRows.map((row, index) => (tickShown[index] ? (
            <text key={`tick-${row}`} x={xFor(timeOf(row))} y={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 15}
              textAnchor={index === tickRows.length - 1 ? "end" : "middle"} className="training-sentence-tick">
              {index === tickRows.length - 1 ? `${formatSeconds(timeOf(row))} s` : formatSeconds(timeOf(row))}
            </text>
          ) : null))}
          {/* the cursor, kept on the axis (it may have been put past the truth's end while a longer sentence was read) */}
          <line x1={xFor(Math.min(cursorS, endS))} x2={xFor(Math.min(cursorS, endS))} y1={HEAD_H - 8}
            y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 4} className="training-sentence-cursor" />
        </svg>
      </div>

      {notesOpen ? (
        <footer className="training-sentence-legend">
          <span>{flight.callsign}: {flightFacts} · {verdictsChip.title}</span>
          {replay !== null && generated === null ? <span>{replay.title}</span> : null}
          {chip !== null ? <span>{chip.title}</span> : null}
          <span>
            A band is a word in force, from the tick where it was issued to the next word of its column; step 0 gives all
            six. Click a band to select its word — here, in the read-back check and in 3D — and again to clear it; ▶ Fly
            flies the selected truth word's segment live (a band click does too, with the panel's "Fly on band click" on). The
            dashed lines: the clearance, the capture of the final, the speed left to the pilot.
          </span>
          {models.length > 0 ? (
            <span>
              The tabs choose the sentence read: the truth, or a model's own — one of its samples, numbered (✗: its flight did
              not land). A model's words are hatched with a dashed edge; the grey hatch before its first step is what it only
              observed; a white tick under a row is where the truth says a word of that column; the solid line in the model's
              colour is where its flight ended, the dashed white one where the observed flight's sentence ends; words under the
              dark shade after it were said to a flight already over (the executor flies on after crossing the threshold
              without the capture, or a stall). The time axis runs on past the observed flight's when the model's flight
              lasts longer. The Read-back and Prior windows read the truth: they are offered on its tab.
            </span>
          ) : null}
          <span>
            The sentence ends {(landing.lastRowBeforeThresholdM / 1000).toFixed(2)} km before the threshold
            {landing.cutAtCrossing ? ", cut before the last passage of the threshold" : ", where the data ends"}.
          </span>
          {executor !== null && generated === null ? (
            <span>
              The executor's verdict on a word, at its band's left: <b style={{ color: TRAINING_EXECUTOR_COLOR }}>●</b> inside,{" "}
              <b style={{ color: TRAINING_OUTSIDE_COLOR }}>●</b> outside, ○ not judged, not reached or superseded; none for a
              word with no check of its own — judged on envelopes re-drawn from where the executor was told the word.
            </span>
          ) : null}
        </footer>
      ) : null}

      {openWindow === "readback" && generated === null ? (
        <TrainingReadbackWindow flight={flight} vocabulary={vocabulary} candidates={candidates} layers={trainingLayers}
          cursorS={cursorS} onCursorChange={setCursorS} column={focusColumn} onColumnChange={setFocusColumn}
          onClose={() => setOpenWindow(null)} executor={executor}
          autopilot={autopilot?.status === "ready" ? autopilot.segment : null} />
      ) : null}
      {openWindow === "prior" && prior && generated === null ? (
        <TrainingPriorWindow flight={flight} vocabulary={vocabulary} candidates={candidates} prior={prior} cursorS={cursorS}
          onCursorChange={setCursorS} column={focusColumn} onColumnChange={setFocusColumn} onClose={() => setOpenWindow(null)} />
      ) : null}
    </section>
  );
}
