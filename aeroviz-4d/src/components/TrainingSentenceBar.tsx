/**
 * TrainingSentenceBar.tsx
 * -----------------------
 * The sentence as a picture: the six columns as six rows — runway pointer, approach, heading, altitude, descent angle,
 * speed, in the vocabulary's order — against the flight's own time. Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 * A BAND IS A WORD IN FORCE, from the step it was issued to the step the next word of its column replaces it. Step 0
 * carries all six columns, so every row opens with a band at 0; after that a column is "unchanged" until its next word,
 * and most steps say nothing at all. The tick at a band's left edge is the issue itself; the numbers along the top are
 * the steps that say anything. The three moments that are not words — the clearance, the capture of the final and the
 * speed becoming "unspecified" — are dashed lines across the rows.
 *
 * WHICH SENTENCE (`trainingSource`): the tabs at the head of the bar — "Truth", then one per model whose own sentences
 * are published for the set (`trainingGenerations`: base, landing, augmented, each in its colour; two runs of one stage
 * are two tabs, named with their run) — choose what the rows draw. A model published at SEVERAL ROUNDS shows its rounds
 * beside the tabs ("r1 r2 …", each titled with how many of this flight's samples landed): a round keeps the sample number
 * read, so one flight is compared round by round; a tab returns to the round last read in it. Two runs of one stage are named with their run, one round exported twice with its overlay id. The TRUTH is the
 * labelled sentence, as above. A MODEL's is one of its samples (the numbered buttons, each marked by how the flight
 * ended), drawn in the same bands — one sentence is read at a time, so what says it is a model's is the frame: the bar's
 * border and a strip down the left edge of its rows in the model's colour, the tab, the sample's chip (the hatching this
 * once had made the bar hard to read, the user 2026-09-26). The rows it only observed (before its first predicted step)
 * are shaded; under each row a white tick wherever the truth says a word of that column, so where the two sentences part
 * is read at a glance; a dashed line where the model cleared the flight, a solid one in its colour where its flight
 * landed — a heavier one in the failure red where it did not (the user, 2026-09-30) — its time written on the axis under
 * it in the same colour, a dashed white one where the observed flight's sentence ends. Words a model said after its flight ended (the executor flies on after three outcomes its judge reads earlier) sit
 * under a grey shade. The
 * time axis is the observed flight's, stretched to the sentence read when that one runs longer (a model's flight that
 * timed out runs on to 1.5× the observed time): the truth is never squeezed by a sample it is not showing. Choosing a
 * model starts at its first sample. A model that does not fly the flight on screen leaves the truth drawn, with its whole
 * header, and says so. The Read-back and Prior windows read the truth, so they are offered only while it is read.
 * A model's words are flown live like the truth's (Fly, or a band click): the backend flies
 * the model's sentence again from its first step (`trainingAutopilot.ts`), which lands on the sample's own track.
 *
 * THE HEADER IS SHORT: the tabs, the callsign (its type, stratum and counts in its tooltip), the runway, the replay's chip
 * ONLY when it went wrong ("Replay · 2 words out", `replayIssueText` — the truth's; its words' dots say the rest), a
 * model's chip (how its sample ended), the live executor's line, the cursor, and the buttons — Fly, Read-back and Prior
 * (the truth's), and ⓘ for the notes. Every chip carries its full reading in its tooltip; the executor's replay of the
 * flight is read in full in the notes. The labeller's own verdicts are TALLIED AT EACH ROW'S END (`rowTally`, the truth's
 * only), lined up on the slash.
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
 * THE EXECUTOR, LIVE (`trainingAutopilot`): the Fly button PICKS the selected word of the sentence read for the live
 * executor (`trainingPick`, with its source; "↻ Fly again" once it has flown), and so does clicking a band (clicking the
 * selected band again clears the pick) — never the cursor. Its
 * line (`TrainingAutopilotStatus`) says whether the word stayed inside its envelope and the two times — the word itself
 * only once the selection has moved off it, so the line leaves the header's buttons on their row. A
 * model's word said after its flight ended has no flight to fly: its Fly button is off. On the flown word's row a small
 * CURSOR says where the executor is (`AutopilotCursor`): at the word's step, pulsing, while the backend flies it; then
 * with the 3D aircraft as it flies the segment out — the same clock, `autopilotPlaybackS` — faded past where the next
 * word of the column was heard (the tail); left at the segment's end, as the aircraft is.
 *
 * THE BAR MEASURES ITSELF: its height is published on the page's root as `--training-bar-height`, so the docks — and
 * Cesium's credits — end above it (`index.css`) instead of under it; the bar itself sits at the bottom edge (Training
 * hides Cesium's clock dial and timeline).
 */

import { useLayoutEffect, useRef, useState } from "react";
import { useApp, useTrainingCursor } from "../context/AppContext";
import TrainingLegend from "./TrainingLegend";
import TrainingTrafficStrip from "./TrainingTrafficStrip";
import { isWindowSentence, windowOnScreen, windowReading } from "../data/trainingTraffic";
import TrainingPriorWindow from "./TrainingPriorWindow";
import TrainingReadbackWindow from "./TrainingReadbackWindow";
import { TrainingAutopilotStatus } from "./TrainingAutopilotStatus";
import useMeasuredWidth from "../hooks/useMeasuredWidth";
import {
  TRAINING_AUTOPILOT_COLOR,
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_FAILURE_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_SURFACE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
  TRAINING_REPLAY_COLOR,
  trainingModelColour,
  trainingWindowReadingColour,
} from "../utils/trainingWordColors";
import {
  AUTOPILOT_TAIL_OPACITY,
  autopilotColour,
  autopilotHasLine,
  autopilotOnScreen,
  autopilotPlaybackS,
  nextPick,
  sameSource,
  type TrainingAutopilotView,
} from "../data/trainingAutopilot";
import {
  executorWordAt,
  executorWordCounts,
  generatedRowAt,
  augmentationText,
  generationOnScreen,
  sentenceAxisEndS,
  generationUnflownReason,
  isAugmentedStart,
  sourceOf,
  overlayOnScreen,
  replayVerdict,
  trainingModelGroups,
  type TrainingExecutorFlight,
  type TrainingExecutorWord,
  type TrainingGeneratedSentence,
  type TrainingGenerationView,
} from "../data/trainingOverlays";
import {
  formatSeconds,
  cursorOnFlight,
  rowAtTime,
  sentenceColumnRuns,
  sentenceWordAt,
  trainingBandLabel,
  trainingClearedValue,
  trainingKindLabel,
  trainingTruthSentence,
  trainingVerdicts,
  trainingWordLabel,
  TRAINING_COLUMN_INDEX,
  TRAINING_COLUMNS,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingSentence,
  type TrainingSentenceEvent,
  type TrainingWordEvent,
} from "../data/trainingSample";
import {
  checkMark, checkText, crossingText, replayIssueText, replayOutsideWords, TRAINING_COLUMN_LABEL, TRAINING_OUTCOME_TAG,
  TRAINING_OUTCOME_TEXT, trainingModelText,
} from "../data/trainingText";

// One SVG unit is one pixel: the bar is as wide as the dock and always VIEW_H tall.
const GUTTER = 96;
/** The right margin: each row's tally of the labeller's verdicts. */
const PAD_R = 44;
const HEAD_H = 22;
/** Where a row's tally puts its slash, past the plot's right end (px): the counts of every row line up on it. */
const TALLY_SLASH = 22;
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
/** The strip down the left edge of a model's rows, in its colour (px). */
const MODEL_STRIP_W = 4;
/** Where a model's flight ended (px): landed, and — heavier, in the failure red — did not. */
const MODEL_END_W = 2;
const FAILED_END_W = 3.5;

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

/** The replay of this flight in words, for the notes: its outcome and its words inside of those judged, and the words it
 *  flew outside their envelopes (``outside``, as `replayOutsideWords` names them). */
function replayText(flight: TrainingExecutorFlight, outside: string[]): string {
  if (!flight.flown) return `The executor's replay does not fly it: ${flight.group}.`;
  const counts = executorWordCounts(flight);
  // the judge's own tally, as the replay gate counts it (the word left to intercept the final on its own counts twice)
  const { wordsInside, wordsJudged } = flight.counts;
  return `The executor (${flight.group}) flew this sentence from row 0, each word said where the observed aircraft ` +
    `heard it: ${TRAINING_OUTCOME_TAG[flight.outcome]}` +
    (flight.crossing === null ? "" : `, ${crossingText(flight.crossing)} at ${formatSeconds(flight.crossing.atS)} s`) +
    ` · ${wordsInside}/${wordsJudged} words inside their envelopes, each re-drawn from where the executor was told it` +
    (counts.notJudged ? `, ${counts.notJudged} not judged` : "") + (counts.notReached ? `, ${counts.notReached} not reached` : "") +
    (counts.superseded ? `, ${counts.superseded} superseded` : "") +
    (outside.length === 0 ? "" : ` · outside: ${outside.join(", ")}`) +
    ` · evaluation ${flight.evaluation.replay} (observed ${flight.evaluation.observed})` +
    (flight.refused === null ? "" : ` · its track refused by the labeller's gate: ${flight.refused}`) + ".";
}

/** A row's tally of the labeller's own checks (Reading.checks), written at its end: the words held of those judged — split
 *  at the slash, which every row lines up on — or, on the approach row, the capture turn's mark; with its reading, for the
 *  tooltip. null for a row with no check of its own (runway, angle; the approach with no capture turn). */
function rowTally(verdicts: ReturnType<typeof trainingVerdicts>, column: TrainingColumn):
  { held: string; of: string; ok: boolean; title: string } | null {
  const count = (held: number, of: number, what: string) => ({ held: `${held}`, of: `/${of}`, ok: held === of, title: `${held} of ${of} ${what}` });
  switch (column) {
    case "heading":
      return count(verdicts.headingContained, verdicts.headingJudged, "heading words inside their bands" +
        (verdicts.headingNotJudged ? ` (${verdicts.headingNotJudged} more with no row of their own: the lead reaches the clearance)` : ""));
    case "altitude":
      return count(verdicts.altitudeContained, verdicts.altitudeWords, "altitude tubes held");
    case "speed":
      return count(verdicts.speedContained, verdicts.speedWords, "speed words held");
    case "approach": {
      const capture = verdicts.captureTurn;
      if (capture === null) return null;
      const ok = capture.progressOk && capture.rateOk;
      return { held: checkMark(ok), of: "", ok,
        title: `the capture turn: monotone ${checkMark(capture.progressOk)}, rate and bank ${checkMark(capture.rateOk)}` };
    }
    default:
      return null;
  }
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
function sampleChip(view: TrainingGenerationView, label: string, sentence: TrainingGeneratedSentence, flight: TrainingFlight,
  runwayName: (index: number) => string, stepS: number): { text: string; title: string } {
  const { generation } = view.overlay;
  const move = view.flight.start?.augmentation ?? null;
  const later = sentence.events.filter((event) => event.row > sentence.firstRow);
  const said = later.filter((event) => event.row * stepS < sentence.endS).length;
  const where = sentence.crossing !== null
    ? ` on ${runwayName(sentence.crossing.runway)}` : `, pointing at ${runwayName(sentence.lastRunway)}`;
  // in a window the judge may end it first: then it flew on silent to its own end, which the crossing is of
  const judged = isWindowSentence(sentence) && sentence.end !== null ? sentence : null;
  const ended = judged === null
    ? `${TRAINING_OUTCOME_TAG[sentence.outcome]}${where} at ${formatSeconds(sentence.endS)} s`
    : `${TRAINING_OUTCOME_TAG[sentence.outcome]} at ${formatSeconds(sentence.endS)} s · then ${TRAINING_OUTCOME_TAG[judged.own]}${where} ` +
      `at ${formatSeconds(judged.ownEndS)} s`;
  return {
    text: `#${sentence.sample + 1}: ${ended} ` +
      (move === null ? `(observed ${formatSeconds(flight.rows * stepS)} s)` : `· moved ${augmentationText(move)}`),
    title: (move === null ? "" : `From an augmented start: the flight's start rotated ${augmentationText(move)} (about the ` +
      `airport, raised, speed ×), drawn with seed ${generation.augment!.seed} in ${view.flight.start!.draws} ` +
      `draw${view.flight.start!.draws === 1 ? "" : "s"}; its time limit ${generation.augment!.timeoutFactor}× the observed ` +
      "flight's remaining time. ") +
      `${label}, sample ${sentence.sample + 1} of ${generation.samples}: it spoke from step ` +
      `${sentence.firstRow}, ${said} words after it before the flight ended` +
      (later.length > said ? ` (and ${later.length - said} after the end, to where the executor stopped)` : "") +
      `; the executor flew each step as it was said and the flight ` +
      (judged === null
        ? `${TRAINING_OUTCOME_TEXT[sentence.outcome]}${sentence.crossing === null ? "" : ` — ${crossingText(sentence.crossing)}`} at ` +
          `${formatSeconds(sentence.endS)} s`
        : `${TRAINING_OUTCOME_TEXT[sentence.outcome]}: the judge ended it at ${formatSeconds(sentence.endS)} s (${judged.end!.kind}, ` +
          `with ${judged.end!.with}); flying on, it ${TRAINING_OUTCOME_TEXT[judged.own]}` +
          `${sentence.crossing === null ? "" : ` — ${crossingText(sentence.crossing)}`} at ${formatSeconds(judged.ownEndS)} s`) +
      `. Runway ${runwayName(sentence.firstRunway)} at its first step` +
      (sentence.lastRunway === sentence.firstRunway ? "" : `, ${runwayName(sentence.lastRunway)} at its end`) +
      ` (${move === null ? "observed" : "the source flight landed on"} ${flight.runway})` + (sentence.runwayChanges ? `, ${sentence.runwayChanges} runway changes` : "") +
      (sentence.goArounds ? `, ${sentence.goArounds} go-around words` : "") +
      `; ${sentence.clearedAtEnd ? "cleared" : "not cleared"} at its end. The masks removed a mean ` +
      Object.entries(sentence.forbiddenMass).map(([column, mass]) => `${column} ${mass.toFixed(4)}`).join(", ") +
      " of its probability a step.",
  };
}

/** The live executor on the flown word's row, in the flight's time: at the word's step, pulsing, while the backend
 *  flies it; then where the 3D aircraft is as it flies the segment out (`autopilotPlaybackS`: the same clock), faded in
 *  the tail; left at the segment's end. A leaf that moves itself each frame, so the bar never re-renders for it. */
function AutopilotCursor({ view, stepS, y, x0, plotW, endS }: {
  view: Extract<TrainingAutopilotView, { status: "flying" | "ready" }>;
  stepS: number;
  /** The top of the flown word's row. */
  y: number;
  /** The bar's time axis: where it starts, how wide it is (px) and the time it ends at (s). A flight flown past the axis's
   *  end (the truth's last word flown on to a landing later than the observed one) holds the cursor there, and its title
   *  says so. */
  x0: number;
  plotW: number;
  endS: number;
}) {
  const node = useRef<SVGRectElement>(null);
  // placed before paint (never a frame at the axis's origin), and again whenever the axis is laid out anew
  useLayoutEffect(() => {
    const rect = node.current!;
    // the bar's xFor, clamped to the axis's end (primitives, so the effect runs again only when the axis changes)
    const put = (seconds: number, faded: boolean) => {
      rect.setAttribute("transform", `translate(${x0 + (Math.min(seconds, endS) / endS) * plotW} 0)`);
      rect.style.opacity = faded ? String(AUTOPILOT_TAIL_OPACITY) : "";
    };
    if (view.status === "flying") {
      put(view.request.row * stepS, false);
      return undefined;
    }
    const { track, tailFrom } = view.segment;
    const totalS = track.tS[track.tS.length - 1] - track.tS[0];
    const tailS = tailFrom === null ? Infinity : track.tS[tailFrom];
    let frame = 0;
    const draw = () => {
      const flownS = autopilotPlaybackS(track, view.playedAt, Date.now());
      put(track.tS[0] + flownS, track.tS[0] + flownS > tailS);
      if (flownS < totalS) frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [view, stepS, x0, plotW, endS]);
  const colour = view.status === "flying" ? TRAINING_AUTOPILOT_COLOR : autopilotColour(view.segment);
  return (
    <rect ref={node} x={-1.5} y={y + 1} width={3} height={ROW_H - 2} rx={1.5} fill={colour}
      className={`training-sentence-autopilot-cursor${view.status === "flying" ? " waiting" : ""}`}>
      <title>{view.status === "flying" ? "the executor flies this word on the backend"
        : "the autopilot's aircraft, flying this word's segment out in 3D" +
          (view.segment.track.tS[view.segment.track.tS.length - 1] > endS
            ? ` (held at the axis's end, ${formatSeconds(endS)} s, while it flies on to ` +
              `${formatSeconds(view.segment.track.tS[view.segment.track.tS.length - 1])} s)` : "")}</title>
    </rect>
  );
}

/** The bar's own height on the page's root (`--training-bar-height`): the docks end above it, and so do Cesium's credits,
 *  which sit outside the workbench. */
function useBarHeight(): (node: HTMLElement | null) => void {
  const [node, setNode] = useState<HTMLElement | null>(null);
  useLayoutEffect(() => {
    if (node === null) return;
    const page = document.documentElement;
    const publish = () => page.style.setProperty("--training-bar-height", `${node.offsetHeight}px`);
    publish();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(publish);
    observer?.observe(node);
    return () => {
      observer?.disconnect();
      page.style.removeProperty("--training-bar-height");
    };
  }, [node]);
  return setNode;
}

export default function TrainingSentenceBar() {
  const {
    mode, trainingSelection: selection, trainingLayers,
    trainingColumn: focusColumn, setTrainingColumn: setFocusColumn,
    trainingExecutor, trainingPrior, trainingAutopilot, trainingPick, setTrainingPick,
    trainingGenerations, trainingSource, setTrainingSource, trainingWindow,
  } = useApp();
  const { trainingCursorS: cursorS, setTrainingCursorS: setCursorS } = useTrainingCursor();
  const [frame, frameW] = useMeasuredWidth(MIN_PLOT_W + GUTTER + PAD_R, DEFAULT_PLOT_W + GUTTER + PAD_R);
  const bar = useBarHeight();
  const [openWindow, setOpenWindow] = useState<"readback" | "prior" | null>(null);
  const [notesOpen, setNotesOpen] = useState<boolean>(false);
  // the round last read in each model's tab (group key → overlay id), wherever it was chosen: its tab returns to it
  const roundRead = useRef<Record<string, string>>({});

  // the Training session outlives a task switch (the panel stays mounted): the bar draws only in Training
  if (!selection || mode !== "training") return null;
  const { flight, vocabulary, candidates } = selection;
  const stepS = vocabulary.stepS;
  const plotW = frameW - GUTTER - PAD_R;
  const { tS } = flight.signals;
  // THE SENTENCE READ: the truth, or the chosen model's sample of this flight (none: a flight its readout does not fly)
  const onScreen = trainingGenerations.flatMap((view) => overlayOnScreen(view, selection) ?? []);
  const read = generationOnScreen(trainingGenerations, trainingSource, selection);
  const model = read?.view ?? null;
  // THE START the models' sentences fly from: the set's own, or augmented ones (the chosen model's kind) — two families,
  // each with its own tabs; the truth is of the set's own start only
  const augmentedStart = model !== null && isAugmentedStart(model.overlay);
  const startsOffered = onScreen.some((view) => isAugmentedStart(view.overlay));
  const models = trainingModelGroups(onScreen.filter((view) => isAugmentedStart(view.overlay) === augmentedStart),
    (view) => view.overlay);
  const modelGroup = models.find((group) => group.members.some((view) => view === model)) ?? null;
  const modelName = model === null ? null : modelGroup!.memberLabel(model);
  // remembered at render, whichever control of the bar chose it (a tab, a round, the start switch): a write the render
  // reads nowhere
  // a model's tab in each start family remembers its own round: the group key is the same in both
  const familyKey = (key: string) => `${augmentedStart ? "augmented start" : "own start"} ${key}`;
  if (model !== null) roundRead.current[familyKey(modelGroup!.key)] = model.overlay.overlayId;
  const generated = read?.sentence ?? null;
  // a sample flown from an augmented start is read: nothing of the truth's own start is drawn against it
  const movedRead = generated !== null && augmentedStart;
  const modelColour = model === null ? null : trainingModelColour(model.overlay.model);
  const sentence: TrainingSentence<TrainingSentenceEvent> = generated ?? trainingTruthSentence(flight);
  // Step r covers [r·step, (r+1)·step): the axis ends where the last step does — of the flight, or of the sentence read
  // when that runs longer.
  const endS = sentenceAxisEndS(flight, stepS, generated);
  const cursorOn = cursorOnFlight(selection, cursorS, endS);
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
  // the axis ends where the longer sentence does, and says so; a model's flight's end is written under its line, in its
  // colour, always — a tick too close to it gives way (the axis's own end, too, when the flight ran to it)
  const tickRows = [...issueRows, Math.round(endS / stepS)];
  const endLabelX = generated === null ? null : xFor(generated.endS);
  const endAnchoredEnd = endLabelX !== null && GUTTER + plotW - endLabelX < TICK_LABEL_GAP / 2;
  // a label's middle: a centred one's x, an end-anchored one's (the axis's last, the end label at the axis's end) half a
  // label width to its left
  const middle = (x: number, anchoredEnd: boolean) => (anchoredEnd ? x - TICK_LABEL_GAP / 2 : x);
  const tickShown = spacedLabels(tickRows.map((row) => xFor(timeOf(row))), TICK_LABEL_GAP, true)
    .map((shown, index) => shown && (endLabelX === null
      || Math.abs(middle(xFor(timeOf(tickRows[index])), index === tickRows.length - 1) - middle(endLabelX, endAnchoredEnd))
        >= TICK_LABEL_GAP));
  // the multi-aircraft window the flight on screen is in, if any
  const traffic = windowOnScreen(trainingWindow, selection);
  // where a model's flight ended: in its colour when it landed, heavier in the failure red when it did not
  const endColour = generated === null ? null : generated.outcome === "landed" ? modelColour! : TRAINING_FAILURE_COLOR;
  const cleared = trainingClearedValue(vocabulary);
  const modelClearance = generated === null ? null
    : generated.events.find((event) => event.column === TRAINING_COLUMN_INDEX.approach && event.value === cleared && event.row > generated.firstRow) ?? null;
  const markers = generated === null ? [
    { at: timeOf(flight.joinRow), key: "cleared", colour: TRAINING_COLUMN_COLOR.approach, dash: "3 3", width: 1,
      title: `cleared to join the final at ${formatSeconds(timeOf(flight.joinRow))} s` },
    { at: timeOf(flight.captureRow), key: "captured", colour: TRAINING_CORRIDOR_COLOR, dash: "3 3", width: 1,
      title: `the final captured at ${formatSeconds(timeOf(flight.captureRow))} s, ` +
        `${(flight.captureBeforeThresholdM / 1000).toFixed(1)} km before the threshold` },
    { at: timeOf(flight.unspecifiedRow), key: "unspecified", colour: TRAINING_COLUMN_COLOR.speed, dash: "3 3", width: 1,
      title: `speed left to the pilot from ${formatSeconds(timeOf(flight.unspecifiedRow))} s` },
  ] : [
    ...(modelClearance === null ? [] : [{ at: timeOf(modelClearance.row), key: "model-cleared", colour: TRAINING_COLUMN_COLOR.approach,
      dash: "3 3", width: 1, title: `${modelName!} cleared the flight to join the final at ${formatSeconds(timeOf(modelClearance.row))} s` }]),
    ...(movedRead ? [] : [{ at: flight.rows * stepS, key: "observed-end", colour: TRAINING_TRACE_COLOR, dash: "2 3", width: 1,
      title: `the observed flight's sentence ends at ${formatSeconds(flight.rows * stepS)} s` }]),
    { at: generated.endS, key: "model-end", colour: endColour!, dash: undefined,
      width: generated.outcome === "landed" ? MODEL_END_W : FAILED_END_W,
      title: `${modelName!}'s flight ${TRAINING_OUTCOME_TEXT[generated.outcome]} at ${formatSeconds(generated.endS)} s` },
  ];
  const landing = flight.envelopes.approach.landing;
  // the overlays and the live executor, when they are of the flight on screen (not the last one's, in flight)
  const executor = overlayOnScreen(trainingExecutor, selection)?.flight ?? null;
  // what went wrong in it, said in the header in the fewest words (the truth's tab only: it flew the truth's sentence)
  const replayFlown = executor !== null && executor.flown && generated === null ? executor : null;
  const replayIssue = replayFlown === null ? null : replayIssueText(replayFlown);
  const replayOutside = replayFlown === null ? [] : replayOutsideWords(replayFlown, vocabulary, candidates);
  const prior = overlayOnScreen(trainingPrior, selection);
  // the sentence read, as the live executor's source: its answer is drawn only over the sentence it flew
  const readSource = sourceOf(read);
  const autopilot = autopilotOnScreen(trainingAutopilot, selection, readSource);
  const flightFacts = `${flight.typecode ?? "type unknown"} · ${flight.stratum} · ${flight.rows} steps · ` +
    `${verdicts.instructionsAfterStep0} words after step 0 · ${verdicts.silentSteps} of ${flight.rows - 1} later steps silent`;
  // the Fly button: the selected word's segment of the sentence read, and what the live executor is doing with it
  const focusRun = focusColumn === null || !cursorOn ? null : sentenceWordAt(sentence, focusColumn, cursorRow);
  /** A model's word said after its flight ended has no flight to fly; a window's aircraft is not flown live at all (the
   *  backend opens read-back sets only). */
  const flyable = (row: number) => selection.liveExecutor && (generated === null || timeOf(row) < generated.endS);
  const pickedHere = focusRun !== null && trainingPick !== null && sameSource(trainingPick.source, readSource)
    && trainingPick.column === focusColumn && trainingPick.row === focusRun.row;
  const flyingHere = pickedHere && autopilot?.status === "flying";
  const flyLabel = flyingHere ? "Flying …" : pickedHere && autopilot !== null ? "↻ Fly again" : "▶ Fly";
  const chip = model !== null && generated !== null ? sampleChip(model, modelName!, generated, flight, runwayName, stepS) : null;
  const truthEvents = flight.words.events;
  const observedW = generated === null ? 0 : xFor(timeOf(generated.firstRow)) - GUTTER;
  // the rows a model spoke on after its flight ended, to where the executor stopped — only when it said at least one
  // whole step more (every sentence's last step runs on past its end by up to a step)
  const afterEndW = generated === null || timeOf(generated.rows) - generated.endS <= stepS ? 0
    : xFor(timeOf(generated.rows)) - xFor(generated.endS);
  // a model chosen that does not fly this flight: the truth is drawn, and said so
  const notFlown = model !== null && generated === null ? model : null;
  /** The chosen model's other start (the same name, round and run), or — none — the first model there; back to the set's own
   *  start with none, the truth. The sample number is kept. */
  const chooseStart = (augmented: boolean) => {
    const family = onScreen.filter((view) => isAugmentedStart(view.overlay) === augmented);
    const same = model === null ? undefined : family.find((view) => view.overlay.model.name === model.overlay.model.name
      && view.overlay.model.round === model.overlay.model.round && view.overlay.model.run === model.overlay.model.run);
    const next = same ?? (augmented ? trainingModelGroups(family, (view) => view.overlay)[0].members[0] : undefined);
    setTrainingSource(next === undefined ? null : { overlayId: next.overlay.overlayId, sample: trainingSource?.sample ?? 0 });
  };
  /** A model's tab: the round last read in it (the first, never read), from its first sample. */
  const chooseModel = (group: (typeof models)[number]) => {
    const last = group.members.find((view) => view.overlay.overlayId === roundRead.current[familyKey(group.key)]) ?? group.members[0];
    setTrainingSource({ overlayId: last.overlay.overlayId, sample: 0 });
  };
  /** A round's tooltip: the model in full, and how its samples of this flight ended. */
  const roundTitle = (view: TrainingGenerationView) => {
    const landed = view.flight.samples.filter((item) => item.outcome === "landed").length;
    return `${trainingModelText(view.overlay.model)}: ` + (view.flight.flown
      ? `${landed} of ${view.flight.samples.length} of its sentences for this flight landed`
      : `does not fly this flight (${generationUnflownReason(view.flight)})`);
  };

  return (
    <section className="training-sentence-bar" aria-label="Sentence bar" ref={bar}
      style={generated === null ? undefined : { borderColor: modelColour! }}>
      <TrainingLegend layers={trainingLayers} vocabulary={vocabulary} executorTrack={executor?.flown === true}
        autopilotColour={autopilot?.status === "ready" && autopilotHasLine(autopilot.segment) ? autopilotColour(autopilot.segment) : null}
        model={model === null || generated === null ? null
          : { label: modelName!, colour: modelColour!, samples: model.flight.samples.length, moved: movedRead }}
        traffic={traffic === null ? null
          : { colour: trainingWindowReadingColour(windowReading(traffic, trainingSource)), commanded: traffic.window.commanded.length }} />
      <TrainingTrafficStrip />
      <header className="training-sentence-head">
        {startsOffered ? (
          <span className="training-source-tabs training-start-tabs" role="group" aria-label="Where the models' sentences start">
            <button type="button" className="training-source-tab" aria-pressed={!augmentedStart}
              title="The set's own starts: each model speaks from the observed flight's state at its first predicted step"
              onClick={() => chooseStart(false)}>
              Real start
            </button>
            <button type="button" className="training-source-tab" aria-pressed={augmentedStart}
              title={"Augmented starts: each flight's start moved as post-training stage 2 moves one — rotated about the airport, " +
                "raised or lowered, sped up or slowed down — the same move for every model; the truth never flew from there"}
              onClick={() => chooseStart(true)}>
              Augmented start
            </button>
          </span>
        ) : null}
        {models.length > 0 ? (
          <span className="training-source-tabs" role="group" aria-label="Which sentence is read">
            {augmentedStart ? null : (
              <button type="button" className="training-source-tab" aria-pressed={model === null}
                title="The labelled sentence of the observed flight" onClick={() => setTrainingSource(null)}>
                Truth
              </button>
            )}
            {models.map((group) => {
              const colour = trainingModelColour(group);
              const on = group === modelGroup;
              const rounds = group.members.map(group.memberLabel).join(", ");
              return (
                <button key={group.key} type="button" className="training-source-tab" aria-pressed={on}
                  style={on ? { borderColor: colour } : undefined}
                  title={group.members.length === 1 ? `${trainingModelText(group.members[0].overlay.model)}, speaking its own sentences`
                    : `${group.title}: ${group.members.length} rounds published (${rounds}), speaking their own sentences`}
                  onClick={() => chooseModel(group)}>
                  <span className="training-source-dot" style={{ background: colour }} />
                  {group.members.length === 1 ? group.memberLabel(group.members[0]) : group.title}
                </button>
              );
            })}
          </span>
        ) : null}
        {modelGroup !== null && modelGroup.members.length > 1 ? (
          <span className="training-source-tabs training-round-tabs" role="group" aria-label={`${modelGroup.title}'s rounds`}>
            {modelGroup.members.map((view) => (
              <button key={view.overlay.overlayId} type="button" className="training-source-tab"
                aria-pressed={view === model} style={view === model ? { borderColor: modelColour! } : undefined}
                aria-label={modelGroup.memberLabel(view)} title={roundTitle(view)}
                onClick={() => setTrainingSource({ overlayId: view.overlay.overlayId, sample: trainingSource!.sample })}>
                r{view.overlay.model.round}
              </button>
            ))}
          </span>
        ) : null}
        {model !== null && model.flight.flown ? (
          <span className="training-sample-buttons" role="group" aria-label={`${modelName}'s samples`}>
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
          <span className="training-chip training-sample-chip" title={`${modelName}'s sentences fly only the ` +
            "flights on their own aircraft dynamics, as its readout does"}>
            {modelName}: not flown ({generationUnflownReason(notFlown.flight)}) — the truth is shown
          </span>
        ) : null}
        {generated === null ? null : (
          <span className="training-chip training-sample-chip" title={chip!.title}
            style={{ borderColor: modelColour!, color: generated.outcome === "landed" ? undefined : TRAINING_FAILURE_COLOR }}>
            {chip!.text}
          </span>
        )}
        {replayFlown !== null && replayIssue !== null ? (
          <span className="training-chip" style={{ color: TRAINING_REPLAY_COLOR[replayVerdict(replayFlown).kind],
            borderColor: "currentColor" }}
            title={`The executor's replay of this sentence: ${TRAINING_OUTCOME_TAG[replayFlown.outcome]}` +
              (replayOutside.length === 0 ? "" : `; outside their envelopes: ${replayOutside.join(", ")} (a red dot at the band's left)`) +
              ". The full reading is behind ⓘ."}>
            Replay · {replayIssue}
          </span>
        ) : null}
        {autopilot ? <TrainingAutopilotStatus view={autopilot} selection={selection}
          named={focusRun === null || autopilot.request.column !== focusColumn || autopilot.request.row !== focusRun.row} /> : null}
        <span className="training-sentence-cursor-readout">
          t = {formatSeconds(cursorS)} s · {cursorOn ? `step ${cursorRow}` : "outside this aircraft"}
        </span>
        {selection.liveExecutor ? <button type="button" className="training-autopilot-fly" disabled={focusRun === null || flyingHere || !flyable(focusRun.row)}
          title={focusRun === null
            ? "Select a word (click its band): the executor then flies that word's segment from where it was said."
            : !flyable(focusRun.row)
              ? `The model said this word after its flight had ended (${formatSeconds(generated!.endS)} s): there is no flight to fly.`
              : generated === null
                ? `The executor flies ${focusColumn} from step ${focusRun.row} now, on the backend, from the observed state there.`
                : `The executor flies the model's sentence again from its first step (${generated.firstRow})` +
                  (movedRead ? ", from its augmented start," : "") + ` to the end of this ` +
                  `${focusColumn} word's segment, on the backend — the sample's own flight — and judges the word.`}
          onClick={() => setTrainingPick(nextPick(trainingPick, readSource, focusColumn!, focusRun!.row))}>
          {flyLabel}
        </button> : null}
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
            : `${modelName!}'s sentence ${generated.sample + 1} for ${flight.callsign}`}>
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
            <g aria-label={`steps 0–${generated.firstRow - 1}: observed${movedRead ? ", moved like the augmented start" : " only"}, ` +
              `the model speaks from step ${generated.firstRow}`}
              className="training-sentence-observed">
              <title>
                steps 0–{generated.firstRow - 1}: observed{movedRead ? ", moved like the augmented start" : " only"} — the model
                speaks from step {generated.firstRow}
              </title>
              <rect x={GUTTER} y={HEAD_H} width={Math.max(observedW, 0)} height={TRAINING_COLUMNS.length * ROW_H}
                fill={TRAINING_RAW_COLOR} fillOpacity={0.12} />
              {observedW >= OBSERVED_LABEL_MIN_W ? (
                <text x={GUTTER + observedW / 2} y={HEAD_H + (TRAINING_COLUMNS.length * ROW_H) / 2 + 4} textAnchor="middle"
                  className="training-sentence-observed-label">{movedRead ? "observed, moved" : "observed"}</text>
              ) : null}
            </g>
          ) : null}

          {TRAINING_COLUMNS.map((column, position) => {
            const y = HEAD_H + position * ROW_H;
            const colour = TRAINING_COLUMN_COLOR[column];
            const selectedColumn = column === focusColumn;
            // the labeller's checks are the truth's: a model's sentence has none
            const tally = generated === null ? rowTally(verdicts, column) : null;
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
                    : `${column} ${label} — said by ${modelName!} at step ${run.row} ` +
                      `(${formatSeconds(timeOf(run.row))} s), in force to ${formatSeconds(timeOf(run.endRow))} s`;
                  const selected = selectedColumn && run.row <= cursorRow && cursorRow < run.endRow;
                  const choose = () => {
                    if (selected) {
                      setFocusColumn(null);
                      setTrainingPick(null);
                      return;
                    }
                    setFocusColumn(column);
                    setCursorS(timeOf(run.row));
                    if (flyable(run.row)) setTrainingPick(nextPick(trainingPick, readSource, column, run.row));
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
                        strokeOpacity={selected ? 1 : 0.6} strokeWidth={selected ? 2 : 1} className="training-sentence-band-fill" />
                      {/* the issue itself */}
                      <rect x={x} y={y + 2} width={2} height={ROW_H - 4} fill={colour} className="training-sentence-issue" />
                      {/* the executor's verdict on this word */}
                      {mark !== null ? (
                        <circle cx={x + 7} cy={y + ROW_H / 2} r={3.2} fill={mark.fill} stroke={mark.stroke} strokeWidth={1.2}
                          className={`training-sentence-verdict training-sentence-verdict-${verdict!.status.replace(/ /g, "-")}`} />
                      ) : null}
                      {width >= LABEL_MIN_W ? (
                        <text x={x + width / 2} y={y + ROW_H / 2 + 4} textAnchor="middle" className="training-sentence-word" fill={colour}>
                          {trainingBandLabel(vocabulary, candidates, column, run.value)}
                        </text>
                      ) : null}
                    </g>
                  );
                })}
                {tally !== null ? (
                  <text y={y + ROW_H / 2 + 4} className="training-sentence-tally" aria-label={`${column}: ${tally.title}`}
                    style={tally.ok ? undefined : { fill: TRAINING_OUTSIDE_COLOR }}>
                    <title>{tally.title}</title>
                    <tspan x={GUTTER + plotW + TALLY_SLASH} textAnchor={tally.of ? "end" : "middle"}>{tally.held}</tspan>
                    {tally.of ? <tspan x={GUTTER + plotW + TALLY_SLASH} textAnchor="start">{tally.of}</tspan> : null}
                  </text>
                ) : null}
                {/* under a model's row: where the truth says a word of this column */}
                {generated !== null && !movedRead ? truthEvents.filter((event) => event.row > 0 && event.column === position).map((event) => (
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

          {/* a model's sentence: a strip in its colour down the left edge of its rows */}
          {generated !== null ? (
            <rect x={GUTTER - MODEL_STRIP_W - 2} y={HEAD_H} width={MODEL_STRIP_W} height={TRAINING_COLUMNS.length * ROW_H} rx={2}
              fill={modelColour!} className="training-sentence-model-strip">
              <title>{modelName!}'s own sentence, sample {generated.sample + 1}</title>
            </rect>
          ) : null}

          {/* the moments that are not words */}
          {markers.map((marker) => (
            <g key={marker.key} aria-label={marker.title}>
              <title>{marker.title}</title>
              <line x1={xFor(marker.at)} x2={xFor(marker.at)} y1={HEAD_H} y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H}
                stroke={marker.colour} strokeDasharray={marker.dash} strokeWidth={marker.width}
                className="training-sentence-marker" />
            </g>
          ))}

          <line x1={GUTTER} x2={GUTTER + plotW} y1={HEAD_H + TRAINING_COLUMNS.length * ROW_H}
            y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H} className="training-sentence-axis" />
          {endLabelX !== null ? (
            <text x={endLabelX} y={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 15}
              textAnchor={endAnchoredEnd ? "end" : "middle"}
              className="training-sentence-tick training-sentence-model-end" style={{ fill: endColour! }}
              aria-label={`${modelName!}'s flight ended at ${formatSeconds(generated!.endS)} s`}>
              {formatSeconds(generated!.endS)} s
            </text>
          ) : null}
          {tickRows.map((row, index) => (tickShown[index] ? (
            <text key={`tick-${row}`} x={xFor(timeOf(row))} y={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 15}
              textAnchor={index === tickRows.length - 1 ? "end" : "middle"} className="training-sentence-tick">
              {index === tickRows.length - 1 ? `${formatSeconds(timeOf(row))} s` : formatSeconds(timeOf(row))}
            </text>
          ) : null))}
          {/* the live executor on its word's row, in the flight's time (none for a segment with no line: 3D flies nothing) */}
          {autopilot !== null && (autopilot.status === "flying"
            || (autopilot.status === "ready" && autopilotHasLine(autopilot.segment))) ? (
            <AutopilotCursor view={autopilot} stepS={stepS} x0={GUTTER} plotW={plotW} endS={endS}
              y={HEAD_H + TRAINING_COLUMN_INDEX[autopilot.request.column] * ROW_H} />
          ) : null}
          {/* the cursor, kept on the axis (it may have been put past the truth's end while a longer sentence was read) — on a
              window's clock, none while the window's time is outside this aircraft's */}
          {cursorOn ? (
            <line x1={xFor(Math.min(cursorS, endS))} x2={xFor(Math.min(cursorS, endS))} y1={HEAD_H - 8}
              y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 4} className="training-sentence-cursor" />
          ) : null}
        </svg>
      </div>

      {notesOpen ? (
        <footer className="training-sentence-legend">
          <span>{flight.callsign}: {flightFacts}</span>
          {generated === null ? (
            <span>
              At a row's end: the labeller's own checks of this flight's envelopes (Reading.checks) — heading words inside
              their bands, altitude tubes held, speed words held, of those judged (red when one is not); the capture turn's
              mark on the approach row.
            </span>
          ) : null}
          {executor !== null && generated === null ? <span>{replayText(executor, replayOutside)}</span> : null}
          {chip !== null ? <span>{chip.title}</span> : null}
          <span>
            A band is a word in force, from the tick where it was issued to the next word of its column; step 0 gives all
            six. Click a band to select its word — here, in the read-back check and in 3D — and again to clear it; ▶ Fly
            flies the selected word's segment live (a band click does too): a short bar on that word's row says where the
            executor is — pulsing at the word's step while the backend flies it, then moving with the 3D aircraft, faded
            past where it heard the next word of the column, left at the segment's end.
            The dashed lines: the clearance, the capture of the final, the speed left to the pilot.
          </span>
          {models.length > 0 ? (
            <span>
              The tabs choose the sentence read: the truth, or a model's own — one of its samples, numbered (✗: its flight did
              not land). Its words are drawn like the truth's; the frame in the model's colour — the bar's border, the strip
              down the rows — says whose they are; the grey before its first step is what it only observed; a white tick under
              a row is where the truth says a word of that column; the solid line in the model's colour is where its flight
              landed (a heavier red one: where it did not), its time written under it on the axis, the dashed white one where the observed flight's sentence ends; its
              words fly live like the truth's: the executor flies its sentence again from its first step — the sample's own
              flight — except a word said after its flight ended; words under the
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
          cursorS={cursorS} cursorOn={cursorOn} onCursorChange={setCursorS} column={focusColumn} onColumnChange={setFocusColumn}
          onClose={() => setOpenWindow(null)} executor={executor}
          autopilot={autopilot?.status === "ready" ? autopilot.segment : null} />
      ) : null}
      {openWindow === "prior" && prior && generated === null ? (
        <TrainingPriorWindow flight={flight} vocabulary={vocabulary} candidates={candidates} prior={prior} cursorS={cursorS}
          cursorOn={cursorOn} onCursorChange={setCursorS} column={focusColumn} onColumnChange={setFocusColumn} onClose={() => setOpenWindow(null)} />
      ) : null}
    </section>
  );
}
