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
 * THE HEADER IS SHORT: the callsign (its type, stratum and counts in its tooltip), the runway, one chip of the labeller's
 * own verdicts, one of the executor's replay when it is on, the live executor's line, the cursor, and the buttons —
 * Fly, Read-back, Prior, and ⓘ for the notes. Every chip carries its full reading in its tooltip.
 *
 * The cursor is in flight time and is moved by clicking a band or a step number: what it reports is the artefact's own
 * step, never a rounded pixel. It does NOT drive `viewer.clock`: Training loads no CZML, and the clock belongs to
 * Observe's playback.
 *
 * ONE COLUMN IS HIGHLIGHTED, NEVER A STEP. Clicking a band selects its word class (`trainingColumn`) and puts the cursor
 * at its issue; every view then highlights that column's word in force at the cursor, and only it. Clicking the selected
 * band again clears it.
 *
 * THE OVERLAYS, when the panel publishes them: the EXECUTOR's verdict on each word as a dot at the band's left (teal
 * inside its envelope, red outside, hollow grey not judged, not reached or superseded; none for a word with no check of
 * its own); the PRIOR's window behind its button (`TrainingPriorWindow`). One window is open at a time.
 *
 * THE EXECUTOR, LIVE (`trainingAutopilot`): the Fly button PICKS the selected word for the live executor (`trainingPick`;
 * "↻ Fly again" once it has flown), and so does clicking a band while the panel's switch is on (`trainingAutopilotAuto`;
 * clicking the selected band again clears the pick) — never the cursor. Its line (`TrainingAutopilotStatus`) names the
 * word it flew, whether it stayed inside its envelope and the two times.
 */

import { useState } from "react";
import { useApp } from "../context/AppContext";
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
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import { autopilotColour, autopilotHasLine, autopilotOnScreen, nextPick } from "../data/trainingAutopilot";
import {
  executorWordAt,
  executorWordCounts,
  overlayOnScreen,
  type TrainingExecutorFlight,
  type TrainingExecutorWord,
} from "../data/trainingOverlays";
import {
  formatSeconds,
  rowAtTime,
  trainingColumnRuns,
  trainingKindLabel,
  trainingVerdicts,
  trainingWordAt,
  trainingWordLabel,
  TRAINING_COLUMNS,
  type TrainingFlight,
} from "../data/trainingSample";
import { checkMark, checkText, crossingText, TRAINING_COLUMN_LABEL, TRAINING_OUTCOME_TAG } from "../data/trainingText";

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

export default function TrainingSentenceBar() {
  const {
    trainingSelection: selection, trainingLayers,
    trainingCursorS: cursorS, setTrainingCursorS: setCursorS,
    trainingColumn: focusColumn, setTrainingColumn: setFocusColumn,
    trainingExecutor, trainingPrior, trainingAutopilot, trainingPick, setTrainingPick, trainingAutopilotAuto,
  } = useApp();
  const [frame, frameW] = useMeasuredWidth(MIN_PLOT_W + GUTTER + PAD_R, DEFAULT_PLOT_W + GUTTER + PAD_R);
  const [openWindow, setOpenWindow] = useState<"readback" | "prior" | null>(null);
  const [notesOpen, setNotesOpen] = useState<boolean>(false);

  if (!selection) return null;
  const { flight, vocabulary, candidates } = selection;
  const plotW = frameW - GUTTER - PAD_R;
  const { tS } = flight.signals;
  // Step r covers [r·step, (r+1)·step): the axis ends where the last step does.
  const endS = flight.rows * vocabulary.stepS;
  const timeOf = (row: number) => row * vocabulary.stepS;
  const xFor = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  const verdicts = trainingVerdicts(flight);
  const cursorRow = rowAtTime(tS, cursorS);
  const toggle = (name: "readback" | "prior") => setOpenWindow((open) => (open === name ? null : name));

  // The steps that SAY something, numbered; step 0 is the opening and is always complete.
  const issueRows = [...new Set(flight.words.events.map((event) => event.row))].sort((a, b) => a - b);
  const stepShown = spacedLabels(issueRows.map((row) => xFor(timeOf(row))), STEP_LABEL_GAP);
  const tickRows = [...issueRows, flight.rows];
  const tickShown = spacedLabels(tickRows.map((row) => xFor(timeOf(row))), TICK_LABEL_GAP, true);
  const markers = [
    { row: flight.joinRow, key: "cleared", colour: TRAINING_COLUMN_COLOR.approach,
      title: `cleared to join the final at ${formatSeconds(timeOf(flight.joinRow))} s` },
    { row: flight.captureRow, key: "captured", colour: TRAINING_CORRIDOR_COLOR,
      title: `the final captured at ${formatSeconds(timeOf(flight.captureRow))} s, ` +
        `${(flight.captureBeforeThresholdM / 1000).toFixed(1)} km before the threshold` },
    { row: flight.unspecifiedRow, key: "unspecified", colour: TRAINING_COLUMN_COLOR.speed,
      title: `speed left to the pilot from ${formatSeconds(timeOf(flight.unspecifiedRow))} s` },
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
  // the Fly button: the selected word's segment, and what the live executor is doing with it
  const focusRun = focusColumn === null ? null : trainingWordAt(flight, focusColumn, cursorRow);
  const pickedHere = focusRun !== null && trainingPick !== null && trainingPick.column === focusColumn
    && trainingPick.row === focusRun.row;
  const flyingHere = pickedHere && autopilot?.status === "flying";
  const flyLabel = flyingHere ? "Flying …" : pickedHere && autopilot !== null ? "↻ Fly again" : "▶ Fly";

  return (
    <section className="training-sentence-bar" aria-label="Sentence bar">
      <TrainingLegend layers={trainingLayers} vocabulary={vocabulary} executorTrack={executor?.flown === true}
        autopilotColour={autopilot?.status === "ready" && autopilotHasLine(autopilot.segment) ? autopilotColour(autopilot.segment) : null} />
      <header className="training-sentence-head">
        <strong title={flightFacts}>{flight.callsign}</strong>
        <span className="training-chip">runway {flight.runway}</span>
        <span className="training-chip" title={verdictsChip.title}>{verdictsChip.text}</span>
        {replay !== null ? (
          <span className="training-chip training-sentence-executor" title={replay.title}
            style={{ color: replay.ok ? TRAINING_EXECUTOR_COLOR : executor!.flown ? TRAINING_OUTSIDE_COLOR : TRAINING_RAW_COLOR }}>
            {replay.text}
          </span>
        ) : null}
        {autopilot ? <TrainingAutopilotStatus view={autopilot} selection={selection} /> : null}
        <span className="training-sentence-cursor-readout">t = {formatSeconds(cursorS)} s · step {cursorRow}</span>
        <button type="button" className="training-autopilot-fly" disabled={focusRun === null || flyingHere}
          title={focusRun === null
            ? "Select a word (click its band): the executor then flies that word's segment from where it was said."
            : `The executor flies ${focusColumn} from step ${focusRun.row} now, on the backend, from the observed state there.`}
          onClick={() => setTrainingPick(nextPick(trainingPick, focusColumn!, focusRun!.row))}>
          {flyLabel}
        </button>
        <button type="button" className="training-sentence-readback-button" aria-pressed={openWindow === "readback"}
          title="The sentence against its track, envelope by envelope" onClick={() => toggle("readback")}>
          Read-back
        </button>
        {prior ? (
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
          aria-label={`The sentence of ${flight.callsign} on runway ${flight.runway}`}>
          {/* the steps that say something: numbers along the top, each a button */}
          {issueRows.map((row, index) => {
            const left = index === 0 ? GUTTER : xFor((timeOf(issueRows[index - 1]) + timeOf(row)) / 2);
            const right = index === issueRows.length - 1 ? GUTTER + plotW : xFor((timeOf(row) + timeOf(issueRows[index + 1])) / 2);
            const words = flight.words.events.filter((event) => event.row === row);
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
                {trainingColumnRuns(flight, column).map((run) => {
                  const x = xFor(timeOf(run.row));
                  const width = xFor(timeOf(run.endRow)) - x;
                  const label = trainingWordLabel(vocabulary, candidates, column, run.value);
                  const verdict = executor === null ? null : executorWordAt(executor, run.row, column);
                  const mark = verdict === null ? null : verdictMark(verdict.status);
                  const title = `${column} ${label} — ${trainingKindLabel(run.event.kind)}, issued at step ${run.row} ` +
                    `(${formatSeconds(timeOf(run.row))} s), in force to ${formatSeconds(timeOf(run.endRow))} s` +
                    (verdict === null ? "" : `\n${executorVerdictText(verdict)}`);
                  const selected = selectedColumn && run.row <= cursorRow && cursorRow < run.endRow;
                  const choose = () => {
                    if (selected) {
                      setFocusColumn(null);
                      setTrainingPick(null);
                      return;
                    }
                    setFocusColumn(column);
                    setCursorS(timeOf(run.row));
                    if (trainingAutopilotAuto) setTrainingPick(nextPick(trainingPick, column, run.row));
                  };
                  return (
                    <g key={`${column}-${run.row}`} role="button" tabIndex={0} aria-label={title} aria-pressed={selected}
                      className="training-sentence-band" onClick={choose}
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
                          {label}
                        </text>
                      ) : null}
                    </g>
                  );
                })}
              </g>
            );
          })}

          {/* the moments that are not words */}
          {markers.map((marker) => (
            <g key={marker.key} aria-label={marker.title}>
              <title>{marker.title}</title>
              <line x1={xFor(timeOf(marker.row))} x2={xFor(timeOf(marker.row))} y1={HEAD_H} y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H}
                stroke={marker.colour} strokeDasharray="3 3" className="training-sentence-marker" />
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
          <line x1={xFor(cursorS)} x2={xFor(cursorS)} y1={HEAD_H - 8} y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 4}
            className="training-sentence-cursor" />
        </svg>
      </div>

      {notesOpen ? (
        <footer className="training-sentence-legend">
          <span>{flight.callsign}: {flightFacts} · {verdictsChip.title}</span>
          {replay !== null ? <span>{replay.title}</span> : null}
          <span>
            A band is a word in force, from the tick where it was issued to the next word of its column; step 0 gives all
            six. Click a band to select its word — here, in the read-back check and in 3D — and again to clear it; ▶ Fly
            flies the selected word's segment live (a band click does too, with the panel's "Fly on band click" on). The
            dashed lines: the clearance, the capture of the final, the speed left to the pilot.
          </span>
          <span>
            The sentence ends {(landing.lastRowBeforeThresholdM / 1000).toFixed(2)} km before the threshold
            {landing.cutAtCrossing ? ", cut before the last passage of the threshold" : ", where the data ends"}.
          </span>
          {executor !== null ? (
            <span>
              The executor's verdict on a word, at its band's left: <b style={{ color: TRAINING_EXECUTOR_COLOR }}>●</b> inside,{" "}
              <b style={{ color: TRAINING_OUTSIDE_COLOR }}>●</b> outside, ○ not judged, not reached or superseded; none for a
              word with no check of its own — judged on envelopes re-drawn from where the executor was told the word.
            </span>
          ) : null}
        </footer>
      ) : null}

      {openWindow === "readback" ? (
        <TrainingReadbackWindow flight={flight} vocabulary={vocabulary} candidates={candidates} layers={trainingLayers}
          cursorS={cursorS} onCursorChange={setCursorS} column={focusColumn} onColumnChange={setFocusColumn}
          onClose={() => setOpenWindow(null)} executor={executor}
          autopilot={autopilot?.status === "ready" ? autopilot.segment : null} />
      ) : null}
      {openWindow === "prior" && prior ? (
        <TrainingPriorWindow flight={flight} vocabulary={vocabulary} candidates={candidates} prior={prior} cursorS={cursorS}
          onCursorChange={setCursorS} column={focusColumn} onColumnChange={setFocusColumn} onClose={() => setOpenWindow(null)} />
      ) : null}
    </section>
  );
}
