/**
 * TrainingSentenceBar.tsx
 * -----------------------
 * The sentence as a picture: the five columns as five rows — runway (with go-around), heading (relative to the course of
 * the runway in force), altitude (a level above the airport elevation E, or "no level-off"), angle, speed, in the
 * vocabulary's order — against the flight's own time. Design: `4dTrajectory/ts_transformer/docs/two_tier/design/vocabulary.md`
 * §12.1 A23; the files: `data/trainingSample.ts`.
 *
 * A BAND IS A WORD IN FORCE, from the row it was said at to the row the next word of its column replaces it. Row 0 says
 * every column, so every row opens with a band; after that a column is "unchanged" until its next word. The tick at a
 * band's left edge is the word being said; the numbers along the top are the rows that say anything.
 *
 * WHICH SENTENCE (`trainingIntervalS`): the tabs at the head of the bar — "Labelled", the labeller's open-loop reading of
 * the observed flight on the 2 s rows, then one per row interval Δ of the set: the CLOSED-LOOP sentence at Δ, the words
 * the closed-loop reading says to the executor from the first predicted step on, Δ apart. The closed-loop bar opens at
 * the first predicted step (the rows before it are shaded: observed only); a word the reading ADDED is a CORRECTION and is
 * marked three ways — an orange, dashed band with an orange tick, never colour alone. The flown path of that sentence is
 * drawn beside the observed track in the charts and in 3D; the bar says how the flown flight ended (a chip: the judge's
 * outcome) and its decision-altitude check (a chip: passed or failed, the values in its tooltip) and marks the end and the
 * DA point on the axis.
 *
 * THE HEADER IS SHORT: the tabs, the callsign (its type, stratum and counts in its tooltip), the runway, the outcome and
 * DA chips, the corrections' count, the live executor's line, the cursor, and the buttons — Fly, Read-back and ⓘ for the
 * notes. Every chip carries its full reading in its tooltip. The judge's verdicts are TALLIED AT EACH ROW'S END
 * (`rowTally`), lined up on the slash, and a word with an envelope carries its verdict as a dot at its band's left.
 *
 * The cursor is in flight time and is moved by clicking a band or a step number: what it reports is the artefact's own
 * row, never a rounded pixel. It does NOT drive `viewer.clock`: Training loads no CZML, and the clock belongs to
 * Evaluation's playback.
 *
 * ONE COLUMN IS HIGHLIGHTED, NEVER A STEP. Clicking a band selects its word class (`trainingColumn`) and puts the cursor
 * at its row; every view then highlights that column's word in force at the cursor, and only it. Clicking the selected
 * band again clears it.
 *
 * THE EXECUTOR, LIVE (`trainingAutopilot`): in a closed-loop sentence the Fly button PICKS the selected word for the live
 * executor (`trainingPick`, "↻ Fly again" once it has flown), and so does clicking a band (clicking the selected band
 * again clears the pick) — never the cursor. Its line (`TrainingAutopilotStatus`) says how the flight went and the two
 * times. On the flown word's row a small CURSOR says where the executor is (`AutopilotCursor`): at the word's row, pulsing,
 * while the backend flies it; then with the 3D aircraft as it flies the segment out — the same clock,
 * `autopilotPlaybackS`; left at the segment's end, as the aircraft is.
 *
 * THE BAR MEASURES ITSELF: its height is published on the page's root as `--training-bar-height`, so the docks — and
 * Cesium's credits — end above it (`index.css`) instead of under it; the bar itself sits at the bottom edge (Training
 * hides Cesium's clock dial and timeline).
 */

import { useLayoutEffect, useRef, useState } from "react";
import { useApp, useTrainingCursor } from "../context/AppContext";
import TrainingLegend from "./TrainingLegend";
import TrainingReadbackWindow from "./TrainingReadbackWindow";
import { TrainingAutopilotStatus } from "./TrainingAutopilotStatus";
import useMeasuredWidth from "../hooks/useMeasuredWidth";
import {
  TRAINING_AUTOPILOT_COLOR,
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRECTION_COLOR,
  TRAINING_DECISION_FAIL_COLOR,
  TRAINING_DECISION_PASS_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_WORD_COLOR,
  trainingOutcomeColour,
} from "../utils/trainingWordColors";
import {
  AUTOPILOT_PLAYBACK_MIN_SPEEDUP,
  autopilotColour,
  autopilotHasLine,
  autopilotOnScreen,
  autopilotPlaybackS,
  nextPick,
  type TrainingAutopilotView,
} from "../data/trainingAutopilot";
import {
  correctionCount,
  formatSeconds,
  readingRowAt,
  readingRowTimeS,
  sentenceColumnRuns,
  sentenceWordAt,
  trainingBandLabel,
  trainingEnvelopeIndex,
  trainingReadingOf,
  trainingWordLabel,
  TRAINING_COLUMN_INDEX,
  TRAINING_COLUMNS,
  type TrainingColumn,
  type TrainingEnvelopes,
  type TrainingReading,
  type TrainingWordRun,
} from "../data/trainingSample";
import {
  checkMark, decisionText, replayText, TRAINING_COLUMN_LABEL, TRAINING_COLUMN_MEANING, TRAINING_OUTCOME_TAG,
} from "../data/trainingText";

// One SVG unit is one pixel: the bar is as wide as the dock and always VIEW_H tall.
const GUTTER = 96;
/** The right margin: each row's tally of the judge's verdicts. */
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

/** A word's envelope verdict as the dot at its band's left reads it: held (true), not held (false), or none (null). */
function wordHeld(envelopes: TrainingEnvelopes | null, column: TrainingColumn, index: number | null): boolean | null {
  if (envelopes === null || index === null) return null;
  switch (column) {
    case "heading": return envelopes.heading[index].inside.every(Boolean);
    case "altitude": return envelopes.altitude[index].contained;
    case "speed": return envelopes.speed[index].contained;
    default: return null;
  }
}

/** A row's tally of the judge's checks, written at its end: the envelopes held of those judged — split at the slash, which
 *  every row lines up on — with its reading, for the tooltip. null for a row with no envelope (runway, angle). */
function rowTally(envelopes: TrainingEnvelopes | null, column: TrainingColumn, flown: boolean):
  { held: string; of: string; ok: boolean; title: string } | null {
  if (envelopes === null) return null;
  const count = (held: number, of: number, what: string) => ({
    held: `${held}`, of: `/${of}`, ok: held === of, title: `${held} of ${of} ${what}${flown ? " (on the flown path)" : ""}`,
  });
  switch (column) {
    case "heading": {
      const judged = envelopes.heading.filter((band) => band.stopRow > band.firstRow);
      return count(judged.filter((band) => band.inside.every(Boolean)).length, judged.length, "heading words inside their bands");
    }
    case "altitude":
      return count(envelopes.altitude.filter((tube) => tube.contained).length, envelopes.altitude.length, "altitude tubes held");
    case "speed":
      return count(envelopes.speed.filter((span) => span.contained).length, envelopes.speed.length, "speed words held");
    default:
      return null;
  }
}

/** The live executor on the flown word's row, in flight time: at the word's row, pulsing, while the backend flies it; then
 *  where the 3D aircraft is as it flies the segment out (`autopilotPlaybackS`: the same clock); left at the segment's end.
 *  A leaf that moves itself each frame, so the bar never re-renders for it. */
function AutopilotCursor({ view, reading, y, x0, plotW, endS }: {
  view: Extract<TrainingAutopilotView, { status: "flying" | "ready" }>;
  reading: TrainingReading;
  /** The top of the flown word's row. */
  y: number;
  /** The bar's time axis: where it starts, how wide it is (px) and the time it ends at (s). */
  x0: number;
  plotW: number;
  endS: number;
}) {
  const node = useRef<SVGRectElement>(null);
  // placed before paint (never a frame at the axis's origin), and again whenever the axis is laid out anew
  useLayoutEffect(() => {
    const rect = node.current!;
    const put = (seconds: number) => rect.setAttribute("transform", `translate(${x0 + (Math.min(seconds, endS) / endS) * plotW} 0)`);
    if (view.status === "flying") {
      put(readingRowTimeS(reading, view.request.row));
      return undefined;
    }
    const { track } = view.segment;
    const totalS = track.tS[track.tS.length - 1] - track.tS[0];
    let frame = 0;
    const draw = () => {
      const flownS = autopilotPlaybackS(track, view.playedAt, Date.now());
      put(track.tS[0] + flownS);
      if (flownS < totalS) frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [view, reading, x0, plotW, endS]);
  const colour = view.status === "flying" ? TRAINING_AUTOPILOT_COLOR : autopilotColour(view.segment);
  return (
    <rect ref={node} x={-1.5} y={y + 1} width={3} height={ROW_H - 2} rx={1.5} fill={colour}
      className={`training-sentence-autopilot-cursor${view.status === "flying" ? " waiting" : ""}`}>
      <title>{view.status === "flying" ? "the executor flies this word on the backend"
        : `the autopilot's aircraft, flying this word's segment out in 3D (×${AUTOPILOT_PLAYBACK_MIN_SPEEDUP} or faster)`}</title>
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
    mode, trainingSelection: selection, trainingIntervalS: intervalS, setTrainingIntervalS, trainingLayers,
    trainingColumn: focusColumn, setTrainingColumn: setFocusColumn, trainingAutopilot, trainingPick, setTrainingPick,
  } = useApp();
  const { trainingCursorS: cursorS, setTrainingCursorS: setCursorS } = useTrainingCursor();
  const [frame, frameW] = useMeasuredWidth(MIN_PLOT_W + GUTTER + PAD_R, DEFAULT_PLOT_W + GUTTER + PAD_R);
  const bar = useBarHeight();
  const [readbackOpen, setReadbackOpen] = useState<boolean>(false);
  const [notesOpen, setNotesOpen] = useState<boolean>(false);

  // the Training session outlives a task switch (the panel stays mounted): the bar draws only in Training
  if (!selection || mode !== "training") return null;
  const { flight, vocabulary } = selection;
  const stepS = vocabulary.stepS;
  const plotW = frameW - GUTTER - PAD_R;
  const reading = trainingReadingOf(flight, stepS, intervalS);
  const { closed } = reading;
  // the axis is the flight's own time from 0 to where the sentence (or the flown track) ends
  const endS = Math.max(reading.endS, reading.judged.tS[reading.judged.tS.length - 1]);
  const xFor = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  const timeOf = (row: number) => readingRowTimeS(reading, row);
  const cursorRow = readingRowAt(reading, cursorS);
  const corrections = correctionCount(reading.events);
  const replay = closed === null ? null : closed.replay;
  const decision = replay?.crossing?.decision ?? null;

  // The rows that SAY something, numbered; the sentence's first row is its opening and is always complete.
  const issueRows = [...new Set(reading.events.map((event) => event.row))].sort((a, b) => a - b);
  const stepShown = spacedLabels(issueRows.map((row) => xFor(timeOf(row))), STEP_LABEL_GAP);
  const tickTimes = [...issueRows.map(timeOf), endS];
  const tickShown = spacedLabels(tickTimes.map(xFor), TICK_LABEL_GAP, true);

  // the moments that are not words
  const markers = reading.loop === "open" ? [
    { at: timeOf(reading.open.captureRow), key: "captured", colour: TRAINING_COLUMN_COLOR.heading, dash: "3 3", width: 1,
      title: `the final captured at ${formatSeconds(timeOf(reading.open.captureRow))} s` },
    { at: timeOf(reading.open.unspecifiedRow), key: "unspecified", colour: TRAINING_COLUMN_COLOR.speed, dash: "3 3", width: 1,
      title: `speed left to the pilot from ${formatSeconds(timeOf(reading.open.unspecifiedRow))} s` },
    ...reading.open.goAroundRows.map((row) => ({ at: timeOf(row), key: `go-around-${row}`, colour: TRAINING_COLUMN_COLOR.runway,
      dash: "3 3", width: 1.5, title: `go-around said at ${formatSeconds(timeOf(row))} s` })),
  ] : [
    ...(decision === null ? [] : [{ at: closed!.startS + decision.cycle, key: "decision", width: 1.5, dash: "2 3",
      colour: decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR, title: decisionText(decision) }]),
    { at: closed!.startS + replay!.endCycle, key: "flown-end", colour: trainingOutcomeColour(replay!.outcome), dash: undefined,
      width: 2, title: `the flown flight ended at ${formatSeconds(closed!.startS + replay!.endCycle)} s: ${replayText(replay!)}` },
  ];

  // the live executor, when it is of the flight and the Δ on screen
  const autopilot = autopilotOnScreen(trainingAutopilot, selection, intervalS);
  const focusRun: TrainingWordRun | null = focusColumn === null ? null : sentenceWordAt(reading, focusColumn, cursorRow);
  const pickedHere = focusRun !== null && trainingPick !== null && trainingPick.column === focusColumn && trainingPick.row === focusRun.row;
  const flyingHere = pickedHere && autopilot?.status === "flying";
  const flyLabel = flyingHere ? "Flying …" : pickedHere && autopilot !== null ? "↻ Fly again" : "▶ Fly";
  const flightFacts = `${flight.typecode ?? "type unknown"} · ${flight.stratum} · ${flight.kind} · ${flight.group} · ` +
    `${reading.events.length} words in ${reading.rows} rows` + (corrections === 0 ? "" : ` (${corrections} added by the closed-loop reading)`);
  const observedW = closed === null ? 0 : xFor(closed.startS) - GUTTER;
  const intervals = vocabulary.rowIntervalsS;

  return (
    <section className="training-sentence-bar" aria-label="Sentence bar" ref={bar}>
      <TrainingLegend layers={trainingLayers} vocabulary={vocabulary} closed={closed !== null} corrections={corrections > 0}
        autopilotColour={autopilot?.status === "ready" && autopilotHasLine(autopilot.segment) ? autopilotColour(autopilot.segment) : null} />
      <header className="training-sentence-head">
        <span className="training-source-tabs" role="group" aria-label="Which sentence is read">
          <button type="button" className="training-source-tab" aria-pressed={intervalS === null}
            title="The labeller's reading of the observed flight (open loop): its words on the 2 s rows from the first row, and the envelopes of its words on the observed track"
            onClick={() => setTrainingIntervalS(null)}>
            Labelled
          </button>
          {intervals.map((interval) => (
            <button key={interval} type="button" className="training-source-tab" aria-pressed={intervalS === interval}
              title={`The closed-loop sentence at Δ = ${interval} s: the words the closed-loop reading says to the executor from the first ` +
                `predicted step on, ${interval} s apart, the ones it added marked; the executor's flown path is drawn beside the observed track`}
              onClick={() => setTrainingIntervalS(interval)}>
              Δ {interval} s
            </button>
          ))}
        </span>
        <strong title={flightFacts}>{flight.callsign}</strong>
        <span className="training-chip">runway {flight.runway}</span>
        {replay !== null ? (
          <span className="training-chip" style={{ color: trainingOutcomeColour(replay.outcome), borderColor: "currentColor" }}
            title={`The flown flight: ${replayText(replay)}`}>
            {TRAINING_OUTCOME_TAG[replay.outcome]}
          </span>
        ) : null}
        {decision !== null ? (
          <span className="training-chip" title={decisionText(decision)}
            style={{ color: decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR, borderColor: "currentColor" }}>
            DA {decision.passed ? "passed" : "failed"} {checkMark(decision.passed)}
          </span>
        ) : null}
        {closed !== null ? (
          <span className="training-chip" style={{ color: corrections === 0 ? undefined : TRAINING_CORRECTION_COLOR }}
            title={`${corrections} of the ${reading.events.length} words were added by the closed-loop reading (corrections, drawn dashed and orange)`}>
            {corrections} correction{corrections === 1 ? "" : "s"}
          </span>
        ) : null}
        {autopilot ? <TrainingAutopilotStatus view={autopilot} selection={selection}
          named={focusRun === null || autopilot.request.column !== focusColumn || autopilot.request.row !== focusRun.row} /> : null}
        <span className="training-sentence-cursor-readout">
          t = {formatSeconds(cursorS)} s · {cursorRow === null ? "before the sentence" : `row ${cursorRow}`}
        </span>
        <button type="button" className="training-autopilot-fly" disabled={closed === null || focusRun === null || flyingHere}
          title={closed === null
            ? "The live executor flies closed-loop sentences: choose a Δ."
            : focusRun === null
              ? "Select a word (click its band): the executor then flies that word's segment of the closed-loop sentence."
              : `The executor flies the closed-loop sentence again from its first predicted step to the end of this ${focusColumn} word's ` +
                "segment, on the backend, and the segment is drawn here and in 3D."}
          onClick={() => setTrainingPick(nextPick(trainingPick, intervalS!, focusColumn!, focusRun!.row))}>
          {flyLabel}
        </button>
        <button type="button" className="training-sentence-readback-button" aria-pressed={readbackOpen}
          title="The sentence against its track, envelope by envelope" onClick={() => setReadbackOpen((open) => !open)}>
          Read-back
        </button>
        <button type="button" className="training-sentence-notes-toggle" aria-expanded={notesOpen}
          aria-label={notesOpen ? "Hide the notes" : "How to read the bar"} onClick={() => setNotesOpen((open) => !open)}>
          ⓘ
        </button>
      </header>

      <div className="training-sentence-frame" ref={frame}>
        <svg className="training-sentence-svg" width={GUTTER + plotW + PAD_R} height={VIEW_H}
          viewBox={`0 0 ${GUTTER + plotW + PAD_R} ${VIEW_H}`} role="group"
          aria-label={reading.loop === "open" ? `The labelled sentence of ${flight.callsign} on runway ${flight.runway}`
            : `The closed-loop sentence of ${flight.callsign} at Δ = ${reading.intervalS} s`}>
          {/* the rows that say something: numbers along the top, each a button */}
          {issueRows.map((row, index) => {
            const left = index === 0 ? GUTTER : xFor((timeOf(issueRows[index - 1]) + timeOf(row)) / 2);
            const right = index === issueRows.length - 1 ? GUTTER + plotW : xFor((timeOf(row) + timeOf(issueRows[index + 1])) / 2);
            const words = reading.events.filter((event) => event.row === row);
            const name = `Row ${row} at ${formatSeconds(timeOf(row))} s: ` + words.map((event) =>
              `${TRAINING_COLUMNS[event.column]} ${trainingBandLabel(event.says)}${event.correction ? " (correction)" : ""}`).join(", ");
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

          {/* a closed-loop sentence: the rows before the first predicted step, which the observed flight only supplies */}
          {closed !== null ? (
            <g aria-label={`0–${formatSeconds(closed.startS)} s: observed only, the sentence opens at the first predicted step`}
              className="training-sentence-observed">
              <title>0–{formatSeconds(closed.startS)} s: observed only — the closed-loop sentence opens at the first predicted step</title>
              <rect x={GUTTER} y={HEAD_H} width={Math.max(observedW, 0)} height={TRAINING_COLUMNS.length * ROW_H}
                fill={TRAINING_RAW_COLOR} fillOpacity={0.12} />
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
            const tally = rowTally(reading.envelopes, column, closed !== null);
            return (
              <g key={column} aria-label={`${column} row`}>
                <rect x={GUTTER} y={y} width={plotW} height={ROW_H} className={`training-sentence-row-bg${position % 2 ? " odd" : ""}`} />
                <g>
                  <title>{TRAINING_COLUMN_LABEL[column]}: {TRAINING_COLUMN_MEANING[column]}</title>
                  <text x={GUTTER - 8} y={y + ROW_H / 2 + 4} textAnchor="end" className="training-sentence-row-label"
                    style={selectedColumn ? { fill: TRAINING_WORD_COLOR, fontWeight: 600 } : undefined}>
                    {TRAINING_COLUMN_LABEL[column]}
                  </text>
                </g>
                {sentenceColumnRuns(reading, column).map((run) => {
                  const x = xFor(timeOf(run.row));
                  const width = xFor(timeOf(run.endRow)) - x;
                  const { event } = run;
                  const label = trainingBandLabel(event.says);
                  const envelopeIndex = trainingEnvelopeIndex(reading, stepS, column, run);
                  const held = wordHeld(reading.envelopes, column, envelopeIndex);
                  const kind = reading.loop === "open"
                    ? reading.open.kinds.find((item) => item.row === event.row && item.column === event.column)?.kind : undefined;
                  const title = `${column} ${trainingWordLabel(event.says)}` +
                    (event.correction ? " — a CORRECTION: added by the closed-loop reading" : "") +
                    (kind === undefined ? "" : ` — labelled as ${kind}`) +
                    `, said at row ${run.row} (${formatSeconds(timeOf(run.row))} s), in force to ${formatSeconds(timeOf(run.endRow))} s` +
                    (held === null ? "" : `\nthe judge: ${held ? "held" : "not held"} ${checkMark(held)}`) +
                    (closed !== null && column !== "runway" && closed.lateralM[run.row] !== null
                      ? `\nflown path vs observed when said: ${Math.abs(closed.lateralM[run.row]!).toFixed(1)} m lateral` +
                        (closed.verticalM[run.row] === null ? "" : `, ${closed.verticalM[run.row]!.toFixed(1)} m vertical`) : "");
                  const selected = selectedColumn && cursorRow !== null && run.row <= cursorRow && cursorRow < run.endRow;
                  const choose = () => {
                    if (selected) {
                      setFocusColumn(null);
                      setTrainingPick(null);
                      return;
                    }
                    setFocusColumn(column);
                    setCursorS(timeOf(run.row));
                    if (intervalS !== null) setTrainingPick(nextPick(trainingPick, intervalS, column, run.row));
                  };
                  const stroke = selected ? TRAINING_WORD_COLOR : event.correction ? TRAINING_CORRECTION_COLOR : colour;
                  return (
                    <g key={`${column}-${run.row}`} role="button" tabIndex={0} aria-label={title} aria-pressed={selected}
                      className={`training-sentence-band${event.correction ? " correction" : ""}`} onClick={choose}
                      onKeyDown={(keyEvent) => {
                        if (keyEvent.key === "Enter" || keyEvent.key === " ") choose();
                      }}>
                      <title>{title}</title>
                      <rect x={x + 1} y={y + 4} width={Math.max(width - 2, 1)} height={ROW_H - 8} rx={3}
                        fill={event.correction ? TRAINING_CORRECTION_COLOR : colour}
                        fillOpacity={selected ? 0.4 : event.correction ? 0.3 : 0.16} stroke={stroke}
                        strokeOpacity={selected ? 1 : 0.7} strokeWidth={selected ? 2 : 1}
                        strokeDasharray={event.correction ? "3 2" : undefined} className="training-sentence-band-fill" />
                      {/* the word being said */}
                      <rect x={x} y={y + 2} width={2} height={ROW_H - 4} fill={event.correction ? TRAINING_CORRECTION_COLOR : colour}
                        className="training-sentence-issue" />
                      {event.correction ? (
                        <polygon points={`${x + 1},${y + 3} ${x + 7},${y + 3} ${x + 1},${y + 9}`} fill={TRAINING_CORRECTION_COLOR}
                          className="training-sentence-correction-mark" />
                      ) : null}
                      {/* the judge's verdict on this word's envelope */}
                      {held !== null ? (
                        <circle cx={x + 11} cy={y + ROW_H / 2} r={3.2} fill={held ? TRAINING_EXECUTOR_COLOR : TRAINING_OUTSIDE_COLOR}
                          stroke={held ? TRAINING_EXECUTOR_COLOR : TRAINING_OUTSIDE_COLOR} strokeWidth={1.2}
                          className={`training-sentence-verdict training-sentence-verdict-${held ? "inside" : "outside"}`} />
                      ) : null}
                      {width >= LABEL_MIN_W ? (
                        <text x={x + width / 2} y={y + ROW_H / 2 + 4} textAnchor="middle" className="training-sentence-word"
                          fill={event.correction ? TRAINING_CORRECTION_COLOR : colour}>
                          {label}
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
              </g>
            );
          })}

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
          {tickTimes.map((t, index) => (tickShown[index] ? (
            <text key={`tick-${index}`} x={xFor(t)} y={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 15}
              textAnchor={index === tickTimes.length - 1 ? "end" : "middle"} className="training-sentence-tick">
              {index === tickTimes.length - 1 ? `${formatSeconds(t)} s` : formatSeconds(t)}
            </text>
          ) : null))}
          {/* the live executor on its word's row, in flight time (none for a segment with no line: 3D flies nothing) */}
          {autopilot !== null && (autopilot.status === "flying"
            || (autopilot.status === "ready" && autopilotHasLine(autopilot.segment))) ? (
            <AutopilotCursor view={autopilot} reading={reading} x0={GUTTER} plotW={plotW} endS={endS}
              y={HEAD_H + TRAINING_COLUMN_INDEX[autopilot.request.column] * ROW_H} />
          ) : null}
          {/* the cursor, kept on the axis */}
          <line x1={xFor(Math.min(cursorS, endS))} x2={xFor(Math.min(cursorS, endS))} y1={HEAD_H - 8}
            y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 4} className="training-sentence-cursor" />
        </svg>
      </div>

      {notesOpen ? (
        <footer className="training-sentence-legend">
          <span>{flight.callsign}: {flightFacts}</span>
          {reading.envelopes !== null ? (
            <span>
              At a row's end: the judge's checks of this sentence's envelopes{closed === null ? " on the observed track"
                : " on the flown path"} — heading words inside their bands, altitude tubes held, speed words held, of those judged
              (red when one is not); a dot at a band's left is the same verdict for that word.
            </span>
          ) : null}
          <span>
            A band is a word in force, from the tick where it was said to the next word of its column; row 0 says all five
            columns. Click a band to select its word — here, in the read-back check and in 3D — and again to clear it; in a
            closed-loop sentence ▶ Fly flies the selected word's segment live (a band click does too): a short bar on that
            word's row says where the executor is — pulsing at the word's row while the backend flies it, then moving with the
            3D aircraft, left at the segment's end.
          </span>
          {closed !== null ? (
            <span>
              The closed-loop sentence at Δ = {reading.intervalS} s: the words the closed-loop reading says to the executor from the
              first predicted step ({formatSeconds(closed.startS)} s) on, one row every {reading.intervalS} s. An orange, dashed band
              with an orange mark at its left is a correction — a word the reading added to bring the flown path back to the
              observed one. The solid line at the right is where the flown flight ended: {replayText(closed.replay)}.
            </span>
          ) : (
            <span>
              The labelled sentence is the labeller's reading of the observed flight (open loop), on the 2 s rows from the
              first row; the dashed lines are the final captured, the speed left to the pilot, and a go-around when there is one.
            </span>
          )}
        </footer>
      ) : null}

      {readbackOpen ? (
        <TrainingReadbackWindow selection={selection} reading={reading} layers={trainingLayers} cursorS={cursorS}
          onCursorChange={setCursorS} column={focusColumn} onColumnChange={setFocusColumn} onClose={() => setReadbackOpen(false)}
          autopilot={autopilot?.status === "ready" ? autopilot.segment : null} />
      ) : null}
    </section>
  );
}
