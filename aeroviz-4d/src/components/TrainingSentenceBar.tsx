/**
 * TrainingSentenceBar.tsx
 * -----------------------
 * The sentence as a picture: the six columns as six rows — runway pointer, approach, heading,
 * altitude, angle, speed, in the vocabulary's order — against the flight's own time. Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 * A BAND IS A WORD IN FORCE, from the step it was issued to the step the next word of its column
 * replaces it. Step 0 carries all six columns, so every row opens with a band at 0; after that a
 * column is "unchanged" until its next word, and most steps say nothing at all. The tick at a
 * band's left edge is the issue itself; the numbers along the top count the steps that say
 * anything, and the header counts the silent ones.
 *
 * The three moments that are not words — the clearance, the capture of the final and the speed
 * becoming "unspecified" — are marked as lines across the rows. The header reads out the
 * labeller's own verdicts; nothing here recomputes one.
 *
 * The cursor is in flight time and is moved by clicking a band or a step number: what it
 * reports is the artefact's own step, never a rounded pixel. It does NOT drive `viewer.clock`:
 * Training loads no CZML, and the clock belongs to Observe's playback.
 *
 * ONE COLUMN IS HIGHLIGHTED, NEVER A STEP. Clicking a band selects its word class
 * (`trainingColumn`) and puts the cursor at its issue; every view then highlights that column's word
 * in force at the cursor, and only it. The other columns' words at the same step are not "the same
 * moment" — their runs begin and end elsewhere. Clicking the selected band again clears it.
 *
 * THE OVERLAYS, when the panel publishes them: the EXECUTOR's verdict on each word as a dot at the band's
 * left (teal inside its envelope, red outside, hollow grey not judged, not reached or superseded; none for
 * a word with no check of its own) and its outcome in the header; the PRIOR's likelihood of this flight in
 * the header, with the button that opens its window (`TrainingPriorWindow`).
 */

import { useLayoutEffect, useRef, useState } from "react";
import { useApp } from "../context/AppContext";
import TrainingLegend from "./TrainingLegend";
import TrainingPriorWindow from "./TrainingPriorWindow";
import TrainingReadbackWindow from "./TrainingReadbackWindow";
import {
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  executorWordAt,
  executorWordCounts,
  type TrainingExecutorFlight,
  type TrainingExecutorWord,
} from "../data/trainingOverlays";
import {
  formatSeconds,
  rowAtTime,
  TRAINING_WEAK_HOLD_WIDTH_M,
  trainingColumnRuns,
  trainingKindLabel,
  trainingVerdicts,
  trainingWordLabel,
  TRAINING_COLUMNS,
  type TrainingColumn,
} from "../data/trainingSample";

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

const ROW_LABEL: Record<TrainingColumn, string> = {
  runway: "Runway",
  approach: "Approach",
  heading: "Heading",
  altitude: "Altitude",
  angle: "Angle",
  speed: "Speed",
};

/**
 * Which of these ascending x positions may carry a text label: greedy from the left, keeping one
 * only when it clears the last kept by `minGap`. With `keepLast` the last position is always
 * kept, evicting the previous keeper if they would collide. The label is dropped, never the
 * step: its hit area and tooltip stay.
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
export function executorVerdictText(word: TrainingExecutorWord): string {
  const checks = word.checks.map((check) =>
    `${check.name} ${check.ok ? "✓" : "✗"}${check.rows === null ? "" : ` (${check.inside}/${check.rows})`}`).join(", ");
  const told = word.flownRow === null ? "" : `, told at its step ${word.flownRow}`;
  return `the executor: ${word.status}${told}${checks ? ` — ${checks}` : ""}${word.reason ? ` — ${word.reason}` : ""}`;
}

/** The header's line on the executor's replay of this flight. */
export function executorSummary(flight: TrainingExecutorFlight): string {
  if (!flight.flown) return `the executor: not flown — ${flight.group}`;
  const counts = executorWordCounts(flight);
  const outcome = (flight.outcome ?? "").replace(/_/g, " ");
  const crossing = flight.crossing === null ? ""
    : ` ${Math.abs(flight.crossing.crossM).toFixed(1)} m ${flight.crossing.crossM >= 0 ? "right" : "left"} of the centreline, ` +
      `${flight.crossing.heightM.toFixed(1)} m above the threshold at ${formatSeconds(flight.crossing.atS)} s`;
  const refused = flight.refused === null ? "" : ` · its track refused by the labeller's gate: ${flight.refused}`;
  // the judge's own tally, as the replay gate counts it (the word left to intercept the final on its own counts twice)
  const { wordsInside, wordsJudged } = flight.counts!;
  return `the executor (${flight.group}): ${outcome}${crossing} · ${wordsInside}/${wordsJudged} words ` +
    `inside their envelopes${counts.notJudged ? `, ${counts.notJudged} not judged` : ""}` +
    `${counts.notReached ? `, ${counts.notReached} not reached` : ""}${counts.superseded ? `, ${counts.superseded} superseded` : ""}` +
    ` · evaluation ${flight.evaluation?.replay} (observed ${flight.evaluation?.observed})${refused}`;
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
    trainingSelection, trainingLayers,
    trainingCursorS: cursorS, setTrainingCursorS: setCursorS,
    trainingColumn: focusColumn, setTrainingColumn: setFocusColumn,
    trainingExecutor, trainingPrior,
  } = useApp();
  const frameRef = useRef<HTMLDivElement>(null);
  const [plotW, setPlotW] = useState<number>(DEFAULT_PLOT_W);
  const [readbackOpen, setReadbackOpen] = useState<boolean>(false);
  const [priorOpen, setPriorOpen] = useState<boolean>(false);
  const [notesOpen, setNotesOpen] = useState<boolean>(false);
  const flightKey = trainingSelection?.flight.flightKey ?? null;

  useLayoutEffect(() => {
    const node = frameRef.current;
    if (!node) return;
    const measure = () => setPlotW(Math.max(node.clientWidth - GUTTER - PAD_R, MIN_PLOT_W));
    measure();
    if (typeof ResizeObserver === "undefined") return;   // jsdom has no layout to observe
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [flightKey]);

  if (!trainingSelection) return null;
  const { flight, vocabulary, candidates } = trainingSelection;
  const { tS } = flight.signals;
  // Step r covers [r·step, (r+1)·step): the axis ends where the last step does.
  const endS = flight.rows * vocabulary.stepS;
  const timeOf = (row: number) => row * vocabulary.stepS;
  const xFor = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  const verdicts = trainingVerdicts(flight);
  const cursorRow = rowAtTime(tS, cursorS);

  // The steps that SAY something, numbered; step 0 is the opening and is always complete.
  const issueRows = [...new Set(flight.words.events.map((event) => event.row))].sort((a, b) => a - b);
  const stepShown = spacedLabels(issueRows.map((row) => xFor(timeOf(row))), STEP_LABEL_GAP);
  const tickRows = [...issueRows, flight.rows];
  const tickShown = spacedLabels(tickRows.map((row) => xFor(timeOf(row))), TICK_LABEL_GAP, true);
  const markers = [
    { row: flight.joinRow, label: "cleared", title: `cleared to join the final at ${formatSeconds(timeOf(flight.joinRow))} s` },
    {
      row: flight.captureRow, label: "captured",
      title: `the final captured at ${formatSeconds(timeOf(flight.captureRow))} s, ` +
        `${(flight.captureBeforeThresholdM / 1000).toFixed(1)} km before the threshold`,
    },
    { row: flight.unspecifiedRow, label: "speed unspecified", title: `speed left to the pilot from ${formatSeconds(timeOf(flight.unspecifiedRow))} s` },
  ];
  const tick = (ok: boolean) => (ok ? "✓" : "✗");
  const capture = verdicts.captureTurn;
  const landing = flight.envelopes.approach.landing;
  // the overlays are published for the selected flight; a stale one (a flight switch in flight) is not drawn
  const executor = trainingExecutor?.flight.flightKey === flight.flightKey ? trainingExecutor.flight : null;
  const prior = trainingPrior?.flight.flightKey === flight.flightKey ? trainingPrior : null;

  return (
    <section className="training-sentence-bar" aria-label="Sentence bar">
      <TrainingLegend layers={trainingLayers} vocabulary={vocabulary} executorTrack={executor?.track != null} />
      <header className="training-sentence-head">
        <strong>{flight.callsign}</strong>
        <span>{flight.typecode}</span>
        <span>runway {flight.runway}</span>
        <span>{flight.stratum}</span>
        <span>
          {flight.rows} steps · {verdicts.instructionsAfterStep0} words after step 0 ·{" "}
          {verdicts.silentSteps} of {flight.rows - 1} later steps silent
        </span>
        <span
          className="training-sentence-arrival"
          title="The labeller's own checks of this flight's envelopes (Reading.checks)."
        >
          turns {verdicts.turnsProgressOk}/{verdicts.turns} monotone, {verdicts.turnsRateOk}/{verdicts.turns} rate ·
          holds {verdicts.holdsContained}/{verdicts.holdsJudged} in their funnel
          {verdicts.holdsNotJudged || verdicts.holdsWeak
            ? ` (${[
              verdicts.holdsNotJudged ? `${verdicts.holdsNotJudged} not judged` : "",
              verdicts.holdsWeak ? `${verdicts.holdsWeak} weak: the funnel starts over ${TRAINING_WEAK_HOLD_WIDTH_M / 1000} km wide` : "",
            ].filter(Boolean).join("; ")})`
            : ""} ·
          capture turn {capture === null ? "none (on the final at entry)" : `${tick(capture.progressOk)} ${tick(capture.rateOk)}`} ·
          altitude {verdicts.altitudeContained}/{verdicts.altitudeWords} · speed {verdicts.speedContained}/{verdicts.speedWords}
        </span>
        {executor ? (
          <span className="training-sentence-executor" style={{ color: TRAINING_EXECUTOR_COLOR }}
            title="The executor flew this sentence from row 0, each word said where the observed aircraft heard it; its verdicts are the judge's, each word's envelope re-drawn from where the executor was told it.">
            {executorSummary(executor)}
          </span>
        ) : null}
        {prior ? (
          <span className="training-sentence-prior"
            title="The negative log-likelihood of this flight's truth sentence under the prior (teacher-forced), per 2 s step.">
            the prior: {prior.flight.nllPerStep.toFixed(3)} nats per step here ({prior.overlay.readout.split}{" "}
            {prior.overlay.readout.model.nllPerStep.toFixed(4)})
          </span>
        ) : null}
        <span className="training-sentence-cursor-readout">
          t = {formatSeconds(cursorS)} s · step {cursorRow}
        </span>
        <button
          type="button"
          className="training-sentence-readback-button"
          onClick={() => setReadbackOpen((open) => !open)}
        >
          {readbackOpen ? "Close read-back check" : "Read-back check"}
        </button>
        {prior ? (
          <button type="button" className="training-sentence-readback-button" onClick={() => setPriorOpen((open) => !open)}>
            {priorOpen ? "Close prior predictions" : "Prior predictions"}
          </button>
        ) : null}
      </header>

      <div className="training-sentence-frame" ref={frameRef}>
        <svg
          className="training-sentence-svg"
          width={GUTTER + plotW + PAD_R}
          height={VIEW_H}
          viewBox={`0 0 ${GUTTER + plotW + PAD_R} ${VIEW_H}`}
          role="img"
          aria-label={`The sentence of ${flight.callsign} on runway ${flight.runway}`}
        >
          {/* the steps that say something: numbers along the top, each a button */}
          {issueRows.map((row, index) => {
            const left = index === 0 ? GUTTER : xFor((timeOf(issueRows[index - 1]) + timeOf(row)) / 2);
            const right = index === issueRows.length - 1
              ? GUTTER + plotW
              : xFor((timeOf(row) + timeOf(issueRows[index + 1])) / 2);
            const words = flight.words.events.filter((event) => event.row === row);
            const name =
              `Step ${row} at ${formatSeconds(timeOf(row))} s: ` +
              words.map((event) => `${TRAINING_COLUMNS[event.column]} ${trainingWordLabel(vocabulary, candidates, TRAINING_COLUMNS[event.column], event.value)}`).join(", ");
            return (
              <g
                key={`step-${row}`}
                role="button"
                tabIndex={0}
                aria-label={name}
                className="training-sentence-event"
                onClick={() => setCursorS(timeOf(row))}
                onKeyDown={(keyEvent) => {
                  if (keyEvent.key === "Enter" || keyEvent.key === " ") setCursorS(timeOf(row));
                }}
              >
                <title>{name}</title>
                <rect x={left} y={2} width={Math.max(right - left, 1)} height={HEAD_H - 6} fill="transparent" />
                <line x1={xFor(timeOf(row))} x2={xFor(timeOf(row))} y1={HEAD_H - 6} y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H}
                  className="training-sentence-event-line" />
                {stepShown[index] ? (
                  <text x={xFor(timeOf(row))} y={HEAD_H - 9} textAnchor="middle" className="training-sentence-event-number">
                    {row}
                  </text>
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
                <rect x={GUTTER} y={y} width={plotW} height={ROW_H}
                  className={`training-sentence-row-bg${position % 2 ? " odd" : ""}`} />
                <text x={GUTTER - 8} y={y + ROW_H / 2 + 4} textAnchor="end" className="training-sentence-row-label"
                  style={selectedColumn ? { fill: TRAINING_WORD_COLOR, fontWeight: 600 } : undefined}>
                  {ROW_LABEL[column]}
                </text>
                {trainingColumnRuns(flight, column).map((run) => {
                  const x = xFor(timeOf(run.row));
                  const width = xFor(timeOf(run.endRow)) - x;
                  const label = trainingWordLabel(vocabulary, candidates, column, run.value);
                  const split = column === "heading"
                    ? flight.envelopes.heading.find((item) => item.row === run.row)?.split ?? null
                    : null;
                  const verdict = executor === null ? null : executorWordAt(executor, run.row, column);
                  const mark = verdict === null ? null : verdictMark(verdict.status);
                  const title =
                    `${column} ${label} — ${trainingKindLabel(run.event.kind, split)}, issued at step ${run.row} ` +
                    `(${formatSeconds(timeOf(run.row))} s), in force to ${formatSeconds(timeOf(run.endRow))} s` +
                    (verdict === null ? "" : `\n${executorVerdictText(verdict)}`);
                  const selected = selectedColumn && run.row <= cursorRow && cursorRow < run.endRow;
                  const choose = () => {
                    if (selected) {
                      setFocusColumn(null);
                      return;
                    }
                    setFocusColumn(column);
                    setCursorS(timeOf(run.row));
                  };
                  return (
                    <g
                      key={`${column}-${run.row}`}
                      role="button"
                      tabIndex={0}
                      aria-label={title}
                      aria-pressed={selected}
                      className="training-sentence-band"
                      onClick={choose}
                      onKeyDown={(keyEvent) => {
                        if (keyEvent.key === "Enter" || keyEvent.key === " ") choose();
                      }}
                    >
                      <title>{title}</title>
                      <rect
                        x={x + 1}
                        y={y + 4}
                        width={Math.max(width - 2, 1)}
                        height={ROW_H - 8}
                        rx={3}
                        fill={colour}
                        fillOpacity={selected ? 0.4 : 0.16}
                        stroke={selected ? TRAINING_WORD_COLOR : colour}
                        strokeOpacity={selected ? 1 : 0.6}
                        strokeWidth={selected ? 2 : 1}
                      />
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
                          {run.event.kind.startsWith("intercept") ? " ⤳" : ""}
                          {split ? ` ${split.part}/${split.parts}` : ""}
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
            <g key={marker.label} aria-label={marker.title}>
              <title>{marker.title}</title>
              <line
                x1={xFor(timeOf(marker.row))}
                x2={xFor(timeOf(marker.row))}
                y1={HEAD_H}
                y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H}
                stroke={TRAINING_CORRIDOR_COLOR}
                strokeDasharray="3 3"
                className="training-sentence-marker"
              />
            </g>
          ))}

          <line x1={GUTTER} x2={GUTTER + plotW} y1={HEAD_H + TRAINING_COLUMNS.length * ROW_H}
            y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H} className="training-sentence-axis" />
          {tickRows.map((row, index) =>
            tickShown[index] ? (
              <text
                key={`tick-${row}`}
                x={xFor(timeOf(row))}
                y={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 15}
                textAnchor={index === tickRows.length - 1 ? "end" : "middle"}
                className="training-sentence-tick"
              >
                {index === tickRows.length - 1 ? `${formatSeconds(timeOf(row))} s` : formatSeconds(timeOf(row))}
              </text>
            ) : null,
          )}
          <line
            x1={xFor(cursorS)}
            x2={xFor(cursorS)}
            y1={HEAD_H - 8}
            y2={HEAD_H + TRAINING_COLUMNS.length * ROW_H + 4}
            className="training-sentence-cursor"
          />
        </svg>
      </div>

      <footer className="training-sentence-legend">
        <span>
          A band is a WORD IN FORCE, from the step it was issued (the tick at its left edge) to the
          next word of its column; step 0 gives all six. Click a band to select that word: it alone
          is highlighted here, in the read-back check and in 3D (click it again to clear). The dashed
          lines mark the clearance, the capture of the final and where the speed is left to the pilot.
          <button
            type="button"
            className="training-sentence-notes-toggle"
            aria-expanded={notesOpen}
            onClick={() => setNotesOpen((open) => !open)}
          >
            {notesOpen ? "Fewer notes" : "More notes"}
          </button>
        </span>
        {notesOpen ? (
          <>
            <span>
              Runway is a POINTER at one of {candidates.length} candidate thresholds (
              {candidates.map((candidate) => candidate.ident).join(", ")}); heading is an absolute ground
              track flown the shorter way; altitude a geometric MSL target or "descend to land"; angle the
              class of the descent (or level, or climb); speed a ground speed or "unspecified". ⤳ marks an
              intercept heading the labeller inserted; n/m a part of a split turn.
            </span>
            <span>
              The sentence ends before the landing: its last step is{" "}
              {(landing.lastRowBeforeThresholdM / 1000).toFixed(2)} km before the threshold
              {landing.cutAtCrossing ? ", cut before the last passage of the threshold" : ", where the data ends"}.
            </span>
            <span>
              With the executor's replay on, a dot at a band's left edge is the executor's verdict on that word:{" "}
              <b style={{ color: TRAINING_EXECUTOR_COLOR }}>●</b> flown inside its envelope,{" "}
              <b style={{ color: TRAINING_OUTSIDE_COLOR }}>●</b> outside, ○ not judged, not reached or superseded (the
              tooltip says which and why); a word with no check of its own (the runway pointer, an angle word, "not
              cleared", "unspecified") has none. The executor is judged against each word's envelope re-drawn from where
              IT was told the word, not the observed flight's. With the prior's predictions on, "Prior predictions" shows
              what it gives each column at each step.
            </span>
          </>
        ) : null}
      </footer>

      {readbackOpen ? (
        <TrainingReadbackWindow
          flight={flight}
          vocabulary={vocabulary}
          candidates={candidates}
          layers={trainingLayers}
          cursorS={cursorS}
          onCursorChange={setCursorS}
          column={focusColumn}
          onColumnChange={setFocusColumn}
          onClose={() => setReadbackOpen(false)}
          executor={executor}
        />
      ) : null}
      {priorOpen && prior ? (
        <TrainingPriorWindow
          flight={flight}
          vocabulary={vocabulary}
          candidates={candidates}
          prior={prior}
          cursorS={cursorS}
          onCursorChange={setCursorS}
          column={focusColumn}
          onColumnChange={setFocusColumn}
          onClose={() => setPriorOpen(false)}
        />
      ) : null}
    </section>
  );
}
