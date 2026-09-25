/**
 * TrainingPriorWindow.tsx
 * -----------------------
 * The prior's predictions for one flight, against its truth sentence (`data/trainingOverlays.ts`). Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §4.6.
 *
 *  • AT THE CURSOR: per column, what the truth sentence does at this step (a word, or nothing; at the first predicted
 *    step the word in force, which the prior must say — `priorTruthAt`), the probability the prior gives that, the
 *    probability it gives a word being said at all, and — were one said — its most likely words. Before the first
 *    predicted step the prior only observes: nothing to read.
 *  • ALONG THE FLIGHT: one strip per column from the first predicted step — the probability of a word being said (the
 *    column's colour) and the probability of the truth (grey), step by step; a tick at every word the truth sentence
 *    says after the first predicted step, in the executor's teal when the prior's most likely word there is the word
 *    said, red when it is another.
 *
 * TEACHER-FORCED: every step sees the truth sentence's words before it — this is how the prior was trained and read
 * out, not a sentence it says on its own. Nothing is recomputed: every probability is the exporter's.
 */

import {
  TRAINING_COLUMN_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  formatSeconds,
  rowAtTime,
  TRAINING_COLUMN_INDEX,
  TRAINING_COLUMNS,
  TRAINING_UNCHANGED,
  trainingColumnRuns,
  trainingWordLabel,
  type TrainingCandidate,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingVocabulary,
} from "../data/trainingSample";
import { priorStep, type TrainingPriorView } from "../data/trainingOverlays";
import useMeasuredWidth from "../hooks/useMeasuredWidth";
import TrainingWindow from "./training/TrainingWindow";
import { ChartFrame } from "./training/chartKit";

const DEFAULT_W = 960;
const MIN_W = 420;
const GUTTER = 84;
const PAD_R = 14;
const STRIP_H = 34;
const STRIP_GAP = 6;
const STRIP_TOP = 4;
const AXIS_H = 22;

export interface TrainingPriorWindowProps {
  flight: TrainingFlight;
  vocabulary: TrainingVocabulary;
  candidates: TrainingCandidate[];
  prior: TrainingPriorView;
  cursorS: number;
  onCursorChange: (seconds: number) => void;
  column: TrainingColumn | null;
  onColumnChange: (column: TrainingColumn) => void;
  onClose: () => void;
}

const p3 = (value: number) => value.toFixed(3);

export default function TrainingPriorWindow({
  flight, vocabulary, candidates, prior, cursorS, onCursorChange, column, onColumnChange, onClose,
}: TrainingPriorWindowProps) {
  const { overlay, flight: predicted } = prior;
  const { readout } = overlay;
  const [frame, width] = useMeasuredWidth(MIN_W, DEFAULT_W);
  const row = rowAtTime(flight.signals.tS, cursorS);
  const label = (name: TrainingColumn, value: number) => trainingWordLabel(vocabulary, candidates, name, value);
  const plotW = width - GUTTER - PAD_R;
  const endS = flight.rows * vocabulary.stepS;
  const x = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  const timeAtX = (px: number) => Math.min(Math.max(((px - GUTTER) / plotW) * endS, 0), endS);
  const height = TRAINING_COLUMNS.length * (STRIP_H + STRIP_GAP) + AXIS_H;
  const stripTop = (index: number) => index * (STRIP_H + STRIP_GAP) + STRIP_TOP;
  /** The strip under a height, or none (a gap, the axis). */
  const stripAt = (y: number) => {
    const index = Math.floor((y - STRIP_TOP) / (STRIP_H + STRIP_GAP));
    return index >= 0 && index < TRAINING_COLUMNS.length && y - stripTop(index) <= STRIP_H ? TRAINING_COLUMNS[index] : null;
  };
  const y = (index: number, p: number) => stripTop(index) + (1 - p) * STRIP_H;
  const first = predicted.firstPredictedRow;
  const polyline = (index: number, values: number[]) =>
    values.map((value, step) => `${x((first + step) * vocabulary.stepS)},${y(index, value)}`).join(" ");

  return (
    <TrainingWindow title="Prior predictions" closeLabel="Close the prior predictions" className="training-prior-window"
      cursorS={cursorS} cursorRow={row} onClose={onClose}
      chips={<>
        <span>{flight.callsign}</span>
        <span>runway {flight.runway}</span>
        <span title={`the negative log-likelihood of this flight's truth sentence under the prior, per predicted ` +
          `${formatSeconds(vocabulary.stepS)} s step; ${readout.split} as a whole: ${readout.model.nllPerStep.toFixed(4)}`}>
          this flight {predicted.nllPerStep.toFixed(3)} nats per step
        </span>
      </>}>
      <table className="training-prior-table" aria-label={`The prior at step ${row}`}>
        <thead>
          <tr>
            <th scope="col">column</th>
            <th scope="col">the truth at this step</th>
            <th scope="col" title="the probability the prior gives what the truth sentence does here">P(truth)</th>
            <th scope="col" title="the probability the prior gives a word being said at this step">P(a word now)</th>
            <th scope="col">were a word said: the most likely</th>
          </tr>
        </thead>
        <tbody>
          {TRAINING_COLUMNS.map((name) => {
            const step = priorStep(flight, predicted, name, row);
            const selected = name === column;
            const inForce = flight.words.inForce[TRAINING_COLUMN_INDEX[name]][row];
            return (
              <tr key={name} className={selected ? "selected" : undefined} tabIndex={0} aria-selected={selected}
                onClick={() => onColumnChange(name)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") onColumnChange(name);
                }}
                style={selected ? { outline: `1px solid ${TRAINING_WORD_COLOR}` } : undefined}>
                <th scope="row" style={{ color: TRAINING_COLUMN_COLOR[name] }}>{name}</th>
                {step === null ? (
                  <td colSpan={4} className="training-prior-muted">observed only — the prior speaks from step {first}</td>
                ) : (
                  <>
                    <td>
                      {step.truth === TRAINING_UNCHANGED
                        ? <>unchanged <span className="training-prior-muted">(in force: {label(name, inForce)})</span></>
                        : <>says <b>{label(name, step.truth)}</b></>}
                    </td>
                    <td>{p3(step.truthP)}</td>
                    <td>{p3(step.changeP)}</td>
                    <td>
                      {step.ranked.map(({ value, p }, rank) => (
                        <span key={rank} className="training-prior-word"
                          style={value === step.truth ? { color: TRAINING_EXECUTOR_COLOR, fontWeight: 600 } : undefined}>
                          {rank ? " · " : ""}{label(name, value)} {p3(p)}
                        </span>
                      ))}
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="training-readback-frame" ref={frame}>
        <ChartFrame label="The prior along the flight" width={width} height={height}
          onPointer={(px) => onCursorChange(timeAtX(px))}
          onPick={(px, py) => {
            onCursorChange(timeAtX(px));
            const strip = stripAt(py);
            if (strip !== null) onColumnChange(strip);
          }}>
          {TRAINING_COLUMNS.map((name, index) => {
            const values = predicted.columns[index];
            const colour = TRAINING_COLUMN_COLOR[name];
            const words = trainingColumnRuns(flight, name).filter((run) => run.row > first);
            return (
              <g key={name} aria-label={`${name} strip`}>
                <rect x={GUTTER} y={stripTop(index)} width={plotW} height={STRIP_H}
                  className={`training-sentence-row-bg${index % 2 ? " odd" : ""}`} />
                <text x={GUTTER - 8} y={stripTop(index) + STRIP_H / 2 + 4} textAnchor="end" className="training-sentence-row-label"
                  style={name === column ? { fill: TRAINING_WORD_COLOR, fontWeight: 600 } : undefined}>
                  {name}
                </text>
                <polyline points={polyline(index, values.truthP)} fill="none" stroke={TRAINING_RAW_COLOR} strokeOpacity={0.7}
                  strokeWidth={1} className="training-prior-truth" />
                <polyline points={polyline(index, values.changeP)} fill="none" stroke={colour} strokeWidth={1.4}
                  className="training-prior-change" />
                {words.map((run) => {
                  const step = priorStep(flight, predicted, name, run.row)!;             // run.row > first: predicted
                  const right = step.ranked[0]?.value === run.value;
                  const at = x(run.row * vocabulary.stepS);
                  return (
                    <g key={run.row} className="training-prior-tick" aria-label={`${name} ${label(name, run.value)} at step ${run.row}`}>
                      <title>
                        step {run.row}: the truth says {label(name, run.value)}; the prior gives a word {p3(step.changeP)}, this word{" "}
                        {p3(step.truthP)}; its most likely word is {step.ranked[0] ? label(name, step.ranked[0].value) : "—"}
                      </title>
                      <line x1={at} x2={at} y1={stripTop(index)} y2={stripTop(index) + STRIP_H}
                        stroke={right ? TRAINING_EXECUTOR_COLOR : TRAINING_OUTSIDE_COLOR} strokeWidth={1.4} />
                      <circle cx={at} cy={y(index, step.changeP)} r={2.6} fill={right ? TRAINING_EXECUTOR_COLOR : TRAINING_OUTSIDE_COLOR} />
                    </g>
                  );
                })}
              </g>
            );
          })}
          <line x1={x(cursorS)} x2={x(cursorS)} y1={0} y2={height - AXIS_H} className="training-readback-cursor-line" />
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
            <text key={fraction} x={x(fraction * endS)} y={height - 6} textAnchor="middle" className="training-readback-tick">
              {formatSeconds(Math.round(fraction * endS))}
            </text>
          ))}
        </ChartFrame>
      </div>

      <footer className="training-readback-legend"
        title={`Teacher-forced: at every step the prior sees the flight so far and the truth sentence's words before the step ` +
          `— how it was trained and read out (${readout.split}, best epoch ${readout.bestEpoch}); not a sentence of its own. ` +
          `The first ${first} steps are only observed; at step ${first} it says every column's word in force.`}>
        <span>
          <b style={{ color: TRAINING_COLUMN_COLOR.heading }}>——</b> P(a word said) ·{" "}
          <b style={{ color: TRAINING_RAW_COLOR }}>——</b> P(the truth) · a tick at each word said after step {first}:{" "}
          <b style={{ color: TRAINING_EXECUTOR_COLOR }}>teal</b> the prior's first choice, <b style={{ color: TRAINING_OUTSIDE_COLOR }}>red</b>{" "}
          another · teacher-forced
        </span>
      </footer>
    </TrainingWindow>
  );
}
