/**
 * TrainingPriorWindow.tsx
 * -----------------------
 * The prior's predictions for one flight, against its truth sentence (`data/trainingOverlays.ts`). Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §4.6.
 *
 *  • AT THE CURSOR: per column, what the truth sentence does at this step (a word, or nothing; at the first predicted
 *    step the word in force, which the prior must say), the probability the prior gives that, the probability it gives
 *    a word being said at all, and — were one said — its most likely words. Before the first predicted step the prior
 *    only observes: nothing to read.
 *  • ALONG THE FLIGHT: one strip per column from the first predicted step — the probability of a word being said (the
 *    column's colour) and the probability of the truth (grey), step by step; a tick at every word the truth sentence
 *    says after the first predicted step, in the executor's teal when the prior's most likely word there is the word
 *    said, red when it is another.
 *
 * TEACHER-FORCED: every step sees the truth sentence's words before it — this is how the prior was trained and read
 * out, not a sentence it says on its own. Nothing is recomputed: every probability is the exporter's.
 *
 * It renders through a PORTAL into `document.body` (AV7), as the read-back window does.
 */

import { createPortal } from "react-dom";
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

const WIDTH = 960;
const GUTTER = 84;
const PAD_R = 14;
const STRIP_H = 34;
const STRIP_GAP = 6;
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
  const { tS } = flight.signals;
  const row = rowAtTime(tS, cursorS);
  const label = (name: TrainingColumn, value: number) => trainingWordLabel(vocabulary, candidates, name, value);
  const plotW = WIDTH - GUTTER - PAD_R;
  const endS = flight.rows * vocabulary.stepS;
  const x = (seconds: number) => GUTTER + (seconds / endS) * plotW;
  const timeAtX = (px: number) => Math.min(Math.max(((px - GUTTER) / plotW) * endS, 0), endS);
  const height = TRAINING_COLUMNS.length * (STRIP_H + STRIP_GAP) + AXIS_H;
  const stripTop = (index: number) => index * (STRIP_H + STRIP_GAP) + 4;
  const y = (index: number, p: number) => stripTop(index) + (1 - p) * STRIP_H;
  const first = predicted.firstPredictedRow;
  const polyline = (index: number, values: number[]) =>
    values.map((value, step) => `${x((first + step) * vocabulary.stepS)},${y(index, value)}`).join(" ");

  return createPortal(
    <div className="training-readback-backdrop">
      <div className="training-readback-window training-prior-window" role="dialog" aria-label="Prior predictions"
        aria-modal="false" tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") onClose();
        }}>
        <header className="training-readback-head">
          <strong>Prior predictions</strong>
          <span>{flight.callsign}</span>
          <span>runway {flight.runway}</span>
          <span title="negative log-likelihood of this flight's truth sentence under the prior, per predicted 2 s step">
            this flight {predicted.nllPerStep.toFixed(3)} nats per step · {readout.split}: prior {readout.model.nllPerStep.toFixed(4)},
            repeat {readout.baselines.repeat.all.toFixed(4)}, previous word {readout.baselines.previousWord.all.toFixed(4)}
          </span>
          <span className="training-readback-cursor">t = {formatSeconds(cursorS)} s · step {row}</span>
          <button type="button" onClick={onClose} aria-label="Close the prior predictions">×</button>
        </header>

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
              const cells = (() => {
                if (step === null) {
                  return <td colSpan={4} className="training-prior-muted">observed only — the prior speaks from step {first}</td>;
                }
                const { truth } = step;
                const inForce = flight.words.inForce[TRAINING_COLUMN_INDEX[name]][row];
                return (
                  <>
                    <td>
                      {truth === TRAINING_UNCHANGED
                        ? <>unchanged <span className="training-prior-muted">(in force: {label(name, inForce)})</span></>
                        : <>says <b>{label(name, truth)}</b></>}
                    </td>
                    <td>{p3(step.truthP)}</td>
                    <td>{p3(step.changeP)}</td>
                    <td>
                      {step.ranked.map(({ value, p }, rank) => (
                        <span key={rank} className="training-prior-word"
                          style={value === truth ? { color: TRAINING_EXECUTOR_COLOR, fontWeight: 600 } : undefined}>
                          {rank ? " · " : ""}{label(name, value)} {p3(p)}
                        </span>
                      ))}
                    </td>
                  </>
                );
              })();
              return (
                <tr key={name} className={selected ? "selected" : undefined} onClick={() => onColumnChange(name)}
                  style={selected ? { outline: `1px solid ${TRAINING_WORD_COLOR}` } : undefined}>
                  <th scope="row" style={{ color: TRAINING_COLUMN_COLOR[name] }}>{name}</th>
                  {cells}
                </tr>
              );
            })}
          </tbody>
        </table>

        <svg className="training-readback-svg" width={WIDTH} height={height} viewBox={`0 0 ${WIDTH} ${height}`}
          aria-label="The prior along the flight"
          onMouseMove={(event) => onCursorChange(timeAtX(event.nativeEvent.offsetX))}
          onClick={(event) => {
            onCursorChange(timeAtX(event.nativeEvent.offsetX));
            const index = Math.floor(event.nativeEvent.offsetY / (STRIP_H + STRIP_GAP));
            if (index >= 0 && index < TRAINING_COLUMNS.length) onColumnChange(TRAINING_COLUMNS[index]);
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
                      <circle cx={at} cy={y(index, step.changeP)} r={2.6}
                        fill={right ? TRAINING_EXECUTOR_COLOR : TRAINING_OUTSIDE_COLOR} />
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
        </svg>

        <footer className="training-readback-legend">
          <span>
            Teacher-forced: at every step the prior sees the flight so far and the truth sentence's words before the step —
            how it was trained and read out ({readout.split}, best epoch {readout.bestEpoch}); it is not the prior speaking a
            sentence of its own. The first {first} steps are only observed; at step {first} the prior says every column's word
            in force. In each strip, <b style={{ color: TRAINING_COLUMN_COLOR.heading }}>——</b> the probability of a
            word being said at the step (0 at the strip's foot, 1 at its top), <b style={{ color: TRAINING_RAW_COLOR }}>——</b> the
            probability of what the truth sentence does there; a tick marks each word the truth says after step {first} —{" "}
            <b style={{ color: TRAINING_EXECUTOR_COLOR }}>teal</b> when the prior's most likely word there is that word,{" "}
            <b style={{ color: TRAINING_OUTSIDE_COLOR }}>red</b> when it is another. Move the pointer to read a step; click a
            strip or a row to select its column.
          </span>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
