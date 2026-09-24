/**
 * TrainingLegend.tsx
 * ------------------
 * What each colour in the Training 3D scene is — one line per thing drawn, only for the switches
 * that are on. Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §4.4.
 *
 * It sits above the sentence bar, in the scene's lower right corner, and folds away. The read-back
 * window has its own legend in its footer; the colours are the same (`trainingWordColors.ts`).
 */

import { useState } from "react";
import type { TrainingLayers } from "../context/AppContext";
import type { TrainingVocabulary } from "../data/trainingSample";
import {
  TRAINING_CAPTURE_TURN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_ENVELOPE_ALPHA,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";

type Swatch = { kind: "line"; colour: string; dash?: string } | { kind: "area"; colour: string; opacity: number; dash?: string };

function SwatchIcon({ swatch }: { swatch: Swatch }) {
  return (
    <svg className="training-legend-swatch" width={22} height={10} aria-hidden="true">
      {swatch.kind === "line" ? (
        <line x1={1} x2={21} y1={5} y2={5} stroke={swatch.colour} strokeWidth={2.2} strokeDasharray={swatch.dash} />
      ) : (
        <rect x={1} y={1} width={20} height={8} fill={swatch.colour} fillOpacity={swatch.opacity} stroke={swatch.colour}
          strokeWidth={1.2} strokeDasharray={swatch.dash} />
      )}
    </svg>
  );
}

export default function TrainingLegend({ layers, vocabulary, executorTrack = false }: {
  layers: TrainingLayers; vocabulary: TrainingVocabulary;
  /** The executor's flown track is drawn (its overlay is on and the flight was flown). */
  executorTrack?: boolean;
}) {
  const [open, setOpen] = useState<boolean>(true);
  const rows: Array<{ key: string; swatch: Swatch; text: string; shown: boolean }> = [
    { key: "track", swatch: { kind: "line", colour: TRAINING_TRACE_COLOR }, shown: true,
      text: "the track (and, faint on the ground, its ground trace)" },
    { key: "heading", swatch: { kind: "line", colour: TRAINING_HEADING_BAND_COLOR }, shown: layers.headingBands,
      text: `a heading word's judged rows, on the ground: from ${vocabulary.headingLeadS} s after it is said to the next ` +
        `word's, where the track must stay within ±${vocabulary.headingToleranceDeg}° of it` },
    { key: "capture-turn", swatch: { kind: "line", colour: TRAINING_CAPTURE_TURN_COLOR, dash: "4 3" }, shown: layers.corridor,
      text: "the capture turn's rows on the ground: from the clearance onto the course" },
    { key: "corridor", swatch: { kind: "area", colour: TRAINING_CORRIDOR_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.corridor }, shown: layers.corridor,
      text: "the capture corridor to the threshold" },
    { key: "tube", swatch: { kind: "area", colour: TRAINING_TUBE_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.tube }, shown: layers.vertical,
      text: `an altitude word's tube, ±${vocabulary.altitudeToleranceM} m` },
    { key: "outside", swatch: { kind: "line", colour: TRAINING_OUTSIDE_COLOR }, shown: true,
      text: "red: the labeller's check failed (rows outside a heading band or a tube, a capture turn)" },
    { key: "selected", swatch: { kind: "line", colour: TRAINING_WORD_COLOR }, shown: true,
      text: "yellow: the selected word — its envelope and the rows it is in force; the rest fades" },
    { key: "executor", swatch: { kind: "line", colour: TRAINING_EXECUTOR_COLOR }, shown: executorTrack,
      text: "teal: the executor's flown track (dashed on the ground), the truth sentence flown from row 0" +
        (layers.headingBands ? "; red on its ground trace: its rows outside the heading word it was told" : "") },
  ];

  return (
    <aside className="training-legend" aria-label="What the 3D scene shows">
      <button type="button" className="training-legend-toggle" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        {open ? "Legend ▾" : "Legend ▸"}
      </button>
      {open ? (
        <ul>
          {rows.filter((row) => row.shown).map((row) => (
            <li key={row.key}>
              <SwatchIcon swatch={row.swatch} />
              <span>{row.text}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </aside>
  );
}
