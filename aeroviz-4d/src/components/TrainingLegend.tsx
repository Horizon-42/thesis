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
import { TRAINING_WEAK_HOLD_WIDTH_M, type TrainingVocabulary } from "../data/trainingSample";
import {
  TRAINING_CORRIDOR_COLOR,
  TRAINING_ENVELOPE_ALPHA,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_FUNNEL_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_TURN_COLOR,
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
  const weakKm = TRAINING_WEAK_HOLD_WIDTH_M / 1000;
  const rows: Array<{ key: string; swatch: Swatch; text: string; shown: boolean }> = [
    { key: "track", swatch: { kind: "line", colour: TRAINING_TRACE_COLOR }, shown: true,
      text: "the track (and, faint on the ground, its ground trace)" },
    { key: "turns", swatch: { kind: "line", colour: TRAINING_TURN_COLOR }, shown: layers.turnPaths,
      text: `a turn: the fastest, and the slowest (${vocabulary.turnRateMinDegS}°/s, begun ${vocabulary.turnStartDelayMaxS} s ` +
        "late); dashed where the slowest does not finish" },
    { key: "turn-end", swatch: { kind: "area", colour: TRAINING_TURN_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.turnEnd }, shown: layers.turnPaths,
      text: "where the turn may end: between the two turns' ends" },
    { key: "turn-region", swatch: { kind: "area", colour: TRAINING_TURN_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.turn }, shown: layers.turnRegions,
      text: "turn region: the area between the two turns" },
    { key: "funnel", swatch: { kind: "area", colour: TRAINING_FUNNEL_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.funnel }, shown: layers.holdFunnels,
      text: `hold funnel: after the turn, along the new heading ±${vocabulary.headingToleranceDeg}° from anywhere the ` +
        "turn may have ended" },
    { key: "funnel-dashed", swatch: { kind: "line", colour: TRAINING_FUNNEL_COLOR, dash: "4 3" }, shown: layers.holdFunnels,
      text: "a hold the labeller does not judge" },
    { key: "funnel-dotted", swatch: { kind: "line", colour: TRAINING_FUNNEL_COLOR, dash: "1 3" }, shown: layers.holdFunnels,
      text: `a judged hold whose funnel starts over ${weakKm} km wide: a weak check` },
    { key: "corridor", swatch: { kind: "area", colour: TRAINING_CORRIDOR_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.corridor }, shown: layers.corridor,
      text: "the capture corridor to the threshold" },
    { key: "tube", swatch: { kind: "area", colour: TRAINING_TUBE_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.tube }, shown: layers.vertical,
      text: `an altitude word's tube, ±${vocabulary.altitudeToleranceM} m` },
    { key: "outside", swatch: { kind: "line", colour: TRAINING_OUTSIDE_COLOR }, shown: true,
      text: "red: the labeller's check failed (a turn, rows outside a funnel or a tube)" },
    { key: "selected", swatch: { kind: "line", colour: TRAINING_WORD_COLOR }, shown: true,
      text: "yellow: the selected word — its envelope's edge and the rows it is in force; the rest fades" },
    { key: "executor", swatch: { kind: "line", colour: TRAINING_EXECUTOR_COLOR }, shown: executorTrack,
      text: "teal: the executor's flown track (dashed on the ground), the truth sentence flown from row 0" },
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
