/**
 * TrainingLegend.tsx
 * ------------------
 * What each colour in the Training 3D scene is — one short line per thing drawn, only for what is on, the full reading
 * in each line's tooltip. Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §4.4.
 *
 * It sits above the sentence bar, in the scene's lower right corner, folded until it is opened. The read-back window
 * has its own swatches; the colours are the same (`trainingWordColors.ts`).
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
import { SwatchIcon, type Swatch } from "./training/chartKit";

export default function TrainingLegend({ layers, vocabulary, executorTrack, autopilotColour }: {
  layers: TrainingLayers;
  vocabulary: TrainingVocabulary;
  /** The executor's flown track is drawn (its overlay is on and the flight was flown). */
  executorTrack: boolean;
  /** The colour the live executor's segment is drawn in (`autopilotColour`), or null when none is drawn. */
  autopilotColour: string | null;
}) {
  const [open, setOpen] = useState<boolean>(false);
  const rows: Array<{ key: string; swatch: Swatch; text: string; title: string; shown: boolean }> = [
    { key: "track", swatch: { kind: "line", colour: TRAINING_TRACE_COLOR }, shown: true, text: "the track",
      title: "the observed track, and — faint on the ground — its ground trace" },
    { key: "heading", swatch: { kind: "line", colour: TRAINING_HEADING_BAND_COLOR }, shown: layers.headingBands,
      text: "heading word: judged rows",
      title: `a heading word's judged rows, on the ground: from ${vocabulary.headingLeadS} s after it is said to the next ` +
        `word's, where the track must stay within ±${vocabulary.headingToleranceDeg}° of it` },
    { key: "capture-turn", swatch: { kind: "line", colour: TRAINING_CAPTURE_TURN_COLOR, dash: "4 3" }, shown: layers.corridor,
      text: "capture turn", title: "the capture turn's rows on the ground: from the clearance onto the course" },
    { key: "corridor", swatch: { kind: "area", colour: TRAINING_CORRIDOR_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.corridor },
      shown: layers.corridor, text: "capture corridor", title: "the capture corridor to the threshold" },
    { key: "tube", swatch: { kind: "area", colour: TRAINING_TUBE_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.tube }, shown: layers.vertical,
      text: `altitude tube ±${vocabulary.altitudeToleranceM} m`, title: "an altitude word's tube, between its lower and upper edge" },
    { key: "outside", swatch: { kind: "line", colour: TRAINING_OUTSIDE_COLOR }, shown: true, text: "outside a check",
      title: "red: the labeller's check failed — rows outside a heading band or a tube, a capture turn" },
    { key: "selected", swatch: { kind: "line", colour: TRAINING_WORD_COLOR }, shown: true, text: "selected word",
      title: "yellow: the selected word — its envelope and the rows it is in force; the rest fades" },
    { key: "executor", swatch: { kind: "line", colour: TRAINING_EXECUTOR_COLOR }, shown: executorTrack, text: "executor replay",
      title: "teal: the executor's flown track (dashed on the ground), the truth sentence flown from row 0; red on its " +
        "ground trace: its rows outside the heading word it was told" },
    { key: "autopilot", swatch: { kind: "line", colour: autopilotColour ?? TRAINING_TRACE_COLOR }, shown: autopilotColour !== null,
      text: "autopilot segment",
      title: "the picked word's segment, flown live by the executor (dashed on the ground): blue inside the word's " +
        "envelope, red outside it" },
  ];

  return (
    <aside className="training-legend" aria-label="What the 3D scene shows">
      <button type="button" className="training-legend-toggle" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        {open ? "Legend ▾" : "Legend ▸"}
      </button>
      {open ? (
        <ul>
          {rows.filter((row) => row.shown).map((row) => (
            <li key={row.key} title={row.title}>
              <SwatchIcon swatch={row.swatch} />
              <span>{row.text}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </aside>
  );
}
