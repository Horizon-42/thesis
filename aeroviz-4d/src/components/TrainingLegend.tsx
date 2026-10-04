/**
 * TrainingLegend.tsx
 * ------------------
 * What each colour in the Training 3D scene is — one short line per thing drawn, only for what is on, the full reading
 * in each line's tooltip.
 *
 * It sits above the sentence bar, in the scene's lower right corner, folded until it is opened. The read-back window
 * has its own swatches; the colours are the same (`trainingWordColors.ts`).
 */

import { useState } from "react";
import type { TrainingLayers } from "../context/AppContext";
import type { TrainingVocabulary } from "../data/trainingSample";
import {
  TRAINING_CORRECTION_COLOR,
  TRAINING_DECISION_FAIL_COLOR,
  TRAINING_DECISION_PASS_COLOR,
  TRAINING_ENVELOPE_ALPHA,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import { SwatchIcon, type Swatch } from "./training/chartKit";
import NotesToggle, { NotesList } from "./training/NotesToggle";

export default function TrainingLegend({ layers, vocabulary, closed, corrections, autopilotColour }: {
  layers: TrainingLayers;
  vocabulary: TrainingVocabulary;
  /** A closed-loop sentence is read: the flown path and the DA point are drawn. */
  closed: boolean;
  /** The sentence read has correction words. */
  corrections: boolean;
  /** The colour the live executor's segment is drawn in (`autopilotColour`), or null when none is drawn. */
  autopilotColour: string | null;
}) {
  const [open, setOpen] = useState<boolean>(false);
  const rows: Array<{ key: string; swatch: Swatch; text: string; title: string; shown: boolean }> = [
    { key: "track", swatch: { kind: "line", colour: TRAINING_TRACE_COLOR }, shown: true, text: "the observed track",
      title: "the observed track, and — faint on the ground — its ground trace" },
    { key: "flown", swatch: { kind: "line", colour: TRAINING_EXECUTOR_COLOR }, shown: closed, text: "the flown path",
      title: "teal: the closed-loop sentence flown by the executor from the first predicted step (dashed on the ground); the " +
        "envelopes below judge it" },
    { key: "da", swatch: { kind: "point", colour: TRAINING_DECISION_PASS_COLOR, ring: "#000000" }, shown: closed, text: "DA point",
      title: "the decision-altitude check of the flown flight's threshold crossing: green passed, red " +
        `(${TRAINING_DECISION_FAIL_COLOR}) failed; its values are in the label and the sentence bar's chip` },
    { key: "correction", swatch: { kind: "point", colour: TRAINING_CORRECTION_COLOR, ring: "#000000" }, shown: corrections,
      text: "correction word", title: "where, on the flown path, a word the closed-loop reading added is said" },
    { key: "heading", swatch: { kind: "line", colour: TRAINING_HEADING_BAND_COLOR }, shown: layers.headingBands,
      text: "heading word: judged rows",
      title: `a heading word's judged rows, on the ground: from ${vocabulary.headingLeadS} s after it is said, where the track must ` +
        `stay within ±${vocabulary.headingToleranceDeg}° of it` },
    { key: "tube", swatch: { kind: "area", colour: TRAINING_TUBE_COLOR, opacity: TRAINING_ENVELOPE_ALPHA.tube }, shown: layers.vertical,
      text: "altitude tube", title: "an altitude word's tube, between its lower and upper edge (a level above the airport elevation E)" },
    { key: "outside", swatch: { kind: "line", colour: TRAINING_OUTSIDE_COLOR }, shown: true, text: "outside a check",
      title: "red: the judge's check failed — rows outside a heading band, a tube that was not held" },
    { key: "selected", swatch: { kind: "line", colour: TRAINING_WORD_COLOR }, shown: true, text: "selected word",
      title: "yellow: the selected word — its envelope and the rows it is in force; the rest fades" },
    { key: "aircraft", swatch: { kind: "point", colour: TRAINING_TRACE_COLOR, ring: "#000000" }, shown: true,
      text: "the aircraft at the cursor",
      title: "the aircraft model where the observed flight (and, in a closed-loop sentence, the flown path in teal) has it at the " +
        "cursor, turned to its exported attitude: heading, the path angle as its pitch and its bank (wings level for a flight " +
        "the dynamics has no airframe for); the angle of attack is a reading through a clean-wing lift curve — high on a " +
        "flapped final — written under it, never drawn" },
    ...(autopilotColour === null ? [] : [{ key: "autopilot", swatch: { kind: "line", colour: autopilotColour } as Swatch,
      shown: true, text: "autopilot segment",
      title: "the picked word's segment, flown live by the executor (dashed on the ground): blue, or red when the flight was flown " +
        "on to an outcome other than a landing" }]),
  ];

  return (
    <aside className="training-legend" aria-label="What the 3D scene shows">
      <button type="button" className="training-legend-toggle" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        {open ? "Legend ▾" : "Legend ▸"}
      </button>
      {open ? (
        <>
          <ul>
            {rows.filter((row) => row.shown).map((row) => (
              <li key={row.key} title={row.title}>
                <SwatchIcon swatch={row.swatch} />
                <span>{row.text}</span>
              </li>
            ))}
          </ul>
          <NotesToggle label="What each line in 3D is">
            <NotesList items={rows.filter((row) => row.shown).map((row) => ({ key: row.key, name: row.text, text: row.title }))} />
          </NotesToggle>
        </>
      ) : null}
    </aside>
  );
}
