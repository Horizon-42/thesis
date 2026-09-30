/**
 * TrainingLegend.tsx
 * ------------------
 * What each colour in the Training 3D scene is — one short line per thing drawn, only for what is on, the full reading
 * in each line's tooltip. Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §4.4.
 *
 * It sits above the sentence bar, in the scene's lower right corner, folded until it is opened. The read-back window
 * has its own swatches; the colours are the same (`trainingWordColors.ts`). When a model's own sentence is read, its
 * flown tracks are listed too — the truth's track is drawn whichever is read.
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
  TRAINING_LOSS_COLOR,
  TRAINING_OTHER_AIRCRAFT_ALPHA,
  TRAINING_OTHER_AIRCRAFT_COLOR,
  TRAINING_ON_SCREEN_COLOR,
  TRAINING_OTHER_SAMPLE_ALPHA,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import { SwatchIcon, type Swatch } from "./training/chartKit";
import NotesToggle, { NotesList } from "./training/NotesToggle";

export default function TrainingLegend({ layers, vocabulary, executorTrack, autopilotColour, model, traffic }: {
  layers: TrainingLayers;
  vocabulary: TrainingVocabulary;
  /** The executor's flown track is drawn (its overlay is on and the flight was flown). */
  executorTrack: boolean;
  /** The colour the live executor's segment is drawn in (`autopilotColour`), or null when none is drawn. */
  autopilotColour: string | null;
  /** The model whose own sentence is read (its name, colour, how many samples it said and whether from an augmented
   *  start), or null for the truth. */
  model: { label: string; colour: string; samples: number; moved: boolean } | null;
  /** The multi-aircraft window the aircraft on screen is in — how many aircraft it commands, drawn in `colour`, the
   *  reading's (`trainingWindowReadingColour`) — or null for a flight of its own. */
  traffic: { colour: string; commanded: number } | null;
}) {
  const [open, setOpen] = useState<boolean>(false);
  const rows: Array<{ key: string; swatch: Swatch; text: string; title: string; shown: boolean }> = [
    { key: "track", swatch: { kind: "line", colour: TRAINING_TRACE_COLOR }, shown: true, text: "the observed track (truth)",
      title: "the observed track, and — faint on the ground — its ground trace; the envelopes are its labelled sentence's" },
    ...(model === null ? [] : [
      { key: "model", swatch: { kind: "line", colour: model.colour } satisfies Swatch, shown: true, text: `${model.label}: the sample read`,
        title: `${model.label}'s own sentence, the sample read, as the executor flew it from its first predicted step ` +
          "(dashed on the ground); its end is named with how the flight ended" },
      { key: "model-others", swatch: { kind: "line", colour: model.colour, opacity: TRAINING_OTHER_SAMPLE_ALPHA } satisfies Swatch,
        shown: model.samples > 1,
        text: `${model.label}: its other samples`, title: `the other ${model.samples - 1} sentences ${model.label} said over this flight, flown the same way` },
      { key: "model-moved", swatch: { kind: "line", colour: model.colour, dash: "4 3" } satisfies Swatch, shown: model.moved,
        text: "moved observed steps", title: `the observed steps ${model.label} read before it spoke, moved like its augmented ` +
          "start (rotated about the airport, raised, sped up); its samples fly on from their end — the truth is where they were moved from" },
    ]),
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
    ...(traffic === null ? [] : [
      { key: "on-screen", swatch: { kind: "point", colour: "#ffffff", ring: TRAINING_ON_SCREEN_COLOR } satisfies Swatch, shown: true,
        text: "▶ the aircraft on screen", title: "the aircraft the sentence bar reads: its tracks as above, where it is at the " +
          "cursor the aircraft model, white, ringed in the selection's yellow, its callsign and attitude on a yellow chip" },
      { key: "commanded", swatch: { kind: "line", colour: traffic.colour } satisfies Swatch, shown: traffic.commanded > 1,
        text: "commanded aircraft",
        title: "the window's other aircraft the sentence read commands — the model's samples, or the record: its track, " +
          "aircraft model and callsign in the reading's colour" },
      { key: "replayed", swatch: { kind: "line", colour: TRAINING_OTHER_AIRCRAFT_COLOR.replayed,
        opacity: TRAINING_OTHER_AIRCRAFT_ALPHA.replayed } satisfies Swatch, shown: true, text: "replayed arrivals",
        title: "arrivals with a sentence the model does not command, replayed as recorded" },
      { key: "background", swatch: { kind: "line", colour: TRAINING_OTHER_AIRCRAFT_COLOR.background,
        opacity: TRAINING_OTHER_AIRCRAFT_ALPHA.background } satisfies Swatch, shown: true, text: "background arrivals",
        title: "arrivals without a sentence, replayed as recorded" },
      { key: "loss", swatch: { kind: "line", colour: TRAINING_LOSS_COLOR } satisfies Swatch, shown: true, text: "loss of separation",
        title: "a pair under its minimum at the cursor, joined — VISUAL solid, IFR dashed where only IFR has it — with the " +
          "closest it came against its minimum; ✕ where the judge ended an aircraft" },
    ]),
    { key: "aircraft", swatch: { kind: "point", colour: model?.colour ?? TRAINING_TRACE_COLOR, ring: "#000000" } satisfies Swatch,
      shown: true, text: "the aircraft at the cursor",
      title: "the aircraft model where the sentence read has it at the cursor, turned to its exported attitude: heading, the " +
        "path angle as its pitch and its bank (wings level for a flight the dynamics has no airframe for); the angle of " +
        "attack is a reading through a clean-wing lift curve — high on a flapped final — written under it, never drawn" },
    ...(autopilotColour === null ? [] : [{ key: "autopilot", swatch: { kind: "line", colour: autopilotColour } as Swatch,
      shown: true, text: "autopilot segment",
      title: "the picked word's segment, flown live by the executor (dashed on the ground): blue inside the word's " +
        "envelope, red outside it" }]),
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
