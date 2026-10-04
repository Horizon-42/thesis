/**
 * PanelParts.tsx
 * --------------
 * What the Training panel's session shares with the panel — the Draw switches, a readout's one line opening the details
 * page, and the details page's control, which the panel owns (its header ⓘ opens it on the session's overview).
 */

import type { MouseEvent } from "react";
import { useApp, type TrainingLayers } from "../../context/AppContext";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_TUBE_COLOR,
} from "../../utils/trainingWordColors";

/** The details page while it is open — its section and the control that opened it (the focus goes back there) — and how
 *  the panel's controls open, turn and close it. */
export interface DetailsPage {
  shown: { section: string; opener: HTMLElement } | null;
  open: (section: string) => (event: MouseEvent<HTMLElement>) => void;
  show: (section: string) => void;
  close: () => void;
}

/** The Draw switches, in drawing order: a short name in its own colour, what it shows in its tooltip. */
export const LAYER_SWITCHES: Array<{ layer: keyof TrainingLayers; colour: string; text: string; title: string }> = [
  { layer: "headingBands", colour: TRAINING_HEADING_BAND_COLOR, text: "Heading bands",
    title: "each heading word's band over the rows it is judged on, and those rows outside it in red" },
  { layer: "vertical", colour: TRAINING_TUBE_COLOR, text: "Altitude tubes + speed bands",
    title: "each altitude word's tube, and on the speed chart each speed word's band" },
  { layer: "candidates", colour: TRAINING_CANDIDATE_COLOR, text: "Other candidate runways",
    title: "every runway the runway word can point at; the one the flight lands on is always drawn" },
];

/** The envelopes' switches (the Draw box's first lines): each in its colour, its reading in its tooltip. */
export function LayerSwitches() {
  const { trainingLayers, setTrainingLayer } = useApp();
  return (
    <>
      {LAYER_SWITCHES.map(({ layer, colour, text, title }) => (
        <label key={layer} style={{ color: colour }} title={title}>
          <input type="checkbox" checked={trainingLayers[layer]} onChange={(event) => setTrainingLayer(layer, event.target.checked)} />
          {text}
        </label>
      ))}
    </>
  );
}

/** One readout in the panel: its name and its conclusion on one line (in full in its tooltip); it opens the details
 *  page on its section. */
export function DetailsLink({ name, summary, onOpen }: { name: string; summary: string; onOpen: (event: MouseEvent<HTMLElement>) => void }) {
  return (
    <li>
      <button type="button" className="training-details-link" aria-haspopup="dialog" title={`${name}: ${summary}`} onClick={onOpen}>
        <span className="training-details-link-name">{name}</span>
        <span className="training-details-link-summary">{summary}</span>
        <span className="training-details-link-open" aria-hidden="true">›</span>
      </button>
    </li>
  );
}
