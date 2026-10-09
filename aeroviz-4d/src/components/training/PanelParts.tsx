/**
 * PanelParts.tsx
 * --------------
 * What the Training panel's sessions share with the panel (outline §6.2 item 5) — the Draw box and stage A's switches, a
 * readout's one line opening the details page, and the details page's state (`useDetailsPage`), which the panel owns and
 * gives to every session: its header ⓘ opens it on the session's first section, a readout's line on its own, and the
 * sentence bar — a sibling of the dock — asks for it through `requestTrainingDetails`.
 */

import { useCallback, useEffect, useRef, useState, useSyncExternalStore, type MouseEvent, type ReactNode } from "react";
import { useApp, type TrainingLayers } from "../../context/AppContext";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_TRACE_COLOR,
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

/** The details page's two sections in every stage (frontend §3 item 3, D159). */
export const EXPERIMENT_SECTION = "experiment";
export const STATISTICS_SECTION = "statistics";

let request: { section: string; opener: HTMLElement; seq: number } | null = null;
const requestListeners = new Set<() => void>();

/** Open the panel's details page on ``section`` from outside the dock (the sentence bar's ⓘ notes); the focus goes back to
 *  ``opener`` when it closes. */
export function requestTrainingDetails(section: string, opener: HTMLElement): void {
  request = { section, opener, seq: (request?.seq ?? 0) + 1 };
  requestListeners.forEach((listener) => listener());
}

function useDetailsRequest() {
  return useSyncExternalStore(
    (listener) => {
      requestListeners.add(listener);
      return () => requestListeners.delete(listener);
    },
    () => request,
  );
}

/** The details page's state, owned by the panel and given to every session: closed while the panel is hidden (another
 *  task), opened by the panel's own controls or by `requestTrainingDetails`. */
export function useDetailsPage(hidden: boolean): DetailsPage {
  const [shown, setShown] = useState<{ section: string; opener: HTMLElement } | null>(null);
  const show = useCallback((section: string) => setShown((open) => (open === null ? null : { ...open, section })), []);
  const close = useCallback(() => setShown(null), []);
  const open = useCallback((section: string) => (event: MouseEvent<HTMLElement>) =>
    setShown({ section, opener: event.currentTarget }), []);
  // each request opens the page once: one made before the panel mounted, or answered already, is never opened again (a
  // task switch back to Learning, a remount)
  const asked = useDetailsRequest();
  const answered = useRef(request?.seq ?? 0);
  useEffect(() => {
    if (asked === null || asked.seq <= answered.current) return;
    answered.current = asked.seq;
    if (!hidden) setShown({ section: asked.section, opener: asked.opener });
  }, [asked, hidden]);
  // a panel hidden (another task) closes its page: it does not come back unasked
  useEffect(() => {
    if (hidden) setShown(null);
  }, [hidden]);
  return { shown: hidden ? null : shown, open, show, close };
}

/** The Draw box of a stage: its switches under one legend. */
export function DrawBox({ legend = "Draw", children }: { legend?: string; children: ReactNode }) {
  return (
    <fieldset className="training-layers">
      <legend>{legend}</legend>
      {children}
    </fieldset>
  );
}

/** The Draw switches, in drawing order: a short name in its own colour, what it shows in its tooltip. */
const LAYER_SWITCHES: Array<{ layer: keyof TrainingLayers; colour: string; text: string; title: string }> = [
  { layer: "observed", colour: TRAINING_TRACE_COLOR, text: "Observed track",
    title: "the observed (ground-truth) flight: its track and ground trace, its aircraft at the cursor, its lines on the read-back charts, and the labelled sentence's envelopes, which judge it" },
  { layer: "headingBands", colour: TRAINING_HEADING_BAND_COLOR, text: "Heading bands",
    title: "each heading word's band over the rows it is judged on, and those rows outside it in red" },
  { layer: "vertical", colour: TRAINING_TUBE_COLOR, text: "Altitude tubes",
    title: "each altitude word's tube, a wall with its two edges; in the read-back window's charts also each speed word's band" },
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

/** A readout's one line that opens no page (it follows the cursor: outline §6.2 item 1). */
export function ReadoutLine({ name, summary }: { name: string; summary: string }) {
  return (
    <li className="training-readout-line" title={`${name}: ${summary}`}>
      <span className="training-details-link-name">{name}</span>
      <span className="training-details-link-summary">{summary}</span>
    </li>
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
