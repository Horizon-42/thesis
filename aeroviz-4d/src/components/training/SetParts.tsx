/**
 * SetParts.tsx
 * ------------
 * The parts of the left panel every stage's session shows the same way (outline §6.2 items 1, 3, 5): the set chooser —
 * the choice of set and, under it, the set's own line of the intent registry and a SMOKE tag, which opens the details page
 * on "The set and the experiment"; the list of items with the same four columns (the callsign, the runway, the stratum or
 * the window's kind, the outcomes of the item's own sentences); and that first section's body.
 */

import type { MouseEvent, ReactNode } from "react";
import { ExperimentIntentBlock } from "../ExperimentDetails";
import { INTENT_REGISTRY_PATH } from "../ExperimentDetails";
import type { TrainingSetIntentState } from "../../data/trainingSetIntent";
import { NotesList } from "./NotesToggle";

/** A set in the chooser: its id, its count of items, its title, whether its source says smoke. */
export interface SetChoice {
  id: string;
  count: string;
  title: string;
  smoke: boolean;
}

/** The set's own intent line, as one line (the problem by name when the registry has none, or several). */
function intentLine(intent: TrainingSetIntentState): string {
  if (intent.status === "loading") return "Reading the intent …";
  if (intent.status === "absent") return intent.problem;
  return intent.intent.run;
}

export function SetChooser({ sets, setId, onChange, intent, onOpenExperiment }: {
  sets: SetChoice[]; setId: string | null; onChange: (id: string) => void; intent: TrainingSetIntentState;
  onOpenExperiment: (event: MouseEvent<HTMLElement>) => void;
}) {
  const chosen = sets.find((item) => item.id === setId) ?? null;
  return (
    <>
      <label className="training-field" title={chosen?.title}>
        <span>Set</span>
        <select value={setId ?? ""} onChange={(event) => onChange(event.target.value)} disabled={sets.length < 2}>
          {sets.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.count}</option>)}
        </select>
      </label>
      {chosen === null ? null : (
        <button type="button" className={`training-set-intent${intent.status === "absent" ? " is-absent" : ""}`} aria-haspopup="dialog"
          title={`${chosen.title} — the set and the experiment (details)`} onClick={onOpenExperiment}>
          {chosen.smoke ? <span className="training-set-smoke" title="the set's source says smoke: not a result">SMOKE</span> : null}
          <span className="training-set-intent-line">{intentLine(intent)}</span>
          <span className="training-details-link-open" aria-hidden="true">›</span>
        </button>
      )}
    </>
  );
}

/** One item of the list: the four columns, the outcomes of its own sentences in the last column's tooltip. */
export interface ItemChoice {
  key: string;
  callsign: string;
  runway: string;
  runwayTitle?: string;
  stratum: string;
  landed: number;
  of: number;
  /** After the count (stage C: the other aircraft). */
  more?: string;
  outcomes: string;
  title: string;
}

export function ItemList({ label, items, active, onSelect }: {
  label: string; items: ItemChoice[]; active: string | null; onSelect: (key: string) => void;
}) {
  return (
    <ul className="training-flight-list" aria-label={label}>
      {items.map((item) => (
        <li key={item.key}>
          <button type="button" className={item.key === active ? "active" : undefined} title={item.title} onClick={() => onSelect(item.key)}>
            <span className="training-flight-callsign">{item.callsign}</span>
            <span className="training-flight-runway" title={item.runwayTitle}>{item.runway}</span>
            <span className="training-flight-stratum">{item.stratum}</span>
            <span className="training-flight-executor" title={item.outcomes}>
              {item.landed}/{item.of} landed{item.more ? ` · ${item.more}` : ""}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

/** "The set and the experiment" (outline §6.2 item 3): the campaign's title, intent and design, the set's own line — or the
 *  problem by name — then the set's facts (what it was made from, its items and how they were drawn, a claimed readout). */
export function ExperimentSection({ setId, intent, facts }: {
  setId: string; intent: TrainingSetIntentState; facts: Array<{ key: string; name: ReactNode; text: string }>;
}) {
  return (
    <>
      {intent.status === "ready" ? (
        <>
          <p className="training-details-lede"><strong>{intent.intent.groupTitle}</strong> <code>{intent.intent.campaign}</code></p>
          <ExperimentIntentBlock intent={intent.intent} runTag="This set" />
        </>
      ) : (
        <p className="experiment-details-missing">
          {intent.status === "loading" ? "Reading the intent …" : intent.problem} The set's intent is its line in{" "}
          <code>{INTENT_REGISTRY_PATH}</code> (the run key <code>{setId}</code>).
        </p>
      )}
      <h4 className="training-details-subhead">The set</h4>
      <NotesList items={[{ key: "id", name: <code>{setId}</code>, text: "the set's id" }, ...facts]} />
    </>
  );
}
