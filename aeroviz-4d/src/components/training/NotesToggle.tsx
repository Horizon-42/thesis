/**
 * NotesToggle.tsx
 * ---------------
 * An ⓘ that shows, as the page's own text, what a Training view otherwise says only in tooltips — the full reading of a
 * chip, a swatch, a legend row — so it is reachable from the keyboard and read by a screen reader, not only by hovering.
 */

import { useState, type ReactNode } from "react";

/** The readings as a list: each name, and what it is. */
export function NotesList({ items }: { items: Array<{ key: string; name: ReactNode; text: string }> }) {
  return (
    <dl className="training-notes-list">
      {items.map((item) => (
        <div key={item.key}>
          <dt>{item.name}</dt>
          <dd>{item.text}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function NotesToggle({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState<boolean>(false);
  return (
    <>
      <button type="button" className="training-notes-toggle" aria-expanded={open} aria-label={label} title={label}
        onClick={() => setOpen((value) => !value)}>
        ⓘ
      </button>
      {open ? <div className="training-notes" role="note" aria-label={label}>{children}</div> : null}
    </>
  );
}
