/**
 * TrainingDetails.tsx
 * -------------------
 * The Training details page: its two sections (frontend §3 item 3, D159) — the set and the experiment, and the models'
 * statistics — on a page of its own over the scene, wide enough for its table. The panel lists each readout as one line (its conclusion) and opens the page
 * on that section; its header's ⓘ opens it on the first.
 *
 * A modal dialog (the readouts are read, not compared against the scene): a tab per section down the left, the chosen
 * section's body on the right, scrolling on its own. While it is open everything else on the page is `inert` — the Tab
 * key stays inside it. A section with nothing to show stays listed, disabled, and says why (none published, switched
 * off, loading, cannot be read). Escape (wherever the focus is), the ×, or a click on the backdrop closes it, and the focus
 * goes back to the control that opened it (passed in: Safari does not focus a button it clicks).
 *
 * It renders through a PORTAL into `document.body` (AV7): `.flight-ops-panel` carries a `backdrop-filter`, which would
 * make a `position: fixed` descendant position against it.
 */

import { useEffect, useId, useRef, type KeyboardEvent, type ReactElement } from "react";
import { createPortal } from "react-dom";

/** A section: its body, or — when it has nothing to show — why not. */
export type TrainingDetailsSection = { id: string; title: string } & (
  | { body: ReactElement; absent?: undefined }
  | { body: null; absent: string }
);

export interface TrainingDetailsProps {
  /** What the page is about, beside its title: the airport, the set. */
  context: string;
  /** The FIRST section always has a body: it is where a section that loses its body gives way to. */
  sections: TrainingDetailsSection[];
  /** The section shown. */
  sectionId: string;
  onSection: (id: string) => void;
  onClose: () => void;
  /** Where the focus goes back to on closing. */
  opener: HTMLElement;
}

export default function TrainingDetails({ context, sections, sectionId, onSection, onClose, opener }: TrainingDetailsProps) {
  const ids = useId();
  const tabId = (id: string) => `${ids}-tab-${id}`;
  const panelId = `${ids}-panel`;
  const backdrop = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const main = useRef<HTMLDivElement>(null);

  // modal: the rest of the page is inert while it is open, and the focus comes back to the opener after
  useEffect(() => {
    const behind = [...document.body.children].filter((element) => element !== backdrop.current && !element.hasAttribute("inert"));
    behind.forEach((element) => element.setAttribute("inert", ""));
    dialog.current!.focus();
    return () => {
      behind.forEach((element) => element.removeAttribute("inert"));
      opener.focus();
    };
  }, [opener]);
  // Escape closes it wherever the focus is — it is modal, so nothing behind it wants the key
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const asked = sections.find((section) => section.id === sectionId);
  // a section whose body went away while open (its overlay reloading) gives way to the first
  const shown = asked !== undefined && asked.body !== null ? asked : sections[0];
  useEffect(() => {
    if (shown.id !== sectionId) onSection(shown.id);
  }, [shown.id, sectionId, onSection]);
  // a section opens at its top
  useEffect(() => {
    main.current!.scrollTop = 0;
  }, [shown.id]);

  const readable = sections.filter((section) => section.body !== null);
  // Up / Down move between the sections that have a body, as a vertical tab list does.
  const onTabKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const at = readable.findIndex((section) => section.id === shown.id);
    const next = readable[(at + (event.key === "ArrowDown" ? 1 : readable.length - 1)) % readable.length];
    onSection(next.id);
    document.getElementById(tabId(next.id))!.focus();
  };

  return createPortal(
    <div ref={backdrop} className="training-details-backdrop" onMouseDown={(event) => {
      if (event.button === 0 && event.target === event.currentTarget) onClose();
    }}>
      <div ref={dialog} className="training-details" role="dialog" aria-modal="true" aria-label="Training details" tabIndex={-1}>
        <header className="training-details-head">
          <h2>Training details</h2>
          <span className="training-details-context">{context}</span>
          <button type="button" className="training-details-close" onClick={onClose} aria-label="Close the details">×</button>
        </header>
        <div className="training-details-body">
          <div className="training-details-tabs" role="tablist" aria-orientation="vertical" aria-label="Sections">
            {sections.map((section) => {
              const selected = section.id === shown.id;
              return (
                <button key={section.id} id={tabId(section.id)} type="button" role="tab"
                  aria-selected={selected} aria-controls={panelId} tabIndex={selected ? 0 : -1}
                  disabled={section.body === null} title={section.absent}
                  onClick={() => onSection(section.id)} onKeyDown={onTabKey}>
                  <span className="training-details-tab-title">{section.title}</span>
                  {section.body === null ? <span className="training-details-tab-absent">{section.absent}</span> : null}
                </button>
              );
            })}
          </div>
          <div ref={main} id={panelId} className="training-details-main" role="tabpanel"
            aria-labelledby={tabId(shown.id)} tabIndex={0}>
            <h3>{shown.title}</h3>
            {shown.body}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
