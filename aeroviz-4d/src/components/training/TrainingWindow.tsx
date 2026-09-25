/**
 * TrainingWindow.tsx
 * ------------------
 * The Training views' floating window (the read-back check, the prior's predictions): a dialog over the scene, not
 * modal — the sentence bar and the flight list stay usable behind it, because reading one against the other is the
 * point. Its header names it, carries a few chips and the shared cursor, and closes it; Escape closes it too, from the
 * moment it opens (it takes the focus).
 *
 * It renders through a PORTAL into `document.body` (AV7): `.flight-ops-panel` carries a `backdrop-filter`, which would
 * make a `position: fixed` descendant position against it.
 */

import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { formatSeconds } from "../../data/trainingSample";

export interface TrainingWindowProps {
  /** Its name: the dialog's label and the header's title. */
  title: string;
  /** The close button's accessible name. */
  closeLabel: string;
  /** Short facts beside the title. */
  chips: ReactNode;
  cursorS: number;
  cursorRow: number;
  onClose: () => void;
  className?: string;
  children: ReactNode;
}

export default function TrainingWindow({
  title, closeLabel, chips, cursorS, cursorRow, onClose, className, children,
}: TrainingWindowProps) {
  const dialog = useRef<HTMLDivElement>(null);
  useEffect(() => {
    dialog.current?.focus();
  }, []);
  return createPortal(
    <div className="training-readback-backdrop">
      <div ref={dialog} className={`training-readback-window${className ? ` ${className}` : ""}`} role="dialog"
        aria-label={title} aria-modal="false" tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") onClose();
        }}>
        <header className="training-readback-head">
          <strong>{title}</strong>
          {chips}
          <span className="training-readback-cursor">t = {formatSeconds(cursorS)} s · step {cursorRow}</span>
          <button type="button" onClick={onClose} aria-label={closeLabel}>×</button>
        </header>
        {children}
      </div>
    </div>,
    document.body,
  );
}
