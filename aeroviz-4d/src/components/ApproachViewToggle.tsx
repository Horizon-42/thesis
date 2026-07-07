/**
 * ApproachViewToggle.tsx
 * -----------------------
 * A small toggle button that opens/closes the 2D approach PROFILE page
 * (ApproachViewPanel) for a given runway. It is the one source for this
 * control, used from every dock that needs it — the Observe dock (governing the global
 * landing runway), the Optimize dock (governing the target runway), and the bottom-right
 * Procedures panel (per runway group).
 *
 * The approach view renders only when `isApproachViewOpen && selectedRunway`, so opening
 * FOCUSES the global selection on the governed runway (matching the Procedures panel's
 * long-standing focus-on-open convention) — this is what makes the target's profile show
 * even in unconstrained Optimize, where `selectedRunway` may be null or a different runway.
 * The button is disabled when it governs no runway (the page can't render without one).
 * `open` additionally requires a non-null global selection: `runwayMatchesSelection`
 * treats null as match-all (filter semantics), but with no selection the approach view
 * renders nothing, so no toggle may claim it as open.
 *
 * `borrowSelection` (unconstrained Optimize's Target-State toggle): opening from a dock
 * whose governed runway is INDEPENDENT of the global selection would otherwise permanently
 * clobber the user's landing-runway scoping (leaking the Optimize target into Observe).
 * With it set, the toggle saves the pre-open {selectedRunway, isApproachViewOpen} once —
 * only when opening actually CHANGES the selection — and restores it when the approach view is
 * closed from this toggle or the toggle unmounts (leaving the dock). Constrained Optimize
 * does NOT set it: there useForcedProcedureDisplay owns the runway + approach view state.
 */

import { useEffect, useRef } from "react";
import { useApp } from "../context/AppContext";
import { bareRunwayIdent, runwayMatchesSelection } from "../utils/runwayIdent";

interface ApproachViewToggleProps {
  /** The runway this toggle governs (either spelling), or `null` when none is available. */
  runwayIdent: string | null;
  /** Extra classes appended to the shared button class (e.g. dock-specific layout). */
  className?: string;
  /** Return the borrowed global selection (and prior approach view state) on close/unmount. */
  borrowSelection?: boolean;
}

export default function ApproachViewToggle({
  runwayIdent,
  className,
  borrowSelection = false,
}: ApproachViewToggleProps) {
  const { selectedRunway, setSelectedRunway, isApproachViewOpen, setApproachViewOpen } = useApp();

  const open =
    isApproachViewOpen &&
    runwayIdent !== null &&
    selectedRunway !== null &&
    runwayMatchesSelection(selectedRunway, runwayIdent);

  // The pre-borrow display (null = nothing borrowed). Restore also runs on unmount, so it
  // lives in a per-render ref to keep the unmount effect dependency-free.
  const savedRef = useRef<{ selectedRunway: string | null; approachViewOpen: boolean } | null>(null);
  const restore = () => {
    const saved = savedRef.current;
    if (saved === null) return;
    savedRef.current = null;
    setSelectedRunway(saved.selectedRunway);
    setApproachViewOpen(saved.approachViewOpen);
  };
  const restoreRef = useRef(restore);
  restoreRef.current = restore;
  useEffect(() => () => restoreRef.current(), []);

  const toggle = () => {
    if (runwayIdent === null) return;
    if (open) {
      if (borrowSelection && savedRef.current !== null) restore();
      else setApproachViewOpen(false);
      return;
    }
    const alreadyFocused = selectedRunway !== null && runwayMatchesSelection(selectedRunway, runwayIdent);
    if (borrowSelection && savedRef.current === null && !alreadyFocused) {
      savedRef.current = { selectedRunway, approachViewOpen: isApproachViewOpen };
    }
    setSelectedRunway(bareRunwayIdent(runwayIdent));
    setApproachViewOpen(true);
  };

  return (
    <button
      type="button"
      className={`approach-view-toggle-button${open ? " active" : ""}${className ? ` ${className}` : ""}`}
      aria-pressed={open}
      disabled={runwayIdent === null}
      title={
        runwayIdent === null
          ? "Select a landing runway to view its approach view"
          : "Open the runway's 2D approach view (side + plan)"
      }
      onClick={toggle}
    >
      {open ? "Hide view" : "View"}
    </button>
  );
}
