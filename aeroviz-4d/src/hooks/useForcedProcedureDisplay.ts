/**
 * useForcedProcedureDisplay.ts
 * ----------------------------
 * While a caller's condition holds, DRIVE the shared procedure display to a runway's
 * approach — open the Procedures panel, enable the `procedures` geometry layer, and
 * (optionally) scope the global landing runway — so you see the corridors / glidepath /
 * step-down floors that a constrained solve or comparison enforces. The user's PRIOR
 * display is saved once and RESTORED when the condition ends (or the caller unmounts,
 * e.g. leaving the task's dock), so in every other context procedures obey only their
 * own switch. It is reactive (not a one-shot), so entering a caller already in the
 * forced state drives it on mount too.
 *
 * The optimize-vs-observe asymmetry is captured in the single nullable `forceRunway`:
 *   • Optimize passes the target runway (INDEPENDENT of the global selection) → the hook
 *     OWNS selectedRunway: it saves, forces, and restores it. Owning the runway also
 *     means owning the 2D approach view keyed on it (isApproachViewOpen): reverting the
 *     runway under an open profile would silently retarget (or blank) that page, so the
 *     profile open-state is saved and restored together with the runway.
 *   • Observe passes `null` (the scoped runway IS the user-owned global selectedRunway)
 *     → the hook never reads or writes selectedRunway or the approach view state, so it can
 *     neither fight the user changing the top-bar runway nor revert it on exit; only the
 *     panel + layer are driven.
 *
 * Extracted from PilotPanel's original inline effect so both docks share one save/restore.
 * The two dock branches are mutually exclusive (WorkbenchLeftDock), so at most one hook
 * instance is ever active at a time — but a task switch between two FORCING docks (e.g. a
 * constrained Optimize → a constrained-comparison Observe) hands off across an unmount/mount
 * seam: React runs the outgoing dock's cleanup (its restore `setState`) and the incoming
 * dock's mount effect in the SAME passive flush, so the incoming effect would read the
 * outgoing dock's STILL-FORCED display before the restore commits — capturing a polluted
 * baseline and skipping the (apparently-already-done) open. The drive is therefore gated
 * behind a one-render `ready` flag: the incoming hook does nothing on its first commit and
 * only saves+drives on the next, by which point the sibling's restore (batched with the
 * `ready` flip in React 18) has committed, so the captured baseline is the user's REAL
 * display and the open re-fires correctly. The flag flips only when a drive is actually
 * pending, so an inactive mount never pays the extra render.
 */

import { useEffect, useRef, useState } from "react";
import { useApp } from "../context/AppContext";

interface ForcedProcedureDisplayOptions {
  /** Drive the display while true; restore the prior display when it turns false. */
  active: boolean;
  /**
   * The (already-BARED) runway ident to scope selectedRunway to while active, or `null`
   * to leave selectedRunway untouched (the caller relies on the global selection already
   * being the runway). See the module note on the optimize-vs-observe asymmetry.
   */
  forceRunway: string | null;
}

/** The user's display as it stands this render — what a save snapshots, what restore returns to. */
interface DisplaySnapshot {
  proceduresOpen: boolean;
  layerOn: boolean;
  selectedRunway: string | null;
  approachViewOpen: boolean;
}

interface SavedDisplay extends DisplaySnapshot {
  /** Whether this snapshot owns selectedRunway (and the approach view keyed on it) — baked
   *  in at save time so restore stays correct if `forceRunway` flips to null while active
   *  (the hook DID force the runway; a live read would skip the restore and leak it). */
  ownsRunway: boolean;
}

export function useForcedProcedureDisplay({ active, forceRunway }: ForcedProcedureDisplayOptions): void {
  const {
    selectedRunway,
    setSelectedRunway,
    proceduresOpen,
    setProceduresOpen,
    layers,
    toggleLayer,
    isApproachViewOpen,
    setApproachViewOpen,
  } = useApp();

  // Live snapshot of the display, read (not depended on) when saving/restoring — a mid-force
  // manual toggle therefore does NOT re-fire the drive effect (it is discarded on restore,
  // exactly the prior behavior).
  const snapshot: DisplaySnapshot = {
    proceduresOpen,
    layerOn: layers.procedures,
    selectedRunway,
    approachViewOpen: isApproachViewOpen,
  };
  const liveRef = useRef(snapshot);
  liveRef.current = snapshot;

  // The user's pre-force display, held while the caller drives it (null = not driving).
  const savedRef = useRef<SavedDisplay | null>(null);

  // Restore runs from the drive effect's "condition ended" branch AND on unmount; keep it
  // in a ref so the unmount effect stays dependency-free (and never restores stale setters).
  const restore = () => {
    const saved = savedRef.current;
    if (saved === null) return;
    savedRef.current = null;
    if (saved.ownsRunway) {
      setSelectedRunway(saved.selectedRunway);
      setApproachViewOpen(saved.approachViewOpen);
    }
    setProceduresOpen(saved.proceduresOpen);
    if (liveRef.current.layerOn !== saved.layerOn) toggleLayer("procedures");
  };
  const restoreRef = useRef(restore);
  restoreRef.current = restore;

  // Defer the first drive by one commit so a sibling dock's unmount-restore (batched with
  // this flip) settles before we read the baseline — see the module note on the handoff seam.
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (active && !ready) setReady(true);
  }, [active, ready]);

  useEffect(() => {
    if (!ready) return;
    if (active) {
      if (savedRef.current === null) {
        savedRef.current = { ...liveRef.current, ownsRunway: forceRunway !== null };
      }
      if (forceRunway !== null && liveRef.current.selectedRunway !== forceRunway) {
        setSelectedRunway(forceRunway);
      }
      if (!liveRef.current.proceduresOpen) setProceduresOpen(true);
      if (!liveRef.current.layerOn) toggleLayer("procedures");
    } else {
      restoreRef.current();
    }
  }, [ready, active, forceRunway, setSelectedRunway, setProceduresOpen, toggleLayer]);

  useEffect(() => () => restoreRef.current(), []);
}
