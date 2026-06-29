/**
 * WorkbenchLeftDock.tsx
 * ---------------------
 * The left working dock. It shows only the controls for the active task, switching
 * on the global workbench `mode`:
 *   • observe              → trajectory playback/options + the flight list
 *   • procedures           → the RNAV procedure tree (runway-scoped)
 *   • fly / optimize / compare → the PilotPanel, driven in the matching sub-mode
 *
 * This is the seam that unifies the app's two former mode systems into one.
 */

import type { ReactNode } from "react";
import { useApp, type WorkbenchMode } from "../context/AppContext";
import ControlPanel from "./ControlPanel";
import FlightTable from "./FlightTable";
import PilotPanel from "./PilotPanel";

type PilotPanelMode = "pilot" | "trajectory" | "comparison";

const MODE_TO_PILOT: Record<"fly" | "optimize" | "compare", PilotPanelMode> = {
  fly: "pilot",
  optimize: "trajectory",
  compare: "comparison",
};

const PILOT_TO_MODE: Record<PilotPanelMode, WorkbenchMode> = {
  pilot: "fly",
  trajectory: "optimize",
  comparison: "compare",
};

interface WorkbenchLeftDockProps {
  /** Observed-flight ids for the Observe-mode flight list. */
  flightIds: string[];
  /** The RNAV procedure tree, shown in Procedures mode. */
  procedures: ReactNode;
}

export default function WorkbenchLeftDock({ flightIds, procedures }: WorkbenchLeftDockProps) {
  const { mode, setMode } = useApp();

  if (mode === "procedures") {
    return <div className="workbench-left-dock">{procedures}</div>;
  }

  if (mode === "fly" || mode === "optimize" || mode === "compare") {
    return (
      <div className="workbench-left-dock">
        <PilotPanel
          mode={MODE_TO_PILOT[mode]}
          onRequestMode={(next) => setMode(PILOT_TO_MODE[next])}
        />
      </div>
    );
  }

  // observe
  return (
    <div className="workbench-left-dock">
      <ControlPanel />
      <FlightTable flightIds={flightIds} />
    </div>
  );
}
