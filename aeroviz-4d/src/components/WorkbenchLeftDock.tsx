/**
 * WorkbenchLeftDock.tsx
 * ---------------------
 * The left working dock. It shows only the controls for the active task, switching
 * on the global workbench `mode` (the four mutually-exclusive tasks):
 *   • observe                  → trajectory playback/options + the flight list
 *   • fly / optimize / compare → the PilotPanel, driven in the matching sub-mode
 *
 * Procedures is NOT a task — the procedure panel is rendered separately (gated on
 * `proceduresOpen`) so it can coexist with whichever task is active.
 */

import { useApp, type WorkbenchMode } from "../context/AppContext";
import ControlPanel from "./ControlPanel";
import type {
  ObservedEvaluationSummary,
  ObservedVerdicts,
} from "../data/observedTracks";
import FlightTable from "./FlightTable";
import EvaluationSummary from "./EvaluationSummary";
import PilotPanel from "./PilotPanel";
import type { ObservedFlightSummary } from "../utils/observedFlightSummary";

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
  /** Per-flight duration + initial ground speed for the flight list. */
  flightSummaries: Record<string, ObservedFlightSummary>;
  /** Gate-verdict tally for the complete runway-eligible observed roster. */
  observedVerdicts?: ObservedVerdicts;
  /** Compact aggregate returned with the observed trajectory response. */
  observedEvaluation?: ObservedEvaluationSummary | null;
}

export default function WorkbenchLeftDock({
  flightIds,
  flightSummaries,
  observedVerdicts,
  observedEvaluation,
}: WorkbenchLeftDockProps) {
  const { mode, setMode } = useApp();

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
      <ControlPanel observedVerdicts={observedVerdicts} />
      <FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />
      <EvaluationSummary observedEvaluation={observedEvaluation} />
    </div>
  );
}
