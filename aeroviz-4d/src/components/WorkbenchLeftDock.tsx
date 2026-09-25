/**
 * WorkbenchLeftDock.tsx
 * ---------------------
 * The left working dock. It shows only the controls for the active task, switching
 * on the global workbench `mode` (the four mutually-exclusive tasks):
 *   • observe                  → trajectory playback/options + the flight list
 *   • training                 → the TrainingPanel (stage B's intermediate results)
 *   • fly / optimize / compare → the PilotPanel, driven in the matching sub-mode
 *
 * Procedures is NOT a task — the procedure panel is rendered separately (gated on
 * `proceduresOpen`) so it can coexist with whichever task is active.
 *
 * TRAINING KEEPS ITS SESSION: from the first time it is opened, the TrainingPanel stays mounted — hidden in the other
 * tasks — so leaving Training and coming back finds the same set, flight, word and live answer, and downloads nothing
 * again (the index, the sample — KRDU's is 6 MB — and the overlays). Its views draw only in Training (the sentence bar
 * and the 3D layers read `mode`).
 */

import { useState, type ReactNode } from "react";
import { useApp, type WorkbenchMode } from "../context/AppContext";
import ControlPanel from "./ControlPanel";
import type {
  ObservedEvaluationSummary,
  ObservedVerdicts,
} from "../data/observedTracks";
import FlightTable from "./FlightTable";
import EvaluationSummary from "./EvaluationSummary";
import PilotPanel from "./PilotPanel";
import TrainingPanel from "./TrainingPanel";
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
  const [trainingOpened, setTrainingOpened] = useState<boolean>(mode === "training");
  if (mode === "training" && !trainingOpened) setTrainingOpened(true);

  let task: ReactNode = null;
  if (mode === "fly" || mode === "optimize" || mode === "compare") {
    task = (
      <PilotPanel
        mode={MODE_TO_PILOT[mode]}
        onRequestMode={(next) => setMode(PILOT_TO_MODE[next])}
      />
    );
  } else if (mode === "observe") {
    task = (
      <>
        <ControlPanel observedVerdicts={observedVerdicts} />
        <FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />
        <EvaluationSummary observedEvaluation={observedEvaluation} />
      </>
    );
  }

  return (
    <div className="workbench-left-dock">
      {task}
      {trainingOpened ? <TrainingPanel key="training" hidden={mode !== "training"} /> : null}
    </div>
  );
}
