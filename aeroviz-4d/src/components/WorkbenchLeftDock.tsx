/**
 * WorkbenchLeftDock.tsx
 * ---------------------
 * The left working dock. It shows only the controls for the active task, switching
 * on the global workbench `mode` (the four mutually-exclusive tasks):
 *   • evaluation     → trajectory playback/options + the flight list
 *   • training       → the TrainingPanel (stage B's intermediate results)
 *   • fly / optimize → the PilotPanel, which takes the task as its mode
 *
 * Procedures is NOT a task — the procedure panel is rendered separately (gated on
 * `proceduresOpen`) so it can coexist with whichever task is active.
 *
 * TRAINING KEEPS ITS SESSION at the airport it was opened at: the TrainingPanel stays mounted — hidden in the other
 * tasks — so leaving Training and coming back finds the same set, flight, word and live answer, and downloads nothing
 * again (the index, the sample — KRDU's is 6 MB — and the overlays). Another airport opened in another task drops it
 * (the panel is unmounted: a hidden panel would otherwise download each airport's Training files in the background);
 * the next visit to Training opens that airport's afresh. Its views draw only in Training (the sentence bar and the 3D
 * layers read `mode`).
 */

import { useState, type ReactNode } from "react";
import { useApp } from "../context/AppContext";
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

interface WorkbenchLeftDockProps {
  /** Observed-flight ids for the Evaluation-mode flight list. */
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
  const { mode, activeAirportCode } = useApp();
  // the airport whose Training session is kept: taken on entering Training, dropped when another airport is opened
  const [trainingAirport, setTrainingAirport] = useState<string | null>(null);
  if (mode === "training" && trainingAirport !== activeAirportCode) setTrainingAirport(activeAirportCode);
  if (mode !== "training" && trainingAirport !== null && trainingAirport !== activeAirportCode) setTrainingAirport(null);

  let task: ReactNode = null;
  if (mode === "fly" || mode === "optimize") {
    task = <PilotPanel mode={mode} />;
  } else if (mode === "evaluation") {
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
      {trainingAirport !== null ? <TrainingPanel key={`training:${trainingAirport}`} hidden={mode !== "training"} /> : null}
    </div>
  );
}
