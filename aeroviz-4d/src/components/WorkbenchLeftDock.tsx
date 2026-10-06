/**
 * WorkbenchLeftDock.tsx
 * ---------------------
 * The left working dock. It shows only the controls for the active task, switching
 * on the global workbench `mode` (the four mutually-exclusive tasks):
 *   • evaluation     → trajectory playback/options + the flight list
 *   • training       → the TrainingPanel (stage A: the two-tier model's Training view)
 *   • fly / optimize → the PilotPanel, which takes the task as its mode; Optimize first asks for its mode
 *                      (`OptimizeKindSelect`): the PilotPanel's single aircraft, or a multi-aircraft job (`TrafficJobPanel`,
 *                      the PilotPanel then stays mounted, hidden)
 *
 * Procedures is NOT a task — the procedure panel is rendered separately (gated on
 * `proceduresOpen`) so it can coexist with whichever task is active.
 *
 * TRAINING KEEPS ITS SESSION at the airport it was opened at: the TrainingPanel stays mounted — hidden in the other
 * tasks — so leaving Training and coming back finds the same set, flight, word and live answer, and downloads nothing
 * again (the index and the sample — KRDU's is 6 MB). Another airport opened in another task drops it
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
import OptimizeKindSelect, { type OptimizeKind } from "./OptimizeKindSelect";
import PilotPanel from "./PilotPanel";
import TrafficJobPanel from "./TrafficJobPanel";
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
  const [optimizeKind, setOptimizeKind] = useState<OptimizeKind>("single");
  if (mode === "training" && trainingAirport !== activeAirportCode) setTrainingAirport(activeAirportCode);
  if (mode !== "training" && trainingAirport !== null && trainingAirport !== activeAirportCode) setTrainingAirport(null);

  let task: ReactNode = null;
  if (mode === "fly" || mode === "optimize") {
    // The PilotPanel keeps its place in the tree (the last child) in Fly, Optimize (single) AND Optimize (multi-aircraft,
    // where it stays mounted but hidden, its scene off): a switch between them never loses what was set up in it. The job
    // panel is keyed by mode and airport: another one is another job, and the old panel's running job is cancelled with it.
    const multi = mode === "optimize" && optimizeKind !== "single";
    task = (
      <>
        {mode === "optimize" ? <OptimizeKindSelect value={optimizeKind} onChange={setOptimizeKind} /> : null}
        {multi ? <TrafficJobPanel key={`${optimizeKind}:${activeAirportCode}`} kind={optimizeKind} /> : null}
        <PilotPanel mode={mode} hidden={multi} />
      </>
    );
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
