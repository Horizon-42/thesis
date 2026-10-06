/**
 * TrainingPriorScene.tsx
 * ----------------------
 * The 3D scene of a prior set beyond stage A's (`useTrainingProcedureLayer`) as a LEAF that renders nothing, beside
 * `TrainingScene`: it reads the Training selection, which changes with the sentence on screen, and a hook in the app shell
 * would re-render the whole workbench.
 */

import useTrainingProcedureLayer from "../hooks/useTrainingProcedureLayer";

export default function TrainingPriorScene() {
  useTrainingProcedureLayer();
  return null;
}
