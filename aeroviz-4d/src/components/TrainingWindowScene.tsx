/**
 * TrainingWindowScene.tsx
 * -----------------------
 * The 3D scene of a window set beyond stage A's (`useTrainingWindowLayer`) as a LEAF that renders nothing, beside
 * `TrainingScene`: it reads the Training selection and cursor, which change with the window on screen and every chart
 * hover, and a hook in the app shell would re-render the whole workbench.
 */

import useTrainingWindowLayer from "../hooks/useTrainingWindowLayer";

export default function TrainingWindowScene() {
  useTrainingWindowLayer();
  return null;
}
