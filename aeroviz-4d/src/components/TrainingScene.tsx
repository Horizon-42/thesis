/**
 * TrainingScene.tsx
 * -----------------
 * The Training 3D scene (`useTrainingTrackLayer`, and with it the executor's layers) as a LEAF that renders nothing: it
 * reads the Training cursor, which moves on every hover over a chart, and a hook in the app shell would re-render the
 * whole workbench — both docks, the flight list, the HUD — on every mousemove. Here only this leaf re-renders.
 */

import useTrainingTrackLayer from "../hooks/useTrainingTrackLayer";

export default function TrainingScene() {
  useTrainingTrackLayer();
  return null;
}
