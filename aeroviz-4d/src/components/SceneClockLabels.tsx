/**
 * SceneClockLabels.tsx
 * --------------------
 * While a traffic scene is drawn (a job's, or a published M1/M2 category), Cesium's animation widget and timeline show the
 * scene's REAL UTC time (`showSceneTimeOnClockWidgets`), not the display epoch the CZML sits on; they get their own
 * formatters back when the scene goes. Draws nothing. The same condition as the "Scene time (UTC)" readout.
 */

import { useEffect } from "react";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { showSceneTimeOnClockWidgets } from "../utils/sceneTime";

export default function SceneClockLabels() {
  const { viewer, sceneTime, mode, layers } = useApp();
  // A job's scene is Optimize's own; a published category is drawn only while the Trajectories layer is on.
  const drawn = mode === "optimize" || layers.trajectories;

  useEffect(() => {
    if (!viewer || !isCesiumViewerUsable(viewer) || !sceneTime || !drawn) return undefined;
    return showSceneTimeOnClockWidgets(viewer, sceneTime);
  }, [viewer, sceneTime, drawn]);

  return null;
}
