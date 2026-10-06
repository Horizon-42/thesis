/**
 * SceneTimeReadout.tsx
 * --------------------
 * "Scene time (UTC)": the real time at the viewer clock's current position, while a traffic scene is drawn (a job's, or a
 * published M1/M2 category). The clock itself runs on a synthetic display epoch (`utils/sceneTime.ts`); the comparison layer
 * publishes what that epoch is in UTC (`AppContext.sceneTime`) and this leaf reads the clock — a value that changes every
 * frame stays out of the context and out of the rest of the tree.
 */

import { useEffect, useState } from "react";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { formatSceneTime, sceneRealTimeMs } from "../utils/sceneTime";

/** The readout is redrawn at most this often (ms of wall time): a clock at 60× moves a minute per second. */
const REDRAW_MS = 250;

export default function SceneTimeReadout() {
  const { viewer, sceneTime, mode, layers } = useApp();
  const [text, setText] = useState<string | null>(null);
  // A job's scene is Optimize's own; a published category is drawn only while the Trajectories layer is on.
  const drawn = mode === "optimize" || layers.trajectories;

  useEffect(() => {
    if (!viewer || !isCesiumViewerUsable(viewer) || !sceneTime || !drawn) {
      setText(null);
      return undefined;
    }
    const show = () => setText(formatSceneTime(sceneRealTimeMs(sceneTime, viewer.clock.currentTime)));
    show();
    let last = performance.now();
    const remove = viewer.clock.onTick.addEventListener(() => {
      const now = performance.now();
      if (now - last < REDRAW_MS) return;
      last = now;
      show();
    });
    return () => remove();
  }, [viewer, sceneTime, drawn]);

  if (text === null) return null;
  return (
    <div className="scene-time-readout" role="status" aria-label="Scene time (UTC)">
      <span>Scene time (UTC)</span>
      <time>{text}</time>
    </div>
  );
}
