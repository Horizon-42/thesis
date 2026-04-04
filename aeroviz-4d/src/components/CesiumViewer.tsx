/**
 * CesiumViewer.tsx
 * ----------------
 * The root 3D scene component.  Renders a full-screen `<div>` that CesiumJS
 * takes over, then delegates Viewer initialisation to the useCesiumViewer hook.
 *
 * Also activates all the data-layer hooks so layers are loaded once the
 * Viewer is ready.
 *
 * Component hierarchy:
 *   App
 *   └── CesiumViewerComponent   ← this file
 *         (useCesiumViewer)      initialises Viewer, stores in context
 *         (useRunwayLayer)       loads runway.geojson
 *         (useWaypointLayer)     loads waypoints.geojson
 *         (useOcsLayer)          builds OCS geometry
 *         (useCzmlLoader)        loads trajectories.czml
 */

import { useRef } from "react";
import { useCesiumViewer } from "../hooks/useCesiumViewer";
// import { useRunwayLayer } from "../hooks/useRunwayLayer";
// import { useWaypointLayer } from "../hooks/useWaypointLayer";
import { useCzmlLoader } from "../hooks/useCzmlLoader";

/** Path to the CZML trajectory file (served from public/) */
const CZML_URL = "/data/trajectories.czml";

export default function CesiumViewerComponent() {
  // This ref is the DOM anchor for the Cesium canvas.
  // IMPORTANT: the div must be in the DOM before useCesiumViewer runs —
  // that's why we pass the ref to the hook rather than accessing the node directly.
  const containerRef = useRef<HTMLDivElement>(null);

  // ── Initialise the 3D globe ────────────────────────────────────────────────
  // This hook creates the Viewer and stores it in AppContext.
  // All other hooks below read `viewer` from AppContext, so they automatically
  // wait until this hook has finished.
  useCesiumViewer(containerRef);

  // ── Activate data layers ───────────────────────────────────────────────────
  // Each hook runs its own useEffect and manages its own cleanup.
  // useRunwayLayer();
  // useWaypointLayer();
  // TODO — uncomment once you have implemented useOcsLayer:
  // useOcsLayer();

  // const { isLoaded, flightIds, error } = useCzmlLoader(CZML_URL);

  // Log loading status to the browser console during development.
  // if (import.meta.env.DEV) {
  //   if (error) console.error("[CZML]", error);
  //   if (isLoaded) console.log("[CZML] loaded flights:", flightIds);
  // }

  return (
    <div
      ref={containerRef}
      style={{
        position: "absolute",
        inset: 0,        // shorthand for top/right/bottom/left: 0
        width: "100%",
        height: "100%",
      }}
    />
  );
}
