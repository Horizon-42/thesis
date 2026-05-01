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
 *         (useProcedureLayer)    loads procedure-details
 *         (useProcedureSegmentLayer) loads v3 procedure render bundles
 *         (useOcsLayer)          builds OCS geometry
 *         (useCzmlLoader)        loads trajectories.czml
 */

import { useRef } from "react";
import { useCesiumViewer } from "../hooks/useCesiumViewer";
import { useRunwayLayer } from "../hooks/useRunwayLayer";
import { useTerrainLayer } from "../hooks/useTerrainLayer";
import { useDsmTerrainLayer } from "../hooks/useDsmTerrainLayer";
import { useObstacleLayer } from "../hooks/useObstacleLayer";
import { useProcedureLayer } from "../hooks/useProcedureLayer";
import { useProcedureSegmentLayer } from "../hooks/useProcedureSegmentLayer";
import { useOcsLayer } from "../hooks/useOcsLayer";
import { useApp } from "../context/AppContext";

export default function CesiumViewerComponent() {
  // This ref is the DOM anchor for the Cesium canvas.
  // IMPORTANT: the div must be in the DOM before useCesiumViewer runs —
  // that's why we pass the ref to the hook rather than accessing the node directly.
  const containerRef = useRef<HTMLDivElement>(null);
  const { layers, procedureVisualizationMode } = useApp();

  // ── Initialise the 3D globe ────────────────────────────────────────────────
  // This hook creates the Viewer and stores it in AppContext.
  // All other hooks below read `viewer` from AppContext, so they automatically
  // wait until this hook has finished.
  useCesiumViewer(containerRef);

  // ── Activate data layers ───────────────────────────────────────────────────
  // Each hook runs its own useEffect and manages its own cleanup.
  useTerrainLayer();
  useDsmTerrainLayer({ enabled: layers.dsmTerrain });
  useRunwayLayer();
  useObstacleLayer();
  useProcedureLayer({ enabled: procedureVisualizationMode === "legacy" });
  useProcedureSegmentLayer({ enabled: procedureVisualizationMode === "protected" });
  useOcsLayer();
  // Waypoint rendering is intentionally disabled for now.
  // Keep the hook implementation for future use.
  // useWaypointLayer();

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
