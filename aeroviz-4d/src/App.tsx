/**
 * App.tsx
 * -------
 * Root layout component.  Stacks the 3D globe (full-screen) with floating
 * UI docks (WorkbenchShell: top bar, per-task left dock, HUD) layered on top.
 *
 * Layout principle:
 *   - CesiumViewer            → position: absolute, fills 100vw × 100vh
 *   - .cesium-overlay-container → position: absolute, inset: 0, CSS Grid
 *       Panels sit in named grid areas (leftStack / hud / ops); clicks fall
 *       through via pointer-events: none on the container.
 */

import CesiumViewerComponent from "./components/CesiumViewer";
import WorkbenchShell from "./components/WorkbenchShell";
import WorkbenchLeftDock from "./components/WorkbenchLeftDock";
import AirportLocalTerrainDemoPage from "./components/AirportLocalTerrainDemoPage";
import ChartAnnotatedPage from "./components/ChartAnnotatedPage";
import AirportLocalTerrainAlert from "./components/AirportLocalTerrainAlert";
import HUD from "./components/HUD";
import WorkbenchRightInspector from "./components/WorkbenchRightInspector";
import WorkbenchBottomBar from "./components/WorkbenchBottomBar";
import ProcedureDetailsPage from "./components/ProcedureDetailsPage";
import ProcedureAnnotationPopup from "./components/ProcedureAnnotationPopup";
import ProcedurePanel from "./components/ProcedurePanel";
import ApproachViewPanel from "./components/ApproachViewPanel";
import { useApp } from "./context/AppContext";
import { planObservedTracks } from "./data/observedTracks";
import { useObservedTrajectoryLayer } from "./hooks/useObservedTrajectoryLayer";
import { useComparisonTrajectoryLayer } from "./hooks/useComparisonTrajectoryLayer";
import { useLandingsManifest } from "./hooks/useLandingsManifest";
import { AEROVIZ_BACKEND_URL } from "./pilot/pilotClient";
import { useEffect, useState } from "react";

function FlightApp() {
  const {
    activeAirportCode,
    selectedRunway,
    trajectoryComparison,
    trajectorySampleCount,
    observedVerdictFilter,
    mode,
    proceduresOpen,
  } = useApp();
  const {
    manifest: landingsManifest,
    status: landingsStatus,
    error: landingsError,
  } = useLandingsManifest(activeAirportCode ?? "");
  // The independently sampled Baseline source exists only in Observe/Baseline. Comparison
  // loads its exact index-selected references in useComparisonTrajectoryLayer instead.
  const {
    fileUrl: observedFileUrl,
    visible: observedVisible,
  } = planObservedTracks({
    mode,
    activeAirportCode,
    selectedRunway,
    trajectoryComparison,
    trajectorySampleCount,
    observedVerdictFilter,
    backendUrl: AEROVIZ_BACKEND_URL,
    landingsManifest,
    landingsStatus,
  });
  const observedLayer = useObservedTrajectoryLayer(
    observedFileUrl,
    observedVisible,
  );
  const comparisonLayer = useComparisonTrajectoryLayer();
  const activeTrajectoryLayer = trajectoryComparison ? comparisonLayer : observedLayer;
  const {
    observedVerdicts,
    observedEvaluation,
  } = observedLayer;
  const czmlError = landingsError ?? activeTrajectoryLayer.error;
  const czmlStatus = czmlError ?? activeTrajectoryLayer.warning;

  return (
    <>
      {/* Layer 0: the 3D globe canvas */}
      <CesiumViewerComponent />

      {/* Layer 1: the workbench shell — top context bar + a per-task left dock over the
          overlay host (clicks fall through to the globe; each dock re-enables them). */}
      <WorkbenchShell
        left={
          <WorkbenchLeftDock
            flightIds={activeTrajectoryLayer.flightIds}
            flightSummaries={activeTrajectoryLayer.flightSummaries}
            observedVerdicts={observedVerdicts}
            observedEvaluation={observedEvaluation}
          />
        }
        right={
          <WorkbenchRightInspector>
            <HUD />
          </WorkbenchRightInspector>
        }
        bottom={<WorkbenchBottomBar />}
      >
        <AirportLocalTerrainAlert />
        <ProcedureAnnotationPopup />
        <ApproachViewPanel />
        {/* Procedures is an independent panel docked bottom-right (grid-area ops); it
            coexists with the active task and keeps its state across task switches. */}
        {proceduresOpen ? <ProcedurePanel /> : null}
        {czmlStatus ? (
          <div
            className={`czml-status ${
              czmlError ? "czml-status-error" : "czml-status-warning"
            }`}
            role="alert"
          >
            {czmlStatus}
          </div>
        ) : null}
      </WorkbenchShell>
    </>
  );
}

export default function App() {
  const [locationState, setLocationState] = useState(() => ({
    pathname: window.location.pathname,
    hash: window.location.hash,
  }));

  useEffect(() => {
    const syncLocation = () => {
      setLocationState({
        pathname: window.location.pathname,
        hash: window.location.hash,
      });
    };

    window.addEventListener("popstate", syncLocation);
    window.addEventListener("hashchange", syncLocation);
    return () => {
      window.removeEventListener("popstate", syncLocation);
      window.removeEventListener("hashchange", syncLocation);
    };
  }, []);

  const routeToken = locationState.hash.split("?")[0];
  const isLocalTerrainDemo =
    locationState.pathname === "/local-terrain-demo" || routeToken === "#local-terrain-demo";
  const isProcedureDetails =
    locationState.pathname === "/procedure-details" || routeToken === "#procedure-details";
  const isChartAnnotated =
    locationState.pathname === "/chart-annotated" || routeToken === "#chart-annotated";

  if (isLocalTerrainDemo) return <AirportLocalTerrainDemoPage />;
  if (isProcedureDetails) return <ProcedureDetailsPage />;
  if (isChartAnnotated) return <ChartAnnotatedPage />;

  return <FlightApp />;
}
