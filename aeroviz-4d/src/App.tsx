/**
 * App.tsx
 * -------
 * Root layout component.  Stacks the 3D globe (full-screen) with floating
 * UI panels (ControlPanel, FlightOperationsPanel, HUD) layered on top.
 *
 * Layout principle:
 *   - CesiumViewer            → position: absolute, fills 100vw × 100vh
 *   - .cesium-overlay-container → position: absolute, inset: 0, CSS Grid
 *       Panels sit in named grid areas (leftStack / hud / ops); clicks fall
 *       through via pointer-events: none on the container.
 */

import CesiumViewerComponent from "./components/CesiumViewer";
import ControlPanel from "./components/ControlPanel";
import AirportLocalTerrainDemoPage from "./components/AirportLocalTerrainDemoPage";
import ChartAnnotatedPage from "./components/ChartAnnotatedPage";
import AirportLocalTerrainAlert from "./components/AirportLocalTerrainAlert";
import HUD from "./components/HUD";
import FlightOperationsPanel from "./components/FlightOperationsPanel";
import ProcedureDetailsPage from "./components/ProcedureDetailsPage";
import ProcedureAnnotationPopup from "./components/ProcedureAnnotationPopup";
import ProcedurePanel from "./components/ProcedurePanel";
import RunwayTrajectoryProfilePanel from "./components/RunwayTrajectoryProfilePanel";
import { useApp } from "./context/AppContext";
import { airportDataUrl, airportLandingsRunwayUrl } from "./data/airportData";
import { useCzmlLoader } from "./hooks/useCzmlLoader";
import { useComparisonTrajectoryLayer } from "./hooks/useComparisonTrajectoryLayer";
import { useEffect, useState } from "react";

function FlightApp() {
  const { activeAirportCode, selectedRunway, trajectoryComparison, layers } = useApp();
  // The observed-track CZML is large (100+ MB per airport), so it loads only when the
  // Trajectories layer is on (matching the "heavyweight layers opt-in" philosophy) — no
  // eager parse on startup. In comparison mode it is suppressed (empty url unloads it)
  // and useComparisonTrajectoryLayer drives the three-colour optimizer trajectories instead.
  const czmlUrl = activeAirportCode && layers.trajectories && !trajectoryComparison
    ? selectedRunway
      ? airportLandingsRunwayUrl(activeAirportCode, selectedRunway)
      : airportDataUrl(activeAirportCode, "trajectories.czml")
    : "";
  const { flightIds, warning, error } = useCzmlLoader(czmlUrl);
  useComparisonTrajectoryLayer();
  const czmlStatus = error ?? warning;

  return (
    <>
      {/* Layer 0: the 3D globe canvas */}
      <CesiumViewerComponent />

      {/* Layer 1: overlay grid — panels anchored to corners, clicks pass through */}
      <div className="cesium-overlay-container">
        <div className="left-overlay-panel-stack">
          <ControlPanel procedures={<ProcedurePanel />} />
        </div>
        <AirportLocalTerrainAlert />
        <ProcedureAnnotationPopup />
        <RunwayTrajectoryProfilePanel />
        {czmlStatus ? (
          <div
            className={`czml-status ${
              error ? "czml-status-error" : "czml-status-warning"
            }`}
            role="alert"
          >
            {czmlStatus}
          </div>
        ) : null}
        <HUD />
        <FlightOperationsPanel flightIds={flightIds} />
      </div>
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
