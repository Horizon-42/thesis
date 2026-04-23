/**
 * App.tsx
 * -------
 * Root layout component.  Stacks the 3D globe (full-screen) with floating
 * UI panels (ControlPanel, FlightTable, HUD) layered on top.
 *
 * Layout principle:
 *   - CesiumViewer            → position: absolute, fills 100vw × 100vh
 *   - .cesium-overlay-container → position: absolute, inset: 0, CSS Grid
 *       Panels sit in named grid areas (ctrl / proc / hud / tbl); clicks fall
 *       through via pointer-events: none on the container.
 */

import CesiumViewerComponent from "./components/CesiumViewer";
import ControlPanel from "./components/ControlPanel";
import DsmTerrainDemoPage from "./components/DsmTerrainDemoPage";
import HUD from "./components/HUD";
import FlightTable from "./components/FlightTable";
import ProcedureDetailsPage from "./components/ProcedureDetailsPage";
import ProcedurePanel from "./components/ProcedurePanel";
import RunwayTrajectoryProfilePanel from "./components/RunwayTrajectoryProfilePanel";
import { useApp } from "./context/AppContext";
import { airportDataUrl } from "./data/airportData";
import { useCzmlLoader } from "./hooks/useCzmlLoader";
import { useEffect, useState } from "react";

function FlightApp() {
  const { activeAirportCode } = useApp();
  const czmlUrl = activeAirportCode
    ? airportDataUrl(activeAirportCode, "trajectories.czml")
    : "";
  const { flightIds, warning, error } = useCzmlLoader(czmlUrl);
  const czmlStatus = error ?? warning;

  return (
    <>
      {/* Layer 0: the 3D globe canvas */}
      <CesiumViewerComponent />

      {/* Layer 1: overlay grid — panels anchored to corners, clicks pass through */}
      <div className="cesium-overlay-container">
        <ControlPanel />
        <ProcedurePanel />
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
        <FlightTable flightIds={flightIds} />
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
  const isDsmTerrainDemo =
    locationState.pathname === "/dsm-terrain-demo" || routeToken === "#dsm-terrain-demo";
  const isProcedureDetails =
    locationState.pathname === "/procedure-details" || routeToken === "#procedure-details";

  if (isDsmTerrainDemo) return <DsmTerrainDemoPage />;
  if (isProcedureDetails) return <ProcedureDetailsPage />;

  return <FlightApp />;
}
