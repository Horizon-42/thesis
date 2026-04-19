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
import DsmDemoPage from "./components/DsmDemoPage";
import DsmTerrainDemoPage from "./components/DsmTerrainDemoPage";
import HUD from "./components/HUD";
import FlightTable from "./components/FlightTable";
import ProcedurePanel from "./components/ProcedurePanel";
import { useCzmlLoader } from "./hooks/useCzmlLoader";

const CZML_URL = "/data/trajectories.czml";

function FlightApp() {
  const { flightIds, warning, error } = useCzmlLoader(CZML_URL);
  const czmlStatus = error ?? warning;

  return (
    <>
      {/* Layer 0: the 3D globe canvas */}
      <CesiumViewerComponent />

      {/* Layer 1: overlay grid — panels anchored to corners, clicks pass through */}
      <div className="cesium-overlay-container">
        <ControlPanel />
        <ProcedurePanel />
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
  const isDsmDemo =
    window.location.pathname === "/dsm-demo" || window.location.hash === "#dsm-demo";
  const isDsmTerrainDemo =
    window.location.pathname === "/dsm-terrain-demo" ||
    window.location.hash === "#dsm-terrain-demo";

  if (isDsmDemo) return <DsmDemoPage />;
  if (isDsmTerrainDemo) return <DsmTerrainDemoPage />;

  return <FlightApp />;
}
