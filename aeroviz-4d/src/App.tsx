/**
 * App.tsx
 * -------
 * Root layout component.  Stacks the 3D globe (full-screen) with floating
 * UI panels (ControlPanel, FlightTable, HUD) layered on top via CSS z-index.
 *
 * Layout principle:
 *   - CesiumViewer  → position: absolute, fills 100vw × 100vh, z-index: 0
 *   - All overlays  → position: absolute, z-index: 100+
 *   There is no flex/grid layout here because Cesium owns the full viewport.
 */

import CesiumViewerComponent from "./components/CesiumViewer";
import ControlPanel from "./components/ControlPanel";
// import FlightTable from "./components/FlightTable";

export default function App() {
  return (
    <>
      {/* Layer 0: the 3D globe canvas */}
      <CesiumViewerComponent />

      {/* Layer 1: floating UI panels — these render OVER the canvas */}
      <ControlPanel />
      {/* <FlightTable /> */}
    </>
  );
}
