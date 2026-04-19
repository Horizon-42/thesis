/**
 * AppContext.tsx
 * --------------
 * Global application state using React Context + useState.
 *
 * What lives here (and why):
 *   - `viewer`           — the CesiumJS Viewer instance.  Shared so any hook
 *                          or component can add entities without prop-drilling.
 *   - `airport`          — the loaded camera target and airport marker config.
 *   - `selectedFlightId` — which callsign is highlighted in the table and
 *                          tracked by the camera.
 *   - `layers`           — boolean flags that hooks read to show/hide their
 *                          respective data sources.
 *   - `playbackSpeed`    — mirrors viewer.clock.multiplier so the UI stays
 *                          in sync with the Cesium clock.
 *
 * Pattern used: "context + useState" (no Redux, no Zustand).
 * This is intentionally simple — appropriate for a research prototype.
 */

import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import type * as Cesium from "cesium";

// ── Layer names ──────────────────────────────────────────────────────────────
// Extend this union if you add new data layers.
export type LayerKey =
  | "terrain"
  | "dsmTerrain"
  | "runways"
  | "waypoints"
  | "ocsSurfaces"
  | "trajectories"
  | "obstacles";

export interface AirportConfig {
  code: string;
  lon: number;
  lat: number;
  /** Initial camera altitude/range in metres */
  height: number;
}

// ── Context shape ────────────────────────────────────────────────────────────
interface AppState {
  /** The live CesiumJS Viewer, or null before it is mounted */
  viewer: Cesium.Viewer | null;
  setViewer: (v: Cesium.Viewer) => void;

  /** Airport camera target loaded from public/data/airport.json */
  airport: AirportConfig | null;
  setAirport: (airport: AirportConfig) => void;

  /** The currently tracked/selected flight callsign */
  selectedFlightId: string | null;
  setSelectedFlightId: (id: string | null) => void;

  /** Visibility flags for each data layer */
  layers: Record<LayerKey, boolean>;
  toggleLayer: (key: LayerKey) => void;

  /** Current Cesium clock multiplier (mirrors viewer.clock.multiplier) */
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;
}

// ── Create context ────────────────────────────────────────────────────────────
// The default value is `null`; we assert non-null in the `useApp` hook below
// so consumers get a helpful error if they forget to wrap with AppProvider.
const AppContext = createContext<AppState | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────
export function AppProvider({ children }: { children: ReactNode }) {
  const [viewer, setViewerState] = useState<Cesium.Viewer | null>(null);
  const [airport, setAirport] = useState<AirportConfig | null>(null);
  const [selectedFlightId, setSelectedFlightId] = useState<string | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(60);

  // All layers start visible; hooks respect these flags.
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    terrain: true,
    dsmTerrain: true,
    runways: true,
    waypoints: true,
    ocsSurfaces: true,
    trajectories: true,
    obstacles: true,
  });

  // Store the Viewer reference.
  // useCallback prevents creating a new function reference on every render.
  const setViewer = useCallback((v: Cesium.Viewer) => {
    setViewerState(v);
  }, []);

  // Flip a single layer's visibility.
  const toggleLayer = useCallback((key: LayerKey) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  return (
    <AppContext.Provider
      value={{
        viewer,
        setViewer,
        airport,
        setAirport,
        selectedFlightId,
        setSelectedFlightId,
        layers,
        toggleLayer,
        playbackSpeed,
        setPlaybackSpeed,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

// ── Consumer hook ─────────────────────────────────────────────────────────────
/**
 * useApp — call this inside any component or hook to access global state.
 *
 * @example
 *   const { viewer, selectedFlightId } = useApp();
 */
export function useApp(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error(
      "useApp() was called outside of <AppProvider>. " +
        "Wrap your component tree with <AppProvider> in main.tsx."
    );
  }
  return ctx;
}
