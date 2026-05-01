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
  useEffect,
  type ReactNode,
} from "react";
import type * as Cesium from "cesium";
import {
  AIRPORTS_INDEX_URL,
  isAirportsIndexManifest,
  normalizeAirportCode,
  sortAirportCatalog,
  type AirportCatalogItem,
  type AirportConfig,
} from "../data/airportData";
import { fetchJson } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";

// ── Layer names ──────────────────────────────────────────────────────────────
// Extend this union if you add new data layers.
export type LayerKey =
  | "terrain"
  | "dsmTerrain"
  | "runways"
  | "waypoints"
  | "ocsSurfaces"
  | "trajectories"
  | "obstacles"
  | "procedures";

export type RunwayProfileViewMode = "split" | "side-xz" | "top-xy";

// ── Context shape ────────────────────────────────────────────────────────────
interface AppState {
  /** The live CesiumJS Viewer, or null before it is mounted */
  viewer: Cesium.Viewer | null;
  setViewer: (v: Cesium.Viewer | null) => void;

  /** Available airport folders exposed by public/data/airports/index.json */
  airports: AirportCatalogItem[];
  /** Active airport folder key, e.g. KRDU */
  activeAirportCode: string;
  setActiveAirportCode: (code: string) => void;

  /** Airport camera target loaded from public/data/airports/<ICAO>/airport.json */
  airport: AirportConfig | null;
  setAirport: (airport: AirportConfig | null) => void;

  /** The currently tracked/selected flight callsign */
  selectedFlightId: string | null;
  setSelectedFlightId: (id: string | null) => void;

  /** The loaded CZML datasource for trajectory sampling and profile views */
  trajectoryDataSource: Cesium.CzmlDataSource | null;
  setTrajectoryDataSource: (dataSource: Cesium.CzmlDataSource | null) => void;

  /** Visibility flags for each data layer */
  layers: Record<LayerKey, boolean>;
  toggleLayer: (key: LayerKey) => void;

  /** Per-branch visibility for v3 procedure features */
  procedureVisibility: Record<string, boolean>;
  setProcedureBranchVisible: (branchId: string, visible: boolean) => void;
  setProcedureBranchesVisible: (branchIds: string[], visible: boolean) => void;

  /** Current Cesium clock multiplier (mirrors viewer.clock.multiplier) */
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;

  /** Selected runway for the 2D runway profile overlay */
  selectedProfileRunwayIdent: string | null;
  setSelectedProfileRunwayIdent: (runwayIdent: string | null) => void;
  isRunwayProfileOpen: boolean;
  setRunwayProfileOpen: (open: boolean) => void;
  runwayProfileViewMode: RunwayProfileViewMode;
  setRunwayProfileViewMode: (mode: RunwayProfileViewMode) => void;
}

// ── Create context ────────────────────────────────────────────────────────────
// The default value is `null`; we assert non-null in the `useApp` hook below
// so consumers get a helpful error if they forget to wrap with AppProvider.
const AppContext = createContext<AppState | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────
export function AppProvider({ children }: { children: ReactNode }) {
  const [viewer, setViewerState] = useState<Cesium.Viewer | null>(null);
  const [airports, setAirports] = useState<AirportCatalogItem[]>([]);
  const [activeAirportCode, setActiveAirportCodeState] = useState<string>("");
  const [airport, setAirport] = useState<AirportConfig | null>(null);
  const [selectedFlightId, setSelectedFlightId] = useState<string | null>(null);
  const [trajectoryDataSource, setTrajectoryDataSource] =
    useState<Cesium.CzmlDataSource | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(60);
  const [procedureVisibility, setProcedureVisibility] = useState<Record<string, boolean>>({});
  const [selectedProfileRunwayIdent, setSelectedProfileRunwayIdent] = useState<string | null>(
    null,
  );
  const [isRunwayProfileOpen, setRunwayProfileOpen] = useState(false);
  const [runwayProfileViewMode, setRunwayProfileViewMode] =
    useState<RunwayProfileViewMode>("split");

  // All layers start visible; hooks respect these flags.
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    terrain: false,
    dsmTerrain: false,
    runways: true,
    waypoints: true,
    ocsSurfaces: false,
    trajectories: true,
    obstacles: true,
    procedures: true,
  });

  // Store the Viewer reference.
  // useCallback prevents creating a new function reference on every render.
  const setViewer = useCallback((v: Cesium.Viewer | null) => {
    setViewerState(v);
  }, []);

  useEffect(() => {
    let cancelled = false;

    fetchJson<unknown>(AIRPORTS_INDEX_URL)
      .then((manifest: unknown) => {
        if (cancelled) return;
        if (!isAirportsIndexManifest(manifest)) {
          throw new Error(`${AIRPORTS_INDEX_URL} is not a valid airport manifest`);
        }

        const nextAirports = sortAirportCatalog(manifest.airports);
        const defaultAirport = normalizeAirportCode(manifest.defaultAirport);
        setAirports(nextAirports);
        setActiveAirportCodeState((current) => {
          if (current && nextAirports.some((airportItem) => airportItem.code === current)) {
            return current;
          }
          if (nextAirports.some((airportItem) => airportItem.code === defaultAirport)) {
            return defaultAirport;
          }
          return nextAirports[0]?.code ?? defaultAirport;
        });
      })
      .catch((error) => {
        console.error("[AppContext] Failed to load airport manifest:", error);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // Flip a single layer's visibility.
  const toggleLayer = useCallback((key: LayerKey) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const setProcedureBranchVisible = useCallback((branchId: string, visible: boolean) => {
    setProcedureVisibility((prev) => ({ ...prev, [branchId]: visible }));
  }, []);

  const setProcedureBranchesVisible = useCallback((branchIds: string[], visible: boolean) => {
    setProcedureVisibility((prev) => {
      const next = { ...prev };
      branchIds.forEach((branchId) => {
        next[branchId] = visible;
      });
      return next;
    });
  }, []);

  const setActiveAirportCode = useCallback(
    (code: string) => {
      const normalizedCode = normalizeAirportCode(code);
      if (!normalizedCode || normalizedCode === activeAirportCode) return;

      if (isCesiumViewerUsable(viewer)) {
        viewer.trackedEntity = undefined;
      }
      setSelectedFlightId(null);
      setTrajectoryDataSource(null);
      setProcedureVisibility({});
      setSelectedProfileRunwayIdent(null);
      setRunwayProfileOpen(false);
      setAirport(null);
      setActiveAirportCodeState(normalizedCode);
    },
    [activeAirportCode, viewer],
  );

  return (
    <AppContext.Provider
      value={{
        viewer,
        setViewer,
        airports,
        activeAirportCode,
        setActiveAirportCode,
        airport,
        setAirport,
        selectedFlightId,
        setSelectedFlightId,
        trajectoryDataSource,
        setTrajectoryDataSource,
        layers,
        toggleLayer,
        procedureVisibility,
        setProcedureBranchVisible,
        setProcedureBranchesVisible,
        playbackSpeed,
        setPlaybackSpeed,
        selectedProfileRunwayIdent,
        setSelectedProfileRunwayIdent,
        isRunwayProfileOpen,
        setRunwayProfileOpen,
        runwayProfileViewMode,
        setRunwayProfileViewMode,
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
