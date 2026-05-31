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
  useMemo,
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
import type {
  ProcedureDisplayLevel,
  ProcedureEntityAnnotation,
} from "../data/procedureAnnotations";
import { fetchJson } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";

// ── Layer names ──────────────────────────────────────────────────────────────
// Extend this union if you add new data layers.
export type LayerKey =
  | "satelliteImagery"
  | "terrain"
  | "airportLocalTerrain"
  | "terrainHillshade"
  | "runways"
  | "waypoints"
  | "ocsSurfaces"
  | "trajectories"
  | "obstacles"
  | "obstacleLabels"
  | "procedures";

export type RunwayProfileViewMode = "split" | "side-xz" | "top-xy";

export type AirportLocalTerrainStatus =
  | "disabled"
  | "missing"
  | "loading"
  | "preloading"
  | "active"
  | "error";

export interface AirportLocalTerrainState {
  status: AirportLocalTerrainStatus;
  airportCode: string | null;
  sourceLabel: string | null;
  minimumHeightM: number | null;
  maximumHeightM: number | null;
  loadedTiles: number;
  totalTiles: number;
  error: string | null;
}

function airportLocalTerrainStateForLayer(
  airportCode: string | null,
  enabled: boolean,
): AirportLocalTerrainState {
  return {
    status: enabled ? "loading" : "disabled",
    airportCode,
    sourceLabel: null,
    minimumHeightM: null,
    maximumHeightM: null,
    loadedTiles: 0,
    totalTiles: 0,
    error: null,
  };
}

// ── Context shapes ───────────────────────────────────────────────────────────
// useApp still exposes one merged interface for existing callers, but the
// provider below splits state by ownership so airport resets, scene mutations,
// procedure controls, and profile controls no longer share one implicit seam.
interface SceneState {
  /** The live CesiumJS Viewer, or null before it is mounted */
  viewer: Cesium.Viewer | null;
  setViewer: (v: Cesium.Viewer | null) => void;

  /** Visibility flags for each data layer */
  layers: Record<LayerKey, boolean>;
  toggleLayer: (key: LayerKey) => void;

  /** Status of the active airport-scoped local high-resolution terrain source */
  airportLocalTerrain: AirportLocalTerrainState;
  setAirportLocalTerrain: (state: AirportLocalTerrainState) => void;
}

interface AirportSessionState {
  /** Available airport folders exposed by public/data/airports/index.json */
  airports: AirportCatalogItem[];
  /** Active airport folder key, e.g. KRDU */
  activeAirportCode: string;
  setActiveAirportCode: (code: string) => void;

  /** Airport camera target loaded from public/data/airports/<ICAO>/airport.json */
  airport: AirportConfig | null;
  setAirport: (airport: AirportConfig | null) => void;
}

interface FlightSessionState {
  /** The currently tracked/selected flight callsign */
  selectedFlightId: string | null;
  setSelectedFlightId: (id: string | null) => void;

  /** The loaded CZML datasource for trajectory sampling and profile views */
  trajectoryDataSource: Cesium.CzmlDataSource | null;
  setTrajectoryDataSource: (dataSource: Cesium.CzmlDataSource | null) => void;
}

interface ProcedureSessionState {
  /** Per-branch visibility for v3 procedure features */
  procedureVisibility: Record<string, boolean>;
  setProcedureBranchVisible: (branchId: string, visible: boolean) => void;
  setProcedureBranchesVisible: (branchIds: string[], visible: boolean) => void;
  procedureAnnotationEnabled: boolean;
  setProcedureAnnotationEnabled: (enabled: boolean) => void;
  procedureWidthMeasurementEnabled: boolean;
  setProcedureWidthMeasurementEnabled: (enabled: boolean) => void;
  procedureDisplayLevel: ProcedureDisplayLevel;
  setProcedureDisplayLevel: (level: ProcedureDisplayLevel) => void;
  selectedProcedureAnnotation: ProcedureEntityAnnotation | null;
  setSelectedProcedureAnnotation: (annotation: ProcedureEntityAnnotation | null) => void;
}

interface PlaybackState {
  /** Current Cesium clock multiplier (mirrors viewer.clock.multiplier) */
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;
}

interface RunwayProfileSessionState {
  /** Selected runway for the 2D runway profile overlay */
  selectedProfileRunwayIdent: string | null;
  setSelectedProfileRunwayIdent: (runwayIdent: string | null) => void;
  isRunwayProfileOpen: boolean;
  setRunwayProfileOpen: (open: boolean) => void;
  runwayProfileViewMode: RunwayProfileViewMode;
  setRunwayProfileViewMode: (mode: RunwayProfileViewMode) => void;
}

interface AppState extends
  SceneState,
  AirportSessionState,
  FlightSessionState,
  ProcedureSessionState,
  PlaybackState,
  RunwayProfileSessionState {}

// The defaults are `null`; useApp asserts all providers are present so consumers
// get a helpful error if they forget to wrap with AppProvider.
const SceneContext = createContext<SceneState | null>(null);
const AirportSessionContext = createContext<AirportSessionState | null>(null);
const FlightSessionContext = createContext<FlightSessionState | null>(null);
const ProcedureSessionContext = createContext<ProcedureSessionState | null>(null);
const PlaybackContext = createContext<PlaybackState | null>(null);
const RunwayProfileSessionContext = createContext<RunwayProfileSessionState | null>(null);

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
  const [procedureAnnotationEnabled, setProcedureAnnotationEnabled] = useState(false);
  const [procedureWidthMeasurementEnabled, setProcedureWidthMeasurementEnabled] = useState(false);
  const [procedureDisplayLevel, setProcedureDisplayLevel] =
    useState<ProcedureDisplayLevel>("PROTECTION");
  const [selectedProcedureAnnotation, setSelectedProcedureAnnotation] =
    useState<ProcedureEntityAnnotation | null>(null);
  const [selectedProfileRunwayIdent, setSelectedProfileRunwayIdent] = useState<string | null>(
    null,
  );
  const [isRunwayProfileOpen, setRunwayProfileOpen] = useState(false);
  const [runwayProfileViewMode, setRunwayProfileViewMode] =
    useState<RunwayProfileViewMode>("split");
  const [airportLocalTerrain, setAirportLocalTerrain] = useState<AirportLocalTerrainState>({
    status: "disabled",
    airportCode: null,
    sourceLabel: null,
    minimumHeightM: null,
    maximumHeightM: null,
    loadedTiles: 0,
    totalTiles: 0,
    error: null,
  });

  // Keep heavyweight analysis layers opt-in. Local terrain and RNAV procedure
  // geometry can allocate hundreds of MB once loaded, so startup should show the
  // core flight scene first and let the user ask for analysis detail.
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    satelliteImagery: true,
    terrain: false,
    airportLocalTerrain: false,
    terrainHillshade: false,
    runways: true,
    waypoints: false,
    ocsSurfaces: false,
    trajectories: true,
    obstacles: false,
    obstacleLabels: false,
    procedures: false,
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
      setProcedureAnnotationEnabled(false);
      setProcedureWidthMeasurementEnabled(false);
      setProcedureDisplayLevel("PROTECTION");
      setSelectedProcedureAnnotation(null);
      setSelectedProfileRunwayIdent(null);
      setRunwayProfileOpen(false);
      setAirportLocalTerrain(
        airportLocalTerrainStateForLayer(normalizedCode, layers.airportLocalTerrain),
      );
      setAirport(null);
      setActiveAirportCodeState(normalizedCode);
    },
    [activeAirportCode, layers.airportLocalTerrain, viewer],
  );

  const sceneState: SceneState = useMemo(() => ({
    viewer,
    setViewer,
    layers,
    toggleLayer,
    airportLocalTerrain,
    setAirportLocalTerrain,
  }), [airportLocalTerrain, layers, setViewer, toggleLayer, viewer]);
  const airportSessionState: AirportSessionState = useMemo(() => ({
    airports,
    activeAirportCode,
    setActiveAirportCode,
    airport,
    setAirport,
  }), [activeAirportCode, airport, airports, setActiveAirportCode]);
  const flightSessionState: FlightSessionState = useMemo(() => ({
    selectedFlightId,
    setSelectedFlightId,
    trajectoryDataSource,
    setTrajectoryDataSource,
  }), [selectedFlightId, trajectoryDataSource]);
  const procedureSessionState: ProcedureSessionState = useMemo(() => ({
    procedureVisibility,
    setProcedureBranchVisible,
    setProcedureBranchesVisible,
    procedureAnnotationEnabled,
    setProcedureAnnotationEnabled,
    procedureWidthMeasurementEnabled,
    setProcedureWidthMeasurementEnabled,
    procedureDisplayLevel,
    setProcedureDisplayLevel,
    selectedProcedureAnnotation,
    setSelectedProcedureAnnotation,
  }), [
    procedureAnnotationEnabled,
    procedureDisplayLevel,
    procedureVisibility,
    procedureWidthMeasurementEnabled,
    selectedProcedureAnnotation,
    setProcedureBranchVisible,
    setProcedureBranchesVisible,
  ]);
  const playbackState: PlaybackState = useMemo(() => ({
    playbackSpeed,
    setPlaybackSpeed,
  }), [playbackSpeed]);
  const runwayProfileSessionState: RunwayProfileSessionState = useMemo(() => ({
    selectedProfileRunwayIdent,
    setSelectedProfileRunwayIdent,
    isRunwayProfileOpen,
    setRunwayProfileOpen,
    runwayProfileViewMode,
    setRunwayProfileViewMode,
  }), [isRunwayProfileOpen, runwayProfileViewMode, selectedProfileRunwayIdent]);

  return (
    <AirportSessionContext.Provider value={airportSessionState}>
      <SceneContext.Provider value={sceneState}>
        <FlightSessionContext.Provider value={flightSessionState}>
          <ProcedureSessionContext.Provider value={procedureSessionState}>
            <PlaybackContext.Provider value={playbackState}>
              <RunwayProfileSessionContext.Provider value={runwayProfileSessionState}>
                {children}
              </RunwayProfileSessionContext.Provider>
            </PlaybackContext.Provider>
          </ProcedureSessionContext.Provider>
        </FlightSessionContext.Provider>
      </SceneContext.Provider>
    </AirportSessionContext.Provider>
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
  const sceneState = useContext(SceneContext);
  const airportSessionState = useContext(AirportSessionContext);
  const flightSessionState = useContext(FlightSessionContext);
  const procedureSessionState = useContext(ProcedureSessionContext);
  const playbackState = useContext(PlaybackContext);
  const runwayProfileSessionState = useContext(RunwayProfileSessionContext);
  if (
    !sceneState ||
    !airportSessionState ||
    !flightSessionState ||
    !procedureSessionState ||
    !playbackState ||
    !runwayProfileSessionState
  ) {
    throw new Error(
      "useApp() was called outside of <AppProvider>. " +
        "Wrap your component tree with <AppProvider> in main.tsx."
    );
  }
  return {
    ...sceneState,
    ...airportSessionState,
    ...flightSessionState,
    ...procedureSessionState,
    ...playbackState,
    ...runwayProfileSessionState,
  };
}
