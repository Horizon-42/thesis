/**
 * AppContext.tsx
 * --------------
 * Global application state using React Context + useState.
 *
 * What lives here (and why):
 *   - `viewer`           — the CesiumJS Viewer instance.  Shared so any hook
 *                          or component can add entities without prop-drilling.
 *   - `airport`          — the loaded camera target and airport marker config.
 *   - `selectedFlightId` — the selected observed flight's ENTITY id (the flight_key
 *                          `id_runway_icao24_landingTime`, not the callsign — namesakes
 *                          repeat daily), highlighted in the table + camera-tracked.
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
import type { AirportLocalTerrainSourceKind } from "../terrain/airportLocalTerrain";

// ── Layer names ──────────────────────────────────────────────────────────────
// Extend this union if you add new data layers.
export type LayerKey =
  | "satelliteImagery"
  | "terrain"
  | "airportLocalTerrain"
  | "terrainHillshade"
  | "terrainHeightTint"
  | "runways"
  | "waypoints"
  | "ocsSurfaces"
  | "trajectories"
  | "obstacles"
  | "obstacleLabels"
  | "procedures"
  | "rangeRing";

export type ApproachViewMode = "split" | "side-xz" | "top-xy";

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
  sourceKind: AirportLocalTerrainSourceKind | null;
  sourceName: string | null;
  horizontalResolutionM: number | null;
  sourceCrsCode: string | null;
  sourceCrsName: string | null;
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
    sourceKind: null,
    sourceName: null,
    horizontalResolutionM: null,
    sourceCrsCode: null,
    sourceCrsName: null,
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

  /** Radius (km) of the airport-centred range ring drawn by the `rangeRing` layer */
  rangeRingRadiusKm: number;
  setRangeRingRadiusKm: (km: number) => void;
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

/** The three coloured trajectories in an optimizer comparison. */
/**
 * A comparison entity's role. `optimizer`/`simulator` are the two halves of an optimizer
 * run (the NLP's plan and its true-dynamics replay); `predicted` is a learned forecast,
 * which has no such split — one trajectory, no controls. It is a separate kind rather than
 * reusing `optimizer` so the legend cannot claim a prediction is an optimizer plan.
 */
export type ComparisonKind = "reference" | "optimizer" | "simulator" | "predicted";

interface FlightSessionState {
  /** The tracked/selected observed flight's entity id (the flight_key, not the callsign) */
  selectedFlightId: string | null;
  setSelectedFlightId: (id: string | null) => void;

  /** Selected landing runway end (e.g. "23R"); null loads the combined CZML */
  selectedRunway: string | null;
  setSelectedRunway: (runway: string | null) => void;

  /** The loaded observed-track CZML datasource for trajectory sampling and profile views */
  trajectoryDataSource: Cesium.CzmlDataSource | null;
  setTrajectoryDataSource: (dataSource: Cesium.CzmlDataSource | null) => void;

  /** The optimized-trajectory playback CZML datasource (Optimize / Trajectory-Play mode),
   *  exposed so the approach view can plot the optimized track alongside the observed one. */
  optimizedTrajectoryDataSource: Cesium.CzmlDataSource | null;
  setOptimizedTrajectoryDataSource: (dataSource: Cesium.CzmlDataSource | null) => void;

  /** When true, the Trajectories layer shows the 3-colour optimizer comparison instead
   *  of the observed tracks (driven by the same runway selection). */
  trajectoryComparison: boolean;
  setTrajectoryComparison: (enabled: boolean) => void;

  /** Selected optimization category dir (e.g. "asdb"); which comparison set to show. */
  trajectoryComparisonCategory: string | null;
  setTrajectoryComparisonCategory: (categoryDir: string | null) => void;

  /** Per-kind visibility for the 3-colour comparison (reference / optimizer / simulator). */
  trajectoryComparisonKinds: Record<ComparisonKind, boolean>;
  setTrajectoryComparisonKind: (kind: ComparisonKind, visible: boolean) => void;

  /** How many trajectories to render (0 = all); applies to both normal and comparison modes. */
  trajectorySampleCount: number;
  setTrajectorySampleCount: (count: number) => void;
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

  /**
   * Whether trajectory playback loops (LOOP_STOP) or stops at the final state
   * (CLAMPED). Applies to both downloaded and optimized trajectories.
   */
  autoReplay: boolean;
  setAutoReplay: (value: boolean) => void;
}

interface ApproachViewSessionState {
  // The approach view's runway is NOT stored separately — it is the global `selectedRunway`
  // (see FlightSessionState), so the procedure/profile runway and the top-bar Landing
  // Runway selector are always one and the same.
  isApproachViewOpen: boolean;
  setApproachViewOpen: (open: boolean) => void;
  approachViewMode: ApproachViewMode;
  setApproachViewMode: (mode: ApproachViewMode) => void;
}

/**
 * The active top-level task. These four are mutually exclusive — one drives the
 * left dock at a time. `fly`/`optimize`/`compare` map onto the PilotPanel's
 * pilot/trajectory/comparison sub-modes. Procedures is intentionally NOT a mode:
 * it is an independent panel (`proceduresOpen`) that coexists with any task.
 */
export type WorkbenchMode = "observe" | "fly" | "optimize" | "compare";

/**
 * Fly (pilot) mode's transport, published by PilotPanel so the shared bottom bar
 * can drive the MANUAL simulation loop (`isFlying`) — which, unlike the
 * optimize/compare CZML playback, does NOT run on `viewer.clock`, so the generic
 * clock transport can't touch it. `null` unless the pilot panel is active in fly
 * mode. The callbacks are stable (ref-backed); the booleans reflect live sim
 * state so the bar's Play/Pause icon and disabled states track it.
 */
export interface PilotTransport {
  /** The sim loop is running (show Pause) vs paused (show Play). */
  running: boolean;
  /** Play/Pause is disabled (busy, placing the aircraft, or nothing to start). */
  playPauseDisabled: boolean;
  /** Reset is disabled. */
  resetDisabled: boolean;
  /** Toggle play/pause of the manual sim. */
  togglePlay: () => void;
  /** Reset the sim to the start (paused). */
  reset: () => void;
}

interface WorkbenchUiState {
  /** Active task in the workbench shell (one of the four exclusive tasks). */
  mode: WorkbenchMode;
  setMode: (mode: WorkbenchMode) => void;
  /**
   * Whether the RNAV procedure panel is shown. Independent of `mode` so procedures
   * can stay open (and keep their state) across any task — e.g. browse procedures
   * while observing traffic. The panel's own "On" switch controls the 3D geometry.
   */
  proceduresOpen: boolean;
  setProceduresOpen: (open: boolean) => void;
  /** When true, every dock/chrome is hidden, leaving a clean globe (demos/figures). */
  presentationMode: boolean;
  setPresentationMode: (enabled: boolean) => void;
  /** Whether the on-demand Layers drawer is open. */
  layersDrawerOpen: boolean;
  setLayersDrawerOpen: (open: boolean) => void;
  /** Whether the right inspector dock is collapsed. */
  rightInspectorCollapsed: boolean;
  setRightInspectorCollapsed: (collapsed: boolean) => void;
  /**
   * Fly-mode manual-sim transport (see PilotTransport). Published by PilotPanel
   * while in fly mode so the shared bottom bar drives the sim; `null` otherwise.
   */
  pilotTransport: PilotTransport | null;
  setPilotTransport: (transport: PilotTransport | null) => void;
}

interface AppState extends
  SceneState,
  AirportSessionState,
  FlightSessionState,
  ProcedureSessionState,
  PlaybackState,
  ApproachViewSessionState,
  WorkbenchUiState {}

// The defaults are `null`; useApp asserts all providers are present so consumers
// get a helpful error if they forget to wrap with AppProvider.
const SceneContext = createContext<SceneState | null>(null);
const AirportSessionContext = createContext<AirportSessionState | null>(null);
const FlightSessionContext = createContext<FlightSessionState | null>(null);
const ProcedureSessionContext = createContext<ProcedureSessionState | null>(null);
const PlaybackContext = createContext<PlaybackState | null>(null);
const ApproachViewSessionContext = createContext<ApproachViewSessionState | null>(null);
const WorkbenchUiContext = createContext<WorkbenchUiState | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────
export function AppProvider({ children }: { children: ReactNode }) {
  const [viewer, setViewerState] = useState<Cesium.Viewer | null>(null);
  const [airports, setAirports] = useState<AirportCatalogItem[]>([]);
  const [activeAirportCode, setActiveAirportCodeState] = useState<string>("");
  const [airport, setAirport] = useState<AirportConfig | null>(null);
  const [selectedFlightId, setSelectedFlightId] = useState<string | null>(null);
  const [selectedRunway, setSelectedRunway] = useState<string | null>(null);
  const [trajectoryDataSource, setTrajectoryDataSource] =
    useState<Cesium.CzmlDataSource | null>(null);
  const [optimizedTrajectoryDataSource, setOptimizedTrajectoryDataSource] =
    useState<Cesium.CzmlDataSource | null>(null);
  const [trajectoryComparison, setTrajectoryComparison] = useState<boolean>(false);
  const [trajectoryComparisonCategory, setTrajectoryComparisonCategory] =
    useState<string | null>(null);
  const [trajectoryComparisonKinds, setTrajectoryComparisonKinds] =
    useState<Record<ComparisonKind, boolean>>({ reference: true, optimizer: false, simulator: true, predicted: true });
  const setTrajectoryComparisonKind = useCallback((kind: ComparisonKind, visible: boolean) => {
    setTrajectoryComparisonKinds((prev) => ({ ...prev, [kind]: visible }));
  }, []);
  const [trajectorySampleCount, setTrajectorySampleCount] = useState<number>(200);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(60);
  const [autoReplay, setAutoReplay] = useState<boolean>(true);
  const [procedureVisibility, setProcedureVisibility] = useState<Record<string, boolean>>({});
  const [procedureAnnotationEnabled, setProcedureAnnotationEnabled] = useState(false);
  const [procedureWidthMeasurementEnabled, setProcedureWidthMeasurementEnabled] = useState(false);
  const [procedureDisplayLevel, setProcedureDisplayLevel] =
    useState<ProcedureDisplayLevel>("PROTECTION");
  const [selectedProcedureAnnotation, setSelectedProcedureAnnotation] =
    useState<ProcedureEntityAnnotation | null>(null);
  const [isApproachViewOpen, setApproachViewOpen] = useState(false);
  const [approachViewMode, setApproachViewMode] =
    useState<ApproachViewMode>("split");
  const [rangeRingRadiusKm, setRangeRingRadiusKm] = useState<number>(5);
  const [mode, setMode] = useState<WorkbenchMode>("observe");
  const [proceduresOpen, setProceduresOpen] = useState<boolean>(false);
  const [pilotTransport, setPilotTransport] = useState<PilotTransport | null>(null);
  const [presentationMode, setPresentationMode] = useState<boolean>(false);
  const [layersDrawerOpen, setLayersDrawerOpen] = useState<boolean>(false);
  const [rightInspectorCollapsed, setRightInspectorCollapsed] = useState<boolean>(false);
  const [airportLocalTerrain, setAirportLocalTerrain] = useState<AirportLocalTerrainState>({
    status: "disabled",
    airportCode: null,
    sourceLabel: null,
    sourceKind: null,
    sourceName: null,
    horizontalResolutionM: null,
    sourceCrsCode: null,
    sourceCrsName: null,
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
    terrainHeightTint: false,
    runways: true,
    waypoints: false,
    ocsSurfaces: false,
    trajectories: false,
    obstacles: false,
    obstacleLabels: false,
    procedures: false,
    rangeRing: false,
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
      setSelectedRunway(null);
      setTrajectoryDataSource(null);
      setProcedureVisibility({});
      setProcedureAnnotationEnabled(false);
      setProcedureWidthMeasurementEnabled(false);
      setProcedureDisplayLevel("PROTECTION");
      setSelectedProcedureAnnotation(null);
      setApproachViewOpen(false);
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
    rangeRingRadiusKm,
    setRangeRingRadiusKm,
  }), [airportLocalTerrain, layers, rangeRingRadiusKm, setViewer, toggleLayer, viewer]);
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
    selectedRunway,
    setSelectedRunway,
    trajectoryDataSource,
    setTrajectoryDataSource,
    optimizedTrajectoryDataSource,
    setOptimizedTrajectoryDataSource,
    trajectoryComparison,
    setTrajectoryComparison,
    trajectoryComparisonCategory,
    setTrajectoryComparisonCategory,
    trajectoryComparisonKinds,
    setTrajectoryComparisonKind,
    trajectorySampleCount,
    setTrajectorySampleCount,
  }), [selectedFlightId, selectedRunway, trajectoryDataSource, optimizedTrajectoryDataSource,
    trajectoryComparison, trajectoryComparisonCategory, trajectoryComparisonKinds,
    setTrajectoryComparisonKind, trajectorySampleCount]);
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
    autoReplay,
    setAutoReplay,
  }), [playbackSpeed, autoReplay]);
  const approachViewSessionState: ApproachViewSessionState = useMemo(() => ({
    isApproachViewOpen,
    setApproachViewOpen,
    approachViewMode,
    setApproachViewMode,
  }), [isApproachViewOpen, approachViewMode]);
  const workbenchUiState: WorkbenchUiState = useMemo(() => ({
    mode,
    setMode,
    proceduresOpen,
    setProceduresOpen,
    presentationMode,
    setPresentationMode,
    layersDrawerOpen,
    setLayersDrawerOpen,
    rightInspectorCollapsed,
    setRightInspectorCollapsed,
    pilotTransport,
    setPilotTransport,
  }), [mode, proceduresOpen, presentationMode, layersDrawerOpen, rightInspectorCollapsed, pilotTransport]);

  return (
    <AirportSessionContext.Provider value={airportSessionState}>
      <SceneContext.Provider value={sceneState}>
        <FlightSessionContext.Provider value={flightSessionState}>
          <ProcedureSessionContext.Provider value={procedureSessionState}>
            <PlaybackContext.Provider value={playbackState}>
              <ApproachViewSessionContext.Provider value={approachViewSessionState}>
                <WorkbenchUiContext.Provider value={workbenchUiState}>
                  {children}
                </WorkbenchUiContext.Provider>
              </ApproachViewSessionContext.Provider>
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
  const approachViewSessionState = useContext(ApproachViewSessionContext);
  const workbenchUiState = useContext(WorkbenchUiContext);
  if (
    !sceneState ||
    !airportSessionState ||
    !flightSessionState ||
    !procedureSessionState ||
    !playbackState ||
    !approachViewSessionState ||
    !workbenchUiState
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
    ...approachViewSessionState,
    ...workbenchUiState,
  };
}
