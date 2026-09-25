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
import type { ObservedVerdictFilter } from "../data/observedTracks";
import { trainingSelectionKey, type TrainingColumn, type TrainingSelection } from "../data/trainingSample";
import type {
  TrainingExecutorView, TrainingGenerationView, TrainingPriorView, TrainingSource,
} from "../data/trainingOverlays";
import type { TrainingAutopilotView, TrainingPick } from "../data/trainingAutopilot";

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
  error: string | null;
}

/**
 * THE LOCAL TERRAIN'S TILE COUNTS, apart from its state: they change on every tile the preload warms (hundreds per
 * airport), so they are a context of their own that `useApp` does not read — only the HUD, which shows them
 * (`useAirportLocalTerrainProgress`), re-renders with them. The state above changes only when the terrain's phase does.
 */
export interface AirportLocalTerrainProgress {
  loadedTiles: number;
  totalTiles: number;
}

export const NO_TERRAIN_TILES: AirportLocalTerrainProgress = { loadedTiles: 0, totalTiles: 0 };

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

  /** Status of the active airport-scoped local high-resolution terrain source; set when its phase changes */
  airportLocalTerrain: AirportLocalTerrainState;
  setAirportLocalTerrain: (state: AirportLocalTerrainState) => void;
  /** Its tile counts, set on every tile; read with `useAirportLocalTerrainProgress` (not here: see its type) */
  setAirportLocalTerrainProgress: (progress: AirportLocalTerrainProgress) => void;

  /** Sets the radius (km) of the airport-centred range ring drawn by the `rangeRing` layer. The radius itself moves on
   *  every step of the drawer's slider: read it with `useRangeRingRadiusKm`, which `useApp` does not spread. */
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

/** The three coloured trajectories in a prediction comparison. */
/**
 * A comparison entity's role. `optimizer`/`simulator` are the two halves of an optimizer
 * run (the NLP's plan and its true-dynamics replay); `predicted` is a learned forecast,
 * which has no such split — one trajectory, no controls. It is a separate kind rather than
 * reusing `optimizer` so the legend cannot claim a prediction is an optimizer plan.
 */
export type ComparisonKind = "reference" | "optimizer" | "simulator" | "predicted" | "lookback";

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

  /** When true, the Trajectories layer shows the 3-colour prediction comparison instead
   *  of the observed tracks (driven by the same runway selection). */
  trajectoryComparison: boolean;
  setTrajectoryComparison: (enabled: boolean) => void;

  /** Selected optimization category dir (e.g. "asdb"); which comparison set to show. */
  trajectoryComparisonCategory: string | null;
  setTrajectoryComparisonCategory: (categoryDir: string | null) => void;

  /** Per-kind visibility for the 3-colour comparison (reference / optimizer / simulator). */
  trajectoryComparisonKinds: Record<ComparisonKind, boolean>;
  setTrajectoryComparisonKind: (kind: ComparisonKind, visible: boolean) => void;

  /** How many trajectories to render (0 = all); baseline applies it inside the active verdict. */
  trajectorySampleCount: number;
  setTrajectorySampleCount: (count: number) => void;

  /** Verdict pool from which the observed baseline sample is chosen. */
  observedVerdictFilter: ObservedVerdictFilter;
  setObservedVerdictFilter: (filter: ObservedVerdictFilter) => void;
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
 * Training's shared selection. TrainingPanel owns the fetch and publishes the flight it is
 * showing; the full-width sentence bar (a sibling of the left dock, not a child), the read-back
 * window and the 3D layer draw it. One fetch, one selection, no second copy of the sample.
 */
interface TrainingSessionState {
  trainingSelection: TrainingSelection | null;
  setTrainingSelection: (selection: TrainingSelection | null) => void;
  /**
   * THE SELECTED WORD CLASS (a column), or null. What every view highlights is ONE word: this
   * column's word in force at the cursor — never the other columns' words at the same step, whose
   * runs start and end elsewhere. It outlives a change of flight; the cursor does not.
   */
  trainingColumn: TrainingColumn | null;
  setTrainingColumn: (column: TrainingColumn | null) => void;
  /**
   * WHICH ENVELOPES ARE DRAWN. The observed track has no switch: it is what every envelope is
   * read against. The switches reach every view at once (the 3D scene and the read-back plan and
   * charts), because an envelope present in one view and absent in another is how a reader comes
   * to compare two different pictures.
   */
  trainingLayers: TrainingLayers;
  setTrainingLayer: (layer: keyof TrainingLayers, on: boolean) => void;
  /**
   * WHAT ANOTHER MODEL MAKES OF THE SELECTED FLIGHT (`data/trainingOverlays.ts`): the executor's replay of its truth
   * sentence and the prior's predictions over it. The panel publishes each for the selected flight while its switch
   * is on and its overlay is loaded, and null otherwise. Kept apart from `trainingSelection`, so switching one on or
   * off redraws its own marks and never re-frames the flight.
   */
  trainingExecutor: TrainingExecutorView | null;
  setTrainingExecutor: (view: TrainingExecutorView | null) => void;
  trainingPrior: TrainingPriorView | null;
  setTrainingPrior: (view: TrainingPriorView | null) => void;
  /**
   * THE PRIOR'S OWN SENTENCES over the selected flight (`TrainingGenerationView`): one view per model whose sentences
   * are published for the open set and loaded, in the manifest's order; empty when there are none.
   */
  trainingGenerations: TrainingGenerationView[];
  setTrainingGenerations: (views: TrainingGenerationView[]) => void;
  /**
   * WHICH SENTENCE THE VIEWS READ: null for the truth — the labelled sentence of the observed flight — or one sample of
   * one model's own. Chosen in the sentence bar (its tabs and sample buttons) or the panel. It is kept across flights and
   * sets — reading one model's sentences flight after flight is the point — and a flight it has nothing for reads the
   * truth (`generationOnScreen`). The truth's track is drawn in 3D whichever is read.
   */
  trainingSource: TrainingSource | null;
  setTrainingSource: (source: TrainingSource | null) => void;
  /**
   * THE EXECUTOR, LIVE (`data/trainingAutopilot.ts`): the selected word's segment of the selected flight, flown by the
   * backend when a word is picked (`trainingPick`) — in flight, failed, or flown; null otherwise.
   */
  trainingAutopilot: TrainingAutopilotView | null;
  setTrainingAutopilot: (view: TrainingAutopilotView | null) => void;
  /** Fly the answer out again in 3D (a new `playedAt`); nothing when there is no answer. */
  replayTrainingAutopilot: () => void;
  /**
   * THE WORD THE LIVE EXECUTOR FLIES: set only by a CLICK — the sentence bar's "Fly this segment" button, or a band
   * clicked while `trainingAutopilotAuto` is on — and cleared by clicking the selected band again; never by the cursor,
   * which the charts move on hover. It belongs to the flight on screen (`trainingSelectionKey`) and is reset with it, as
   * the cursor is: another flight or another set starts with nothing picked. Leaving Training and coming back keeps
   * both, with the flight: the Training session outlives a task switch (`WorkbenchLeftDock`).
   */
  trainingPick: TrainingPick | null;
  setTrainingPick: (pick: TrainingPick | null) => void;
  /** A band clicked flies its word at once (the panel's switch, on at first); off, only the button flies. */
  trainingAutopilotAuto: boolean;
  setTrainingAutopilotAuto: (on: boolean) => void;
}

export interface TrainingLayers {
  /** The heading words' bands (instruction-v3): each word's target ± the tolerance over the rows it is judged on, on
   *  the heading chart; those rows on the ground in 3D; the rows outside in red wherever the track is drawn. */
  headingBands: boolean;
  /** The capture: its turn (from the clearance onto the course), the corridor, its centreline, and the course band
   *  after the capture. */
  corridor: boolean;
  /** The vertical envelopes: the altitude words' tubes (and, on the speed chart, the bands). */
  vertical: boolean;
  /** Every candidate runway with its extended centreline — the runway pointer's choices. */
  candidates: boolean;
}

/**
 * The active top-level task. These four are mutually exclusive — one drives the
 * left dock at a time. `fly`/`optimize`/`compare` map onto the PilotPanel's
 * pilot/trajectory/comparison sub-modes. Procedures is intentionally NOT a mode:
 * it is an independent panel (`proceduresOpen`) that coexists with any task.
 */
export type WorkbenchMode = "observe" | "training" | "fly" | "optimize" | "compare";

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

/**
 * THE TRAINING CURSOR: flight-relative time shared by the sentence bar, the read-back charts and the 3D scene. It moves
 * on every mousemove over a chart, so it is a context of its own that `useApp` does NOT read: only what draws the cursor
 * (`useTrainingCursor`) re-renders when it moves, never the ~50 other consumers of `useApp` or the app shell. It belongs
 * to the flight on screen and is reset with it (`trainingSelectionKey`).
 */
interface TrainingCursorState {
  trainingCursorS: number;
  setTrainingCursorS: (atS: number) => void;
}

interface AppState extends
  SceneState,
  AirportSessionState,
  FlightSessionState,
  ProcedureSessionState,
  PlaybackState,
  ApproachViewSessionState,
  TrainingSessionState,
  WorkbenchUiState {}

// The defaults are `null`; useApp asserts all providers are present so consumers
// get a helpful error if they forget to wrap with AppProvider.
const SceneContext = createContext<SceneState | null>(null);
const AirportSessionContext = createContext<AirportSessionState | null>(null);
const FlightSessionContext = createContext<FlightSessionState | null>(null);
const ProcedureSessionContext = createContext<ProcedureSessionState | null>(null);
const PlaybackContext = createContext<PlaybackState | null>(null);
const ApproachViewSessionContext = createContext<ApproachViewSessionState | null>(null);
const TrainingSessionContext = createContext<TrainingSessionState | null>(null);
const TrainingCursorContext = createContext<TrainingCursorState | null>(null);
const AirportLocalTerrainProgressContext = createContext<AirportLocalTerrainProgress | null>(null);
const RangeRingRadiusContext = createContext<number | null>(null);
const WorkbenchUiContext = createContext<WorkbenchUiState | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────
export function AppProvider({ children }: { children: ReactNode }) {
  const [viewer, setViewerState] = useState<Cesium.Viewer | null>(null);
  const [airports, setAirports] = useState<AirportCatalogItem[]>([]);
  const [activeAirportCode, setActiveAirportCodeState] = useState<string>("");
  const [airport, setAirport] = useState<AirportConfig | null>(null);
  const [selectedFlightId, setSelectedFlightId] = useState<string | null>(null);
  const [trainingSelection, setTrainingSelection] = useState<TrainingSelection | null>(null);
  // The cursor and the pick belong to the flight on screen, and are reset with it before any consumer paints the new
  // flight with the last one's time or flies the last one's word.
  const trainingScope = trainingSelectionKey(trainingSelection);
  const [trainingCursor, setTrainingCursor] = useState<{ scope: string | null; atS: number }>({ scope: null, atS: 0 });
  if (trainingCursor.scope !== trainingScope) setTrainingCursor({ scope: trainingScope, atS: 0 });
  const trainingCursorS = trainingCursor.scope === trainingScope ? trainingCursor.atS : 0;
  // a setter captured before the flight changed is of the last flight: it writes nothing
  const setTrainingCursorS = useCallback((atS: number) => {
    setTrainingCursor((current) => (current.scope === trainingScope ? { scope: trainingScope, atS } : current));
  }, [trainingScope]);
  const [trainingPicked, setTrainingPicked] = useState<{ scope: string | null; pick: TrainingPick | null }>({
    scope: null, pick: null,
  });
  if (trainingPicked.scope !== trainingScope) setTrainingPicked({ scope: trainingScope, pick: null });
  const trainingPick = trainingPicked.scope === trainingScope ? trainingPicked.pick : null;
  const setTrainingPick = useCallback((pick: TrainingPick | null) => {
    setTrainingPicked((current) => (current.scope === trainingScope ? { scope: trainingScope, pick } : current));
  }, [trainingScope]);
  const [trainingColumn, setTrainingColumn] = useState<TrainingColumn | null>(null);
  const [trainingLayers, setTrainingLayers] = useState<TrainingLayers>({
    headingBands: true, corridor: true, vertical: true, candidates: true,
  });
  const setTrainingLayer = useCallback((layer: keyof TrainingLayers, on: boolean) => {
    setTrainingLayers((current) => ({ ...current, [layer]: on }));
  }, []);
  const [trainingExecutor, setTrainingExecutor] = useState<TrainingExecutorView | null>(null);
  const [trainingPrior, setTrainingPrior] = useState<TrainingPriorView | null>(null);
  const [trainingGenerations, setTrainingGenerations] = useState<TrainingGenerationView[]>([]);
  const [trainingSource, setTrainingSource] = useState<TrainingSource | null>(null);
  const [trainingAutopilot, setTrainingAutopilot] = useState<TrainingAutopilotView | null>(null);
  const replayTrainingAutopilot = useCallback(() => {
    setTrainingAutopilot((view) => (view?.status === "ready" ? { ...view, playedAt: Date.now() } : view));
  }, []);
  const [trainingAutopilotAuto, setTrainingAutopilotAuto] = useState<boolean>(true);
  const [selectedRunway, setSelectedRunway] = useState<string | null>(null);
  const [trajectoryDataSource, setTrajectoryDataSource] =
    useState<Cesium.CzmlDataSource | null>(null);
  const [optimizedTrajectoryDataSource, setOptimizedTrajectoryDataSource] =
    useState<Cesium.CzmlDataSource | null>(null);
  const [trajectoryComparison, setTrajectoryComparison] = useState<boolean>(false);
  const [trajectoryComparisonCategory, setTrajectoryComparisonCategory] =
    useState<string | null>(null);
  const [trajectoryComparisonKinds, setTrajectoryComparisonKinds] =
    useState<Record<ComparisonKind, boolean>>({
      reference: true, optimizer: false, simulator: true, predicted: true, lookback: true });
  const setTrajectoryComparisonKind = useCallback((kind: ComparisonKind, visible: boolean) => {
    setTrajectoryComparisonKinds((prev) => ({ ...prev, [kind]: visible }));
  }, []);
  const [trajectorySampleCount, setTrajectorySampleCount] = useState<number>(200);
  const [observedVerdictFilter, setObservedVerdictFilter] =
    useState<ObservedVerdictFilter>("all");
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
    error: null,
  });
  const [airportLocalTerrainProgress, setAirportLocalTerrainProgress] =
    useState<AirportLocalTerrainProgress>(NO_TERRAIN_TILES);

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
      setAirportLocalTerrainProgress(NO_TERRAIN_TILES);
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
    setAirportLocalTerrainProgress,
    setRangeRingRadiusKm,
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
    observedVerdictFilter,
    setObservedVerdictFilter,
  }), [selectedFlightId, selectedRunway, trajectoryDataSource, optimizedTrajectoryDataSource,
    trajectoryComparison, trajectoryComparisonCategory, trajectoryComparisonKinds,
    setTrajectoryComparisonKind, trajectorySampleCount, observedVerdictFilter]);
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
  const trainingSessionState: TrainingSessionState = useMemo(() => ({
    trainingSelection,
    setTrainingSelection,
    trainingColumn,
    setTrainingColumn,
    trainingLayers,
    setTrainingLayer,
    trainingExecutor,
    setTrainingExecutor,
    trainingPrior,
    setTrainingPrior,
    trainingGenerations,
    setTrainingGenerations,
    trainingSource,
    setTrainingSource,
    trainingAutopilot,
    setTrainingAutopilot,
    replayTrainingAutopilot,
    trainingPick,
    setTrainingPick,
    trainingAutopilotAuto,
    setTrainingAutopilotAuto,
  }), [trainingSelection, trainingColumn, trainingLayers, setTrainingLayer, trainingExecutor, trainingPrior,
    trainingGenerations, trainingSource, trainingAutopilot, replayTrainingAutopilot, trainingPick, setTrainingPick,
    trainingAutopilotAuto]);
  const trainingCursorState: TrainingCursorState = useMemo(() => ({ trainingCursorS, setTrainingCursorS }),
    [trainingCursorS, setTrainingCursorS]);
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
                <TrainingSessionContext.Provider value={trainingSessionState}>
                  <WorkbenchUiContext.Provider value={workbenchUiState}>
                    <TrainingCursorContext.Provider value={trainingCursorState}>
                      <AirportLocalTerrainProgressContext.Provider value={airportLocalTerrainProgress}>
                        <RangeRingRadiusContext.Provider value={rangeRingRadiusKm}>
                          {children}
                        </RangeRingRadiusContext.Provider>
                      </AirportLocalTerrainProgressContext.Provider>
                    </TrainingCursorContext.Provider>
                  </WorkbenchUiContext.Provider>
                </TrainingSessionContext.Provider>
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
  const trainingSessionState = useContext(TrainingSessionContext);
  const workbenchUiState = useContext(WorkbenchUiContext);
  if (
    !sceneState ||
    !airportSessionState ||
    !flightSessionState ||
    !procedureSessionState ||
    !playbackState ||
    !approachViewSessionState ||
    !trainingSessionState ||
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
    ...trainingSessionState,
    ...workbenchUiState,
  };
}

/** The Training cursor (`TrainingCursorState`): only for what draws it — the sentence bar and the 3D scene's leaf. */
export function useTrainingCursor(): TrainingCursorState {
  const cursor = useContext(TrainingCursorContext);
  if (!cursor) throw new Error("useTrainingCursor() was called outside of <AppProvider>.");
  return cursor;
}

/** The local terrain's tile counts (`AirportLocalTerrainProgress`): only for what shows them — the HUD. */
export function useAirportLocalTerrainProgress(): AirportLocalTerrainProgress {
  const progress = useContext(AirportLocalTerrainProgressContext);
  if (!progress) throw new Error("useAirportLocalTerrainProgress() was called outside of <AppProvider>.");
  return progress;
}

/** The range ring's radius (km): only for what draws it — the ring layer and the drawer's slider. */
export function useRangeRingRadiusKm(): number {
  const radiusKm = useContext(RangeRingRadiusContext);
  if (radiusKm === null) throw new Error("useRangeRingRadiusKm() was called outside of <AppProvider>.");
  return radiusKm;
}
