import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { useApp, type WorkbenchMode } from "../context/AppContext";
import {
  fetchRunwayThresholdTargets,
  type RunwayThresholdTarget,
} from "../data/runwayThresholdTargets";
import {
  fetchRnavInitialFixCandidates,
  type RnavInitialFixCandidate,
} from "../data/rnavInitialFixCandidates";
import {
  procedureDetailsDocumentUrl,
  type ProcedureDetailDocument,
} from "../data/procedureDetails";
import {
  buildProcedureConstraint,
  procedureThresholdAnchor,
  type ProcedureConstraint,
} from "../data/procedureConstraint";
import { fetchJson } from "../utils/fetchJson";
import { usePilotAircraft, type PilotAircraftPose } from "../hooks/usePilotAircraft";
import { usePilotTargetGate } from "../hooks/usePilotTargetGate";
import { useOptimizedTrajectoryPlayback } from "../hooks/useOptimizedTrajectoryPlayback";
import { useDynamicsComparisonPlayback } from "../hooks/useDynamicsComparisonPlayback";
import DynamicsComparisonCharts from "./DynamicsComparisonCharts";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import PilotInitialStateOverlay, {
  EnglishNumberInput,
  formatCoord,
  formatNumberInputValue,
  type PilotInitialEditableKey,
} from "./PilotInitialStateOverlay";
import PilotRealtimeStatePanel from "./PilotRealtimeStatePanel";
import PilotTargetStateOverlay, {
  type PilotTargetEditableKey,
  type PilotTargetState,
} from "./PilotTargetStateOverlay";
import {
  usePilotInitialPlacement,
  type PilotInitialPlacementPosition,
} from "../hooks/usePilotInitialPlacement";
import {
  fetchPilotAircraftConfigs,
  resetPilotSimulation,
  stepPilotSimulation,
  type PilotAircraftConfig,
  type PilotAircraftType,
  type PilotControls,
  type PilotResetState,
  type PilotSimulationMode,
  type PilotSnapshot,
} from "../pilot/pilotClient";
import {
  clampHeadingToRunwayTolerance,
  clampTargetSpeedMps,
  defaultTargetSpeedMps,
  knotsToMetresPerSecond,
  runwayAlignedHeadingDeg,
  targetAltitudeMForThreshold,
  targetSpeedBoundsMps,
} from "../pilot/trajectoryTargetConstraints";
import { bareRunwayIdent } from "../utils/runwayIdent";
import { useForcedProcedureDisplay } from "../hooks/useForcedProcedureDisplay";
import ApproachViewToggle from "./ApproachViewToggle";
import {
  runTrajectoryOptimization,
  decomposeOptimizer,
  composeOptimizer,
  validFittingsForFrame,
  type TrajectoryOptimizer,
  type TrajectoryOptimizationResult,
  type TrajectorySample,
  type OptimizerParts,
  type OptimizerFrame,
  type OptimizerTransport,
  type OptimizerFitting,
} from "../pilot/trajectoryOptimizationClient";
import {
  runDynamicsComparison,
  averageDynamicsComparisonHistory,
  clearDynamicsComparisonHistory,
  fetchDynamicsComparisonHistoryCount,
  type DynamicsComparisonAverage,
  type DynamicsComparisonControl,
  type DynamicsComparisonDeltas,
  type DynamicsComparisonResult,
  type DynamicsComparisonSystem,
} from "../pilot/dynamicsComparisonClient";
import {
  openWorkerSession,
  closeWorkerSession,
  beaconCloseWorkerSession,
  type WorkerSessionKind,
} from "../pilot/workerSessionClient";
import { haversineDistanceM } from "../utils/procedureGeoMath";

const DEFAULT_SIMULATION_MODE: PilotSimulationMode = "alpha";
const DEFAULT_BANK_DEG = 45;
const DEFAULT_LOAD_FACTOR = 1.414214;
const DEFAULT_THRUST_N = 67000;
const MIN_LOAD_FACTOR = 0;
const MAX_LOAD_FACTOR = 3;
const DEFAULT_CONTROLS: PanelControls = makeDefaultControls(null);
const DEFAULT_INTEGRATOR_DT_S = 0.2;
const PLAYBACK_FRAME_DT_S = 0.2;
const STEP_INTERVAL_MS = 120;
const MAX_TRAIL_POINTS = 360;
const DEFAULT_TARGET_GAMMA_DEG = -3;
const DEFAULT_MAX_ITERATIONS = 300;
// Constrained (multiphase) control segments per procedure leg — aligned with the backend +
// CollocationOptimizer's own default (3), so a frontend solve matches the batch pipeline.
const DEFAULT_N_SEG_PER_PHASE = 3;
// A full RNAV approach optimizes to ~250-350 s, and for the multiphase optimizer this is the
// per-phase cap (the longest leg alone is ~130 s), so the default must clear that comfortably.
const DEFAULT_ARRIVAL_TIME_S = 600;
const DEFAULT_COMPARISON_DURATION_S = 240;
const DEFAULT_COMPARISON_DT_S = 0.1;
// Hermite-Simpson default: with the ψ corridor both fittings converge everywhere, and the
// measured playback fidelity is ~0.7-0.9 m (HS, 4th order) vs 227-296 m (trapezoidal, 2nd
// order) on doglegged approaches for ~2x the solve time — see CLAUDE.md 2026-07-05.
const DEFAULT_TRAJECTORY_OPTIMIZER: TrajectoryOptimizer = "casadiMultiphaseNormalizedFullTransport";
// The optimizer is edited as orthogonal axes (see trajectoryOptimizationClient): the constraint
// MODE, the base frame, and — for the geodetic frame — transport + normalization.
const OPTIMIZER_CONSTRAINTS_OPTIONS: { value: "none" | "procedure"; label: string }[] = [
  { value: "none", label: "None — direct (initial → target)" },
  { value: "procedure", label: "RNAV procedure — per-leg constraints" },
];
const OPTIMIZER_FRAME_OPTIONS: { value: OptimizerFrame; label: string }[] = [
  { value: "geodetic", label: "Geodetic RHS" },
  { value: "localEnu", label: "Local ENU @ target (fixed tangent, drifts far out)" },
  { value: "reanchoredEnu", label: "Re-anchored ENU (playback model)" },
];
const OPTIMIZER_TRANSPORT_OPTIONS: { value: OptimizerTransport; label: string }[] = [
  { value: "approx", label: "Approx (drops ψ cross term)" },
  { value: "full", label: "Full / exact" },
];
const OPTIMIZER_FITTING_OPTIONS: { value: OptimizerFitting; label: string }[] = [
  { value: "hermiteSimpson", label: "Hermite-Simpson (cubic, 4th order)" },
  { value: "trapezoidal", label: "Trapezoidal (linear, 2nd order)" },
  { value: "shooting", label: "RK4 / shooting (4th order)" },
];
const FALLBACK_MAX_THRUST_N = 240000;
/** Stable empty-samples reference so the playback hook deps don't churn. */
const EMPTY_SAMPLES: TrajectorySample[] = [];
/** The single pseudo-"system" backing the Trajectory-Play target-deviation delta
 * chips — reuses the dynamics comparison's delta strip with one amber chip per row. */
const TARGET_DELTA_KEY = "Δ";
const TARGET_DELTA_SYSTEMS: DynamicsComparisonSystem[] = [
  { key: TARGET_DELTA_KEY, label: "final − target", colorRgba: [251, 191, 36, 255], isReference: false },
];

function usesLoadFactorControl(mode: PilotSimulationMode) {
  return mode === "loadFactor" || mode === "casadi";
}

/** The panel serves two workbench tasks and takes the task itself as its mode. */
type PilotPanelMode = Extract<WorkbenchMode, "fly" | "optimize">;

/**
 * The panel's controls carry BOTH parameterisations' fields (alpha and load factor): the
 * Simulation select decides which one is flown, and the dynamics comparison always flies the
 * load-factor one.
 */
type PanelControls = Required<PilotControls>;

interface PlacementBackup {
  initialState: PilotResetState;
  isInitialPreviewVisible: boolean;
  isEnabled: boolean;
  isFlying: boolean;
  liveSnapshot: PilotSnapshot | null;
  trail: PilotAircraftPose[];
}

interface PilotPanelProps {
  /** The workbench task the panel serves — the top bar's task switcher owns it. */
  mode: PilotPanelMode;
  /**
   * The panel stays mounted but out of sight, its scene off (no playback on the clock, no target gate, no forced procedure
   * display, no live-state readout): Optimize shows a multi-aircraft panel instead, and a task or mode switch must not lose
   * what was set up here. Showing it again is a task switch like Fly ↔ Optimize (the playbacks are released).
   */
  hidden?: boolean;
}

export default function PilotPanel({ mode: activeMode, hidden = false }: PilotPanelProps) {
  const {
    activeAirportCode,
    airport,
    viewer,
    setPilotTransport,
  } = useApp();
  const [isEnabled, setIsEnabled] = useState(false);
  const [isFlying, setIsFlying] = useState(false);
  // Optimized-trajectory playback runs on Cesium's own clock from a backend CZML.
  // `isTrajectoryPlaying` mirrors the intended clock animation (play vs pause).
  // ("Is the CZML loaded" — isTrajectoryPlaybackActive — is DERIVED below from the
  // active mode + a computed result, so the shared clock transport stays bound to
  // the current mode instead of unloading on every switch.)
  const [isTrajectoryPlaying, setIsTrajectoryPlaying] = useState(false);
  const [isInitialEditorOpen, setIsInitialEditorOpen] = useState(false);
  const [isTargetEditorOpen, setIsTargetEditorOpen] = useState(false);
  const [isPlacingInitialPosition, setIsPlacingInitialPosition] = useState(false);
  const [isInitialPreviewVisible, setIsInitialPreviewVisible] = useState(false);
  const [isFollowing, setIsFollowing] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Why the aircraft catalog did not load: held apart from `error`, which every action clears, so it stays on screen
  // for as long as there is no aircraft — in the very modes whose target reads "— (no aircraft)".
  const [catalogError, setCatalogError] = useState<string | null>(null);
  // the catalog is asked for once per mount, and again by the alert's Retry (the backend may come up after the page)
  const [catalogAttempt, setCatalogAttempt] = useState<number>(0);
  const [aircraftConfigs, setAircraftConfigs] = useState<PilotAircraftConfig[]>([]);
  const [simulationMode, setSimulationMode] =
    useState<PilotSimulationMode>(DEFAULT_SIMULATION_MODE);
  const [controls, setControls] = useState<PanelControls>(DEFAULT_CONTROLS);
  const [integratorDtS, setIntegratorDtS] = useState(DEFAULT_INTEGRATOR_DT_S);
  // The live flight's state (the backend sim's), and apart from it the state a CZML playback
  // (Optimize's, Fly's comparison) samples at the clock time — one is never read as the other.
  const [liveSnapshot, setLiveSnapshot] = useState<PilotSnapshot | null>(null);
  const [playbackSnapshot, setPlaybackSnapshot] = useState<PilotSnapshot | null>(null);
  const [trail, setTrail] = useState<PilotAircraftPose[]>([]);
  const [runwayTargets, setRunwayTargets] = useState<RunwayThresholdTarget[]>([]);
  const [targetState, setTargetState] = useState<PilotTargetState>(() =>
    makeDefaultTrajectoryTarget(null, null),
  );
  // The optimizer is edited as orthogonal axes (constraints mode, frame, transport, normalization,
  // fitting). Holding the AXES as the source of truth — not the composed wire string — is what lets
  // the hidden advanced axes survive toggling Constraints (the wire string can't encode them in
  // constrained mode). The wire name is derived from the axes for the request + playback.
  const [optimizerParts, setOptimizerParts] = useState<OptimizerParts>(() =>
    decomposeOptimizer(DEFAULT_TRAJECTORY_OPTIMIZER),
  );
  const trajectoryOptimizer = useMemo(
    () => composeOptimizer(optimizerParts),
    [optimizerParts],
  );
  const [nSegments, setNSegments] = useState(10);
  // Constrained (procedure) mode: control segments PER LEG (total = legs × this).
  const [nSegPerPhase, setNSegPerPhase] = useState(DEFAULT_N_SEG_PER_PHASE);
  // State-collocation subintervals per control segment (M); 0 = the optimizer's auto
  // per-phase density (~3 s state step, capped at 16).
  const [stateSubsteps, setStateSubsteps] = useState(0);
  const [showAdvancedNumerics, setShowAdvancedNumerics] = useState(false);
  const [arrivalTimeS, setArrivalTimeS] = useState(DEFAULT_ARRIVAL_TIME_S);
  const [maxIterations, setMaxIterations] = useState(DEFAULT_MAX_ITERATIONS);
  const [optimizedTrajectory, setOptimizedTrajectory] =
    useState<TrajectoryOptimizationResult | null>(null);

  // Fly's second run, the dynamics comparison: the panel's controls held fixed and the start
  // flown four ways, replayed as a multi-system CZML with a pop-up deviation chart.
  const [comparisonDurationS, setComparisonDurationS] = useState(
    DEFAULT_COMPARISON_DURATION_S,
  );
  const [comparisonDtS, setComparisonDtS] = useState(DEFAULT_COMPARISON_DT_S);
  const [comparisonResult, setComparisonResult] =
    useState<DynamicsComparisonResult | null>(null);
  const [isComparisonPlaying, setIsComparisonPlaying] = useState(false);
  const [hiddenComparisonKeys, setHiddenComparisonKeys] = useState<string[]>([]);
  // A mode's playback is LOADED (its CZML on the clock) exactly while you are in
  // that mode AND it has a computed result. Deriving this — rather than toggling
  // it on Play and clearing it on every mode switch — keeps the shared bottom bar
  // + native clock dial bound to the current mode's trajectory the moment you
  // enter it (the playback hooks load it paused; Play animates).
  const isTrajectoryPlaybackActive =
    !hidden && activeMode === "optimize" && optimizedTrajectory?.playback != null;
  const isComparisonPlaybackActive =
    !hidden && activeMode === "fly" && comparisonResult != null;
  // The state of the run on screen: Fly's live flight, or a loaded playback's sample.
  const snapshot = activeMode === "fly" && !isComparisonPlaybackActive ? liveSnapshot : playbackSnapshot;
  // A/C/D deviations vs the reference B at the current clock time, overlaid on
  // the Live-State readout during a comparison playback.
  const [comparisonDeltas, setComparisonDeltas] =
    useState<DynamicsComparisonDeltas | null>(null);
  const [isChartsOpen, setIsChartsOpen] = useState(false);
  // Run history (#2/#3): count of stored runs + the backend-averaged result.
  // `chartMode` selects which chart the overlay shows: this run vs the average.
  const [comparisonHistoryCount, setComparisonHistoryCount] = useState(0);
  const [averagedComparison, setAveragedComparison] =
    useState<DynamicsComparisonAverage | null>(null);
  const [chartMode, setChartMode] = useState<"run" | "average">("run");

  const controlsRef = useRef(controls);
  const simulationModeRef = useRef(simulationMode);
  const integratorDtRef = useRef(integratorDtS);
  const stepInFlightRef = useRef(false);
  const placementBackupRef = useRef<PlacementBackup | null>(null);

  const [initialState, setInitialState] = useState<PilotResetState>(() =>
    makeDefaultInitialState(null, null),
  );
  // The runway's published RNAV fixes and the runway they were read for (null: none read).
  const [rnavFixes, setRnavFixes] = useState<{
    runwayIdent: string;
    candidates: RnavInitialFixCandidate[];
  } | null>(null);
  const rnavInitialFixCandidates = rnavFixes?.candidates ?? [];
  const [selectedRnavInitialFixKey, setSelectedRnavInitialFixKey] = useState("");

  const pose = snapshotToPose(liveSnapshot);
  const selectedAircraft = aircraftConfigs.find(
    (config) => config.code === initialState.aircraftType,
  ) ?? aircraftConfigs[0] ?? null;
  const selectedMaxThrustN = selectedAircraft?.maxThrustN ?? FALLBACK_MAX_THRUST_N;
  const targetSpeedBounds = selectedAircraft ? targetSpeedBoundsMps(selectedAircraft) : null;
  const selectedTargetRunway = runwayTargets.find(
    (target) => target.id === targetState.runwayThresholdId,
  );
  const placementGuidance = useMemo(
    () =>
      selectedAircraft && selectedTargetRunway
        ? {
            aircraft: selectedAircraft,
            runway: selectedTargetRunway,
          }
        : null,
    [selectedAircraft, selectedTargetRunway],
  );
  const targetGateState = useMemo(
    () =>
      selectedTargetRunway
        ? {
            runwayThresholdId: targetState.runwayThresholdId,
            runwayIdent: selectedTargetRunway.runwayIdent,
            lon: targetState.lon,
            lat: targetState.lat,
            altM: targetState.altM,
            headingDeg: targetState.headingDeg,
          }
        : null,
    [
      selectedTargetRunway,
      targetState.altM,
      targetState.headingDeg,
      targetState.lat,
      targetState.lon,
      targetState.runwayThresholdId,
    ],
  );

  // Invalidate any computed/loaded optimized trajectory. Used whenever an input
  // that feeds the optimizer changes, so a stale CZML never stays on the clock.
  const clearOptimizedPlayback = useCallback(() => {
    setOptimizedTrajectory(null);
    setIsTrajectoryPlaying(false);
  }, []);

  // Invalidate any computed/loaded dynamics comparison. Used whenever an input
  // feeding the comparison changes, so a stale CZML/chart never stays on screen.
  // The run's chart is drawn only beside its result, so it goes with it; the averaged
  // history chart does not depend on the panel's inputs and stays.
  const clearComparisonPlayback = useCallback(() => {
    setComparisonResult(null);
    setIsComparisonPlaying(false);
  }, []);

  const clearSnapshotForInitialEdit = useCallback(() => {
    clearOptimizedPlayback();
    clearComparisonPlayback();
    if (!liveSnapshot && !isEnabled && !isFlying && !isTrajectoryPlaying) return;

    setIsFlying(false);
    setIsEnabled(false);
    setLiveSnapshot(null);
    setTrail([]);
  }, [
    clearOptimizedPlayback,
    clearComparisonPlayback,
    isEnabled,
    isFlying,
    isTrajectoryPlaying,
    liveSnapshot,
  ]);

  const updateInitialPosition = useCallback(
    (position: PilotInitialPlacementPosition) => {
      // Deliberately KEEPS the RNAV IF selection: the selector identifies the PROCEDURE (entry
      // fix); the start is independent. The multiphase optimizer flies a free transition from a
      // custom start and must PASS the selected fix within the leg's k·RNP disc — clearing the
      // selection here used to make a custom-start constrained optimize impossible.
      setInitialState((current) => ({
        ...current,
        lon: clamp(position.lon, -180, 180),
        lat: clamp(position.lat, -90, 90),
        altM: position.altM ?? current.altM,
        headingDeg: position.headingDeg ?? current.headingDeg,
        flightPathDeg: position.flightPathDeg ?? current.flightPathDeg,
        speedMps: selectedAircraft
          ? defaultInitialSpeedMps(selectedAircraft)
          : current.speedMps,
      }));
      clearSnapshotForInitialEdit();
    },
    [clearSnapshotForInitialEdit, selectedAircraft],
  );

  const finishInitialPlacement = useCallback(() => {
    placementBackupRef.current = null;
    setIsInitialPreviewVisible(true);
    setIsPlacingInitialPosition(false);
  }, []);

  const cancelInitialPlacement = useCallback(() => {
    const backup = placementBackupRef.current;
    if (backup) {
      setInitialState(backup.initialState);
      setIsInitialPreviewVisible(backup.isInitialPreviewVisible);
      setIsEnabled(backup.isEnabled);
      setIsFlying(backup.isFlying);
      setLiveSnapshot(backup.liveSnapshot);
      setTrail(backup.trail);
    }

    placementBackupRef.current = null;
    setIsPlacingInitialPosition(false);
    setIsInitialEditorOpen(false);
  }, []);

  const openInitialEditor = useCallback(() => {
    if (isFlying || isTrajectoryPlaying || isBusy || aircraftConfigs.length === 0) return;

    setError(null);
    setIsInitialEditorOpen(true);
    setIsInitialPreviewVisible(true);
    clearSnapshotForInitialEdit();
  }, [
    aircraftConfigs.length,
    clearSnapshotForInitialEdit,
    isBusy,
    isFlying,
    isTrajectoryPlaying,
  ]);

  const closeInitialEditor = useCallback(() => {
    if (isPlacingInitialPosition) {
      cancelInitialPlacement();
      return;
    }

    setIsInitialEditorOpen(false);
  }, [cancelInitialPlacement, isPlacingInitialPosition]);

  const toggleInitialPlacement = useCallback(() => {
    if (isPlacingInitialPosition) {
      cancelInitialPlacement();
      return;
    }

    if (isFlying || isTrajectoryPlaying || isBusy || aircraftConfigs.length === 0) return;

    placementBackupRef.current = {
      initialState,
      isInitialPreviewVisible,
      isEnabled,
      isFlying,
      liveSnapshot,
      trail,
    };
    setIsFlying(false);
    setIsEnabled(false);
    setError(null);
    setIsInitialEditorOpen(true);
    setIsInitialPreviewVisible(true);
    setIsPlacingInitialPosition(true);
  }, [
    cancelInitialPlacement,
    initialState,
    isInitialPreviewVisible,
    isBusy,
    isEnabled,
    isFlying,
    isTrajectoryPlaying,
    isPlacingInitialPosition,
    liveSnapshot,
    trail,
    aircraftConfigs.length,
  ]);

  // Hidden (a multi-aircraft Optimize is shown): no globe click may move this panel's start, so an active placement is
  // cancelled (its backup restored) and neither the placement nor its preview is drawn.
  useEffect(() => {
    if (hidden && isPlacingInitialPosition) cancelInitialPlacement();
  }, [hidden, isPlacingInitialPosition, cancelInitialPlacement]);

  usePilotInitialPlacement({
    enabled: !hidden && isPlacingInitialPosition,
    // The static "START" preview marks the chosen start state while setting up.
    // Hide it once a live flight or a playback holds the screen (a loaded comparison
    // samples its state only while it plays, so without its own guard the START aircraft
    // would sit at the origin while the per-system models fly away).
    previewVisible: !hidden && (isPlacingInitialPosition ||
      ((isInitialEditorOpen || isInitialPreviewVisible) &&
        !isEnabled &&
        !liveSnapshot &&
        !playbackSnapshot &&
        !isComparisonPlaybackActive)),
    initialState,
    placementGuidance,
    onPositionChange: updateInitialPosition,
    onFinish: finishInitialPlacement,
    onCancel: cancelInitialPlacement,
  });

  // The hand-built aircraft + trail are only for the live flight. The comparison and
  // Trajectory Play draw their aircraft and colored trajectories from a CZML instead.
  usePilotAircraft({
    enabled: isEnabled && activeMode === "fly",
    pose,
    trail,
    follow: isFollowing,
  });

  usePilotTargetGate({
    enabled: !hidden && activeMode === "optimize",
    target: targetGateState,
  });

  // Drive the live readout from the optimized rollout sampled at the clock time. Every
  // optimizer emits load-factor controls, so playback always reads in the "casadi" mode.
  const handlePlaybackSample = useCallback(
    (sample: TrajectorySample | null) => {
      if (!sample) {
        setPlaybackSnapshot(null);
        return;
      }
      setPlaybackSnapshot(
        trajectorySampleToSnapshot(
          sample,
          "casadi",
          initialState.aircraftType,
          initialState.massKg,
        ),
      );
    },
    [initialState.aircraftType, initialState.massKg],
  );

  useOptimizedTrajectoryPlayback({
    enabled: isTrajectoryPlaybackActive,
    czml: optimizedTrajectory?.playback?.czml ?? null,
    samples: optimizedTrajectory?.playback?.samples ?? EMPTY_SAMPLES,
    follow: isFollowing,
    onSample: handlePlaybackSample,
  });

  // Drive the live readout from the reference-B rollout (the comparison always
  // uses the casadi/load-factor parameterisation).
  const handleComparisonSample = useCallback(
    (sample: TrajectorySample | null) => {
      if (!sample) {
        setPlaybackSnapshot(null);
        return;
      }
      setPlaybackSnapshot(
        trajectorySampleToSnapshot(
          sample,
          "casadi",
          initialState.aircraftType,
          initialState.massKg,
        ),
      );
    },
    [initialState.aircraftType, initialState.massKg],
  );

  useDynamicsComparisonPlayback({
    enabled: isComparisonPlaybackActive,
    czml: comparisonResult?.playback.czml ?? null,
    hiddenKeys: hiddenComparisonKeys,
    follow: isFollowing,
    samples: comparisonResult?.playback.samples ?? EMPTY_SAMPLES,
    onSample: handleComparisonSample,
    chart: comparisonResult?.chart ?? null,
    onDeltas: setComparisonDeltas,
  });

  // Trajectory Play: the aircraft's live deviation from the requested target,
  // shown as one amber delta chip per state row (the same style as the comparison's
  // deviations) instead of separate "X Error" rows. It tracks the sampled state,
  // so it converges to the final-vs-target error as playback reaches the end.
  const trajectoryTargetDeltas = useMemo<DynamicsComparisonDeltas | null>(() => {
    if (activeMode !== "optimize" || !snapshot) return null;
    const s = snapshot.state;
    return {
      [TARGET_DELTA_KEY]: {
        horiz: haversineDistanceM(
          { latDeg: s.lat, lonDeg: s.lon, altM: 0 },
          { latDeg: targetState.lat, lonDeg: targetState.lon, altM: 0 },
        ),
        alt: s.altM - targetState.altM,
        head: headingMagnitudeDeg(s.headingDeg, targetState.headingDeg),
        speed: s.speedMps - targetState.speedMps,
        fpa: s.flightPathDeg - targetState.flightPathDeg,
      },
    };
  }, [
    activeMode,
    snapshot,
    targetState.lat,
    targetState.lon,
    targetState.altM,
    targetState.headingDeg,
    targetState.speedMps,
    targetState.flightPathDeg,
  ]);

  // Keep the backend's casadi worker resident (warm) while the panel's task is open —
  // Optimize's solver, Fly's dynamics comparison — and decommission it when the task
  // closes, so repeated runs are fast but the worker's memory is reclaimed once the
  // user leaves. A `pagehide` beacon also releases it when the whole tab/window
  // closes, where this cleanup would not run.
  // Hidden (a multi-aircraft Optimize is shown) the panel holds no worker: its memory is free for the job, and a return to
  // Optimize (single) opens it again.
  useEffect(() => {
    if (hidden) return undefined;
    const kind: WorkerSessionKind = activeMode === "optimize" ? "optimizer" : "comparison";
    void openWorkerSession(kind);
    const releaseOnUnload = () => beaconCloseWorkerSession(kind);
    window.addEventListener("pagehide", releaseOnUnload);
    return () => {
      window.removeEventListener("pagehide", releaseOnUnload);
      void closeWorkerSession(kind);
    };
  }, [activeMode, hidden]);

  useEffect(() => {
    const aircraft = aircraftConfigs[0] ?? null;
    placementBackupRef.current = null;
    setInitialState(makeDefaultInitialState(airport, aircraft));
    setRnavFixes(null);
    setSelectedRnavInitialFixKey("");
    setSimulationMode(DEFAULT_SIMULATION_MODE);
    setControls(makeDefaultControls(aircraft));
    setIsInitialEditorOpen(false);
    setIsTargetEditorOpen(false);
    setIsPlacingInitialPosition(false);
    setIsInitialPreviewVisible(false);
    setIsFlying(false);
    setIsTrajectoryPlaying(false);
    setIsEnabled(false);
    setLiveSnapshot(null);
    setTrail([]);
    setOptimizedTrajectory(null);
    setComparisonDurationS(DEFAULT_COMPARISON_DURATION_S);
    setComparisonDtS(DEFAULT_COMPARISON_DT_S);
    setAveragedComparison(null);
    setComparisonHistoryCount(0);
    clearComparisonPlayback();
  }, [airport, aircraftConfigs, clearComparisonPlayback]);

  useEffect(() => {
    if (!selectedAircraft) return;

    setTargetState((current) => {
      const runwayTarget = runwayTargets.find(
        (target) => target.id === current.runwayThresholdId,
      ) ?? runwayTargets[0] ?? null;
      return makeDefaultTrajectoryTarget(runwayTarget, selectedAircraft, current);
    });
  }, [runwayTargets, selectedAircraft]);

  useEffect(() => {
    let cancelled = false;

    void fetchPilotAircraftConfigs()
      .then((configs) => {
        if (cancelled) return;
        setAircraftConfigs(configs);
        setCatalogError(null);
      })
      .catch((configError: unknown) => {
        if (cancelled) return;
        // the list stays as it is (empty: it is set only on success) — a new empty array would re-run the reset
        // effect keyed on it and wipe what the user set up while the catalog was asked for
        setCatalogError(toErrorMessage(configError));
      });

    return () => {
      cancelled = true;
    };
  }, [catalogAttempt]);

  useEffect(() => {
    if (!activeAirportCode) {
      setRunwayTargets([]);
      setTargetState(makeDefaultTrajectoryTarget(null, null));
      return;
    }

    let cancelled = false;
    void fetchRunwayThresholdTargets(activeAirportCode)
      .then((targets) => {
        if (cancelled) return;
        setRunwayTargets(targets);
        setTargetState((current) => {
          const selected = targets.find((target) => target.id === current.runwayThresholdId) ??
            targets[0] ??
            null;
          return makeDefaultTrajectoryTarget(selected, null, current);
        });
      })
      .catch((runwayError: unknown) => {
        if (cancelled) return;
        setRunwayTargets([]);
        setTargetState(makeDefaultTrajectoryTarget(null, null));
        setError(toErrorMessage(runwayError));
      });

    return () => {
      cancelled = true;
    };
  }, [activeAirportCode]);

  // Read per runway, whichever task is open: a switch between Fly and Optimize keeps the
  // list and the picked fix (the start may sit on it).
  useEffect(() => {
    if (!activeAirportCode || !selectedTargetRunway) {
      setRnavFixes(null);
      setSelectedRnavInitialFixKey("");
      return;
    }

    let cancelled = false;
    const runwayIdent = selectedTargetRunway.runwayIdent;
    setRnavFixes(null);
    setSelectedRnavInitialFixKey("");

    void fetchRnavInitialFixCandidates(activeAirportCode, runwayIdent)
      .then((candidates) => {
        if (cancelled) return;
        setRnavFixes({ runwayIdent, candidates });
      })
      .catch((initialError: unknown) => {
        if (cancelled) return;
        setError(toErrorMessage(initialError));
      });

    return () => {
      cancelled = true;
    };
  }, [activeAirportCode, selectedTargetRunway]);

  // Refresh the stored-run count when entering Fly, so the Average button
  // reflects history from earlier sessions too (count is server-side).
  useEffect(() => {
    if (activeMode !== "fly") return;
    let cancelled = false;
    void fetchDynamicsComparisonHistoryCount()
      .then((count) => {
        if (!cancelled) setComparisonHistoryCount(count);
      })
      .catch(() => {
        // Non-critical: leave the count as-is if the backend is unreachable.
      });
    return () => {
      cancelled = true;
    };
  }, [activeMode]);

  useEffect(() => {
    controlsRef.current = controls;
  }, [controls]);

  useEffect(() => {
    simulationModeRef.current = simulationMode;
  }, [simulationMode]);

  useEffect(() => {
    integratorDtRef.current = integratorDtS;
  }, [integratorDtS]);

  const appendTrailPoint = useCallback((nextSnapshot: PilotSnapshot, segmentIndex?: number) => {
    const nextPose = snapshotToPose(nextSnapshot);
    if (!nextPose) return;
    const nextTrailPose =
      segmentIndex === undefined ? nextPose : { ...nextPose, segmentIndex };
    setTrail((current) => [...current.slice(-(MAX_TRAIL_POINTS - 1)), nextTrailPose]);
  }, []);

  useEffect(() => {
    if (!isEnabled || !isFlying) return;

    let cancelled = false;
    const tick = () => {
      if (stepInFlightRef.current) return;
      stepInFlightRef.current = true;

      void stepPilotSimulation(
        controlsRef.current,
        PLAYBACK_FRAME_DT_S,
        simulationModeRef.current,
        integratorDtRef.current,
      )
        .then((nextSnapshot) => {
          if (cancelled) return;
          setLiveSnapshot(nextSnapshot);
          appendTrailPoint(nextSnapshot);
          setError(null);
        })
        .catch((stepError: unknown) => {
          if (cancelled) return;
          setIsFlying(false);
          setError(toErrorMessage(stepError));
        })
        .finally(() => {
          stepInFlightRef.current = false;
        });
    };

    tick();
    const interval = window.setInterval(tick, STEP_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [appendTrailPoint, isEnabled, isFlying]);

  useEffect(() => {
    if (!isEnabled || activeMode !== "fly") return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) return;

      let handled = true;
      switch (event.key.toLowerCase()) {
        case "arrowleft":
        case "a":
          nudgeControl("bankDeg", 3, -45, 45);
          break;
        case "arrowright":
        case "d":
          nudgeControl("bankDeg", -3, -45, 45);
          break;
        case "arrowup":
        case "w":
          nudgeModeControl(1);
          break;
        case "arrowdown":
        case "s":
          nudgeModeControl(-1);
          break;
        case "q":
          nudgeControl("thrustN", -500, 0, selectedMaxThrustN);
          break;
        case "e":
          nudgeControl("thrustN", 500, 0, selectedMaxThrustN);
          break;
        case " ":
          changeControls((current) =>
            usesLoadFactorControl(simulationModeRef.current)
              ? { ...current, bankDeg: 0, loadFactor: DEFAULT_LOAD_FACTOR }
              : { ...current, bankDeg: 0, attackDeg: 0 }
          );
          break;
        default:
          handled = false;
      }

      if (handled) event.preventDefault();
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [activeMode, isEnabled, selectedMaxThrustN]);

  async function startPilot() {
    placementBackupRef.current = null;
    setIsInitialEditorOpen(false);
    setIsPlacingInitialPosition(false);
    setIsTrajectoryPlaying(false);
    clearComparisonPlayback();
    setIsBusy(true);
    setError(null);
    try {
      // Resume only a live session the backend holds (`isEnabled`); otherwise start afresh.
      if (!isEnabled) {
        const nextSnapshot = await resetPilotSimulation(
          initialState,
          controls,
          simulationMode,
        );
        setLiveSnapshot(nextSnapshot);
        const nextPose = snapshotToPose(nextSnapshot);
        setTrail(nextPose ? [nextPose] : []);
      }
      setIsEnabled(true);
      setIsFlying(true);
    } catch (startError: unknown) {
      setIsFlying(false);
      setError(toErrorMessage(startError));
    } finally {
      setIsBusy(false);
    }
  }

  async function resetPilot() {
    placementBackupRef.current = null;
    setIsInitialEditorOpen(false);
    setIsPlacingInitialPosition(false);
    setIsTrajectoryPlaying(false);
    clearComparisonPlayback();
    setIsBusy(true);
    setError(null);
    try {
      const nextSnapshot = await resetPilotSimulation(
        initialState,
        controls,
        simulationMode,
      );
      setLiveSnapshot(nextSnapshot);
      const nextPose = snapshotToPose(nextSnapshot);
      setTrail(nextPose ? [nextPose] : []);
      setIsEnabled(true);
      setIsFlying(false);
    } catch (resetError: unknown) {
      setError(toErrorMessage(resetError));
    } finally {
      setIsBusy(false);
    }
  }

  function stopPilot() {
    placementBackupRef.current = null;
    setIsInitialEditorOpen(false);
    setIsPlacingInitialPosition(false);
    setIsFlying(false);
    setIsTrajectoryPlaying(false);
    setIsEnabled(false);
    setLiveSnapshot(null);
    setTrail([]);
    setError(null);
  }

  // Every edit of the controls goes through here: a computed comparison flew the controls
  // as they were, so it no longer describes the panel once they change.
  function changeControls(update: (current: PanelControls) => PanelControls) {
    setControls(update);
    clearComparisonPlayback();
  }

  function updateControl(
    key: keyof PanelControls,
    value: number,
    min: number,
    max: number,
  ) {
    if (!Number.isFinite(value)) return;
    changeControls((current) => ({ ...current, [key]: clamp(value, min, max) }));
  }

  function updateSimulationMode(value: PilotSimulationMode) {
    setSimulationMode(value);
    clearComparisonPlayback();
  }

  function handleSimulationSelectKeyDown(
    event: ReactKeyboardEvent<HTMLSelectElement>,
  ) {
    if (event.key === "ArrowUp") {
      event.preventDefault();
      nudgeModeControl(1, simulationMode);
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      nudgeModeControl(-1, simulationMode);
    }
  }

  function updateInitialField(
    key: PilotInitialEditableKey,
    value: number,
    min: number,
    max: number,
  ) {
    if (!Number.isFinite(value) || isFlying || isTrajectoryPlaying) return;

    // Field edits keep the RNAV IF selection (same rationale as updateInitialPosition: the
    // selector names the procedure; the start may differ — the optimizer flies to the fix).
    setInitialState((current) => ({ ...current, [key]: clamp(value, min, max) }));
    setIsInitialPreviewVisible(true);
    clearSnapshotForInitialEdit();
  }

  function updateAircraftType(aircraftType: PilotAircraftType) {
    if (isFlying || isTrajectoryPlaying) return;

    const aircraft = aircraftConfigs.find((config) => config.code === aircraftType);
    if (!aircraft) return;

    setInitialState((current) => ({
      ...current,
      aircraftType: aircraft.code,
      massKg: aircraft.massKg,
      speedMps: defaultInitialSpeedMps(aircraft),
    }));
    setControls(makeDefaultControls(aircraft));
    setTargetState((current) =>
      makeDefaultTrajectoryTarget(selectedTargetRunway ?? null, aircraft, current, true)
    );
    setIsInitialPreviewVisible(true);
    clearSnapshotForInitialEdit();
  }

  function updateIntegratorDt(value: number) {
    if (!Number.isFinite(value)) return;
    setIntegratorDtS(clamp(value, 0.02, 0.5));
  }

  // Unload any active playback when leaving a mode, but KEEP the computed
  // results (optimized trajectory / comparison) so returning lets the user
  // replay without recomputing.
  function suspendPlaybacks() {
    setIsTrajectoryPlaying(false);
    setIsComparisonPlaying(false);
    setIsChartsOpen(false);
  }

  // On each task switch (the top bar's): release the clock and close the editors.
  const taskKey = `${activeMode}:${hidden}`;
  const previousModeRef = useRef(taskKey);
  useEffect(() => {
    if (previousModeRef.current === taskKey) return;
    previousModeRef.current = taskKey;
    if (isPlacingInitialPosition) return;
    suspendPlaybacks();
    setIsInitialEditorOpen(false);
    setIsTargetEditorOpen(false);
    if (activeMode !== "fly") setIsFlying(false);
    setError(null);
  }, [taskKey]);

  function openTargetEditor() {
    if (isBusy || isTrajectoryPlaying || runwayTargets.length === 0) return;

    setError(null);
    setIsInitialEditorOpen(false);
    setIsTargetEditorOpen(true);
  }

  function closeTargetEditor() {
    setIsTargetEditorOpen(false);
  }

  function updateTargetRunway(runwayThresholdId: string) {
    const target = runwayTargets.find((candidate) => candidate.id === runwayThresholdId);
    if (!target) return;

    setTargetState((current) =>
      makeDefaultTrajectoryTarget(target, selectedAircraft, current)
    );
    setSelectedRnavInitialFixKey("");
    clearOptimizedPlayback();
  }

  function updateRnavInitialFix(candidateKey: string) {
    if (isFlying || isTrajectoryPlaying) return;

    setSelectedRnavInitialFixKey(candidateKey);
    const candidate = rnavInitialFixCandidates.find(
      (current) => current.key === candidateKey,
    );
    if (!candidate || !selectedAircraft) return;

    try {
      const speedMps = initialSpeedMpsForAircraft(selectedAircraft);
      setInitialState((current) =>
        makeInitialStateFromRnavFix(candidate, selectedAircraft, current, speedMps)
      );
      setIsInitialPreviewVisible(true);
      clearSnapshotForInitialEdit();
      setError(null);
    } catch (initialError: unknown) {
      setError(toErrorMessage(initialError));
    }
  }

  function updateTargetField(
    key: PilotTargetEditableKey,
    value: number,
    min: number,
    max: number,
  ) {
    if (!Number.isFinite(value)) return;

    let nextValue = clamp(value, min, max);
    if (key === "speedMps") {
      // The editor is only rendered once an aircraft is selected (it owns the speed range).
      if (!selectedAircraft) return;
      nextValue = clampTargetSpeedMps(nextValue, selectedAircraft);
    } else if (key === "headingDeg") {
      nextValue = selectedTargetRunway
        ? clampHeadingToRunwayTolerance(value, selectedTargetRunway.psiDeg)
        : runwayAlignedHeadingDeg(nextValue);
    }

    setTargetState((current) => ({ ...current, [key]: nextValue }));
    clearOptimizedPlayback();
  }

  function updateNSegments(value: number) {
    if (!Number.isFinite(value)) return;
    setNSegments(Math.round(clamp(value, 2, 80)));   // CollocationOptimizer requires n_segments >= 2
    clearOptimizedPlayback();
  }

  function updateNSegPerPhase(value: number) {
    if (!Number.isFinite(value)) return;
    setNSegPerPhase(Math.round(clamp(value, 1, 16)));
    clearOptimizedPlayback();
  }

  function updateStateSubsteps(value: number) {
    if (!Number.isFinite(value)) return;
    setStateSubsteps(Math.round(clamp(value, 0, 64)));   // 0 = auto density
    clearOptimizedPlayback();
  }

  /** Edit one axis of the optimizer, snapping the fitting to one the (possibly changed) frame
   * allows. Patches the AXES state directly, so a locked/hidden axis (e.g. Frame in constrained
   * mode) keeps its value across the toggle instead of being flattened by the wire round-trip. */
  function updateOptimizerParts(patch: Partial<OptimizerParts>) {
    setOptimizerParts((prev) => {
      const next: OptimizerParts = { ...prev, ...patch };
      const allowed = validFittingsForFrame(next.constrained ? "geodetic" : next.frame);
      if (!allowed.includes(next.fitting)) next.fitting = allowed[0];
      return next;
    });
    clearOptimizedPlayback();
    // NOTE: the constraint → procedure-display link is NOT here (a one-shot on the
    // switch would miss the default constrained state and never revert). It is a
    // reactive, Optimize-scoped drive/restore effect below.
  }

  function updateArrivalTime(value: number) {
    if (!Number.isFinite(value)) return;
    setArrivalTimeS(clamp(value, 1, 1000));
    clearOptimizedPlayback();
  }

  function updateMaxIterations(value: number) {
    if (!Number.isFinite(value)) return;
    setMaxIterations(Math.round(clamp(value, 1, 10000)));
    clearOptimizedPlayback();
  }

  async function computeTrajectory() {
    if (!hasAircraftConfigs || runwayTargets.length === 0) return;

    setIsBusy(true);
    setIsFlying(false);
    clearOptimizedPlayback();
    setError(null);
    try {
      // The multiphase dynamics REQUIRES the selected approach's procedure constraint (it builds
      // one phase per leg). The selected RNAV initial fix identifies the procedure + branch; the
      // backend enforces each leg's corridor / glidepath / step-down floor as NLP path constraints.
      let procedureConstraint: ProcedureConstraint | undefined;
      let pilotTargetState = trajectoryTargetToPilotState(
        targetState,
        initialState.aircraftType,
        initialState.massKg,
      );
      const { constrained } = optimizerParts;
      if (constrained) {
        const candidate = rnavInitialFixCandidates.find(
          (current) => current.key === selectedRnavInitialFixKey,
        );
        if (!activeAirportCode || !candidate) {
          throw new Error(
            "Select an RNAV initial fix to run the multiphase (per-leg constraints) optimizer.",
          );
        }
        const document = await fetchJson<ProcedureDetailDocument>(
          procedureDetailsDocumentUrl(activeAirportCode, candidate.procedureUid),
        );
        const built = buildProcedureConstraint(document, { branchId: candidate.branchId });
        if (!built) {
          throw new Error(
            "Could not build a procedure constraint for the selected approach.",
          );
        }
        procedureConstraint = built;
        // Anchor the target on the procedure's OWN threshold (CIFP). The backend anchors the
        // constraint (n, e) frame at the target and rejects a procedure that does not end there;
        // the runway.geojson pavement midpoint can sit hundreds of metres from the CIFP landing
        // threshold (displaced thresholds / OurAirports endpoint quality).
        const anchor = procedureThresholdAnchor(built, document);
        pilotTargetState = {
          ...pilotTargetState,
          lon: anchor.lon,
          lat: anchor.lat,
          headingDeg: runwayAlignedHeadingDeg(anchor.psiDeg),
          ...(anchor.elevationM !== null
            ? { altM: targetAltitudeMForThreshold(anchor.elevationM, selectedAircraft ?? null) }
            : {}),
        };
      }

      const result = await runTrajectoryOptimization({
        optimizer: trajectoryOptimizer,
        initialState,
        targetState: pilotTargetState,
        nSegments,
        nSegPerPhase: constrained ? nSegPerPhase : undefined,
        stateSubsteps: stateSubsteps > 0 ? stateSubsteps : undefined,
        arrivalTimeS,
        maxIterations,
        procedureConstraint,
      });
      setOptimizedTrajectory(result);
    } catch (computeError: unknown) {
      setOptimizedTrajectory(null);
      setError(toErrorMessage(computeError));
    } finally {
      setIsBusy(false);
    }
  }

  // Optimize's playback leaves Fly's live session as it is (paused: a task switch stopped
  // it; its aircraft is drawn only in Fly).
  function playOptimizedTrajectory() {
    if (!optimizedTrajectory?.playback) return;

    setError(null);
    setIsTrajectoryPlaying(true);
    if (isCesiumViewerUsable(viewer)) {
      viewer.clock.shouldAnimate = true;
    }
  }

  function pauseOptimizedTrajectory() {
    setIsTrajectoryPlaying(false);
    if (isCesiumViewerUsable(viewer)) {
      viewer.clock.shouldAnimate = false;
    }
  }

  function resetTrajectoryReplay() {
    setIsTrajectoryPlaying(false);
    setError(null);
    if (isCesiumViewerUsable(viewer)) {
      viewer.clock.shouldAnimate = false;
      viewer.clock.currentTime = viewer.clock.startTime.clone();
    }
  }

  // ── Dynamics comparison handlers ──────────────────────────────────────────
  function updateComparisonDuration(value: number) {
    if (!Number.isFinite(value)) return;
    setComparisonDurationS(clamp(value, 5, 600));
    clearComparisonPlayback();
  }

  function updateComparisonDt(value: number) {
    if (!Number.isFinite(value)) return;
    setComparisonDtS(clamp(value, 0.05, 1));
    clearComparisonPlayback();
  }

  async function computeComparison() {
    if (!hasAircraftConfigs || comparisonControl === null) return;

    // The comparison takes the screen from the live flight, which ends.
    stopPilot();
    setIsBusy(true);
    clearComparisonPlayback();
    setError(null);
    try {
      const result = await runDynamicsComparison({
        initialState,
        control: comparisonControl,
        durationS: comparisonDurationS,
        dtS: comparisonDtS,
      });
      setComparisonResult(result);
      setComparisonHistoryCount(result.historyCount);
      setChartMode("run");
      setIsChartsOpen(true);
    } catch (comparisonError: unknown) {
      setComparisonResult(null);
      setError(toErrorMessage(comparisonError));
    } finally {
      setIsBusy(false);
    }
  }

  async function showAveragedHistory() {
    setIsBusy(true);
    setError(null);
    try {
      const averaged = await averageDynamicsComparisonHistory();
      setAveragedComparison(averaged);
      setComparisonHistoryCount(averaged.runCount);
      setChartMode("average");
      setIsChartsOpen(true);
    } catch (averageError: unknown) {
      setError(toErrorMessage(averageError));
    } finally {
      setIsBusy(false);
    }
  }

  async function clearComparisonHistory() {
    setIsBusy(true);
    setError(null);
    try {
      const count = await clearDynamicsComparisonHistory();
      setComparisonHistoryCount(count);
      setAveragedComparison(null);
      if (chartMode === "average") setIsChartsOpen(false);
    } catch (clearError: unknown) {
      setError(toErrorMessage(clearError));
    } finally {
      setIsBusy(false);
    }
  }

  function toggleRunCharts() {
    if (runChartOpen) {
      setIsChartsOpen(false);
      return;
    }
    setChartMode("run");
    setIsChartsOpen(true);
  }

  // Compute already ended the live flight: a loaded comparison and a live session never coexist.
  function playComparison() {
    if (!comparisonResult) return;

    setError(null);
    setIsComparisonPlaying(true);
    if (isCesiumViewerUsable(viewer)) {
      viewer.clock.shouldAnimate = true;
    }
  }

  function pauseComparison() {
    setIsComparisonPlaying(false);
    if (isCesiumViewerUsable(viewer)) {
      viewer.clock.shouldAnimate = false;
    }
  }

  function resetComparisonReplay() {
    // Only meaningful once the comparison CZML is loaded (Effect 1 sets the
    // clock). Before first Play the clock belongs to another mode, so do nothing.
    if (!isComparisonPlaybackActive) return;
    setIsComparisonPlaying(false);
    setError(null);
    if (isCesiumViewerUsable(viewer)) {
      viewer.clock.shouldAnimate = false;
      viewer.clock.currentTime = viewer.clock.startTime.clone();
    }
  }

  function toggleComparisonSystem(key: string) {
    setHiddenComparisonKeys((current) =>
      current.includes(key)
        ? current.filter((existing) => existing !== key)
        : [...current, key],
    );
  }

  function nudgeControl(
    key: keyof PanelControls,
    delta: number,
    min: number,
    max: number,
  ) {
    changeControls((current) => ({
      ...current,
      [key]: clamp(current[key] + delta, min, max),
    }));
  }

  function nudgeModeControl(
    direction: 1 | -1,
    mode: PilotSimulationMode = simulationModeRef.current,
  ) {
    if (usesLoadFactorControl(mode)) {
      nudgeControl(
        "loadFactor",
        direction * 0.05,
        MIN_LOAD_FACTOR,
        MAX_LOAD_FACTOR,
      );
      return;
    }

    nudgeControl("attackDeg", direction * 0.5, -10, 18);
  }

  const statusLabel = isPlacingInitialPosition
    ? "Placing"
    : isBusy
      ? "Computing"
      : isTrajectoryPlaying || isComparisonPlaying
        ? "Playing"
        : isFlying
          ? "Flying"
          : isTrajectoryPlaybackActive || isComparisonPlaybackActive
            ? "Paused"
            : optimizedTrajectory && activeMode === "optimize"
              ? "Ready"
              : snapshot
                ? "Paused"
                : "Standby";
  const hasAircraftConfigs = aircraftConfigs.length > 0;
  const isAnyPlaying = isTrajectoryPlaying || isComparisonPlaying;
  const initialControlsDisabled = isFlying || isAnyPlaying || isBusy || !hasAircraftConfigs;
  const targetControlsDisabled = isBusy || isTrajectoryPlaying || runwayTargets.length === 0;
  const comparisonControlsDisabled = isBusy || isComparisonPlaying || !hasAircraftConfigs;
  // The controls are frozen while a backend request runs (Compute, Start, Reset): the
  // reply then always answers the controls on screen.
  const controlsDisabled = isBusy;
  const runChartOpen = isChartsOpen && chartMode === "run" && comparisonResult !== null;
  // An Optimize start needs the runway's RNAV fixes; Fly can start anywhere (Edit / place).
  const missingRnavFixesRunway =
    activeMode === "optimize" && rnavFixes !== null && rnavFixes.candidates.length === 0
      ? rnavFixes.runwayIdent
      : null;
  // The comparison flies the panel's controls held fixed, on the load-factor
  // parameterisation every compared system shares — so not under Alpha simulation.
  const comparisonControl: DynamicsComparisonControl | null = usesLoadFactorControl(simulationMode)
    ? { thrustN: controls.thrustN, bankDeg: controls.bankDeg, loadFactor: controls.loadFactor }
    : null;
  // ``optimizerParts`` is the axes state (above); the fittings a frame allows drive the Fitting
  // dropdown (re-anchored ENU is shooting-only).
  const allowedFittings = validFittingsForFrame(
    optimizerParts.constrained ? "geodetic" : optimizerParts.frame,
  );
  const trajectorySegmentDurationS = optimizedTrajectory
    ? optimizedTrajectory.finalTimeS / Math.max(1, optimizedTrajectory.controls.length)
    : null;

  // ── Fly's live-flight transport for the shared bottom bar ────────────────────
  // The live aircraft runs on the manual sim loop (`isFlying`), NOT viewer.clock,
  // so the bottom bar's generic clock Play/Reset can't drive it. Publish the sim
  // transport to context — via stable, ref-backed callbacks so the effect doesn't
  // churn — while the live flight is Fly's run on screen; a loaded comparison is a
  // clock playback, so the transport is withdrawn (null) and the bar drives the clock.
  // A layout effect: published before the first paint of Fly, so the bar never shows a
  // frame of the clock transport on entering it.
  const pilotTransportImplRef = useRef({ startPilot, resetPilot, isFlying });
  pilotTransportImplRef.current = { startPilot, resetPilot, isFlying };
  const bottomTogglePilotPlay = useCallback(() => {
    const impl = pilotTransportImplRef.current;
    if (impl.isFlying) setIsFlying(false);
    else void impl.startPilot();
  }, []);
  const bottomResetPilot = useCallback(() => {
    void pilotTransportImplRef.current.resetPilot();
  }, []);
  useLayoutEffect(() => {
    if (activeMode !== "fly" || isComparisonPlaybackActive) return undefined;
    setPilotTransport({
      running: isFlying,
      playPauseDisabled: isBusy || isPlacingInitialPosition || (!isFlying && !hasAircraftConfigs),
      resetDisabled: isBusy || isPlacingInitialPosition || !hasAircraftConfigs,
      togglePlay: bottomTogglePilotPlay,
      reset: bottomResetPilot,
    });
    return () => setPilotTransport(null);
  }, [
    activeMode,
    isComparisonPlaybackActive,
    isFlying,
    isBusy,
    isPlacingInitialPosition,
    hasAircraftConfigs,
    bottomTogglePilotPlay,
    bottomResetPilot,
    setPilotTransport,
  ]);

  // ── Optimize-scoped procedure display (drive & restore) ──────────────────────
  // While in Optimize (trajectory) mode WITH procedure constraints on, drive the
  // shared procedure display to the target runway's approach — open the panel,
  // enable the geometry layer, scope the runway — so you see the corridors /
  // glidepath / step-down floors the solve enforces (the hook saves the user's prior
  // display and restores it when the force ends or this panel unmounts). The target
  // runway is INDEPENDENT of the global selection, so we hand the hook a non-null
  // forceRunway — it then owns selectedRunway; Evaluation passes null (see ControlPanel).
  const forcedRunwayIdent = selectedTargetRunway
    ? bareRunwayIdent(selectedTargetRunway.runwayIdent)
    : null;
  useForcedProcedureDisplay({
    active: !hidden && activeMode === "optimize" && optimizerParts.constrained && forcedRunwayIdent !== null,
    forceRunway: forcedRunwayIdent,
  });

  return (
    <div className="pilot-panel" hidden={hidden}>
      <PilotRealtimeStatePanel
        snapshot={snapshot}
        visible={
          !hidden &&
          (isFlying ||
            isTrajectoryPlaying ||
            (activeMode === "optimize" && snapshot !== null) ||
            (isComparisonPlaybackActive && snapshot !== null))
        }
        showControlReadout={activeMode === "optimize" || isComparisonPlaybackActive}
        simulationMode={snapshot?.simulationMode ?? simulationMode}
        comparisonDeltas={
          isComparisonPlaybackActive
            ? comparisonDeltas
            : activeMode === "optimize"
              ? trajectoryTargetDeltas
              : null
        }
        comparisonSystems={
          isComparisonPlaybackActive
            ? comparisonResult?.systems ?? null
            : activeMode === "optimize" && trajectoryTargetDeltas
              ? TARGET_DELTA_SYSTEMS
              : null
        }
        deltaReferenceLabel={activeMode === "optimize" ? "target" : "B"}
      />

      <header className="pilot-panel-header">
        <div className="pilot-panel-header-main">
          <div className="pilot-panel-title-block">
            <h3>{activeMode === "optimize" ? "Trajectory Play" : "Fly"}</h3>
          </div>
          <span className={`pilot-status pilot-status-${statusLabel.toLowerCase()}`}>
            {statusLabel}
          </span>
        </div>
      </header>

      <section className="pilot-initial-summary" aria-label="Initial aircraft state summary">
        <div className="pilot-initial-summary-header">
          <h4>Initial Aircraft</h4>
          <button
            type="button"
            onClick={openInitialEditor}
            disabled={initialControlsDisabled}
          >
            Edit
          </button>
        </div>

        <dl className="pilot-initial-position">
          <div>
            <dt>Lat</dt>
            <dd>{formatCoord(initialState.lat, "N", "S")}</dd>
          </div>
          <div>
            <dt>Lon</dt>
            <dd>{formatCoord(initialState.lon, "E", "W")}</dd>
          </div>
        </dl>

        <dl className="pilot-initial-readouts">
          <div>
            <dt>Alt</dt>
            <dd>{formatNumberInputValue(initialState.altM)} m</dd>
          </div>
          <div>
            <dt>Type</dt>
            <dd>{initialState.aircraftType}</dd>
          </div>
          <div>
            <dt>Psi</dt>
            <dd>{formatNumberInputValue(initialState.headingDeg)} deg</dd>
          </div>
          <div>
            <dt>Gamma</dt>
            <dd>{formatNumberInputValue(initialState.flightPathDeg)} deg</dd>
          </div>
          <div>
            <dt>V0</dt>
            <dd>{formatNumberInputValue(initialState.speedMps)} m/s</dd>
          </div>
          <div>
            <dt>Mass</dt>
            <dd>{formatNumberInputValue(initialState.massKg)} kg</dd>
          </div>
        </dl>
      </section>

      <PilotInitialStateOverlay
        open={isInitialEditorOpen}
        isPlacing={isPlacingInitialPosition}
        state={initialState}
        aircraftConfigs={aircraftConfigs}
        rnavInitialFixCandidates={rnavInitialFixCandidates}
        selectedRnavInitialFixKey={selectedRnavInitialFixKey}
        disabled={initialControlsDisabled}
        onClose={closeInitialEditor}
        onPlaceToggle={toggleInitialPlacement}
        onFieldChange={updateInitialField}
        onAircraftTypeChange={updateAircraftType}
        onRnavInitialFixChange={updateRnavInitialFix}
      />

      {activeMode === "optimize" ? (
        <>
          <section className="pilot-initial-summary" aria-label="Target aircraft state summary">
            <div className="pilot-initial-summary-header">
              <h4>Target State</h4>
              <div className="pilot-initial-summary-actions">
                {/* Open the target runway's 2D approach view (side + plan). The runway
                    is INDEPENDENT of the global selection, so the toggle focuses it — and
                    BORROWS the selection (restored on close/dock exit) so viewing the
                    target's profile can't permanently clobber the user's runway scoping.
                    Constrained solves don't borrow: useForcedProcedureDisplay already owns
                    the runway + approach view state there. */}
                <ApproachViewToggle
                  runwayIdent={selectedTargetRunway?.runwayIdent ?? null}
                  borrowSelection={!optimizerParts.constrained}
                />
                <button
                  type="button"
                  onClick={openTargetEditor}
                  disabled={targetControlsDisabled}
                >
                  Edit
                </button>
              </div>
            </div>

            <dl className="pilot-initial-position">
              <div>
                <dt>Lat</dt>
                <dd>{formatCoord(targetState.lat, "N", "S")}</dd>
              </div>
              <div>
                <dt>Lon</dt>
                <dd>{formatCoord(targetState.lon, "E", "W")}</dd>
              </div>
            </dl>

            <dl className="pilot-initial-readouts">
              <div>
                <dt>Alt</dt>
                <dd>{formatNumberInputValue(targetState.altM)} m</dd>
              </div>
              <div>
                <dt>Rwy</dt>
                <dd>{selectedTargetRunway?.runwayIdent ?? "-"}</dd>
              </div>
              <div>
                <dt>Psi</dt>
                <dd>{formatNumberInputValue(targetState.headingDeg)} deg</dd>
              </div>
              <div>
                <dt>Vt</dt>
                <dd>
                  {Number.isFinite(targetState.speedMps)
                    ? `${formatNumberInputValue(targetState.speedMps)} m/s`
                    : "— (no aircraft)"}
                </dd>
              </div>
              <div>
                <dt>Gamma</dt>
                <dd>{formatNumberInputValue(targetState.flightPathDeg)} deg</dd>
              </div>
            </dl>
          </section>

          {targetSpeedBounds && (
            <PilotTargetStateOverlay
              open={isTargetEditorOpen}
              state={targetState}
              runwayTargets={runwayTargets}
              speedMinMps={targetSpeedBounds.min}
              speedMaxMps={targetSpeedBounds.max}
              disabled={targetControlsDisabled}
              onClose={closeTargetEditor}
              onRunwayChange={updateTargetRunway}
              onFieldChange={updateTargetField}
            />
          )}

          <section className="pilot-optimization-row" aria-label="Trajectory optimization settings">
            <label>
              <span>Constraints</span>
              <select
                className="pilot-select-input"
                value={optimizerParts.constrained ? "procedure" : "none"}
                disabled={targetControlsDisabled}
                onChange={(event) =>
                  updateOptimizerParts({ constrained: event.target.value === "procedure" })
                }
              >
                {OPTIMIZER_CONSTRAINTS_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Fitting</span>
              <select
                className="pilot-select-input"
                value={optimizerParts.fitting}
                disabled={targetControlsDisabled || allowedFittings.length === 1}
                onChange={(event) =>
                  updateOptimizerParts({ fitting: event.target.value as OptimizerFitting })
                }
              >
                {OPTIMIZER_FITTING_OPTIONS.filter((o) => allowedFittings.includes(o.value)).map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </label>
            {optimizerParts.constrained ? (
              <label>
                <span title="Piecewise-constant control intervals PER procedure leg; the total control count is legs × this.">
                  Control segs / leg
                </span>
                <EnglishNumberInput
                  value={nSegPerPhase}
                  min={1}
                  max={16}
                  step="1"
                  disabled={targetControlsDisabled}
                  onCommit={updateNSegPerPhase}
                />
              </label>
            ) : (
              <label>
                <span title="Piecewise-constant control intervals over the whole trajectory.">
                  Control segments
                </span>
                <EnglishNumberInput
                  value={nSegments}
                  min={2}
                  max={80}
                  step="1"
                  disabled={targetControlsDisabled}
                  onCommit={updateNSegments}
                />
              </label>
            )}
            <label>
              <span title="State-collocation subintervals per control segment (state nodes = segments × this). 0 = auto: a ~3 s state step, capped at 16. Higher = a denser, more dynamically faithful plan; slower solve.">
                State substeps
              </span>
              <EnglishNumberInput
                value={stateSubsteps}
                min={0}
                max={64}
                step="1"
                disabled={targetControlsDisabled}
                onCommit={updateStateSubsteps}
              />
            </label>
            {optimizerParts.constrained && (
              <span
                className="pilot-multiphase-hint"
                title="One phase per procedure leg (start->IAF, then each leg), enforcing that leg's corridor / glidepath / step-down floor as NLP path constraints. Select an RNAV initial fix to identify the approach."
              >
                Per-leg constraints from the selected RNAV approach
              </span>
            )}
            <label>
              <span>Arrival time</span>
              <EnglishNumberInput
                value={arrivalTimeS}
                min={1}
                max={1000}
                step="5"
                disabled={targetControlsDisabled}
                onCommit={updateArrivalTime}
              />
            </label>
            <label>
              <span>Max iter</span>
              <EnglishNumberInput
                value={maxIterations}
                min={1}
                max={10000}
                step="50"
                disabled={targetControlsDisabled}
                onCommit={updateMaxIterations}
              />
            </label>

            <details
              className="pilot-advanced-numerics"
              open={showAdvancedNumerics}
              onToggle={(event) => setShowAdvancedNumerics(event.currentTarget.open)}
            >
              <summary>Advanced dynamics</summary>
              <div className="pilot-advanced-grid">
                <label>
                  <span>Frame</span>
                  <select
                    className="pilot-select-input"
                    value={optimizerParts.frame}
                    disabled={targetControlsDisabled || optimizerParts.constrained}
                    onChange={(event) =>
                      updateOptimizerParts({ frame: event.target.value as OptimizerFrame })
                    }
                  >
                    {OPTIMIZER_FRAME_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>{o.label}</option>
                    ))}
                  </select>
                </label>
                {optimizerParts.frame === "geodetic" && (
                  <>
                    <label>
                      <span>Transport</span>
                      <select
                        className="pilot-select-input"
                        value={optimizerParts.transport}
                        disabled={targetControlsDisabled || optimizerParts.constrained}
                        onChange={(event) =>
                          updateOptimizerParts({ transport: event.target.value as OptimizerTransport })
                        }
                      >
                        {OPTIMIZER_TRANSPORT_OPTIONS.map((o) => (
                          <option key={o.value} value={o.value}>{o.label}</option>
                        ))}
                      </select>
                    </label>
                    <label className="pilot-advanced-checkbox">
                      <input
                        type="checkbox"
                        checked={optimizerParts.normalized}
                        disabled={targetControlsDisabled || optimizerParts.constrained}
                        onChange={(event) =>
                          updateOptimizerParts({ normalized: event.target.checked })
                        }
                      />
                      <span title="Decision STATE is metric position offsets from the target — a pure change of variables that conditions the NLP well (robust on loose windows / fine meshes).">
                        Normalized state (well-conditioned)
                      </span>
                    </label>
                  </>
                )}
                {optimizerParts.constrained && (
                  <span className="pilot-locked-hint">
                    Locked to geodetic · full transport · normalized — the per-leg path constraints
                    live on the metric-position state.
                  </span>
                )}
              </div>
            </details>
          </section>

          <div className="pilot-actions">
            <button
              className="pilot-primary-button"
              onClick={computeTrajectory}
              disabled={
                isBusy ||
                isTrajectoryPlaying ||
                isPlacingInitialPosition ||
                !hasAircraftConfigs ||
                runwayTargets.length === 0
              }
            >
              Optimize
            </button>
            <button
              onClick={playOptimizedTrajectory}
              disabled={
                isBusy ||
                isTrajectoryPlaying ||
                isPlacingInitialPosition ||
                !optimizedTrajectory?.playback
              }
            >
              Play
            </button>
            <button
              onClick={pauseOptimizedTrajectory}
              disabled={!isTrajectoryPlaying || isBusy || isPlacingInitialPosition}
            >
              Pause
            </button>
            <button
              onClick={resetTrajectoryReplay}
              disabled={
                isBusy ||
                isPlacingInitialPosition ||
                (!isTrajectoryPlaybackActive && !optimizedTrajectory)
              }
            >
              Reset
            </button>
          </div>

          <section className="pilot-control-zone" aria-label="Trajectory play controls">
            <div className="pilot-options-row">
              <label className="pilot-checkbox-label">
                <input
                  type="checkbox"
                  checked={isFollowing}
                  onChange={(event) => setIsFollowing(event.target.checked)}
                />
                Follow camera
              </label>
            </div>

            {optimizedTrajectory ? (
              <dl className="pilot-plan-readouts">
                <div>
                  <dt>Final</dt>
                  <dd>{formatNumberInputValue(optimizedTrajectory.finalTimeS)} s</dd>
                </div>
                <div>
                  <dt>Segment</dt>
                  <dd>{formatNumberInputValue(trajectorySegmentDurationS ?? 0)} s</dd>
                </div>
                {optimizedTrajectory.timings ? (
                  <div>
                    <dt title="Wall-clock time for the whole optimization: NLP build + solve + playback rollout.">
                      Solve time
                    </dt>
                    <dd>{optimizedTrajectory.timings.totalS.toFixed(2)} s</dd>
                  </div>
                ) : null}
              </dl>
            ) : null}
          </section>
        </>
      ) : (
        <>
          <section className="pilot-optimization-row" aria-label="Start fix runway">
            <label>
              <span>RNAV runway</span>
              <select
                className="pilot-select-input"
                value={targetState.runwayThresholdId}
                disabled={initialControlsDisabled || runwayTargets.length === 0}
                onChange={(event) => updateTargetRunway(event.target.value)}
              >
                {runwayTargets.length === 0 ? <option value="">—</option> : null}
                {runwayTargets.map((target) => (
                  <option key={target.id} value={target.id}>
                    {target.runwayIdent}
                  </option>
                ))}
              </select>
            </label>
          </section>
          <p className="dyncmp-hint">
            Set the start via <strong>Edit</strong> above (fields / place on map); the
            runway&apos;s published RNAV fixes are offered there as starts.
          </p>

          <section className="pilot-control-zone" aria-label="Pilot controls">
            <h4 className="pilot-panel-section-title">Controls</h4>
            <div className="pilot-stepper-row">
              <button
                disabled={controlsDisabled}
                onClick={() => nudgeControl("bankDeg", 5, -45, 45)}
                title="Bank left"
              >
                &lt;
              </button>
              <label>
                <span>Bank</span>
                <EnglishNumberInput
                  value={controls.bankDeg}
                  min={-45}
                  max={45}
                  step="1"
                  disabled={controlsDisabled}
                  onCommit={(value) => updateControl("bankDeg", value, -45, 45)}
                />
              </label>
              <button
                disabled={controlsDisabled}
                onClick={() => nudgeControl("bankDeg", -5, -45, 45)}
                title="Bank right"
              >
                &gt;
              </button>
            </div>

            {usesLoadFactorControl(simulationMode) ? (
              <div className="pilot-stepper-row">
                <button
                  disabled={controlsDisabled}
                  onClick={() =>
                    nudgeControl(
                      "loadFactor",
                      -0.05,
                      MIN_LOAD_FACTOR,
                      MAX_LOAD_FACTOR,
                    )
                  }
                  title="Reduce load factor"
                >
                  -
                </button>
                <label>
                  <span>Load factor</span>
                  <EnglishNumberInput
                    value={controls.loadFactor}
                    min={MIN_LOAD_FACTOR}
                    max={MAX_LOAD_FACTOR}
                    step="0.05"
                    disabled={controlsDisabled}
                    onCommit={(value) =>
                      updateControl(
                        "loadFactor",
                        value,
                        MIN_LOAD_FACTOR,
                        MAX_LOAD_FACTOR,
                      )
                    }
                  />
                </label>
                <button
                  disabled={controlsDisabled}
                  onClick={() =>
                    nudgeControl(
                      "loadFactor",
                      0.05,
                      MIN_LOAD_FACTOR,
                      MAX_LOAD_FACTOR,
                    )
                  }
                  title="Increase load factor"
                >
                  +
                </button>
              </div>
            ) : (
              <div className="pilot-stepper-row">
                <button
                  disabled={controlsDisabled}
                  onClick={() => nudgeControl("attackDeg", -0.5, -10, 18)}
                  title="Reduce alpha"
                >
                  -
                </button>
                <label>
                  <span>Alpha</span>
                  <EnglishNumberInput
                    value={controls.attackDeg}
                    min={-10}
                    max={18}
                    step="0.5"
                    disabled={controlsDisabled}
                    onCommit={(value) => updateControl("attackDeg", value, -10, 18)}
                  />
                </label>
                <button
                  disabled={controlsDisabled}
                  onClick={() => nudgeControl("attackDeg", 0.5, -10, 18)}
                  title="Increase alpha"
                >
                  +
                </button>
              </div>
            )}

            <div className="pilot-stepper-row">
              <button
                disabled={controlsDisabled}
                onClick={() => nudgeControl("thrustN", -1000, 0, selectedMaxThrustN)}
                title="Reduce thrust"
              >
                -
              </button>
              <label>
                <span>Thrust</span>
                <EnglishNumberInput
                  value={controls.thrustN}
                  min={0}
                  max={selectedMaxThrustN}
                  step="500"
                  disabled={controlsDisabled}
                  onCommit={(value) =>
                    updateControl("thrustN", value, 0, selectedMaxThrustN)
                  }
                />
              </label>
              <button
                disabled={controlsDisabled}
                onClick={() => nudgeControl("thrustN", 1000, 0, selectedMaxThrustN)}
                title="Increase thrust"
              >
                +
              </button>
            </div>

            <div className="pilot-options-row">
              <label>
                <span>Simulation</span>
                <select
                  className="pilot-select-input"
                  value={simulationMode}
                  disabled={isPlacingInitialPosition || controlsDisabled}
                  onKeyDown={handleSimulationSelectKeyDown}
                  onChange={(event) =>
                    updateSimulationMode(event.target.value as PilotSimulationMode)
                  }
                >
                  <option value="alpha">Alpha</option>
                  <option value="loadFactor">Load factor</option>
                  <option value="casadi">CasADi</option>
                </select>
              </label>
              <label className="pilot-checkbox-label">
                <input
                  type="checkbox"
                  checked={isFollowing}
                  onChange={(event) => setIsFollowing(event.target.checked)}
                />
                Follow camera
              </label>
            </div>
          </section>

          <section className="pilot-control-zone" aria-label="Live flight">
            <h4 className="pilot-panel-section-title">Live flight</h4>
            <div className="pilot-actions">
              <button
                className="pilot-primary-button"
                onClick={startPilot}
                disabled={isBusy || isFlying || isPlacingInitialPosition || !hasAircraftConfigs}
              >
                {isEnabled ? "Resume" : "Start"}
              </button>
              <button
                onClick={() => setIsFlying(false)}
                disabled={!isFlying || isBusy || isPlacingInitialPosition}
              >
                Pause
              </button>
              <button
                onClick={resetPilot}
                disabled={isBusy || isPlacingInitialPosition || !hasAircraftConfigs}
              >
                Reset
              </button>
              <button
                onClick={stopPilot}
                disabled={!isEnabled || isPlacingInitialPosition}
              >
                End
              </button>
            </div>
            <div className="pilot-options-row">
              <label>
                <span>dt</span>
                <EnglishNumberInput
                  value={integratorDtS}
                  min={0.02}
                  max={0.5}
                  step="0.02"
                  disabled={false}
                  onCommit={updateIntegratorDt}
                />
              </label>
            </div>
          </section>

          <section className="pilot-control-zone" aria-label="Dynamics comparison">
            <h4 className="pilot-panel-section-title">Compare dynamics</h4>
            {comparisonControl === null ? (
              <p className="dyncmp-hint">
                The comparison flies a load-factor control: set Simulation to Load factor or
                CasADi.
              </p>
            ) : null}
            <div className="pilot-optimization-row">
              <label>
                <span>Duration</span>
                <EnglishNumberInput
                  value={comparisonDurationS}
                  min={5}
                  max={600}
                  step="10"
                  disabled={comparisonControlsDisabled}
                  onCommit={updateComparisonDuration}
                />
              </label>
              <label>
                <span>dt</span>
                <EnglishNumberInput
                  value={comparisonDtS}
                  min={0.05}
                  max={1}
                  step="0.05"
                  disabled={comparisonControlsDisabled}
                  onCommit={updateComparisonDt}
                />
              </label>
            </div>

            <div className="pilot-actions">
              <button
                className="pilot-primary-button"
                onClick={computeComparison}
                disabled={
                  isBusy ||
                  isComparisonPlaying ||
                  isPlacingInitialPosition ||
                  !hasAircraftConfigs ||
                  comparisonControl === null
                }
              >
                Compute
              </button>
              <button
                onClick={playComparison}
                disabled={
                  isBusy ||
                  isComparisonPlaying ||
                  isPlacingInitialPosition ||
                  !comparisonResult
                }
              >
                Play
              </button>
              <button
                onClick={pauseComparison}
                disabled={!isComparisonPlaying || isBusy || isPlacingInitialPosition}
              >
                Pause
              </button>
              <button
                onClick={resetComparisonReplay}
                disabled={isBusy || isPlacingInitialPosition || !isComparisonPlaybackActive}
              >
                Reset
              </button>
            </div>

            <div className="pilot-options-row">
              <button
                type="button"
                onClick={toggleRunCharts}
                disabled={!comparisonResult}
              >
                {runChartOpen ? "Hide charts" : "Show charts"}
              </button>
            </div>

            <div className="pilot-actions dyncmp-history-actions">
              <button
                type="button"
                onClick={showAveragedHistory}
                disabled={isBusy || comparisonHistoryCount === 0}
                title="Average the deviation of all stored runs (computed on the backend)"
              >
                Average history ({comparisonHistoryCount})
              </button>
              <button
                type="button"
                onClick={clearComparisonHistory}
                disabled={isBusy || comparisonHistoryCount === 0}
              >
                Clear history
              </button>
            </div>

            {comparisonResult ? (
              <>
                <ul className="dyncmp-panel-legend" aria-label="Trajectory visibility">
                  {comparisonResult.systems.map((system) => {
                    const isHidden = hiddenComparisonKeys.includes(system.key);
                    const [r, g, b, a] = system.colorRgba;
                    return (
                      <li key={system.key}>
                        <label className={`dyncmp-legend-item${isHidden ? " is-hidden" : ""}`}>
                          <input
                            type="checkbox"
                            checked={!isHidden}
                            onChange={() => toggleComparisonSystem(system.key)}
                          />
                          <span
                            className="dyncmp-legend-swatch"
                            style={{ background: `rgba(${r}, ${g}, ${b}, ${(a / 255).toFixed(3)})` }}
                          />
                          <span className="dyncmp-legend-label">{system.label}</span>
                        </label>
                      </li>
                    );
                  })}
                </ul>
                <dl className="pilot-plan-readouts">
                  <div>
                    <dt>Duration</dt>
                    <dd>{formatNumberInputValue(comparisonResult.durationS)} s</dd>
                  </div>
                  <div>
                    <dt>dt</dt>
                    <dd>{formatNumberInputValue(comparisonResult.dtS)} s</dd>
                  </div>
                  <div>
                    <dt>Speed</dt>
                    <dd>{comparisonResult.playback.multiplier}x</dd>
                  </div>
                </dl>
                {comparisonResult.durationS < comparisonResult.requestedDurationS - 0.5 ? (
                  <p className="dyncmp-hint dyncmp-hint-warn" role="status">
                    Flight reached the ground after {formatNumberInputValue(comparisonResult.durationS)} s
                    (requested {formatNumberInputValue(comparisonResult.requestedDurationS)} s) — horizon truncated.
                  </p>
                ) : null}
              </>
            ) : (
              <p className="dyncmp-hint">
                Compute holds the controls above fixed for the duration and flies the start
                four ways (fixed tangent, re-anchored, geodetic ±transport) to compare the
                drift. It ends a live flight; editing a control clears the result.
              </p>
            )}
          </section>
        </>
      )}

      {catalogError ? (
        <div className="pilot-error" role="alert">
          The aircraft catalog did not load: {catalogError}{" "}
          <button type="button" onClick={() => setCatalogAttempt((attempt) => attempt + 1)}>Retry</button>
        </div>
      ) : null}
      {missingRnavFixesRunway !== null ? (
        <div className="pilot-error" role="alert">
          No RNAV IF points are available for {activeAirportCode} {missingRnavFixesRunway}.
        </div>
      ) : null}
      {error ? <div className="pilot-error" role="alert">{error}</div> : null}

      {activeMode === "fly" && isChartsOpen && chartMode === "average" && averagedComparison ? (
        <DynamicsComparisonCharts
          chart={averagedComparison.chart}
          systems={averagedComparison.systems}
          hiddenKeys={hiddenComparisonKeys}
          onToggleSystem={toggleComparisonSystem}
          onClose={() => setIsChartsOpen(false)}
          subtitle={`Mean deviation across ${averagedComparison.runCount} stored run${
            averagedComparison.runCount === 1 ? "" : "s"
          }, resampled onto a common distance grid (backend-averaged).`}
        />
      ) : activeMode === "fly" && runChartOpen && comparisonResult ? (
        <DynamicsComparisonCharts
          chart={comparisonResult.chart}
          systems={comparisonResult.systems}
          hiddenKeys={hiddenComparisonKeys}
          onToggleSystem={toggleComparisonSystem}
          onClose={() => setIsChartsOpen(false)}
        />
      ) : null}
    </div>
  );
}

function makeDefaultInitialState(
  airport: { lon: number; lat: number } | null,
  aircraft: PilotAircraftConfig | null,
): PilotResetState {
  return {
    lon: airport?.lon ?? -78.7873,
    lat: airport?.lat ?? 35.878659,
    altM: 1000,
    speedMps: defaultInitialSpeedMps(aircraft),
    headingDeg: 0,
    flightPathDeg: 0,
    massKg: aircraft?.massKg ?? 0,
    aircraftType: aircraft?.code ?? "",
  };
}

function makeDefaultControls(aircraft: PilotAircraftConfig | null): PanelControls {
  return {
    thrustN: Math.min(DEFAULT_THRUST_N, aircraft?.maxThrustN ?? DEFAULT_THRUST_N),
    bankDeg: DEFAULT_BANK_DEG,
    attackDeg: 5.783,
    loadFactor: DEFAULT_LOAD_FACTOR,
  };
}

function makeDefaultTrajectoryTarget(
  runwayTarget: RunwayThresholdTarget | null,
  aircraft: PilotAircraftConfig | null,
  fallback?: PilotTargetState,
  resetSpeed = false,
): PilotTargetState {
  const fallbackSpeed = resetSpeed ? undefined : fallback?.speedMps;
  // No aircraft yet: no target speed (NaN) until the catalog names one; then the published
  // speed, or the kept speed clamped into that aircraft's range.
  const keptSpeed =
    fallbackSpeed !== undefined && Number.isFinite(fallbackSpeed) ? fallbackSpeed : undefined;
  return {
    runwayThresholdId: runwayTarget?.id ?? fallback?.runwayThresholdId ?? "",
    lon: runwayTarget?.lon ?? fallback?.lon ?? 0,
    lat: runwayTarget?.lat ?? fallback?.lat ?? 0,
    altM: runwayTarget
      ? targetAltitudeMForThreshold(runwayTarget.altM, aircraft)
      : fallback?.altM ?? 0,
    speedMps: aircraft
      ? clampTargetSpeedMps(keptSpeed ?? defaultTargetSpeedMps(aircraft), aircraft)
      : fallbackSpeed ?? Number.NaN,
    headingDeg: runwayTarget
      ? runwayAlignedHeadingDeg(runwayTarget.psiDeg)
      : runwayAlignedHeadingDeg(fallback?.headingDeg ?? 0),
    flightPathDeg: fallback?.flightPathDeg ?? DEFAULT_TARGET_GAMMA_DEG,
  };
}

function defaultInitialSpeedMps(aircraft: PilotAircraftConfig | null): number {
  return aircraft
    ? initialSpeedMpsForAircraft(aircraft)
    : knotsToMetresPerSecond(170);
}

function initialSpeedMpsForAircraft(aircraft: PilotAircraftConfig): number {
  if (!Number.isFinite(aircraft.terminalSpeedKt)) {
    throw new Error(
      `Aircraft spec ${aircraft.code} is missing terminalSpeedKt; cannot set RNAV IF initial speed.`,
    );
  }
  return knotsToMetresPerSecond(aircraft.terminalSpeedKt + 25);
}

function makeInitialStateFromRnavFix(
  candidate: RnavInitialFixCandidate,
  aircraft: PilotAircraftConfig,
  fallback: PilotResetState,
  speedMps: number,
): PilotResetState {
  return {
    ...fallback,
    lon: candidate.lon,
    lat: candidate.lat,
    altM: candidate.altM,
    headingDeg: runwayAlignedHeadingDeg(candidate.headingDeg),
    flightPathDeg: 0,
    speedMps,
    massKg: aircraft.massKg,
    aircraftType: aircraft.code,
  };
}

/** Shortest-arc magnitude between two headings in degrees (0..180). */
function headingMagnitudeDeg(a: number, b: number): number {
  return Math.abs(((a - b + 540) % 360) - 180);
}

function trajectoryTargetToPilotState(
  target: PilotTargetState,
  aircraftType: PilotAircraftType,
  massKg: number,
): PilotResetState {
  return {
    lon: target.lon,
    lat: target.lat,
    altM: target.altM,
    speedMps: target.speedMps,
    headingDeg: target.headingDeg,
    flightPathDeg: target.flightPathDeg,
    massKg,
    aircraftType,
  };
}

function trajectorySampleToSnapshot(
  sample: TrajectorySample,
  simulationMode: PilotSimulationMode,
  aircraftType: PilotAircraftType,
  massKg: number,
): PilotSnapshot {
  const control: PilotControls = {
    thrustN: sample.thrustN,
    bankDeg: sample.bankDeg,
    attackDeg: 0,
  };
  if (sample.loadFactor !== undefined) {
    control.loadFactor = sample.loadFactor;
  }
  return {
    ok: true,
    elapsedS: sample.t,
    simulationMode,
    state: {
      lon: sample.lon,
      lat: sample.lat,
      altM: sample.altM,
      speedMps: sample.speedMps,
      headingDeg: sample.headingDeg,
      flightPathDeg: sample.flightPathDeg,
      massKg,
      aircraftType,
    },
    control,
    aero: {
      liftCoefficient: sample.liftCoefficient,
      dragCoefficient: sample.dragCoefficient,
      actualLoadFactor: sample.actualLoadFactor,
    },
  };
}

function snapshotToPose(snapshot: PilotSnapshot | null): PilotAircraftPose | null {
  if (!snapshot) return null;
  return {
    lon: snapshot.state.lon,
    lat: snapshot.state.lat,
    altM: snapshot.state.altM,
    headingDeg: snapshot.state.headingDeg,
    flightPathDeg: snapshot.state.flightPathDeg,
    bankDeg: snapshot.control.bankDeg,
    attackDeg: snapshot.control.attackDeg,
  };
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function isEditableTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLTextAreaElement;
}
