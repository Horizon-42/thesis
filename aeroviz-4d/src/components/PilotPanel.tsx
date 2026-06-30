import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { useApp } from "../context/AppContext";
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
import {
  runTrajectoryOptimization,
  optimizerToParts,
  partsToOptimizer,
  validFittingsForDynamics,
  type TrajectoryOptimizer,
  type TrajectoryOptimizationResult,
  type TrajectorySample,
  type OptimizerDynamics,
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
const DEFAULT_CONTROLS: PilotControls = makeDefaultControls(null);
const DEFAULT_INTEGRATOR_DT_S = 0.2;
const DEFAULT_TRAJECTORY_DT_S = 0.5;
const PLAYBACK_FRAME_DT_S = 0.2;
const STEP_INTERVAL_MS = 120;
const MAX_TRAIL_POINTS = 360;
const DEFAULT_TARGET_GAMMA_DEG = -3;
const DEFAULT_MAX_ITERATIONS = 300;
// A full RNAV approach optimizes to ~250-350 s, and for the multiphase optimizer this is the
// per-phase cap (the longest leg alone is ~130 s), so the default must clear that comfortably.
const DEFAULT_ARRIVAL_TIME_S = 600;
const DEFAULT_COMPARISON_DURATION_S = 240;
const DEFAULT_COMPARISON_DT_S = 0.1;
const DEFAULT_COMPARISON_CONTROL: DynamicsComparisonControl = {
  thrustN: 70000,
  bankDeg: 0,
  loadFactor: 1,
};
const DEFAULT_TRAJECTORY_OPTIMIZER: TrajectoryOptimizer = "casadiMultiphaseNormalizedFullTransport";
const OPTIMIZER_DYNAMICS_OPTIONS: { value: OptimizerDynamics; label: string }[] = [
  { value: "geodetic", label: "Geodetic RHS (+transport, approx)" },
  { value: "reanchoredEnu", label: "Re-anchored ENU (playback model)" },
  { value: "localEnu", label: "Local ENU @ target (fixed tangent, drifts far out)" },
  { value: "geodeticNormalized", label: "Geodetic RHS (normalized, robust)" },
  { value: "geodeticFullTransport", label: "Geodetic RHS (+transport, full/exact)" },
  { value: "geodeticNormalizedFullTransport", label: "Geodetic RHS (normalized + full/exact transport)" },
  { value: "geodeticMultiphase", label: "Multiphase (per-leg procedure constraints)" },
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
 * chips — reuses the Compare-mode delta strip with one amber chip per row. */
const TARGET_DELTA_KEY = "Δ";
const TARGET_DELTA_SYSTEMS: DynamicsComparisonSystem[] = [
  { key: TARGET_DELTA_KEY, label: "final − target", colorRgba: [251, 191, 36, 255], isReference: false },
];

function usesLoadFactorControl(mode: PilotSimulationMode) {
  return mode === "loadFactor" || mode === "casadi";
}

function trajectoryOptimizerSimulationMode(
  optimizer: TrajectoryOptimizer,
): PilotSimulationMode {
  // The CasADi optimisers (IPOPT and every direct-collocation defect-scheme
  // variant) emit LoadFactorControl-shaped controls (T, mu, n_cmd), so
  // playback must run the "casadi" simulation mode to interpret them.  All
  // other optimisers emit alpha-based controls and play back via alpha.
  return optimizer === "casadiIpopt" ||
    optimizer.startsWith("casadiDirectCollocation")
    ? "casadi"
    : "alpha";
}

type PilotPanelMode = "pilot" | "trajectory" | "comparison";

interface PlacementBackup {
  initialState: PilotResetState;
  isInitialPreviewVisible: boolean;
  isEnabled: boolean;
  isFlying: boolean;
  snapshot: PilotSnapshot | null;
  trail: PilotAircraftPose[];
}

interface PilotPanelProps {
  /**
   * When provided, the panel's mode is controlled by the workbench shell (the global
   * task switcher) and the panel's own tab row is hidden. Omitted → the panel keeps its
   * own internal tab switching (used standalone / in tests).
   */
  mode?: PilotPanelMode;
  onRequestMode?: (mode: PilotPanelMode) => void;
}

export default function PilotPanel({ mode: controlledMode, onRequestMode }: PilotPanelProps = {}) {
  const { activeAirportCode, airport, viewer } = useApp();
  const [internalActiveMode, setActiveMode] = useState<PilotPanelMode>("pilot");
  const activeMode = controlledMode ?? internalActiveMode;
  const [isEnabled, setIsEnabled] = useState(false);
  const [isFlying, setIsFlying] = useState(false);
  // Optimized-trajectory playback now runs on Cesium's own clock from a backend
  // CZML.  `isTrajectoryPlaybackActive` means the CZML is loaded into the scene;
  // `isTrajectoryPlaying` mirrors the intended clock animation (play vs pause).
  const [isTrajectoryPlaybackActive, setIsTrajectoryPlaybackActive] = useState(false);
  const [isTrajectoryPlaying, setIsTrajectoryPlaying] = useState(false);
  const [isInitialEditorOpen, setIsInitialEditorOpen] = useState(false);
  const [isTargetEditorOpen, setIsTargetEditorOpen] = useState(false);
  const [isPlacingInitialPosition, setIsPlacingInitialPosition] = useState(false);
  const [isInitialPreviewVisible, setIsInitialPreviewVisible] = useState(false);
  const [isFollowing, setIsFollowing] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aircraftConfigs, setAircraftConfigs] = useState<PilotAircraftConfig[]>([]);
  const [simulationMode, setSimulationMode] =
    useState<PilotSimulationMode>(DEFAULT_SIMULATION_MODE);
  const [controls, setControls] = useState<PilotControls>(DEFAULT_CONTROLS);
  const [integratorDtS, setIntegratorDtS] = useState(DEFAULT_INTEGRATOR_DT_S);
  const [snapshot, setSnapshot] = useState<PilotSnapshot | null>(null);
  const [trail, setTrail] = useState<PilotAircraftPose[]>([]);
  const [runwayTargets, setRunwayTargets] = useState<RunwayThresholdTarget[]>([]);
  const [targetState, setTargetState] = useState<PilotTargetState>(() =>
    makeDefaultTrajectoryTarget(null, null),
  );
  const [trajectoryOptimizer, setTrajectoryOptimizer] =
    useState<TrajectoryOptimizer>(DEFAULT_TRAJECTORY_OPTIMIZER);
  const [nSegments, setNSegments] = useState(10);
  const [arrivalTimeS, setArrivalTimeS] = useState(DEFAULT_ARRIVAL_TIME_S);
  const [trajectoryDtS, setTrajectoryDtS] = useState(DEFAULT_TRAJECTORY_DT_S);
  const [maxIterations, setMaxIterations] = useState(DEFAULT_MAX_ITERATIONS);
  const [optimizedTrajectory, setOptimizedTrajectory] =
    useState<TrajectoryOptimizationResult | null>(null);

  // Dynamics Comparison mode: one trajectory flown four ways under one constant
  // control, replayed as a multi-system CZML with a pop-up deviation chart.
  const [comparisonControl, setComparisonControl] =
    useState<DynamicsComparisonControl>(DEFAULT_COMPARISON_CONTROL);
  const [comparisonDurationS, setComparisonDurationS] = useState(
    DEFAULT_COMPARISON_DURATION_S,
  );
  const [comparisonDtS, setComparisonDtS] = useState(DEFAULT_COMPARISON_DT_S);
  const [comparisonResult, setComparisonResult] =
    useState<DynamicsComparisonResult | null>(null);
  const [isComparisonPlaybackActive, setIsComparisonPlaybackActive] = useState(false);
  const [isComparisonPlaying, setIsComparisonPlaying] = useState(false);
  const [hiddenComparisonKeys, setHiddenComparisonKeys] = useState<string[]>([]);
  // A/C/D deviations vs the reference B at the current clock time, overlaid on
  // the Live-State readout during a Compare playback.
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
  const [rnavInitialFixCandidates, setRnavInitialFixCandidates] = useState<
    RnavInitialFixCandidate[]
  >([]);
  const [selectedRnavInitialFixKey, setSelectedRnavInitialFixKey] = useState("");

  const pose = snapshotToPose(snapshot);
  const selectedAircraft = aircraftConfigs.find(
    (config) => config.code === initialState.aircraftType,
  ) ?? aircraftConfigs[0] ?? null;
  const selectedMaxThrustN = selectedAircraft?.maxThrustN ?? FALLBACK_MAX_THRUST_N;
  const targetSpeedBounds = targetSpeedBoundsMps(selectedAircraft);
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
    setIsTrajectoryPlaybackActive(false);
  }, []);

  // Invalidate any computed/loaded dynamics comparison. Used whenever an input
  // feeding the comparison changes, so a stale CZML/chart never stays on screen.
  const clearComparisonPlayback = useCallback(() => {
    setComparisonResult(null);
    setIsComparisonPlaying(false);
    setIsComparisonPlaybackActive(false);
    setIsChartsOpen(false);
    setHiddenComparisonKeys([]);
    setChartMode("run");
  }, []);

  const clearSnapshotForInitialEdit = useCallback(() => {
    clearOptimizedPlayback();
    clearComparisonPlayback();
    if (!snapshot && !isEnabled && !isFlying && !isTrajectoryPlaying) return;

    setIsFlying(false);
    setIsEnabled(false);
    setSnapshot(null);
    setTrail([]);
  }, [
    clearOptimizedPlayback,
    clearComparisonPlayback,
    isEnabled,
    isFlying,
    isTrajectoryPlaying,
    snapshot,
  ]);

  const updateInitialPosition = useCallback(
    (position: PilotInitialPlacementPosition) => {
      setSelectedRnavInitialFixKey("");
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
      setSnapshot(backup.snapshot);
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
      snapshot,
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
    snapshot,
    trail,
    aircraftConfigs.length,
  ]);

  usePilotInitialPlacement({
    enabled: isPlacingInitialPosition,
    // The static "START" preview marks the chosen start state while setting up.
    // Hide it once a comparison is loaded/playing (Compare never sets `snapshot`,
    // so without this guard the START aircraft would sit at the origin while the
    // per-system models fly away).
    previewVisible: isPlacingInitialPosition ||
      ((isInitialEditorOpen || isInitialPreviewVisible) &&
        !isEnabled &&
        !snapshot &&
        !isComparisonPlaybackActive),
    initialState,
    placementGuidance,
    onPositionChange: updateInitialPosition,
    onFinish: finishInitialPlacement,
    onCancel: cancelInitialPlacement,
  });

  // The hand-built aircraft + trail are only for live Pilot mode. In Trajectory
  // Play the aircraft and the colored trajectory come from the CZML instead.
  usePilotAircraft({
    enabled: isEnabled && activeMode === "pilot",
    pose,
    trail,
    follow: isFollowing,
  });

  usePilotTargetGate({
    enabled: activeMode === "trajectory",
    target: targetGateState,
  });

  // Drive the live readout from the optimized rollout sampled at the clock time.
  const playbackOptimizer = optimizedTrajectory?.optimizer ?? DEFAULT_TRAJECTORY_OPTIMIZER;
  const handlePlaybackSample = useCallback(
    (sample: TrajectorySample | null) => {
      if (!sample) {
        setSnapshot(null);
        return;
      }
      setSnapshot(
        trajectorySampleToSnapshot(
          sample,
          trajectoryOptimizerSimulationMode(playbackOptimizer),
          initialState.aircraftType,
          initialState.massKg,
        ),
      );
    },
    [playbackOptimizer, initialState.aircraftType, initialState.massKg],
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
        setSnapshot(null);
        return;
      }
      setSnapshot(
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
    follow: isFollowing && activeMode === "comparison",
    samples: comparisonResult?.playback.samples ?? EMPTY_SAMPLES,
    onSample: handleComparisonSample,
    chart: comparisonResult?.chart ?? null,
    onDeltas: setComparisonDeltas,
  });

  // Trajectory Play: the aircraft's live deviation from the requested target,
  // shown as one amber delta chip per state row (the same style as the Compare
  // deviations) instead of separate "X Error" rows. It tracks the sampled state,
  // so it converges to the final-vs-target error as playback reaches the end.
  const trajectoryTargetDeltas = useMemo<DynamicsComparisonDeltas | null>(() => {
    if (activeMode !== "trajectory" || !snapshot) return null;
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

  // Keep the backend's casadi solver worker resident (warm) while the Optimize
  // (trajectory) or Compare (comparison) tab is open, and decommission it when
  // the tab closes — so repeated solves are fast but the worker's memory is
  // reclaimed once the user leaves. A `pagehide` beacon also releases it when
  // the whole tab/window closes, where this cleanup would not run.
  useEffect(() => {
    const kind: WorkerSessionKind | null =
      activeMode === "trajectory"
        ? "optimizer"
        : activeMode === "comparison"
          ? "comparison"
          : null;
    if (kind === null) {
      return undefined;
    }
    void openWorkerSession(kind);
    const releaseOnUnload = () => beaconCloseWorkerSession(kind);
    window.addEventListener("pagehide", releaseOnUnload);
    return () => {
      window.removeEventListener("pagehide", releaseOnUnload);
      void closeWorkerSession(kind);
    };
  }, [activeMode]);

  useEffect(() => {
    const aircraft = aircraftConfigs[0] ?? null;
    placementBackupRef.current = null;
    setInitialState(makeDefaultInitialState(airport, aircraft));
    setRnavInitialFixCandidates([]);
    setSelectedRnavInitialFixKey("");
    setSimulationMode(DEFAULT_SIMULATION_MODE);
    setControls(makeDefaultControls(aircraft));
    setActiveMode("pilot");
    setIsInitialEditorOpen(false);
    setIsTargetEditorOpen(false);
    setIsPlacingInitialPosition(false);
    setIsInitialPreviewVisible(false);
    setIsFlying(false);
    setIsTrajectoryPlaying(false);
    setIsTrajectoryPlaybackActive(false);
    setIsEnabled(false);
    setSnapshot(null);
    setTrail([]);
    setOptimizedTrajectory(null);
    setComparisonControl(DEFAULT_COMPARISON_CONTROL);
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
        setError(null);
      })
      .catch((configError: unknown) => {
        if (cancelled) return;
        setAircraftConfigs([]);
        setError(toErrorMessage(configError));
      });

    return () => {
      cancelled = true;
    };
  }, []);

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

  useEffect(() => {
    if (
      (activeMode !== "trajectory" && activeMode !== "comparison") ||
      !activeAirportCode ||
      !selectedTargetRunway
    ) {
      setRnavInitialFixCandidates([]);
      setSelectedRnavInitialFixKey("");
      return;
    }

    let cancelled = false;
    setRnavInitialFixCandidates([]);
    setSelectedRnavInitialFixKey("");

    void fetchRnavInitialFixCandidates(
      activeAirportCode,
      selectedTargetRunway.runwayIdent,
    )
      .then((candidates) => {
        if (cancelled) return;
        setRnavInitialFixCandidates(candidates);
        // In Compare mode an RNAV fix is an optional convenience (the start can
        // also be edited / placed), so an empty list is not an error there.
        if (candidates.length === 0 && activeMode === "trajectory") {
          setError(
            `No RNAV IF points are available for ${activeAirportCode} ${selectedTargetRunway.runwayIdent}.`,
          );
          return;
        }
        setError(null);
      })
      .catch((initialError: unknown) => {
        if (cancelled) return;
        setRnavInitialFixCandidates([]);
        setError(toErrorMessage(initialError));
      });

    return () => {
      cancelled = true;
    };
  }, [
    activeAirportCode,
    activeMode,
    selectedTargetRunway,
  ]);

  // Refresh the stored-run count when entering Compare mode, so the Average
  // button reflects history from earlier sessions too (count is server-side).
  useEffect(() => {
    if (activeMode !== "comparison") return;
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
          setSnapshot(nextSnapshot);
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
    if (!isEnabled || activeMode !== "pilot") return;

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
          setControls((current) =>
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
    setIsBusy(true);
    setError(null);
    try {
      if (!snapshot) {
        const nextSnapshot = await resetPilotSimulation(
          initialState,
          controls,
          simulationMode,
        );
        setSnapshot(nextSnapshot);
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
    setIsBusy(true);
    setError(null);
    try {
      const nextSnapshot = await resetPilotSimulation(
        initialState,
        controls,
        simulationMode,
      );
      setSnapshot(nextSnapshot);
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
    setSnapshot(null);
    setTrail([]);
    setError(null);
  }

  function updateControl(
    key: keyof PilotControls,
    value: number,
    min: number,
    max: number,
  ) {
    if (!Number.isFinite(value)) return;
    setControls((current) => ({ ...current, [key]: clamp(value, min, max) }));
  }

  function updateSimulationMode(value: PilotSimulationMode) {
    setSimulationMode(value);
    if (!usesLoadFactorControl(value)) return;

    setControls((current) =>
      current.loadFactor === undefined
        ? { ...current, loadFactor: DEFAULT_LOAD_FACTOR }
        : current
    );
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

    setInitialState((current) => ({ ...current, [key]: clamp(value, min, max) }));
    setSelectedRnavInitialFixKey("");
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
    setIsTrajectoryPlaybackActive(false);
    setIsComparisonPlaying(false);
    setIsComparisonPlaybackActive(false);
    setIsChartsOpen(false);
  }

  // When the panel is shell-controlled, mode changes are requested upward (the shell
  // updates the global mode → controlledMode flows back in); otherwise switch locally.
  function applyMode(next: PilotPanelMode) {
    if (onRequestMode) onRequestMode(next);
    else setActiveMode(next);
  }

  function openTrajectoryMode() {
    if (isPlacingInitialPosition) return;
    setIsFlying(false);
    suspendPlaybacks();
    setIsInitialEditorOpen(false);
    applyMode("trajectory");
    setError(null);
  }

  function openPilotMode() {
    if (isPlacingInitialPosition) return;
    suspendPlaybacks();
    setIsTargetEditorOpen(false);
    applyMode("pilot");
    setError(null);
  }

  function openComparisonMode() {
    if (isPlacingInitialPosition) return;
    setIsFlying(false);
    suspendPlaybacks();
    setIsTargetEditorOpen(false);
    setIsInitialEditorOpen(false);
    applyMode("comparison");
    setError(null);
  }

  // When the shell drives the mode, run the same side-effects the internal tab handlers
  // would (release the clock, close editors) on each external mode change.
  const previousControlledModeRef = useRef(controlledMode);
  useEffect(() => {
    if (controlledMode === undefined || previousControlledModeRef.current === controlledMode) {
      previousControlledModeRef.current = controlledMode;
      return;
    }
    previousControlledModeRef.current = controlledMode;
    if (isPlacingInitialPosition) return;
    suspendPlaybacks();
    setIsInitialEditorOpen(false);
    setIsTargetEditorOpen(false);
    if (controlledMode !== "pilot") setIsFlying(false);
    setError(null);
  }, [controlledMode]);

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
    setNSegments(Math.round(clamp(value, 1, 80)));
    clearOptimizedPlayback();
  }

  function updateArrivalTime(value: number) {
    if (!Number.isFinite(value)) return;
    setArrivalTimeS(clamp(value, 1, 1000));
    clearOptimizedPlayback();
  }

  function updateTrajectoryDt(value: number) {
    if (!Number.isFinite(value)) return;
    setTrajectoryDtS(clamp(value, 0.02, 2));
    clearOptimizedPlayback();
  }

  function updateMaxIterations(value: number) {
    if (!Number.isFinite(value)) return;
    setMaxIterations(Math.round(clamp(value, 1, 10000)));
    clearOptimizedPlayback();
  }

  function updateTrajectoryOptimizer(value: TrajectoryOptimizer) {
    setTrajectoryOptimizer(value);
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
      const { dynamics } = optimizerToParts(trajectoryOptimizer);
      if (dynamics === "geodeticMultiphase") {
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
      }

      const result = await runTrajectoryOptimization({
        optimizer: trajectoryOptimizer,
        initialState,
        targetState: trajectoryTargetToPilotState(
          targetState,
          initialState.aircraftType,
          initialState.massKg,
        ),
        nSegments,
        arrivalTimeS,
        dtS: trajectoryDtS,
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

  function playOptimizedTrajectory() {
    if (!optimizedTrajectory?.playback) return;

    setError(null);
    setIsFlying(false);
    setIsEnabled(false);
    setIsTrajectoryPlaybackActive(true);
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
  function updateComparisonControl(
    key: keyof DynamicsComparisonControl,
    value: number,
    min: number,
    max: number,
  ) {
    if (!Number.isFinite(value)) return;
    setComparisonControl((current) => ({ ...current, [key]: clamp(value, min, max) }));
    clearComparisonPlayback();
  }

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
    if (!hasAircraftConfigs) return;

    setIsBusy(true);
    setIsFlying(false);
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
    if (isChartsOpen && chartMode === "run") {
      setIsChartsOpen(false);
      return;
    }
    setChartMode("run");
    setIsChartsOpen(true);
  }

  function playComparison() {
    if (!comparisonResult) return;

    setError(null);
    setIsFlying(false);
    setIsEnabled(false);
    setIsComparisonPlaybackActive(true);
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
    key: keyof PilotControls,
    delta: number,
    min: number,
    max: number,
  ) {
    setControls((current) => ({
      ...current,
      [key]: clamp((current[key] ?? defaultControlValue(key)) + delta, min, max),
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
    : isBusy && (activeMode === "trajectory" || activeMode === "comparison")
      ? "Computing"
      : isTrajectoryPlaying || isComparisonPlaying
        ? "Playing"
        : isFlying
          ? "Flying"
          : isTrajectoryPlaybackActive || isComparisonPlaybackActive
            ? "Paused"
            : (optimizedTrajectory && activeMode === "trajectory") ||
                (comparisonResult && activeMode === "comparison")
              ? "Ready"
              : snapshot
                ? "Paused"
                : "Standby";
  const hasAircraftConfigs = aircraftConfigs.length > 0;
  const isAnyPlaying = isTrajectoryPlaying || isComparisonPlaying;
  const initialControlsDisabled = isFlying || isAnyPlaying || isBusy || !hasAircraftConfigs;
  const targetControlsDisabled = isBusy || isTrajectoryPlaying || runwayTargets.length === 0;
  const comparisonControlsDisabled = isBusy || isComparisonPlaying || !hasAircraftConfigs;
  // The single optimizer name is shown as two dropdowns: dynamics × fitting.
  const { dynamics: optimizerDynamics, fitting: optimizerFitting } =
    optimizerToParts(trajectoryOptimizer);
  const allowedFittings = validFittingsForDynamics(optimizerDynamics);
  const trajectorySegmentDurationS = optimizedTrajectory
    ? optimizedTrajectory.finalTimeS / Math.max(1, optimizedTrajectory.controls.length)
    : null;

  return (
    <div className="pilot-panel">
      <PilotRealtimeStatePanel
        snapshot={snapshot}
        visible={
          isFlying ||
          isTrajectoryPlaying ||
          (activeMode === "trajectory" && snapshot !== null) ||
          (activeMode === "comparison" && isComparisonPlaybackActive && snapshot !== null)
        }
        showControlReadout={activeMode === "trajectory" || activeMode === "comparison"}
        simulationMode={snapshot?.simulationMode ?? simulationMode}
        comparisonDeltas={
          activeMode === "comparison"
            ? comparisonDeltas
            : activeMode === "trajectory"
              ? trajectoryTargetDeltas
              : null
        }
        comparisonSystems={
          activeMode === "comparison"
            ? comparisonResult?.systems ?? null
            : activeMode === "trajectory" && trajectoryTargetDeltas
              ? TARGET_DELTA_SYSTEMS
              : null
        }
        deltaReferenceLabel={activeMode === "trajectory" ? "target" : "B"}
      />

      <header className="pilot-panel-header">
        <div className="pilot-panel-header-main">
          <div className="pilot-panel-title-block">
            <h3>
              {activeMode === "comparison"
                ? "Dynamics Compare"
                : activeMode === "trajectory"
                  ? "Trajectory Play"
                  : "Pilot Mode"}
            </h3>
          </div>
          <span className={`pilot-status pilot-status-${statusLabel.toLowerCase()}`}>
            {statusLabel}
          </span>
        </div>
        {controlledMode === undefined ? (
          <div className="pilot-panel-mode-row pilot-panel-mode-switch" role="group" aria-label="Panel mode">
            {([
              { mode: "pilot", label: "Pilot", onClick: openPilotMode },
              { mode: "trajectory", label: "Trajectory", onClick: openTrajectoryMode },
              { mode: "comparison", label: "Compare", onClick: openComparisonMode },
            ] as const).map((entry) => (
              <button
                key={entry.mode}
                type="button"
                className={`pilot-mode-toggle${activeMode === entry.mode ? " active" : ""}`}
                onClick={entry.onClick}
                disabled={isPlacingInitialPosition}
                aria-pressed={activeMode === entry.mode}
              >
                {entry.label}
              </button>
            ))}
          </div>
        ) : null}
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
        rnavInitialFixCandidates={
          activeMode === "trajectory" || activeMode === "comparison"
            ? rnavInitialFixCandidates
            : []
        }
        selectedRnavInitialFixKey={selectedRnavInitialFixKey}
        disabled={initialControlsDisabled}
        onClose={closeInitialEditor}
        onPlaceToggle={toggleInitialPlacement}
        onFieldChange={updateInitialField}
        onAircraftTypeChange={updateAircraftType}
        onRnavInitialFixChange={updateRnavInitialFix}
      />

      {activeMode === "trajectory" ? (
        <>
          <section className="pilot-initial-summary" aria-label="Target aircraft state summary">
            <div className="pilot-initial-summary-header">
              <h4>Target State</h4>
              <button
                type="button"
                onClick={openTargetEditor}
                disabled={targetControlsDisabled}
              >
                Edit
              </button>
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
                <dd>{formatNumberInputValue(targetState.speedMps)} m/s</dd>
              </div>
              <div>
                <dt>Gamma</dt>
                <dd>{formatNumberInputValue(targetState.flightPathDeg)} deg</dd>
              </div>
            </dl>
          </section>

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

          <section className="pilot-optimization-row" aria-label="Trajectory optimization settings">
            <label>
              <span>Dynamics</span>
              <select
                className="pilot-select-input"
                value={optimizerDynamics}
                disabled={targetControlsDisabled}
                onChange={(event) =>
                  updateTrajectoryOptimizer(
                    partsToOptimizer(event.target.value as OptimizerDynamics, optimizerFitting),
                  )
                }
              >
                {OPTIMIZER_DYNAMICS_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Fitting</span>
              <select
                className="pilot-select-input"
                value={optimizerFitting}
                disabled={targetControlsDisabled || allowedFittings.length === 1}
                onChange={(event) =>
                  updateTrajectoryOptimizer(
                    partsToOptimizer(optimizerDynamics, event.target.value as OptimizerFitting),
                  )
                }
              >
                {OPTIMIZER_FITTING_OPTIONS.filter((o) => allowedFittings.includes(o.value)).map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </label>
            {optimizerDynamics === "geodeticMultiphase" && (
              <span
                className="pilot-multiphase-hint"
                title="Multiphase optimises one phase per procedure leg (start->IAF, then each leg), enforcing that leg's corridor / glidepath / step-down floor as NLP path constraints. Select an RNAV initial fix to identify the approach."
              >
                Per-leg constraints from the selected RNAV approach
              </span>
            )}
            <label>
              <span>Segments</span>
              <EnglishNumberInput
                value={nSegments}
                min={1}
                max={80}
                step="1"
                disabled={targetControlsDisabled}
                onCommit={updateNSegments}
              />
            </label>
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
              <span>dt</span>
              <EnglishNumberInput
                value={trajectoryDtS}
                min={0.02}
                max={2}
                step="0.02"
                disabled={targetControlsDisabled}
                onCommit={updateTrajectoryDt}
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
                  <dt>dt</dt>
                  <dd>{formatNumberInputValue(optimizedTrajectory.dtS)} s</dd>
                </div>
                <div>
                  <dt>Segment</dt>
                  <dd>{formatNumberInputValue(trajectorySegmentDurationS ?? 0)} s</dd>
                </div>
              </dl>
            ) : null}
          </section>
        </>
      ) : activeMode === "comparison" ? (
        <>
          <p className="dyncmp-hint">
            Set the start state via <strong>Edit</strong> above (fields / place on
            map) or pick a published RNAV fix from a runway below.
          </p>
          <section
            className="pilot-optimization-row"
            aria-label="Dynamics comparison settings"
          >
            <label>
              <span>RNAV runway</span>
              <select
                className="pilot-select-input"
                value={targetState.runwayThresholdId}
                disabled={comparisonControlsDisabled || runwayTargets.length === 0}
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
            <label>
              <span>Thrust</span>
              <EnglishNumberInput
                value={comparisonControl.thrustN}
                min={0}
                max={selectedMaxThrustN}
                step="500"
                disabled={comparisonControlsDisabled}
                onCommit={(value) =>
                  updateComparisonControl("thrustN", value, 0, selectedMaxThrustN)
                }
              />
            </label>
            <label>
              <span>Bank</span>
              <EnglishNumberInput
                value={comparisonControl.bankDeg}
                min={-60}
                max={60}
                step="1"
                disabled={comparisonControlsDisabled}
                onCommit={(value) => updateComparisonControl("bankDeg", value, -60, 60)}
              />
            </label>
            <label>
              <span>Load factor</span>
              <EnglishNumberInput
                value={comparisonControl.loadFactor}
                min={MIN_LOAD_FACTOR}
                max={MAX_LOAD_FACTOR}
                step="0.05"
                disabled={comparisonControlsDisabled}
                onCommit={(value) =>
                  updateComparisonControl("loadFactor", value, MIN_LOAD_FACTOR, MAX_LOAD_FACTOR)
                }
              />
            </label>
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
          </section>

          <div className="pilot-actions">
            <button
              className="pilot-primary-button"
              onClick={computeComparison}
              disabled={
                isBusy ||
                isComparisonPlaying ||
                isPlacingInitialPosition ||
                !hasAircraftConfigs
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

          <section className="pilot-control-zone" aria-label="Dynamics comparison playback">
            <div className="pilot-options-row">
              <label className="pilot-checkbox-label">
                <input
                  type="checkbox"
                  checked={isFollowing}
                  onChange={(event) => setIsFollowing(event.target.checked)}
                />
                Follow B
              </label>
              <button
                type="button"
                onClick={toggleRunCharts}
                disabled={!comparisonResult}
              >
                {isChartsOpen && chartMode === "run" ? "Hide charts" : "Show charts"}
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
                Set a constant control + horizon, then Compute to fly the start state
                four ways (fixed tangent, re-anchored, geodetic ±transport) and compare
                the drift.
              </p>
            )}
          </section>
        </>
      ) : (
        <>
          <div className="pilot-actions">
            <button
              className="pilot-primary-button"
              onClick={startPilot}
              disabled={isBusy || isFlying || isPlacingInitialPosition || !hasAircraftConfigs}
            >
              {snapshot ? "Resume" : "Start"}
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
              disabled={(!isEnabled && !snapshot) || isPlacingInitialPosition}
            >
              End
            </button>
          </div>

          <section className="pilot-control-zone" aria-label="Pilot controls">
            <div className="pilot-stepper-row">
              <button
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
                  disabled={false}
                  onCommit={(value) => updateControl("bankDeg", value, -45, 45)}
                />
              </label>
              <button
                onClick={() => nudgeControl("bankDeg", -5, -45, 45)}
                title="Bank right"
              >
                &gt;
              </button>
            </div>

            {usesLoadFactorControl(simulationMode) ? (
              <div className="pilot-stepper-row">
                <button
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
                    value={controls.loadFactor ?? DEFAULT_LOAD_FACTOR}
                    min={MIN_LOAD_FACTOR}
                    max={MAX_LOAD_FACTOR}
                    step="0.05"
                    disabled={false}
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
                    disabled={false}
                    onCommit={(value) => updateControl("attackDeg", value, -10, 18)}
                  />
                </label>
                <button
                  onClick={() => nudgeControl("attackDeg", 0.5, -10, 18)}
                  title="Increase alpha"
                >
                  +
                </button>
              </div>
            )}

            <div className="pilot-stepper-row">
              <button
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
                  disabled={false}
                  onCommit={(value) =>
                    updateControl("thrustN", value, 0, selectedMaxThrustN)
                  }
                />
              </label>
              <button
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
                  disabled={isPlacingInitialPosition}
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
        </>
      )}

      {error ? <div className="pilot-error" role="alert">{error}</div> : null}

      {activeMode === "comparison" && isChartsOpen && chartMode === "average" && averagedComparison ? (
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
      ) : activeMode === "comparison" && isChartsOpen && chartMode === "run" && comparisonResult ? (
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

function makeDefaultControls(aircraft: PilotAircraftConfig | null): PilotControls {
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
  return {
    runwayThresholdId: runwayTarget?.id ?? fallback?.runwayThresholdId ?? "",
    lon: runwayTarget?.lon ?? fallback?.lon ?? 0,
    lat: runwayTarget?.lat ?? fallback?.lat ?? 0,
    altM: runwayTarget
      ? targetAltitudeMForThreshold(runwayTarget.altM, aircraft)
      : fallback?.altM ?? 0,
    speedMps: clampTargetSpeedMps(
      fallbackSpeed ?? defaultTargetSpeedMps(aircraft),
      aircraft,
    ),
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
    attackDeg: sample.attackDeg ?? 0,
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

function defaultControlValue(key: keyof PilotControls): number {
  return key === "loadFactor" ? DEFAULT_LOAD_FACTOR : 0;
}

function isEditableTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLTextAreaElement;
}
