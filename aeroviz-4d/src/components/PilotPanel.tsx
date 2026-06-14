import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../context/AppContext";
import {
  fetchRunwayThresholdTargets,
  type RunwayThresholdTarget,
} from "../data/runwayThresholdTargets";
import { usePilotAircraft, type PilotAircraftPose } from "../hooks/usePilotAircraft";
import { usePilotTargetGate } from "../hooks/usePilotTargetGate";
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
  AEROVIZ_BACKEND_URL,
  fetchPilotAircraftConfigs,
  resetPilotSimulation,
  stepPilotSimulation,
  type PilotAircraftConfig,
  type PilotAircraftType,
  type PilotControls,
  type PilotResetState,
  type PilotSnapshot,
} from "../pilot/pilotClient";
import {
  TARGET_APPROACH_SPEED_MPS,
  clampHeadingToRunwayTolerance,
  clampTargetSpeedMps,
  runwayAlignedHeadingDeg,
  targetAltitudeMForThreshold,
} from "../pilot/trajectoryTargetConstraints";
import {
  runTrajectoryOptimization,
  type TrajectoryOptimizer,
  type TrajectoryOptimizationResult,
} from "../pilot/trajectoryOptimizationClient";

const DEFAULT_CONTROLS: PilotControls = {
  thrustN: 43441,
  bankDeg: 0,
  attackDeg: 5.783,
};
const DEFAULT_FRAME_DT_S = 0.2;
const STEP_INTERVAL_MS = 120;
const MAX_TRAIL_POINTS = 360;
const DEFAULT_TARGET_SPEED_MPS = TARGET_APPROACH_SPEED_MPS;
const DEFAULT_TARGET_GAMMA_DEG = -3;
const DEFAULT_TARGET_ALPHA_DEG = 4;
const DEFAULT_MAX_ITERATIONS = 300;
const DEFAULT_ARRIVAL_TIME_S = 100;
const DEFAULT_TRAJECTORY_OPTIMIZER: TrajectoryOptimizer = "transcription";

type PilotPanelMode = "pilot" | "trajectory";

interface PlacementBackup {
  initialState: PilotResetState;
  isInitialPreviewVisible: boolean;
  isEnabled: boolean;
  isFlying: boolean;
  snapshot: PilotSnapshot | null;
  trail: PilotAircraftPose[];
}

export default function PilotPanel() {
  const { activeAirportCode, airport } = useApp();
  const [activeMode, setActiveMode] = useState<PilotPanelMode>("pilot");
  const [isEnabled, setIsEnabled] = useState(false);
  const [isFlying, setIsFlying] = useState(false);
  const [isTrajectoryPlaying, setIsTrajectoryPlaying] = useState(false);
  const [isInitialEditorOpen, setIsInitialEditorOpen] = useState(false);
  const [isTargetEditorOpen, setIsTargetEditorOpen] = useState(false);
  const [isPlacingInitialPosition, setIsPlacingInitialPosition] = useState(false);
  const [isInitialPreviewVisible, setIsInitialPreviewVisible] = useState(false);
  const [isFollowing, setIsFollowing] = useState(true);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aircraftConfigs, setAircraftConfigs] = useState<PilotAircraftConfig[]>([]);
  const [controls, setControls] = useState<PilotControls>(DEFAULT_CONTROLS);
  const [frameDtS, setFrameDtS] = useState(DEFAULT_FRAME_DT_S);
  const [snapshot, setSnapshot] = useState<PilotSnapshot | null>(null);
  const [trail, setTrail] = useState<PilotAircraftPose[]>([]);
  const [runwayTargets, setRunwayTargets] = useState<RunwayThresholdTarget[]>([]);
  const [targetState, setTargetState] = useState<PilotTargetState>(() =>
    makeDefaultTrajectoryTarget(null),
  );
  const [trajectoryOptimizer, setTrajectoryOptimizer] =
    useState<TrajectoryOptimizer>(DEFAULT_TRAJECTORY_OPTIMIZER);
  const [nSegments, setNSegments] = useState(10);
  const [arrivalTimeS, setArrivalTimeS] = useState(DEFAULT_ARRIVAL_TIME_S);
  const [trajectoryDtS, setTrajectoryDtS] = useState(DEFAULT_FRAME_DT_S);
  const [maxIterations, setMaxIterations] = useState(DEFAULT_MAX_ITERATIONS);
  const [optimizedTrajectory, setOptimizedTrajectory] =
    useState<TrajectoryOptimizationResult | null>(null);

  const controlsRef = useRef(controls);
  const frameDtRef = useRef(frameDtS);
  const stepInFlightRef = useRef(false);
  const trajectoryPlanRef = useRef<TrajectoryOptimizationResult | null>(null);
  const trajectoryReplayIndexRef = useRef(0);
  const trajectorySegmentElapsedSRef = useRef(0);
  const trajectoryStepInFlightRef = useRef(false);
  const placementBackupRef = useRef<PlacementBackup | null>(null);

  const [initialState, setInitialState] = useState<PilotResetState>(() =>
    makeDefaultInitialState(null, null),
  );

  const pose = snapshotToPose(snapshot);
  const selectedTargetRunway = runwayTargets.find(
    (target) => target.id === targetState.runwayThresholdId,
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

  const clearSnapshotForInitialEdit = useCallback(() => {
    setOptimizedTrajectory(null);
    if (!snapshot && !isEnabled && !isFlying && !isTrajectoryPlaying) return;

    setIsFlying(false);
    setIsTrajectoryPlaying(false);
    setIsEnabled(false);
    setSnapshot(null);
    setTrail([]);
  }, [isEnabled, isFlying, isTrajectoryPlaying, snapshot]);

  const updateInitialPosition = useCallback(
    (position: PilotInitialPlacementPosition) => {
      setInitialState((current) => ({
        ...current,
        lon: clamp(position.lon, -180, 180),
        lat: clamp(position.lat, -90, 90),
      }));
      clearSnapshotForInitialEdit();
    },
    [clearSnapshotForInitialEdit],
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
    previewVisible: (isInitialEditorOpen || isPlacingInitialPosition || isInitialPreviewVisible) &&
      !isEnabled &&
      !snapshot,
    initialState,
    onPositionChange: updateInitialPosition,
    onFinish: finishInitialPlacement,
    onCancel: cancelInitialPlacement,
  });

  usePilotAircraft({
    enabled: isEnabled,
    pose,
    trail,
    follow: isFollowing,
  });

  usePilotTargetGate({
    enabled: activeMode === "trajectory",
    target: targetGateState,
  });

  useEffect(() => {
    placementBackupRef.current = null;
    trajectoryReplayIndexRef.current = 0;
    trajectorySegmentElapsedSRef.current = 0;
    setInitialState(makeDefaultInitialState(airport, aircraftConfigs[0] ?? null));
    setActiveMode("pilot");
    setIsInitialEditorOpen(false);
    setIsTargetEditorOpen(false);
    setIsPlacingInitialPosition(false);
    setIsInitialPreviewVisible(false);
    setIsFlying(false);
    setIsTrajectoryPlaying(false);
    setIsEnabled(false);
    setSnapshot(null);
    setTrail([]);
    setOptimizedTrajectory(null);
  }, [airport, aircraftConfigs]);

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
      setTargetState(makeDefaultTrajectoryTarget(null));
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
          return makeDefaultTrajectoryTarget(selected, current);
        });
      })
      .catch((runwayError: unknown) => {
        if (cancelled) return;
        setRunwayTargets([]);
        setTargetState(makeDefaultTrajectoryTarget(null));
        setError(toErrorMessage(runwayError));
      });

    return () => {
      cancelled = true;
    };
  }, [activeAirportCode]);

  useEffect(() => {
    controlsRef.current = controls;
  }, [controls]);

  useEffect(() => {
    frameDtRef.current = frameDtS;
  }, [frameDtS]);

  useEffect(() => {
    trajectoryPlanRef.current = optimizedTrajectory;
  }, [optimizedTrajectory]);

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

      void stepPilotSimulation(controlsRef.current, frameDtRef.current)
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
    if (!isEnabled || !isTrajectoryPlaying) return;

    let cancelled = false;
    const tick = () => {
      const plan = trajectoryPlanRef.current;
      if (!plan || trajectoryStepInFlightRef.current) return;

      const segmentIndex = trajectoryReplayIndexRef.current;
      const control = plan.controls[segmentIndex];
      if (!control) {
        setIsTrajectoryPlaying(false);
        return;
      }

      const segmentDurationS = plan.finalTimeS / Math.max(1, plan.controls.length);
      const remainingSegmentS = segmentDurationS - trajectorySegmentElapsedSRef.current;
      if (remainingSegmentS <= 1e-9) {
        trajectoryReplayIndexRef.current += 1;
        trajectorySegmentElapsedSRef.current = 0;
        if (trajectoryReplayIndexRef.current >= plan.controls.length) {
          setIsTrajectoryPlaying(false);
        }
        return;
      }

      const replayDtS = Math.min(plan.dtS, remainingSegmentS);
      trajectoryStepInFlightRef.current = true;
      void stepPilotSimulation(control, replayDtS)
        .then((nextSnapshot) => {
          if (cancelled) return;
          setSnapshot(nextSnapshot);
          appendTrailPoint(nextSnapshot, segmentIndex);
          setError(null);
          trajectorySegmentElapsedSRef.current += replayDtS;
          if (trajectorySegmentElapsedSRef.current >= segmentDurationS - 1e-9) {
            trajectoryReplayIndexRef.current += 1;
            trajectorySegmentElapsedSRef.current = 0;
          }
          if (trajectoryReplayIndexRef.current >= plan.controls.length) {
            setIsTrajectoryPlaying(false);
          }
        })
        .catch((stepError: unknown) => {
          if (cancelled) return;
          setIsTrajectoryPlaying(false);
          setError(toErrorMessage(stepError));
        })
        .finally(() => {
          trajectoryStepInFlightRef.current = false;
        });
    };

    tick();
    const interval = window.setInterval(tick, STEP_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [appendTrailPoint, isEnabled, isTrajectoryPlaying]);

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
          nudgeControl("attackDeg", 0.5, -10, 18);
          break;
        case "arrowdown":
        case "s":
          nudgeControl("attackDeg", -0.5, -10, 18);
          break;
        case "q":
          nudgeControl("thrustN", -500, 0, 60000);
          break;
        case "e":
          nudgeControl("thrustN", 500, 0, 60000);
          break;
        case " ":
          setControls((current) => ({ ...current, bankDeg: 0, attackDeg: 0 }));
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
  }, [activeMode, isEnabled]);

  async function startPilot() {
    placementBackupRef.current = null;
    setIsInitialEditorOpen(false);
    setIsPlacingInitialPosition(false);
    setIsTrajectoryPlaying(false);
    setIsBusy(true);
    setError(null);
    try {
      if (!snapshot) {
        const nextSnapshot = await resetPilotSimulation(initialState, controls);
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
      const nextSnapshot = await resetPilotSimulation(initialState, controls);
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

  function updateInitialField(
    key: PilotInitialEditableKey,
    value: number,
    min: number,
    max: number,
  ) {
    if (!Number.isFinite(value) || isFlying || isTrajectoryPlaying) return;

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
    }));
    setIsInitialPreviewVisible(true);
    clearSnapshotForInitialEdit();
  }

  function updateFrameDt(value: number) {
    if (!Number.isFinite(value)) return;
    setFrameDtS(clamp(value, 0.02, 0.5));
  }

  function openTrajectoryMode() {
    if (isPlacingInitialPosition) return;
    setIsFlying(false);
    setIsInitialEditorOpen(false);
    setActiveMode("trajectory");
    setError(null);
  }

  function openPilotMode() {
    if (isPlacingInitialPosition) return;
    setIsTrajectoryPlaying(false);
    setIsTargetEditorOpen(false);
    setActiveMode("pilot");
    setError(null);
  }

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

    setTargetState((current) => makeDefaultTrajectoryTarget(target, current));
    setOptimizedTrajectory(null);
    setIsTrajectoryPlaying(false);
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
      nextValue = clampTargetSpeedMps(nextValue);
    } else if (key === "headingDeg") {
      nextValue = selectedTargetRunway
        ? clampHeadingToRunwayTolerance(value, selectedTargetRunway.psiDeg)
        : runwayAlignedHeadingDeg(nextValue);
    }

    setTargetState((current) => ({ ...current, [key]: nextValue }));
    setOptimizedTrajectory(null);
    setIsTrajectoryPlaying(false);
  }

  function updateNSegments(value: number) {
    if (!Number.isFinite(value)) return;
    setNSegments(Math.round(clamp(value, 1, 80)));
    setOptimizedTrajectory(null);
    setIsTrajectoryPlaying(false);
  }

  function updateArrivalTime(value: number) {
    if (!Number.isFinite(value)) return;
    setArrivalTimeS(clamp(value, 1, 1000));
    setOptimizedTrajectory(null);
    setIsTrajectoryPlaying(false);
  }

  function updateTrajectoryDt(value: number) {
    if (!Number.isFinite(value)) return;
    setTrajectoryDtS(clamp(value, 0.02, 0.5));
    setOptimizedTrajectory(null);
    setIsTrajectoryPlaying(false);
  }

  function updateMaxIterations(value: number) {
    if (!Number.isFinite(value)) return;
    setMaxIterations(Math.round(clamp(value, 1, 10000)));
    setOptimizedTrajectory(null);
    setIsTrajectoryPlaying(false);
  }

  function updateTrajectoryOptimizer(value: TrajectoryOptimizer) {
    setTrajectoryOptimizer(value);
    setOptimizedTrajectory(null);
    setIsTrajectoryPlaying(false);
  }

  async function computeTrajectory() {
    if (!hasAircraftConfigs || runwayTargets.length === 0) return;

    setIsBusy(true);
    setIsFlying(false);
    setIsTrajectoryPlaying(false);
    setError(null);
    try {
      const result = await runTrajectoryOptimization({
        optimizer: trajectoryOptimizer,
        initialState,
        targetState: trajectoryTargetToPilotState(
          targetState,
          initialState.aircraftType,
          initialState.massKg,
        ),
        targetControl: { attackDeg: targetState.attackDeg },
        nSegments,
        arrivalTimeS,
        dtS: trajectoryDtS,
        maxIterations,
      });
      setOptimizedTrajectory(result);
      trajectoryReplayIndexRef.current = 0;
      trajectorySegmentElapsedSRef.current = 0;
    } catch (computeError: unknown) {
      setOptimizedTrajectory(null);
      setError(toErrorMessage(computeError));
    } finally {
      setIsBusy(false);
    }
  }

  async function playOptimizedTrajectory() {
    if (!optimizedTrajectory || optimizedTrajectory.controls.length === 0) return;

    setIsBusy(true);
    setIsFlying(false);
    setIsTrajectoryPlaying(false);
    setError(null);
    try {
      const firstControl = optimizedTrajectory.controls[0] ?? controls;
      const nextSnapshot = await resetPilotSimulation(initialState, firstControl);
      setSnapshot(nextSnapshot);
      const nextPose = snapshotToPose(nextSnapshot);
      setTrail(nextPose ? [{ ...nextPose, segmentIndex: 0 }] : []);
      trajectoryReplayIndexRef.current = 0;
      trajectorySegmentElapsedSRef.current = 0;
      setIsEnabled(true);
      setIsTrajectoryPlaying(true);
    } catch (playError: unknown) {
      setError(toErrorMessage(playError));
    } finally {
      setIsBusy(false);
    }
  }

  function resetTrajectoryReplay() {
    trajectoryReplayIndexRef.current = 0;
    trajectorySegmentElapsedSRef.current = 0;
    setIsTrajectoryPlaying(false);
    setIsFlying(false);
    setIsEnabled(false);
    setSnapshot(null);
    setTrail([]);
    setError(null);
  }

  function nudgeControl(
    key: keyof PilotControls,
    delta: number,
    min: number,
    max: number,
  ) {
    setControls((current) => ({
      ...current,
      [key]: clamp(current[key] + delta, min, max),
    }));
  }

  const statusLabel = isPlacingInitialPosition
    ? "Placing"
    : isBusy && activeMode === "trajectory"
      ? "Computing"
      : isTrajectoryPlaying
        ? "Playing"
        : isFlying
          ? "Flying"
          : optimizedTrajectory && activeMode === "trajectory"
            ? "Ready"
            : snapshot
              ? "Paused"
              : "Standby";
  const hasAircraftConfigs = aircraftConfigs.length > 0;
  const initialControlsDisabled = isFlying || isTrajectoryPlaying || isBusy || !hasAircraftConfigs;
  const targetControlsDisabled = isBusy || isTrajectoryPlaying || runwayTargets.length === 0;
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
          (activeMode === "trajectory" && snapshot !== null)
        }
        showControlReadout={activeMode === "trajectory"}
        targetState={activeMode === "trajectory" ? targetState : null}
      />

      <header className="pilot-panel-header">
        <div className="pilot-panel-header-main">
          <div className="pilot-panel-title-block">
            <h3>{activeMode === "trajectory" ? "Trajectory Play" : "Pilot Mode"}</h3>
            <span className="pilot-panel-server">{AEROVIZ_BACKEND_URL}</span>
          </div>
          <span className={`pilot-status pilot-status-${statusLabel.toLowerCase()}`}>
            {statusLabel}
          </span>
        </div>
        <div className="pilot-panel-mode-row">
          <button
            type="button"
            className="pilot-mode-toggle"
            onClick={activeMode === "trajectory" ? openPilotMode : openTrajectoryMode}
            disabled={isPlacingInitialPosition}
          >
            {activeMode === "trajectory" ? "Pilot" : "Trajectory"}
          </button>
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
        disabled={initialControlsDisabled}
        onClose={closeInitialEditor}
        onPlaceToggle={toggleInitialPlacement}
        onFieldChange={updateInitialField}
        onAircraftTypeChange={updateAircraftType}
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
              <div>
                <dt>Alpha</dt>
                <dd>{formatNumberInputValue(targetState.attackDeg)} deg</dd>
              </div>
            </dl>
          </section>

          <PilotTargetStateOverlay
            open={isTargetEditorOpen}
            state={targetState}
            runwayTargets={runwayTargets}
            disabled={targetControlsDisabled}
            onClose={closeTargetEditor}
            onRunwayChange={updateTargetRunway}
            onFieldChange={updateTargetField}
          />

          <section className="pilot-optimization-row" aria-label="Trajectory optimization settings">
            <label>
              <span>Optimizer</span>
              <select
                className="pilot-select-input"
                value={trajectoryOptimizer}
                disabled={targetControlsDisabled}
                onChange={(event) =>
                  updateTrajectoryOptimizer(event.target.value as TrajectoryOptimizer)
                }
              >
                <option value="transcription">Transcription</option>
                <option value="leastSquaresTranscription">Least squares</option>
                <option value="singleShooting">Single shooting</option>
              </select>
            </label>
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
                max={0.5}
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
                !optimizedTrajectory
              }
            >
              Play
            </button>
            <button
              onClick={() => setIsTrajectoryPlaying(false)}
              disabled={!isTrajectoryPlaying || isBusy || isPlacingInitialPosition}
            >
              Pause
            </button>
            <button
              onClick={resetTrajectoryReplay}
              disabled={
                isBusy ||
                isPlacingInitialPosition ||
                (!snapshot && !optimizedTrajectory)
              }
            >
              Reset
            </button>
          </div>

          <section className="pilot-control-zone" aria-label="Trajectory play controls">
            <div className="pilot-options-row">
              <label>
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

            <div className="pilot-stepper-row">
              <button
                onClick={() => nudgeControl("thrustN", -1000, 0, 60000)}
                title="Reduce thrust"
              >
                -
              </button>
              <label>
                <span>Thrust</span>
                <EnglishNumberInput
                  value={controls.thrustN}
                  min={0}
                  max={60000}
                  step="500"
                  disabled={false}
                  onCommit={(value) => updateControl("thrustN", value, 0, 60000)}
                />
              </label>
              <button
                onClick={() => nudgeControl("thrustN", 1000, 0, 60000)}
                title="Increase thrust"
              >
                +
              </button>
            </div>

            <div className="pilot-options-row">
              <label>
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
                  value={frameDtS}
                  min={0.02}
                  max={0.5}
                  step="0.02"
                  disabled={false}
                  onCommit={updateFrameDt}
                />
              </label>
            </div>
          </section>
        </>
      )}

      {error ? <div className="pilot-error" role="alert">{error}</div> : null}
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
    speedMps: 120,
    headingDeg: 0,
    flightPathDeg: 0,
    massKg: aircraft?.massKg ?? 0,
    aircraftType: aircraft?.code ?? "",
  };
}

function makeDefaultTrajectoryTarget(
  runwayTarget: RunwayThresholdTarget | null,
  fallback?: PilotTargetState,
): PilotTargetState {
  return {
    runwayThresholdId: runwayTarget?.id ?? fallback?.runwayThresholdId ?? "",
    lon: runwayTarget?.lon ?? fallback?.lon ?? 0,
    lat: runwayTarget?.lat ?? fallback?.lat ?? 0,
    altM: runwayTarget
      ? targetAltitudeMForThreshold(runwayTarget.altM)
      : fallback?.altM ?? 0,
    speedMps: clampTargetSpeedMps(fallback?.speedMps ?? DEFAULT_TARGET_SPEED_MPS),
    headingDeg: runwayTarget
      ? runwayAlignedHeadingDeg(runwayTarget.psiDeg)
      : runwayAlignedHeadingDeg(fallback?.headingDeg ?? 0),
    flightPathDeg: fallback?.flightPathDeg ?? DEFAULT_TARGET_GAMMA_DEG,
    attackDeg: fallback?.attackDeg ?? DEFAULT_TARGET_ALPHA_DEG,
  };
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
