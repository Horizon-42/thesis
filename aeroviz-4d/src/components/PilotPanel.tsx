import { useCallback, useEffect, useRef, useState } from "react";
import { useApp } from "../context/AppContext";
import { usePilotAircraft, type PilotAircraftPose } from "../hooks/usePilotAircraft";
import PilotInitialStateOverlay, {
  EnglishNumberInput,
  formatCoord,
  formatNumberInputValue,
  type PilotInitialEditableKey,
} from "./PilotInitialStateOverlay";
import {
  usePilotInitialPlacement,
  type PilotInitialPlacementPosition,
} from "../hooks/usePilotInitialPlacement";
import {
  PILOT_SERVER_URL,
  resetPilotSimulation,
  stepPilotSimulation,
  type PilotControls,
  type PilotResetState,
  type PilotSnapshot,
} from "../pilot/pilotClient";

const DEFAULT_CONTROLS: PilotControls = {
  thrustN: 12000,
  bankDeg: 0,
  loadFactor: 1,
};
const DEFAULT_FRAME_DT_S = 0.2;
const STEP_INTERVAL_MS = 120;
const MAX_TRAIL_POINTS = 360;

interface PlacementBackup {
  initialState: PilotResetState;
  isInitialPreviewVisible: boolean;
  isEnabled: boolean;
  isFlying: boolean;
  snapshot: PilotSnapshot | null;
  trail: PilotAircraftPose[];
}

export default function PilotPanel() {
  const { airport } = useApp();
  const [isEnabled, setIsEnabled] = useState(false);
  const [isFlying, setIsFlying] = useState(false);
  const [isInitialEditorOpen, setIsInitialEditorOpen] = useState(false);
  const [isPlacingInitialPosition, setIsPlacingInitialPosition] = useState(false);
  const [isInitialPreviewVisible, setIsInitialPreviewVisible] = useState(false);
  const [isFollowing, setIsFollowing] = useState(true);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [controls, setControls] = useState<PilotControls>(DEFAULT_CONTROLS);
  const [frameDtS, setFrameDtS] = useState(DEFAULT_FRAME_DT_S);
  const [snapshot, setSnapshot] = useState<PilotSnapshot | null>(null);
  const [trail, setTrail] = useState<PilotAircraftPose[]>([]);

  const controlsRef = useRef(controls);
  const frameDtRef = useRef(frameDtS);
  const stepInFlightRef = useRef(false);
  const placementBackupRef = useRef<PlacementBackup | null>(null);

  const [initialState, setInitialState] = useState<PilotResetState>(() =>
    makeDefaultInitialState(null),
  );

  const pose = snapshotToPose(snapshot);

  const clearSnapshotForInitialEdit = useCallback(() => {
    if (!snapshot && !isEnabled && !isFlying) return;

    setIsFlying(false);
    setIsEnabled(false);
    setSnapshot(null);
    setTrail([]);
  }, [isEnabled, isFlying, snapshot]);

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
    setIsInitialEditorOpen(false);
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
    if (isFlying || isBusy) return;

    setError(null);
    setIsInitialEditorOpen(true);
    setIsInitialPreviewVisible(true);
    clearSnapshotForInitialEdit();
  }, [clearSnapshotForInitialEdit, isBusy, isFlying]);

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

    if (isFlying || isBusy) return;

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
    isPlacingInitialPosition,
    snapshot,
    trail,
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

  useEffect(() => {
    placementBackupRef.current = null;
    setInitialState(makeDefaultInitialState(airport));
    setIsInitialEditorOpen(false);
    setIsPlacingInitialPosition(false);
    setIsInitialPreviewVisible(false);
    setIsFlying(false);
    setIsEnabled(false);
    setSnapshot(null);
    setTrail([]);
  }, [airport]);

  useEffect(() => {
    controlsRef.current = controls;
  }, [controls]);

  useEffect(() => {
    frameDtRef.current = frameDtS;
  }, [frameDtS]);

  const appendTrailPoint = useCallback((nextSnapshot: PilotSnapshot) => {
    const nextPose = snapshotToPose(nextSnapshot);
    if (!nextPose) return;
    setTrail((current) => [...current.slice(-(MAX_TRAIL_POINTS - 1)), nextPose]);
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
    if (!isEnabled) return;

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
          nudgeControl("loadFactor", 0.04, 0.4, 2.2);
          break;
        case "arrowdown":
        case "s":
          nudgeControl("loadFactor", -0.04, 0.4, 2.2);
          break;
        case "q":
          nudgeControl("thrustN", -500, 0, 60000);
          break;
        case "e":
          nudgeControl("thrustN", 500, 0, 60000);
          break;
        case " ":
          setControls((current) => ({ ...current, bankDeg: 0, loadFactor: 1 }));
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
  }, [isEnabled]);

  async function startPilot() {
    placementBackupRef.current = null;
    setIsInitialEditorOpen(false);
    setIsPlacingInitialPosition(false);
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
    if (!Number.isFinite(value) || isFlying) return;

    setInitialState((current) => ({ ...current, [key]: clamp(value, min, max) }));
    setIsInitialPreviewVisible(true);
    clearSnapshotForInitialEdit();
  }

  function updateFrameDt(value: number) {
    if (!Number.isFinite(value)) return;
    setFrameDtS(clamp(value, 0.02, 0.5));
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

  const statusLabel = error
    ? "Error"
    : isPlacingInitialPosition
      ? "Placing"
      : isFlying
        ? "Flying"
        : snapshot
          ? "Paused"
          : "Standby";
  const initialControlsDisabled = isFlying || isBusy;

  return (
    <div className="pilot-panel">
      <header className="pilot-panel-header">
        <div>
          <h3>Pilot Mode</h3>
          <span className="pilot-panel-server">{PILOT_SERVER_URL}</span>
        </div>
        <span className={`pilot-status pilot-status-${statusLabel.toLowerCase()}`}>
          {statusLabel}
        </span>
      </header>

      <section className="pilot-initial-summary" aria-label="Initial aircraft state summary">
        <div>
          <h4>Initial Aircraft</h4>
          <span>{formatCoord(initialState.lat, "N", "S")}</span>
          <span>{formatCoord(initialState.lon, "E", "W")}</span>
        </div>
        <button
          type="button"
          onClick={openInitialEditor}
          disabled={initialControlsDisabled}
        >
          Initial State
        </button>
        <dl>
          <div>
            <dt>Psi</dt>
            <dd>{formatNumberInputValue(initialState.headingDeg)}</dd>
          </div>
          <div>
            <dt>Gamma</dt>
            <dd>{formatNumberInputValue(initialState.flightPathDeg)}</dd>
          </div>
          <div>
            <dt>V0</dt>
            <dd>{formatNumberInputValue(initialState.speedMps)}</dd>
          </div>
        </dl>
      </section>

      <PilotInitialStateOverlay
        open={isInitialEditorOpen}
        isPlacing={isPlacingInitialPosition}
        state={initialState}
        disabled={initialControlsDisabled}
        onClose={closeInitialEditor}
        onPlaceToggle={toggleInitialPlacement}
        onFieldChange={updateInitialField}
      />

      <div className="pilot-actions">
        <button
          className="pilot-primary-button"
          onClick={startPilot}
          disabled={isBusy || isFlying || isPlacingInitialPosition}
        >
          {snapshot ? "Resume" : "Start"}
        </button>
        <button
          onClick={() => setIsFlying(false)}
          disabled={!isFlying || isBusy || isPlacingInitialPosition}
        >
          Pause
        </button>
        <button onClick={resetPilot} disabled={isBusy || isPlacingInitialPosition}>
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
            onClick={() => nudgeControl("loadFactor", -0.05, 0.4, 2.2)}
            title="Reduce load factor"
          >
            -
          </button>
          <label>
            <span>Load</span>
            <EnglishNumberInput
              value={controls.loadFactor}
              min={0.4}
              max={2.2}
              step="0.01"
              disabled={false}
              onCommit={(value) => updateControl("loadFactor", value, 0.4, 2.2)}
            />
          </label>
          <button
            onClick={() => nudgeControl("loadFactor", 0.05, 0.4, 2.2)}
            title="Increase load factor"
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

      <section className="pilot-state-zone" aria-live="polite">
        <Readout label="LAT" value={snapshot ? formatCoord(snapshot.state.lat, "N", "S") : "--"} />
        <Readout label="LON" value={snapshot ? formatCoord(snapshot.state.lon, "E", "W") : "--"} />
        <Readout label="ALT" value={snapshot ? `${snapshot.state.altM.toFixed(0)} m` : "--"} />
        <Readout label="SPD" value={snapshot ? `${snapshot.state.speedMps.toFixed(1)} m/s` : "--"} />
        <Readout label="HDG" value={snapshot ? `${snapshot.state.headingDeg.toFixed(1)} deg` : "--"} />
        <Readout label="GAM" value={snapshot ? `${snapshot.state.flightPathDeg.toFixed(2)} deg` : "--"} />
        <Readout label="CL" value={snapshot ? snapshot.aero.liftCoefficient.toFixed(3) : "--"} />
        <Readout label="CD" value={snapshot ? snapshot.aero.dragCoefficient.toFixed(3) : "--"} />
      </section>

      {error ? <div className="pilot-error" role="alert">{error}</div> : null}
    </div>
  );
}

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div className="pilot-readout">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function makeDefaultInitialState(
  airport: { lon: number; lat: number } | null,
): PilotResetState {
  return {
    lon: airport?.lon ?? -78.7873,
    lat: airport?.lat ?? 35.878659,
    altM: 1000,
    speedMps: 120,
    headingDeg: 0,
    flightPathDeg: 0,
    massKg: 10000,
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
