import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../context/AppContext";
import { usePilotAircraft, type PilotAircraftPose } from "../hooks/usePilotAircraft";
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

export default function PilotPanel() {
  const { airport } = useApp();
  const [isEnabled, setIsEnabled] = useState(false);
  const [isFlying, setIsFlying] = useState(false);
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

  const initialState = useMemo<PilotResetState>(() => ({
    lon: airport?.lon ?? -78.7873,
    lat: airport?.lat ?? 35.878659,
    altM: 1000,
    speedMps: 120,
    headingDeg: 0,
    flightPathDeg: 0,
    massKg: 10000,
  }), [airport]);

  const pose = snapshotToPose(snapshot);

  usePilotAircraft({
    enabled: isEnabled,
    pose,
    trail,
    follow: isFollowing,
  });

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
          nudgeControl("bankDeg", -3, -45, 45);
          break;
        case "arrowright":
        case "d":
          nudgeControl("bankDeg", 3, -45, 45);
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
    setIsFlying(false);
    setIsEnabled(false);
    setSnapshot(null);
    setTrail([]);
    setError(null);
  }

  function updateControl(key: keyof PilotControls, value: number) {
    setControls((current) => ({ ...current, [key]: value }));
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
    : isFlying
      ? "Flying"
      : snapshot
        ? "Paused"
        : "Standby";

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

      <div className="pilot-actions">
        <button
          className="pilot-primary-button"
          onClick={startPilot}
          disabled={isBusy || isFlying}
        >
          {snapshot ? "Resume" : "Start"}
        </button>
        <button onClick={() => setIsFlying(false)} disabled={!isFlying || isBusy}>
          Pause
        </button>
        <button onClick={resetPilot} disabled={isBusy}>
          Reset
        </button>
        <button onClick={stopPilot} disabled={!isEnabled && !snapshot}>
          End
        </button>
      </div>

      <section className="pilot-control-zone" aria-label="Pilot controls">
        <div className="pilot-stepper-row">
          <button
            onClick={() => nudgeControl("bankDeg", -5, -45, 45)}
            title="Bank left"
          >
            &lt;
          </button>
          <label>
            <span>Bank</span>
            <input
              type="range"
              min="-45"
              max="45"
              step="1"
              value={controls.bankDeg}
              onChange={(event) => updateControl("bankDeg", Number(event.target.value))}
            />
            <output>{controls.bankDeg.toFixed(0)} deg</output>
          </label>
          <button
            onClick={() => nudgeControl("bankDeg", 5, -45, 45)}
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
            <input
              type="range"
              min="0.4"
              max="2.2"
              step="0.01"
              value={controls.loadFactor}
              onChange={(event) => updateControl("loadFactor", Number(event.target.value))}
            />
            <output>{controls.loadFactor.toFixed(2)}</output>
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
            <input
              type="range"
              min="0"
              max="60000"
              step="500"
              value={controls.thrustN}
              onChange={(event) => updateControl("thrustN", Number(event.target.value))}
            />
            <output>{Math.round(controls.thrustN).toLocaleString()} N</output>
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
            Follow
          </label>
          <label>
            <span>dt</span>
            <input
              type="range"
              min="0.02"
              max="0.5"
              step="0.02"
              value={frameDtS}
              onChange={(event) => setFrameDtS(Number(event.target.value))}
            />
            <output>{frameDtS.toFixed(2)} s</output>
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

function formatCoord(deg: number, pos: string, neg: string): string {
  return `${Math.abs(deg).toFixed(5)} ${deg >= 0 ? pos : neg}`;
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
