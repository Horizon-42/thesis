import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { PilotSimulationMode, PilotSnapshot } from "../pilot/pilotClient";
import { formatCoord } from "./PilotInitialStateOverlay";

interface PilotRealtimeStatePanelProps {
  snapshot: PilotSnapshot | null;
  visible: boolean;
  showControlReadout?: boolean;
  simulationMode?: PilotSimulationMode;
  targetState?: Pick<PilotSnapshot["state"], "lat" | "lon" | "altM"> | null;
}

export default function PilotRealtimeStatePanel({
  snapshot,
  visible,
  showControlReadout = false,
  simulationMode,
  targetState = null,
}: PilotRealtimeStatePanelProps) {
  const [portalTarget, setPortalTarget] = useState<HTMLElement | null>(null);

  useEffect(() => {
    setPortalTarget(
      document.querySelector<HTMLElement>(".cesium-overlay-container"),
    );
  }, []);

  if (!visible || !snapshot || !portalTarget) return null;
  const effectiveSimulationMode = simulationMode ?? snapshot.simulationMode ?? "alpha";
  const showLoadFactor = effectiveSimulationMode === "loadFactor" &&
    snapshot.control.loadFactor !== undefined;
  const loadFactor = snapshot.control.loadFactor ?? 0;

  return createPortal(
    <aside className="pilot-realtime-panel" aria-label="Realtime aircraft state">
      <span className="pilot-realtime-title">Live State</span>
      <dl className="pilot-realtime-readouts" aria-live="polite">
        <RealtimeReadout label="Time" value={`${snapshot.elapsedS.toFixed(1)} s`} />
        <RealtimeReadout label="Latitude" value={formatCoord(snapshot.state.lat, "N", "S")} />
        <RealtimeReadout label="Longitude" value={formatCoord(snapshot.state.lon, "E", "W")} />
        <RealtimeReadout label="Altitude" value={`${snapshot.state.altM.toFixed(0)} m`} />
        {targetState ? (
          <>
            <RealtimeReadout
              label="Lat Error"
              value={formatSignedDelta(snapshot.state.lat - targetState.lat, 6, "deg")}
            />
            <RealtimeReadout
              label="Lon Error"
              value={formatSignedDelta(snapshot.state.lon - targetState.lon, 6, "deg")}
            />
            <RealtimeReadout
              label="Alt Error"
              value={formatSignedDelta(snapshot.state.altM - targetState.altM, 1, "m")}
            />
          </>
        ) : null}
        <RealtimeReadout label="Speed" value={`${snapshot.state.speedMps.toFixed(1)} m/s`} />
        <RealtimeReadout
          label="Heading Angle (psi)"
          value={`${snapshot.state.headingDeg.toFixed(1)} deg`}
        />
        <RealtimeReadout
          label="Flight Path Angle (gamma)"
          value={`${snapshot.state.flightPathDeg.toFixed(2)} deg`}
        />
        {showControlReadout ? (
          <RealtimeReadout
            label="Control"
            value={showLoadFactor
              ? `bank ${snapshot.control.bankDeg.toFixed(1)} deg | n ${loadFactor.toFixed(2)} | thrust ${snapshot.control.thrustN.toFixed(0)} N`
              : `bank ${snapshot.control.bankDeg.toFixed(1)} deg | alpha ${snapshot.control.attackDeg.toFixed(2)} deg | thrust ${snapshot.control.thrustN.toFixed(0)} N`}
            wide
          />
        ) : null}
        {showLoadFactor ? (
          <RealtimeReadout
            label="Load Factor"
            value={`${loadFactor.toFixed(2)} g`}
          />
        ) : (
          <RealtimeReadout
            label="Attack Angle (alpha)"
            value={`${snapshot.control.attackDeg.toFixed(2)} deg`}
          />
        )}
        <RealtimeReadout
          label="Lift Coefficient"
          value={snapshot.aero.liftCoefficient.toFixed(3)}
        />
        <RealtimeReadout
          label="Drag Coefficient"
          value={snapshot.aero.dragCoefficient.toFixed(3)}
        />
      </dl>
    </aside>,
    portalTarget,
  );
}

function formatSignedDelta(value: number, fractionDigits: number, unit: string): string {
  const zeroThreshold = 0.5 * 10 ** -fractionDigits;
  const normalizedValue = Math.abs(value) < zeroThreshold ? 0 : value;
  const prefix = normalizedValue > 0 ? "+" : "";
  return `${prefix}${normalizedValue.toFixed(fractionDigits)} ${unit}`;
}

function RealtimeReadout({
  label,
  value,
  wide = false,
}: {
  label: string;
  value: string;
  wide?: boolean;
}) {
  return (
    <div className={wide ? "pilot-realtime-readout pilot-realtime-readout-wide" : "pilot-realtime-readout"}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
