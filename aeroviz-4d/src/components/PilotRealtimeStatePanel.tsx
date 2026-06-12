import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { PilotSnapshot } from "../pilot/pilotClient";
import { formatCoord } from "./PilotInitialStateOverlay";

interface PilotRealtimeStatePanelProps {
  snapshot: PilotSnapshot | null;
  visible: boolean;
}

export default function PilotRealtimeStatePanel({
  snapshot,
  visible,
}: PilotRealtimeStatePanelProps) {
  const [portalTarget, setPortalTarget] = useState<HTMLElement | null>(null);

  useEffect(() => {
    setPortalTarget(
      document.querySelector<HTMLElement>(".cesium-overlay-container"),
    );
  }, []);

  if (!visible || !snapshot || !portalTarget) return null;

  return createPortal(
    <aside className="pilot-realtime-panel" aria-label="Realtime aircraft state">
      <span className="pilot-realtime-title">Live State</span>
      <dl className="pilot-realtime-readouts" aria-live="polite">
        <RealtimeReadout label="Time" value={`${snapshot.elapsedS.toFixed(1)} s`} />
        <RealtimeReadout label="Latitude" value={formatCoord(snapshot.state.lat, "N", "S")} />
        <RealtimeReadout label="Longitude" value={formatCoord(snapshot.state.lon, "E", "W")} />
        <RealtimeReadout label="Altitude" value={`${snapshot.state.altM.toFixed(0)} m`} />
        <RealtimeReadout label="Speed" value={`${snapshot.state.speedMps.toFixed(1)} m/s`} />
        <RealtimeReadout
          label="Heading Angle (psi)"
          value={`${snapshot.state.headingDeg.toFixed(1)} deg`}
        />
        <RealtimeReadout
          label="Flight Path Angle (gamma)"
          value={`${snapshot.state.flightPathDeg.toFixed(2)} deg`}
        />
        <RealtimeReadout
          label="Attack Angle (alpha)"
          value={`${snapshot.control.attackDeg.toFixed(2)} deg`}
        />
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

function RealtimeReadout({ label, value }: { label: string; value: string }) {
  return (
    <div className="pilot-realtime-readout">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
