import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { PilotSimulationMode, PilotSnapshot } from "../pilot/pilotClient";
import type {
  DynamicsComparisonDelta,
  DynamicsComparisonDeltas,
  DynamicsComparisonSystem,
} from "../pilot/dynamicsComparisonClient";
import { formatCoord } from "./PilotInitialStateOverlay";

interface PilotRealtimeStatePanelProps {
  snapshot: PilotSnapshot | null;
  visible: boolean;
  showControlReadout?: boolean;
  simulationMode?: PilotSimulationMode;
  /**
   * Per-row deviations appended as coloured chips. Compare mode passes the A/C/D
   * errors vs the reference B; Trajectory Play passes the single live error vs the
   * target. The main rows keep showing the subject's state and each chip is its
   * keyed deviation.
   */
  comparisonDeltas?: DynamicsComparisonDeltas | null;
  /** The systems backing ``comparisonDeltas`` (for the per-chip colours/labels). */
  comparisonSystems?: DynamicsComparisonSystem[] | null;
  /** What the deltas are measured against — used in the Horiz Err row label. */
  deltaReferenceLabel?: string;
}

export default function PilotRealtimeStatePanel({
  snapshot,
  visible,
  showControlReadout = false,
  simulationMode,
  comparisonDeltas = null,
  comparisonSystems = null,
  deltaReferenceLabel = "B",
}: PilotRealtimeStatePanelProps) {
  const [portalTarget, setPortalTarget] = useState<HTMLElement | null>(null);

  useEffect(() => {
    setPortalTarget(
      document.querySelector<HTMLElement>(".cesium-overlay-container"),
    );
  }, []);

  if (!visible || !snapshot || !portalTarget) return null;
  const effectiveSimulationMode = simulationMode ?? snapshot.simulationMode ?? "alpha";
  const showLoadFactor =
    (effectiveSimulationMode === "loadFactor" || effectiveSimulationMode === "casadi") &&
    snapshot.control.loadFactor !== undefined;
  const loadFactor = snapshot.control.loadFactor ?? 0;

  // In Compare mode, build the colored A/C/D deviation strip for a given error
  // field; returns null outside Compare so the rows render exactly as before.
  const showDeltas = comparisonDeltas !== null && comparisonSystems !== null;
  const compareExtra = (
    field: keyof DynamicsComparisonDelta,
    fractionDigits: number,
    signed: boolean,
  ): ReactNode =>
    comparisonDeltas !== null && comparisonSystems !== null ? (
      <ComparisonDeltaStrip
        systems={comparisonSystems}
        deltas={comparisonDeltas}
        field={field}
        fractionDigits={fractionDigits}
        signed={signed}
      />
    ) : null;

  return createPortal(
    <aside className="pilot-realtime-panel" aria-label="Realtime aircraft state">
      <span className="pilot-realtime-title">Live State</span>
      <dl className="pilot-realtime-readouts" aria-live="polite">
        <RealtimeReadout label="Time" value={`${snapshot.elapsedS.toFixed(1)} s`} />
        <RealtimeReadout label="Latitude" value={formatCoord(snapshot.state.lat, "N", "S")} />
        <RealtimeReadout label="Longitude" value={formatCoord(snapshot.state.lon, "E", "W")} />
        <RealtimeReadout
          label="Altitude"
          value={`${snapshot.state.altM.toFixed(0)} m`}
          extra={compareExtra("alt", 1, true)}
        />
        {showDeltas ? (
          <RealtimeReadout
            label={`Horiz Err (vs ${deltaReferenceLabel})`}
            value="0 m"
            extra={compareExtra("horiz", 1, false)}
          />
        ) : null}
        <RealtimeReadout
          label="Speed"
          value={`${snapshot.state.speedMps.toFixed(1)} m/s`}
          extra={compareExtra("speed", 1, true)}
        />
        <RealtimeReadout
          label="Heading Angle (psi)"
          value={`${snapshot.state.headingDeg.toFixed(1)} deg`}
          extra={compareExtra("head", 2, false)}
        />
        <RealtimeReadout
          label="Flight Path Angle (gamma)"
          value={`${snapshot.state.flightPathDeg.toFixed(2)} deg`}
          extra={compareExtra("fpa", 2, true)}
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
          label="Actual n"
          value={`${snapshot.aero.actualLoadFactor.toFixed(3)} g`}
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

/** Signed value with a "+" on positives and small magnitudes collapsed to 0
 * (no trailing unit), e.g. "+5.2", "-2.1", "0.0". */
function formatSignedCompact(value: number, fractionDigits: number): string {
  const zeroThreshold = 0.5 * 10 ** -fractionDigits;
  const normalizedValue = Math.abs(value) < zeroThreshold ? 0 : value;
  const prefix = normalizedValue > 0 ? "+" : "";
  return `${prefix}${normalizedValue.toFixed(fractionDigits)}`;
}

function RealtimeReadout({
  label,
  value,
  wide = false,
  extra = null,
}: {
  label: string;
  value: string;
  wide?: boolean;
  extra?: ReactNode;
}) {
  return (
    <div className={wide ? "pilot-realtime-readout pilot-realtime-readout-wide" : "pilot-realtime-readout"}>
      <dt>{label}</dt>
      <dd>{value}</dd>
      {extra}
    </div>
  );
}

/** Compare mode: one colored chip per compared system (A/C/D) showing its error
 * vs the reference B for a single field, appended under the B value. */
function ComparisonDeltaStrip({
  systems,
  deltas,
  field,
  fractionDigits,
  signed,
}: {
  systems: DynamicsComparisonSystem[];
  deltas: DynamicsComparisonDeltas;
  field: keyof DynamicsComparisonDelta;
  fractionDigits: number;
  signed: boolean;
}) {
  return (
    <div className="pilot-realtime-deltas">
      {systems.map((system) => {
        const delta = deltas[system.key];
        if (delta === undefined) return null; // B (the reference) has no delta
        const [r, g, b] = system.colorRgba;
        const value = delta[field];
        // Decimals only help small deviations; for large ones (e.g. a km-scale
        // horizontal error early in a trajectory) they are noise — drop them.
        const digits = Math.abs(value) >= 100 ? 0 : fractionDigits;
        return (
          <span
            key={system.key}
            className="pilot-realtime-delta"
            style={{ color: `rgb(${r}, ${g}, ${b})` }}
            title={system.label}
          >
            {system.key}{" "}
            {signed ? formatSignedCompact(value, digits) : Math.abs(value).toFixed(digits)}
          </span>
        );
      })}
    </div>
  );
}
