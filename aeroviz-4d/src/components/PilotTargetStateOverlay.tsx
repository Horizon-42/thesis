import { createPortal } from "react-dom";
import type { RunwayThresholdTarget } from "../data/runwayThresholdTargets";
import {
  EnglishNumberInput,
  formatCoord,
  formatNumberInputValue,
} from "./PilotInitialStateOverlay";

const UNBOUNDED_MIN = Number.NEGATIVE_INFINITY;
const UNBOUNDED_MAX = Number.POSITIVE_INFINITY;

export type PilotTargetEditableKey =
  | "speedMps"
  | "headingDeg"
  | "flightPathDeg";

export interface PilotTargetState {
  runwayThresholdId: string;
  lon: number;
  lat: number;
  altM: number;
  speedMps: number;
  headingDeg: number;
  flightPathDeg: number;
}

interface PilotTargetStateOverlayProps {
  open: boolean;
  state: PilotTargetState;
  runwayTargets: RunwayThresholdTarget[];
  speedMinMps: number;
  speedMaxMps: number;
  disabled: boolean;
  onClose: () => void;
  onRunwayChange: (runwayThresholdId: string) => void;
  onFieldChange: (
    key: PilotTargetEditableKey,
    value: number,
    min: number,
    max: number,
  ) => void;
}

export default function PilotTargetStateOverlay({
  open,
  state,
  runwayTargets,
  speedMinMps,
  speedMaxMps,
  disabled,
  onClose,
  onRunwayChange,
  onFieldChange,
}: PilotTargetStateOverlayProps) {
  if (!open) return null;

  const selectedRunway = runwayTargets.find(
    (target) => target.id === state.runwayThresholdId,
  );

  return createPortal(
    <aside className="pilot-initial-overlay" aria-label="Target state setup">
      <header className="pilot-initial-overlay-header">
        <div>
          <h3>Target State</h3>
          <span>Runway threshold gate and terminal constraints</span>
        </div>
        <button type="button" onClick={onClose} title="Close target state setup">
          Close
        </button>
      </header>

      <div className="pilot-initial-overlay-body">
        <section className="pilot-initial-place-card pilot-target-gate-card">
          <label>
            <span>Threshold Gate</span>
            <select
              className="pilot-select-input"
              value={state.runwayThresholdId}
              disabled={disabled || runwayTargets.length === 0}
              onChange={(event) => onRunwayChange(event.target.value)}
            >
              {runwayTargets.length === 0 ? (
                <option value="">Unavailable</option>
              ) : null}
              {runwayTargets.map((target) => (
                <option key={target.id} value={target.id}>
                  {target.runwayIdent}
                </option>
              ))}
            </select>
          </label>
          <div className="pilot-initial-coordinates">
            <output>{formatCoord(state.lat, "N", "S")}</output>
            <output>{formatCoord(state.lon, "E", "W")}</output>
          </div>
          <div className="pilot-target-gate-readouts">
            <output>{formatNumberInputValue(state.altM)} m</output>
            <output>{selectedRunway?.runwayPairIdent ?? "-"}</output>
          </div>
        </section>

        <section className="pilot-initial-fields" aria-label="Target state values">
          <label>
            <span>Vt m/s</span>
            <EnglishNumberInput
              value={state.speedMps}
              min={speedMinMps}
              max={speedMaxMps}
              step="0.1"
              disabled={disabled}
              onCommit={(value) =>
                onFieldChange(
                  "speedMps",
                  value,
                  speedMinMps,
                  speedMaxMps,
                )
              }
            />
          </label>
          <label>
            <span>Psi deg</span>
            <EnglishNumberInput
              value={state.headingDeg}
              min={UNBOUNDED_MIN}
              max={UNBOUNDED_MAX}
              step="1"
              disabled={disabled}
              onCommit={(value) =>
                onFieldChange("headingDeg", value, UNBOUNDED_MIN, UNBOUNDED_MAX)
              }
            />
          </label>
          <label>
            <span>Gamma deg</span>
            <EnglishNumberInput
              value={state.flightPathDeg}
              min={-15}
              max={10}
              step="0.1"
              disabled={disabled}
              onCommit={(value) => onFieldChange("flightPathDeg", value, -15, 10)}
            />
          </label>
        </section>
      </div>
    </aside>,
    document.body,
  );
}
