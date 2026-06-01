import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";
import type { PilotResetState } from "../pilot/pilotClient";

const UNBOUNDED_MIN = Number.NEGATIVE_INFINITY;
const UNBOUNDED_MAX = Number.POSITIVE_INFINITY;

export type PilotInitialEditableKey =
  | "altM"
  | "headingDeg"
  | "flightPathDeg"
  | "speedMps"
  | "massKg";

interface PilotInitialStateOverlayProps {
  open: boolean;
  isPlacing: boolean;
  state: PilotResetState;
  disabled: boolean;
  onClose: () => void;
  onPlaceToggle: () => void;
  onFieldChange: (
    key: PilotInitialEditableKey,
    value: number,
    min: number,
    max: number,
  ) => void;
}

interface EnglishNumberInputProps {
  value: number;
  min: number;
  max: number;
  step: string;
  disabled: boolean;
  onCommit: (value: number) => void;
}

export default function PilotInitialStateOverlay({
  open,
  isPlacing,
  state,
  disabled,
  onClose,
  onPlaceToggle,
  onFieldChange,
}: PilotInitialStateOverlayProps) {
  if (!open) return null;

  return createPortal(
    <aside className="pilot-initial-overlay" aria-label="Initial aircraft setup">
      <header className="pilot-initial-overlay-header">
        <div>
          <h3>Initial Aircraft</h3>
          <span>{isPlacing ? "Drag on the globe, release to set" : "Position and entry state"}</span>
        </div>
        <button type="button" onClick={onClose} title="Close initial aircraft setup">
          Close
        </button>
      </header>

      <div className="pilot-initial-overlay-body">
        <section className="pilot-initial-place-card">
          <button
            type="button"
            className={isPlacing ? "pilot-placement-button active" : "pilot-placement-button"}
            onClick={onPlaceToggle}
            disabled={!isPlacing && disabled}
          >
            {isPlacing ? "Cancel Place" : "Place Aircraft"}
          </button>
          <div className="pilot-initial-coordinates">
            <output>{formatCoord(state.lat, "N", "S")}</output>
            <output>{formatCoord(state.lon, "E", "W")}</output>
          </div>
        </section>

        <section className="pilot-initial-fields" aria-label="Initial state values">
          <label>
            <span>Alt m</span>
            <EnglishNumberInput
              value={state.altM}
              min={-500}
              max={14000}
              step="1"
              disabled={disabled}
              onCommit={(value) => onFieldChange("altM", value, -500, 14000)}
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
              min={UNBOUNDED_MIN}
              max={UNBOUNDED_MAX}
              step="1"
              disabled={disabled}
              onCommit={(value) =>
                onFieldChange("flightPathDeg", value, UNBOUNDED_MIN, UNBOUNDED_MAX)
              }
            />
          </label>
          <label>
            <span>V0 m/s</span>
            <EnglishNumberInput
              value={state.speedMps}
              min={1}
              max={350}
              step="1"
              disabled={disabled}
              onCommit={(value) => onFieldChange("speedMps", value, 1, 350)}
            />
          </label>
          <label>
            <span>Mass kg</span>
            <EnglishNumberInput
              value={state.massKg}
              min={1}
              max={UNBOUNDED_MAX}
              step="1"
              disabled={disabled}
              onCommit={(value) => onFieldChange("massKg", value, 1, UNBOUNDED_MAX)}
            />
          </label>
        </section>
      </div>
    </aside>,
    document.body,
  );
}

export function EnglishNumberInput({
  value,
  min,
  max,
  step,
  disabled,
  onCommit,
}: EnglishNumberInputProps) {
  const [text, setText] = useState(formatNumberInputValue(value));
  const isFocusedRef = useRef(false);

  useEffect(() => {
    if (!isFocusedRef.current) {
      setText(formatNumberInputValue(value));
    }
  }, [value]);

  const commit = () => {
    const parsed = parseEnglishNumber(text);
    if (parsed === null) {
      setText(formatNumberInputValue(value));
      return;
    }

    const nextValue = clamp(parsed, min, max);
    onCommit(nextValue);
    setText(formatNumberInputValue(nextValue));
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.currentTarget.blur();
    } else if (event.key === "Escape") {
      setText(formatNumberInputValue(value));
      event.currentTarget.blur();
    } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      const parsed = parseEnglishNumber(text);
      const baseValue = parsed ?? value;
      const stepValue = parseEnglishNumber(step) ?? 1;
      const direction = event.key === "ArrowUp" ? 1 : -1;
      const nextValue = clamp(baseValue + direction * stepValue, min, max);

      event.preventDefault();
      onCommit(nextValue);
      setText(formatNumberInputValue(nextValue));
    }
  };

  return (
    <input
      className="pilot-number-input"
      type="text"
      inputMode="decimal"
      pattern="-?[0-9]*(\\.[0-9]+)?"
      step={step}
      value={text}
      disabled={disabled}
      onFocus={() => {
        isFocusedRef.current = true;
      }}
      onChange={(event) => setText(event.target.value.replace(",", "."))}
      onBlur={() => {
        isFocusedRef.current = false;
        commit();
      }}
      onKeyDown={handleKeyDown}
    />
  );
}

export function formatNumberInputValue(value: number): string {
  return Number(value.toFixed(6)).toString();
}

function parseEnglishNumber(value: string): number | null {
  const trimmed = value.trim();
  if (!/^-?(?:\d+\.?\d*|\.\d+)$/.test(trimmed)) return null;

  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatCoord(deg: number, pos: string, neg: string): string {
  return `${Math.abs(deg).toFixed(5)} ${deg >= 0 ? pos : neg}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
