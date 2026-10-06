/**
 * OptimizeKindSelect.tsx
 * ----------------------
 * The Optimize task's mode (design §10.1): one aircraft optimized alone (the present mode, `PilotPanel`), or the
 * multi-aircraft modes — one aircraft controlled in its recorded traffic (M1), every aircraft of a block controlled (M2).
 */

export type OptimizeKind = "single" | "m1" | "m2";

export const OPTIMIZE_KIND_LABELS: Record<OptimizeKind, string> = {
  single: "Single aircraft",
  m1: "Multi-aircraft: one controlled",
  m2: "Multi-aircraft: all controlled",
};

export default function OptimizeKindSelect({
  value,
  onChange,
}: {
  value: OptimizeKind;
  onChange: (kind: OptimizeKind) => void;
}) {
  return (
    <label className="optimize-kind-select">
      <span>Mode</span>
      <select
        className="pilot-select-input"
        value={value}
        onChange={(event) => onChange(event.target.value as OptimizeKind)}
      >
        {(Object.keys(OPTIMIZE_KIND_LABELS) as OptimizeKind[]).map((kind) => (
          <option key={kind} value={kind}>{OPTIMIZE_KIND_LABELS[kind]}</option>
        ))}
      </select>
    </label>
  );
}
