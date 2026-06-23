import { memo, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { createPortal } from "react-dom";
import type {
  DynamicsComparisonChart,
  DynamicsComparisonSystem,
} from "../pilot/dynamicsComparisonClient";

/**
 * DynamicsComparisonCharts.tsx
 * ----------------------------
 * Pop-up stats overlay for the Dynamics Comparison mode. Plots each compared
 * system's deviation from the reference B over distance flown — horizontal,
 * altitude, heading and speed error — plus a final-value summary (incl. the
 * final speed deviation). Lines are colored to match the 3D paths and respect
 * the per-system hide toggles.
 */

interface Props {
  chart: DynamicsComparisonChart;
  systems: DynamicsComparisonSystem[];
  hiddenKeys: readonly string[];
  onToggleSystem: (key: string) => void;
  onClose: () => void;
  /** Header subtitle override (e.g. "Averaged over N runs"). */
  subtitle?: string;
}

type MetricField = "horiz" | "alt" | "head" | "speed";

const METRICS: { field: MetricField; title: string; unit: string }[] = [
  { field: "horiz", title: "Horizontal error", unit: "m" },
  { field: "alt", title: "Altitude error", unit: "m" },
  { field: "head", title: "Heading error", unit: "°" },
  { field: "speed", title: "Speed error", unit: "m/s" },
];

function rgba([r, g, b, a]: [number, number, number, number]): string {
  return `rgba(${r}, ${g}, ${b}, ${(a / 255).toFixed(3)})`;
}

/** Signed deviation (alt, speed): "+x"/"-x", but never a misleading "-0.0". */
function formatSigned(value: number, digits: number): string {
  const rounded = Number(value.toFixed(digits));
  if (rounded === 0) return (0).toFixed(digits);
  return rounded > 0 ? `+${rounded.toFixed(digits)}` : rounded.toFixed(digits);
}

/** Non-negative magnitude (horizontal distance, |heading|): no sign, no "-0.0". */
function formatMagnitude(value: number, digits: number): string {
  return Math.abs(value).toFixed(digits);
}

function niceTickStep(span: number, targetTicks: number): number {
  const rough = Math.max(1e-9, span / Math.max(1, targetTicks));
  const power = 10 ** Math.floor(Math.log10(rough));
  const scaled = rough / power;
  const nice = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return nice * power;
}

function buildTicks(min: number, max: number, targetTicks: number): number[] {
  const step = niceTickStep(Math.max(1e-9, max - min), targetTicks);
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.25; v += step) {
    // round away tiny FP dust so labels read cleanly
    ticks.push(Number(v.toPrecision(12)));
  }
  return ticks;
}

interface ChartLine {
  key: string;
  color: string;
  values: number[];
}

const MetricChart = memo(function MetricChart({
  title,
  unit,
  xValues,
  lines,
}: {
  title: string;
  unit: string;
  xValues: number[];
  lines: ChartLine[];
}) {
  const width = 320;
  const height = 184;
  const marginLeft = 46;
  const marginRight = 12;
  const marginTop = 24;
  const marginBottom = 34;
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;

  const { minY, maxY } = useMemo(() => {
    let lo = 0;
    let hi = 0;
    lines.forEach((line) => {
      line.values.forEach((v) => {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      });
    });
    if (lo === hi) {
      hi = lo + 1;
    }
    const pad = (hi - lo) * 0.08;
    return { minY: lo - pad, maxY: hi + pad };
  }, [lines]);

  const maxX = xValues.length > 0 ? xValues[xValues.length - 1] : 1;
  const minX = xValues.length > 0 ? xValues[0] : 0;
  const xSpan = Math.max(1e-9, maxX - minX);
  const ySpan = Math.max(1e-9, maxY - minY);

  const plotX = (value: number) => marginLeft + ((value - minX) / xSpan) * plotWidth;
  const plotY = (value: number) => marginTop + ((maxY - value) / ySpan) * plotHeight;

  const xTicks = useMemo(() => buildTicks(minX, maxX, 4), [minX, maxX]);
  const yTicks = useMemo(() => buildTicks(minY, maxY, 4), [minY, maxY]);
  const zeroY = plotY(0);

  return (
    <figure className="dyncmp-chart">
      <figcaption>
        {title} <span className="dyncmp-chart-unit">({unit})</span>
      </figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <rect
          x={marginLeft}
          y={marginTop}
          width={plotWidth}
          height={plotHeight}
          className="dyncmp-chart-frame"
        />

        {yTicks.map((tick) => {
          const y = plotY(tick);
          if (y < marginTop - 0.5 || y > marginTop + plotHeight + 0.5) return null;
          return (
            <g key={`y-${tick}`}>
              <line
                x1={marginLeft}
                y1={y}
                x2={marginLeft + plotWidth}
                y2={y}
                className="dyncmp-chart-grid"
              />
              <text x={marginLeft - 5} y={y + 3} textAnchor="end" className="dyncmp-chart-tick">
                {Math.abs(tick) >= 1000 ? tick.toFixed(0) : Number(tick.toPrecision(3))}
              </text>
            </g>
          );
        })}

        {xTicks.map((tick) => {
          const x = plotX(tick);
          if (x < marginLeft - 0.5 || x > marginLeft + plotWidth + 0.5) return null;
          return (
            <g key={`x-${tick}`}>
              <line
                x1={x}
                y1={marginTop + plotHeight}
                x2={x}
                y2={marginTop + plotHeight + 4}
                className="dyncmp-chart-axis"
              />
              <text
                x={x}
                y={marginTop + plotHeight + 16}
                textAnchor="middle"
                className="dyncmp-chart-tick"
              >
                {Number(tick.toPrecision(3))}
              </text>
            </g>
          );
        })}

        {zeroY >= marginTop && zeroY <= marginTop + plotHeight ? (
          <line
            x1={marginLeft}
            y1={zeroY}
            x2={marginLeft + plotWidth}
            y2={zeroY}
            className="dyncmp-chart-zero"
          />
        ) : null}

        {lines.map((line) => {
          const d = line.values
            .map((v, i) => `${i === 0 ? "M" : "L"} ${plotX(xValues[i]).toFixed(1)} ${plotY(v).toFixed(1)}`)
            .join(" ");
          return (
            <path key={line.key} d={d} fill="none" stroke={line.color} strokeWidth={2} />
          );
        })}

        <text
          x={marginLeft + plotWidth / 2}
          y={height - 4}
          textAnchor="middle"
          className="dyncmp-chart-axis-label"
        >
          distance flown (km)
        </text>
      </svg>
    </figure>
  );
});

export default function DynamicsComparisonCharts({
  chart,
  systems,
  hiddenKeys,
  onToggleSystem,
  onClose,
  subtitle,
}: Props) {
  const hiddenSet = useMemo(() => new Set(hiddenKeys), [hiddenKeys]);
  const colorByKey = useMemo(() => {
    const map = new Map<string, [number, number, number, number]>();
    systems.forEach((system) => map.set(system.key, system.colorRgba));
    return map;
  }, [systems]);
  // Color is keyed by a system that always exists (comparedKeys derive from the
  // same `systems`), so a missing key is a real bug — fail loudly, no fallback.
  const colorFor = (key: string) => rgba(colorByKey.get(key)!);

  // Compared systems are those with an error series (A/C/D); keep system order.
  const comparedKeys = useMemo(
    () => systems.map((system) => system.key).filter((key) => key in chart.series),
    [systems, chart],
  );
  const visibleKeys = useMemo(
    () => comparedKeys.filter((key) => !hiddenSet.has(key)),
    [comparedKeys, hiddenSet],
  );
  // Memoize the per-metric line sets so dragging the window (which re-renders
  // this component via setPosition) does not rebuild them and re-render the
  // memoized charts; only a data/visibility change does.
  const metricCharts = useMemo(
    () =>
      METRICS.map((metric) => ({
        ...metric,
        lines: visibleKeys.map((key) => ({
          key,
          color: colorFor(key),
          values: chart.series[key][metric.field],
        })),
      })),
    // colorFor closes over colorByKey; both derive from `systems`.
    [visibleKeys, chart, colorByKey], // eslint-disable-line react-hooks/exhaustive-deps
  );

  // Draggable floating window: starts centered (position === null), then tracks
  // the dragged title bar. So it never blocks the panel — the user can move it
  // anywhere or close it (and reopen from the panel).
  const windowRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ dx: number; dy: number } | null>(null);
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);

  function onHeaderPointerDown(event: ReactPointerEvent<HTMLElement>) {
    // A click on the Close button must not start a drag.
    if ((event.target as HTMLElement).closest("button")) return;
    const win = windowRef.current;
    if (!win) return;
    const rect = win.getBoundingClientRect();
    dragRef.current = { dx: event.clientX - rect.left, dy: event.clientY - rect.top };
    setPosition({ x: rect.left, y: rect.top });
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  }

  function onHeaderPointerMove(event: ReactPointerEvent<HTMLElement>) {
    const drag = dragRef.current;
    const win = windowRef.current;
    if (!drag || !win) return;
    const maxX = Math.max(0, window.innerWidth - win.offsetWidth);
    const maxY = Math.max(0, window.innerHeight - win.offsetHeight);
    setPosition({
      x: Math.min(Math.max(0, event.clientX - drag.dx), maxX),
      y: Math.min(Math.max(0, event.clientY - drag.dy), maxY),
    });
  }

  function onHeaderPointerUp(event: ReactPointerEvent<HTMLElement>) {
    if (!dragRef.current) return;
    dragRef.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
  }

  // Portal to <body>: the Pilot panel ancestor has `backdrop-filter`, which makes
  // it the containing block for position:fixed AND clips overflow — so a fixed
  // window rendered inside it is pinned to the panel, not the viewport. The
  // portal lets the floating window position against the page instead.
  return createPortal(
    <div
      ref={windowRef}
      className={`dyncmp-charts-overlay${position ? "" : " is-centered"}`}
      role="dialog"
      aria-label="Dynamics comparison deviation charts"
      style={position ? { left: position.x, top: position.y } : undefined}
    >
      <header
        className="dyncmp-charts-header"
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
      >
        <div className="dyncmp-charts-titles">
          <h3>Dynamics deviation vs reference B</h3>
          <p>{subtitle ?? "Each system's error against the per-step re-anchored ENU reference, over distance flown."}</p>
        </div>
        <button type="button" className="dyncmp-charts-close" onClick={onClose}>
          Close
        </button>
      </header>

      <div className="dyncmp-charts-body">
      <div className="dyncmp-charts-legend">
        {systems.map((system) => {
          const isHidden = hiddenSet.has(system.key);
          return (
            <label
              key={system.key}
              className={`dyncmp-legend-item${isHidden ? " is-hidden" : ""}`}
            >
              <input
                type="checkbox"
                checked={!isHidden}
                onChange={() => onToggleSystem(system.key)}
              />
              <span className="dyncmp-legend-swatch" style={{ background: rgba(system.colorRgba) }} />
              <span className="dyncmp-legend-label">
                {system.label}
                {system.isReference ? " · reference (not plotted)" : ""}
              </span>
            </label>
          );
        })}
      </div>

      <div className="dyncmp-charts-grid">
        {metricCharts.map((metric) => (
          <MetricChart
            key={metric.field}
            title={metric.title}
            unit={metric.unit}
            xValues={chart.distanceKm}
            lines={metric.lines}
          />
        ))}
      </div>

      <table className="dyncmp-final-table">
        <caption>Final deviation at end of flight</caption>
        <thead>
          <tr>
            <th scope="col">System</th>
            <th scope="col">Horiz (m)</th>
            <th scope="col">Alt (m)</th>
            <th scope="col">Heading (°)</th>
            <th scope="col">Speed (m/s)</th>
          </tr>
        </thead>
        <tbody>
          {comparedKeys.map((key) => {
            const final = chart.final[key];
            return (
              <tr key={key} className={hiddenSet.has(key) ? "is-hidden" : ""}>
                <th scope="row">
                  <span className="dyncmp-legend-swatch" style={{ background: colorFor(key) }} />
                  {key}
                </th>
                <td>{formatMagnitude(final.horiz, 1)}</td>
                <td>{formatSigned(final.alt, 1)}</td>
                <td>{formatMagnitude(final.head, 3)}</td>
                <td>{formatSigned(final.speed, 2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
    </div>,
    document.body,
  );
}
