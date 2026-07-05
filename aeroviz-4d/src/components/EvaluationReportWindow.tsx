/**
 * EvaluationReportWindow.tsx
 * --------------------------
 * The detailed evaluation view behind the Optimization block's "Details" button:
 * a draggable floating window (same shell/pattern as the Dynamics-Comparison
 * charts) rendering the backend evaluation report — summary cards, gate note,
 * aggregate table, per-flight deviation charts and the full verdict table.
 *
 * SINGLE SOURCE: every number shown comes from the published
 * `evaluation_report.json` (`python -m evaluation` output copied verbatim by the
 * comparison builder). This component only sorts/formats/plots — the standalone
 * `python -m evaluation.visualize` HTML shows the same data outside the app.
 * Track overlays are deliberately NOT duplicated here: the 3D scene already
 * renders every flight's observed/optimized paths.
 */

import { memo, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { createPortal } from "react-dom";
import type { EvaluationReport, EvaluationRow } from "../data/evaluationReport";

interface Props {
  report: EvaluationReport;
  /** e.g. "KRDU · Runway target (constrained)" */
  subtitle: string;
  onClose: () => void;
}

const OK_COLOR = "#3fbf72";
const FAIL_COLOR = "#e05b5b";
const GATE_COLOR = "#e05b5b";
const BAND_COLOR = "rgba(63, 191, 114, 0.16)";

function formatNum(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function formatPct(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(100 * value).toFixed(1)}%`;
}

/** Shared axes math for the small SVG charts. */
const CHART = { width: 320, height: 170, left: 46, right: 10, top: 12, bottom: 26 };

function plotFrame() {
  return {
    plotWidth: CHART.width - CHART.left - CHART.right,
    plotHeight: CHART.height - CHART.top - CHART.bottom,
  };
}

/**
 * Per-flight deviation bars, sorted descending (a presentation choice — the
 * values themselves come straight from the report rows). Optional log scale
 * (deviations span 0.1 m … km), gate line, and window band.
 */
const DeviationBars = memo(function DeviationBars({
  title,
  values,
  log = false,
  gate,
  band,
}: {
  title: string;
  values: { label: string; value: number }[];
  log?: boolean;
  /** Horizontal limit line (e.g. the lateral gate). */
  gate?: number;
  /** Shaded acceptance window [low, high] (e.g. the vertical WCH window). */
  band?: [number, number];
}) {
  const { plotWidth, plotHeight } = plotFrame();
  const sorted = useMemo(
    () => [...values].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
    [values],
  );
  const transform = (v: number) => (log ? Math.log10(Math.max(Math.abs(v), 0.01)) : v);
  const transformed = sorted.map((v) => transform(v.value));
  let lo = Math.min(0, ...transformed, band ? transform(band[0]) : 0);
  let hi = Math.max(0, ...transformed, gate != null ? transform(gate) : 0, band ? transform(band[1]) : 0);
  if (lo === hi) hi = lo + 1;
  const span = hi - lo;
  const y = (v: number) => CHART.top + ((hi - v) / span) * plotHeight;
  const barW = Math.max(1, plotWidth / Math.max(1, sorted.length) - 0.5);

  return (
    <figure className="dyncmp-chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${CHART.width} ${CHART.height}`} role="img" aria-label={title}>
        <rect x={CHART.left} y={CHART.top} width={plotWidth} height={plotHeight} className="dyncmp-chart-frame" />
        {band ? (
          <rect
            x={CHART.left}
            y={y(transform(band[1]))}
            width={plotWidth}
            height={Math.max(1, y(transform(band[0])) - y(transform(band[1])))}
            fill={BAND_COLOR}
          />
        ) : null}
        {[lo, (lo + hi) / 2, hi].map((tick) => (
          <text key={tick} x={CHART.left - 5} y={y(tick) + 3} textAnchor="end" className="dyncmp-chart-tick">
            {log ? formatNum(10 ** tick, 10 ** tick >= 10 ? 0 : 1) : formatNum(tick, Math.abs(hi - lo) >= 10 ? 0 : 1)}
          </text>
        ))}
        {sorted.map((item, i) => {
          const v = transform(item.value);
          const y0 = y(Math.max(0, Math.min(v, hi)));
          const y1 = y(Math.max(lo, Math.min(0, v)));
          const inside =
            (gate == null || Math.abs(item.value) <= gate) &&
            (band == null || (item.value >= band[0] && item.value <= band[1]));
          return (
            <rect
              key={item.label}
              x={CHART.left + (i / sorted.length) * plotWidth}
              y={Math.min(y0, y1)}
              width={barW}
              height={Math.max(1, Math.abs(y1 - y0))}
              fill={inside ? OK_COLOR : FAIL_COLOR}
            >
              <title>{`${item.label}: ${formatNum(item.value, 2)} m`}</title>
            </rect>
          );
        })}
        {gate != null ? (
          <line
            x1={CHART.left} x2={CHART.left + plotWidth}
            y1={y(transform(gate))} y2={y(transform(gate))}
            stroke={GATE_COLOR} strokeDasharray="4 3" strokeWidth={1.4}
          />
        ) : null}
        <text x={CHART.left + plotWidth / 2} y={CHART.height - 4} textAnchor="middle" className="dyncmp-chart-axis-label">
          solved flights (sorted by |deviation|) · m
        </text>
      </svg>
    </figure>
  );
});

/** Optimized vs observed flight-time scatter (raw row values; diagonal = equal). */
const TimeScatter = memo(function TimeScatter({
  points,
}: {
  points: { label: string; observed: number; optimized: number; success: boolean }[];
}) {
  const { plotWidth, plotHeight } = plotFrame();
  const all = points.flatMap((p) => [p.observed, p.optimized]);
  const lo = Math.min(...all) * 0.95;
  const hi = Math.max(...all) * 1.05;
  const span = Math.max(1e-9, hi - lo);
  const x = (v: number) => CHART.left + ((v - lo) / span) * plotWidth;
  const y = (v: number) => CHART.top + ((hi - v) / span) * plotHeight;

  return (
    <figure className="dyncmp-chart">
      <figcaption>Flight time: optimized vs observed (s)</figcaption>
      <svg viewBox={`0 0 ${CHART.width} ${CHART.height}`} role="img" aria-label="Flight time scatter">
        <rect x={CHART.left} y={CHART.top} width={plotWidth} height={plotHeight} className="dyncmp-chart-frame" />
        <line x1={x(lo)} y1={y(lo)} x2={x(hi)} y2={y(hi)} className="dyncmp-chart-grid" strokeDasharray="4 3" />
        {[lo, (lo + hi) / 2, hi].map((tick) => (
          <g key={tick}>
            <text x={CHART.left - 5} y={y(tick) + 3} textAnchor="end" className="dyncmp-chart-tick">
              {formatNum(tick, 0)}
            </text>
            <text x={x(tick)} y={CHART.top + plotHeight + 12} textAnchor="middle" className="dyncmp-chart-tick">
              {formatNum(tick, 0)}
            </text>
          </g>
        ))}
        {points.map((p) => (
          <circle
            key={p.label}
            cx={x(p.observed)} cy={y(p.optimized)} r={3}
            fill={p.success ? OK_COLOR : FAIL_COLOR} fillOpacity={0.85}
          >
            <title>{`${p.label}: observed ${formatNum(p.observed, 0)} s → optimized ${formatNum(p.optimized, 0)} s`}</title>
          </circle>
        ))}
        <text x={CHART.left + plotWidth / 2} y={CHART.height - 4} textAnchor="middle" className="dyncmp-chart-axis-label">
          observed (s) — diagonal = equal
        </text>
      </svg>
    </figure>
  );
});

function rowWhy(row: EvaluationRow): string {
  return row.reason ?? row.violations.join("; ");
}

export default function EvaluationReportWindow({ report, subtitle, onClose }: Props) {
  const solvedRows = useMemo(() => report.trajectories.filter((r) => r.solved), [report]);
  const referenceRows = useMemo(
    () => solvedRows.filter((r) => r.reference?.flight_time_delta_s !== undefined),
    [solvedRows],
  );

  const lateralValues = useMemo(
    () => solvedRows.map((r) => ({ label: r.id, value: r.lateral_m ?? 0 })),
    [solvedRows],
  );
  const verticalValues = useMemo(
    () => solvedRows.map((r) => ({ label: r.id, value: r.vertical_m ?? 0 })),
    [solvedRows],
  );
  const timePoints = useMemo(
    () =>
      referenceRows.map((r) => ({
        label: r.id,
        observed: r.reference!.flight_time_s,
        optimized: r.final_time_s ?? 0,
        success: r.success,
      })),
    [referenceRows],
  );

  const th = report.thresholds;

  // Draggable floating window (same pattern as DynamicsComparisonCharts):
  // starts centered, then tracks the dragged title bar; portal to <body> so the
  // dock's backdrop-filter can't pin/clip it.
  const windowRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ dx: number; dy: number } | null>(null);
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);

  function onHeaderPointerDown(event: ReactPointerEvent<HTMLElement>) {
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

  const cards: { value: string; label: string; tone?: "ok" | "bad" }[] = [
    { value: String(report.total), label: "total trajectories" },
    {
      value: `${report.solved}/${report.total}`,
      label: `solve rate ${formatPct(report.solve_rate)}`,
      tone: report.solve_rate >= 0.9 ? "ok" : "bad",
    },
    {
      value: `${report.successful}/${report.total}`,
      label: `success rate ${formatPct(report.success_rate)}`,
      tone: report.success_rate >= 0.9 ? "ok" : "bad",
    },
  ];
  if (report.success_rate_among_solved != null) {
    cards.push({ value: formatPct(report.success_rate_among_solved), label: "success among solved" });
  }
  if (report.final_time_s) {
    cards.push({ value: `${formatNum(report.final_time_s.mean)} s`, label: "mean flight time" });
  }
  if (report.reference) {
    cards.push({
      value: `${formatNum(report.reference.flight_time_delta_s.mean)} s`,
      label: "mean Δt vs observed (optimized − flown)",
    });
  }

  return createPortal(
    <div
      ref={windowRef}
      className={`dyncmp-charts-overlay eval-report-window${position ? "" : " is-centered"}`}
      role="dialog"
      aria-label="Evaluation report"
      style={position ? { left: position.x, top: position.y } : undefined}
    >
      <header
        className="dyncmp-charts-header"
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
      >
        <div className="dyncmp-charts-titles">
          <h3>Evaluation report</h3>
          <p>{subtitle}</p>
        </div>
        <button type="button" className="dyncmp-charts-close" onClick={onClose}>
          Close
        </button>
      </header>

      <div className="dyncmp-charts-body">
        <div className="eval-report-cards">
          {cards.map((card) => (
            <div key={card.label} className={`eval-report-card${card.tone ? ` is-${card.tone}` : ""}`}>
              <div className="eval-report-card-value">{card.value}</div>
              <div className="eval-report-card-label">{card.label}</div>
            </div>
          ))}
        </div>

        <p className="eval-report-gates">
          Gates (FAA Order 8260.58D): lateral ≤ {formatNum(th.lateral_max_m, 2)} m — the LPV
          (Localizer Performance with Vertical guidance) course semiwidth floor at the threshold;
          vertical −{formatNum(th.vertical_below_max_m, 2)}/+{formatNum(th.vertical_above_max_m, 2)} m
          — the WCH (Wheel Crossing Height) window about the published TCH (Threshold Crossing Height).
        </p>

        <table className="dyncmp-final-table eval-report-aggregates">
          <caption>Aggregates (over solved flights)</caption>
          <thead>
            <tr>
              <th scope="col">metric</th>
              <th scope="col">mean</th>
              <th scope="col">p95 / min</th>
              <th scope="col">max</th>
            </tr>
          </thead>
          <tbody>
            {report.lateral_m ? (
              <tr>
                <th scope="row">final lateral deviation (m)</th>
                <td>{formatNum(report.lateral_m.mean)}</td>
                <td>{formatNum(report.lateral_m.p95)}</td>
                <td>{formatNum(report.lateral_m.max)}</td>
              </tr>
            ) : null}
            {report.vertical_m ? (
              <tr>
                <th scope="row">final vertical |deviation| (m)</th>
                <td>{formatNum(report.vertical_m.mean_abs)}</td>
                <td>{formatNum(report.vertical_m.p95_abs)}</td>
                <td>{formatNum(report.vertical_m.max_abs)}</td>
              </tr>
            ) : null}
            {report.final_time_s ? (
              <tr>
                <th scope="row">flight time (s)</th>
                <td>{formatNum(report.final_time_s.mean)}</td>
                <td>{formatNum(report.final_time_s.min)} (min)</td>
                <td>{formatNum(report.final_time_s.max)} (max)</td>
              </tr>
            ) : null}
            {report.reference ? (
              <>
                <tr>
                  <th scope="row">Δt vs observed (s, {report.reference.compared} flights)</th>
                  <td>{formatNum(report.reference.flight_time_delta_s.mean)}</td>
                  <td>{formatNum(report.reference.flight_time_delta_s.min)} (min)</td>
                  <td>{formatNum(report.reference.flight_time_delta_s.max)} (max)</td>
                </tr>
                <tr>
                  <th scope="row">path-shape deviation, lateral (m)</th>
                  <td>{formatNum(report.reference.path_lateral_m.mean)}</td>
                  <td />
                  <td>{formatNum(report.reference.path_lateral_m.max)} (max)</td>
                </tr>
              </>
            ) : null}
          </tbody>
        </table>

        {solvedRows.length > 0 ? (
          <div className="dyncmp-charts-grid">
            <DeviationBars
              title="Final lateral deviation (log; dashed = gate)"
              values={lateralValues}
              log
              gate={th.lateral_max_m}
            />
            <DeviationBars
              title="Final vertical deviation (band = WCH window)"
              values={verticalValues}
              band={[-th.vertical_below_max_m, th.vertical_above_max_m]}
            />
            {timePoints.length > 0 ? <TimeScatter points={timePoints} /> : null}
          </div>
        ) : (
          <p className="eval-report-empty">No solved trajectories to chart.</p>
        )}

        <div className="eval-report-rows">
          <table className="dyncmp-final-table eval-report-table">
            <caption>Per-trajectory verdicts</caption>
            <thead>
              <tr>
                <th scope="col">flight</th>
                <th scope="col">solved</th>
                <th scope="col">success</th>
                <th scope="col">lateral (m)</th>
                <th scope="col">vertical (m)</th>
                <th scope="col">T (s)</th>
                <th scope="col">Δt vs obs (s)</th>
                <th scope="col">notes</th>
              </tr>
            </thead>
            <tbody>
              {report.trajectories.map((row, i) => (
                <tr
                  key={`${row.file ?? row.id}-${i}`}
                  className={!row.solved ? "eval-row-unsolved" : row.success ? "" : "eval-row-fail"}
                >
                  <th scope="row">{row.id}</th>
                  <td>{row.solved ? "✓" : "✗"}</td>
                  <td>{row.success ? "✓" : "✗"}</td>
                  <td>{formatNum(row.lateral_m, 2)}</td>
                  <td>{formatNum(row.vertical_m, 2)}</td>
                  <td>{formatNum(row.final_time_s)}</td>
                  <td>{formatNum(row.reference?.flight_time_delta_s)}</td>
                  <td className="eval-row-why">{rowWhy(row)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>,
    document.body,
  );
}
