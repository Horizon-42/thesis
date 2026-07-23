/**
 * EvaluationReportWindow.tsx
 * --------------------------
 * The detailed evaluation view behind the Optimization block's "Details" button:
 * a draggable floating window (same shell/pattern as the Dynamics-Comparison
 * charts) rendering the backend evaluation report — summary cards, gate note,
 * aggregate table, per-flight deviation charts and the full verdict table.
 *
 * SINGLE SOURCE: every number shown comes from the published
 * evaluation-report artifact (`python -m evaluation` output copied verbatim by the
 * comparison builder and named by its index). This component only sorts/formats/plots — the standalone
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
const CHART = { width: 320, height: 186, left: 50, right: 12, top: 14, bottom: 30 };

function plotFrame() {
  return {
    plotWidth: CHART.width - CHART.left - CHART.right,
    plotHeight: CHART.height - CHART.top - CHART.bottom,
  };
}

const LOG_FLOOR = 0.01;
/** log10 magnitude (deviations span 0.1 m … km). */
const logScale = (v: number) => Math.log10(Math.max(Math.abs(v), LOG_FLOOR));
/** Signed symlog: linear feel near 0, log beyond — keeps the ±m window visible
 * next to km-scale outliers. */
const symlogScale = (v: number) => Math.sign(v) * Math.log10(1 + Math.abs(v));

interface ProfilePoint {
  label: string;
  value: number;
  /** Whether this flight passes THIS chart's own gate (colours the dot). */
  pass: boolean;
}

/**
 * Per-flight deviation profile: one dot per solved flight, RANKED by deviation
 * (a presentation choice — the values come straight from the report rows), on a
 * log/symlog axis so metre-scale landings and km-scale misses share one plot.
 * The chart's own gate is drawn (dashed line or shaded window) and each dot is
 * coloured by whether the flight passes THAT gate.
 */
const DeviationProfile = memo(function DeviationProfile({
  title,
  points,
  scale,
  gate,
  band,
}: {
  title: string;
  points: ProfilePoint[];
  scale: "log" | "symlog";
  /** Dashed limit line (the lateral gate), in metres. */
  gate?: number;
  /** Shaded acceptance window [low, high] (the vertical WCH window), in metres. */
  band?: [number, number];
}) {
  const { plotWidth, plotHeight } = plotFrame();
  const transform = scale === "log" ? logScale : symlogScale;
  const sorted = useMemo(
    () =>
      scale === "log"
        ? [...points].sort((a, b) => b.value - a.value)      // worst on the left
        : [...points].sort((a, b) => a.value - b.value),     // low → high, band in view
    [points, scale],
  );
  const transformed = sorted.map((p) => transform(p.value));
  const anchors = [
    ...(gate != null ? [transform(gate)] : []),
    ...(band ? [transform(band[0]), transform(band[1])] : []),
  ];
  let lo = Math.min(...transformed, ...anchors);
  let hi = Math.max(...transformed, ...anchors);
  if (lo === hi) hi = lo + 1;
  const pad = (hi - lo) * 0.08;
  lo -= pad;
  hi += pad;
  const y = (t: number) => CHART.top + ((hi - t) / (hi - lo)) * plotHeight;
  const x = (i: number) =>
    CHART.left + (sorted.length > 1 ? (i / (sorted.length - 1)) * plotWidth : plotWidth / 2);

  // Decade ticks (…0.1, 1, 10, 100, 1000…), signed for the symlog axis + 0.
  // On the symlog axis every |v| < 1 collapses onto ~0 (log10(1+|v|) ≈ 0), so the
  // sub-1 decades (±0.01, ±0.1) would stack their labels on top of the 0 tick —
  // skip them there and keep only 0, ±1, ±10, ±100…, which space out cleanly.
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let e = -2; e <= 5; e += 1) {
      const v = 10 ** e;
      if (scale === "log") {
        if (logScale(v) >= lo && logScale(v) <= hi) out.push(v);
      } else if (e >= 0) {
        for (const s of [v, -v]) {
          if (symlogScale(s) >= lo && symlogScale(s) <= hi) out.push(s);
        }
      }
    }
    if (scale === "symlog" && 0 >= lo && 0 <= hi) out.push(0);
    return out.sort((a, b) => a - b);
  }, [scale, lo, hi]);

  const failures = points.filter((p) => !p.pass).length;

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
        {ticks.map((tick) => {
          const ty = y(transform(tick));
          return (
            <g key={tick}>
              <line x1={CHART.left} x2={CHART.left + plotWidth} y1={ty} y2={ty} className="dyncmp-chart-grid" />
              <text x={CHART.left - 5} y={ty + 3} textAnchor="end" className="dyncmp-chart-tick">
                {Math.abs(tick) >= 1000 ? `${tick / 1000}k` : tick}
              </text>
            </g>
          );
        })}
        {gate != null ? (
          <g>
            <line
              x1={CHART.left} x2={CHART.left + plotWidth}
              y1={y(transform(gate))} y2={y(transform(gate))}
              stroke={GATE_COLOR} strokeDasharray="4 3" strokeWidth={1.4}
            />
            <text
              x={CHART.left + plotWidth - 2} y={y(transform(gate)) - 3}
              textAnchor="end" fill={GATE_COLOR} fontSize={9}
            >
              gate {formatNum(gate, 2)} m
            </text>
          </g>
        ) : null}
        {band ? (
          <text
            x={CHART.left + 4} y={y(transform(band[1])) - 3}
            fill={OK_COLOR} fontSize={9}
          >
            window −{formatNum(Math.abs(band[0]), 2)} / +{formatNum(band[1], 2)} m
          </text>
        ) : null}
        {sorted.map((p, i) => (
          <circle key={p.label + i} cx={x(i)} cy={y(transformed[i])} r={2}
                  fill={p.pass ? OK_COLOR : FAIL_COLOR} fillOpacity={0.85}>
            <title>{`${p.label}: ${formatNum(p.value, 2)} m`}</title>
          </circle>
        ))}
        <text x={CHART.left + plotWidth / 2} y={CHART.height - 4} textAnchor="middle" className="dyncmp-chart-axis-label">
          one dot per solved flight, ranked · m ({scale} scale)
        </text>
      </svg>
      <p className="eval-chart-note">
        <span style={{ color: OK_COLOR }}>●</span> passes this gate&nbsp;&nbsp;
        <span style={{ color: FAIL_COLOR }}>●</span> outside ({failures} of {points.length})
      </p>
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
          observed (s) — below the diagonal = optimized is faster
        </text>
      </svg>
      <p className="eval-chart-note">
        <span style={{ color: OK_COLOR }}>●</span> successful (all gates)&nbsp;&nbsp;
        <span style={{ color: FAIL_COLOR }}>●</span> off target (failed a gate)
      </p>
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

  const th = report.thresholds;
  const lateralValues = useMemo(
    () =>
      solvedRows.map((r) => ({
        label: r.id,
        value: r.lateral_m ?? 0,
        pass: (r.lateral_m ?? 0) <= th.lateral_max_m,
      })),
    [solvedRows, th],
  );
  const verticalValues = useMemo(
    () =>
      solvedRows.map((r) => ({
        label: r.id,
        value: r.vertical_m ?? 0,
        pass:
          (r.vertical_m ?? 0) >= -th.vertical_below_max_m &&
          (r.vertical_m ?? 0) <= th.vertical_above_max_m,
      })),
    [solvedRows, th],
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

  // Cards are deliberately NEUTRAL: red/green in this window means exactly one thing —
  // a per-flight gate verdict (see the chart legends) — not an arbitrary rate threshold.
  //
  // An OBSERVED batch is labelled differently, and not merely for wording: every
  // observed track trivially "has states", so its solve rate is 1.0 by construction and
  // says nothing. The established rate (did the flight fly a fittable, stabilised final
  // approach?) replaces it rather than joining it.
  const observed = report.observed;
  const cards: { value: string; label: string }[] = [
    { value: String(report.total), label: observed ? "observed flights" : "total trajectories" },
    observed
      ? {
          value: `${observed.established}/${report.total}`,
          label: `established rate ${formatPct(observed.established_rate)}`,
        }
      : {
          value: `${report.solved}/${report.total}`,
          label: `solve rate ${formatPct(report.solve_rate)}`,
        },
    {
      value: `${report.successful}/${report.measured ?? report.total}`,
      label: observed ? "inside both gates" : `success rate ${formatPct(report.success_rate)}`,
    },
  ];
  if (observed) {
    // Stated, never hidden: with 25 ft altitude quantisation against a 9.15 m window,
    // most real approaches sit within noise of a gate boundary.
    cards.push({ value: String(observed.marginal), label: "marginal (undecidable)" });
    if (observed.not_established > 0) {
      cards.push({ value: String(observed.not_established), label: "not established" });
    }
  } else if (report.success_rate_among_solved != null) {
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
            <div key={card.label} className="eval-report-card">
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
              <th scope="col">p95</th>
              <th scope="col">min</th>
              <th scope="col">max</th>
            </tr>
          </thead>
          <tbody>
            {report.lateral_m ? (
              <tr>
                <th scope="row">final lateral deviation (m)</th>
                <td>{formatNum(report.lateral_m.mean)}</td>
                <td>{formatNum(report.lateral_m.p95)}</td>
                <td className="eval-report-na">—</td>
                <td>{formatNum(report.lateral_m.max)}</td>
              </tr>
            ) : null}
            {report.vertical_m ? (
              <>
                <tr>
                  <th scope="row">final vertical |deviation| (m)</th>
                  <td>{formatNum(report.vertical_m.mean_abs)}</td>
                  <td>{formatNum(report.vertical_m.p95_abs)}</td>
                  <td className="eval-report-na">—</td>
                  <td>{formatNum(report.vertical_m.max_abs)}</td>
                </tr>
                {/* The row above is computed over |value|, so it cannot show a
                    high/low bias — cancellation is exactly what it discards.
                    mean_signed is the only signed statistic the backend emits
                    (signed_spread has no p95/min/max counterpart), hence one
                    mean and three dashes. */}
                <tr>
                  <th scope="row">final vertical deviation, signed (m, + = high)</th>
                  <td>{formatNum(report.vertical_m.mean_signed)}</td>
                  <td className="eval-report-na">—</td>
                  <td className="eval-report-na">—</td>
                  <td className="eval-report-na">—</td>
                </tr>
              </>
            ) : null}
            {report.final_time_s ? (
              <tr>
                <th scope="row">flight time (s)</th>
                <td>{formatNum(report.final_time_s.mean)}</td>
                <td className="eval-report-na">—</td>
                <td>{formatNum(report.final_time_s.min)}</td>
                <td>{formatNum(report.final_time_s.max)}</td>
              </tr>
            ) : null}
            {report.reference ? (
              <>
                <tr>
                  <th scope="row">Δt vs observed (s, {report.reference.compared} flights)</th>
                  <td>{formatNum(report.reference.flight_time_delta_s.mean)}</td>
                  <td className="eval-report-na">—</td>
                  <td>{formatNum(report.reference.flight_time_delta_s.min)}</td>
                  <td>{formatNum(report.reference.flight_time_delta_s.max)}</td>
                </tr>
                <tr>
                  <th scope="row">path-shape deviation, lateral (m)</th>
                  <td>{formatNum(report.reference.path_lateral_m.mean)}</td>
                  <td className="eval-report-na">—</td>
                  <td className="eval-report-na">—</td>
                  <td>{formatNum(report.reference.path_lateral_m.max)}</td>
                </tr>
              </>
            ) : null}
          </tbody>
        </table>

        {solvedRows.length > 0 ? (
          <div className="dyncmp-charts-grid">
            <DeviationProfile
              title="Final lateral deviation, worst → best"
              points={lateralValues}
              scale="log"
              gate={th.lateral_max_m}
            />
            <DeviationProfile
              title="Final vertical deviation, low → high"
              points={verticalValues}
              scale="symlog"
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
