/**
 * EvaluationReportWindow.tsx
 * --------------------------
 * The detailed evaluation view behind the Observe evaluation block's "Details" button:
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

import {
  memo,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { createPortal } from "react-dom";
import {
  isLegacyEvaluationReport,
  type EvaluationComponentResult,
  type EvaluationReport,
  type EvaluationRow,
  type EvaluationVerdict,
} from "../data/evaluationReport";
import {
  createDeviationScatterRenderer,
  type DeviationOrbitView,
  type DeviationScatterDatum,
  type DeviationScatterHit,
  type DeviationScatterRenderer,
} from "../utils/deviationScatterWebgl";

interface Props {
  report: EvaluationReport;
  /** Subject-aware heading, e.g. "Observed Baseline Evaluation Report". */
  title: string;
  /** e.g. "KRDU · Runway target (constrained)" */
  subtitle: string;
  onClose: () => void;
}

const OK_COLOR = "#3fbf72";
const FAIL_COLOR = "#e05b5b";
const INDETERMINATE_COLOR = "#989da6";
const GATE_COLOR = "#e05b5b";
const BAND_COLOR = "rgba(63, 191, 114, 0.16)";

function verdictColor(verdict: EvaluationVerdict): string {
  return verdict === "pass"
    ? OK_COLOR
    : verdict === "fail"
      ? FAIL_COLOR
      : INDETERMINATE_COLOR;
}

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
  result: EvaluationComponentResult;
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
  /** Shaded acceptance window [low, high] when a common vertical bound exists. */
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

  const failures = points.filter((p) => p.result === "fail").length;
  const indeterminate = points.filter((p) => p.result === "indeterminate").length;

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
                  fill={p.result === "pass" ? OK_COLOR : p.result === "fail" ? FAIL_COLOR : INDETERMINATE_COLOR}
                  fillOpacity={0.85}>
            <title>{`${p.label}: ${formatNum(p.value, 2)} m`}</title>
          </circle>
        ))}
        <text x={CHART.left + plotWidth / 2} y={CHART.height - 4} textAnchor="middle" className="dyncmp-chart-axis-label">
          one dot per solved flight, ranked · m ({scale} scale)
        </text>
      </svg>
      <p className="eval-chart-note">
        <span style={{ color: OK_COLOR }}>●</span> passes this gate&nbsp;&nbsp;
        <span style={{ color: FAIL_COLOR }}>●</span> outside ({failures})&nbsp;&nbsp;
        <span style={{ color: INDETERMINATE_COLOR }}>●</span> indeterminate ({indeterminate})
      </p>
    </figure>
  );
});

/** Optimized vs observed flight-time scatter (raw row values; diagonal = equal). */
const TimeScatter = memo(function TimeScatter({
  points,
}: {
  points: {
    label: string;
    observed: number;
    optimized: number;
    verdict: EvaluationVerdict;
  }[];
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
            fill={verdictColor(p.verdict)} fillOpacity={0.85}
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
        <span style={{ color: FAIL_COLOR }}>●</span> failed a gate&nbsp;&nbsp;
        <span style={{ color: INDETERMINATE_COLOR }}>●</span> indeterminate
      </p>
    </figure>
  );
});

const DEFAULT_3D_VIEW: DeviationOrbitView = {
  yaw: -0.68,
  pitch: -0.42,
  distance: 3.25,
};

/** GPU-backed, orbitable 3D view of lateral × vertical deviation by flight. */
const DeviationScatter3D = memo(function DeviationScatter3D({
  points,
  lateralGate,
  verticalBand,
}: {
  points: DeviationScatterDatum[];
  lateralGate: number;
  verticalBand: [number, number];
}) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<DeviationScatterRenderer | null>(null);
  const viewRef = useRef<DeviationOrbitView>({ ...DEFAULT_3D_VIEW });
  const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);
  const drawFrameRef = useRef<number | null>(null);
  const [hovered, setHovered] = useState<DeviationScatterHit | null>(null);
  const [webglAvailable, setWebglAvailable] = useState<boolean | null>(null);

  function requestDraw() {
    if (drawFrameRef.current !== null) return;
    drawFrameRef.current = window.requestAnimationFrame(() => {
      drawFrameRef.current = null;
      rendererRef.current?.draw(viewRef.current);
    });
  }

  function clearHover() {
    setHovered(null);
  }

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;

    // This is intentionally a native, non-passive boundary on the complete
    // stage rather than a React onWheel on the canvas. It covers the legend,
    // reset button and empty stage space, and stops the event before it can
    // reach the report's scroll container.
    const onStageWheel = (event: WheelEvent) => {
      viewRef.current.distance = Math.min(
        6.5,
        Math.max(2.15, viewRef.current.distance + event.deltaY * 0.003),
      );
      clearHover();
      requestDraw();
      event.preventDefault();
      event.stopPropagation();
    };
    const onStageContextMenu = (event: MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();
    };
    stage.addEventListener("wheel", onStageWheel, { passive: false });
    stage.addEventListener("contextmenu", onStageContextMenu);
    return () => {
      stage.removeEventListener("wheel", onStageWheel);
      stage.removeEventListener("contextmenu", onStageContextMenu);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (typeof WebGLRenderingContext === "undefined") {
      setWebglAvailable(false);
      return;
    }
    const renderer = createDeviationScatterRenderer(
      canvas,
      points,
      lateralGate,
      verticalBand,
    );
    if (!renderer) {
      setWebglAvailable(false);
      return;
    }
    rendererRef.current = renderer;
    setWebglAvailable(true);

    const resizeAndDraw = () => {
      const rect = canvas.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      renderer.resize(
        Math.max(1, Math.round(rect.width * pixelRatio)),
        Math.max(1, Math.round(rect.height * pixelRatio)),
      );
      renderer.draw(viewRef.current);
    };
    resizeAndDraw();

    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(resizeAndDraw);
      resizeObserver.observe(canvas);
    } else {
      window.addEventListener("resize", resizeAndDraw);
    }

    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener("resize", resizeAndDraw);
      if (drawFrameRef.current !== null) {
        window.cancelAnimationFrame(drawFrameRef.current);
        drawFrameRef.current = null;
      }
      renderer.dispose();
      rendererRef.current = null;
    };
  }, [points, lateralGate, verticalBand[0], verticalBand[1]]);

  function onCanvasPointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (event.button !== 0) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    dragRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.currentTarget.classList.add("is-dragging");
    clearHover();
    event.preventDefault();
    event.stopPropagation();
  }

  function onCanvasPointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (drag && drag.pointerId === event.pointerId) {
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      drag.x = event.clientX;
      drag.y = event.clientY;
      viewRef.current.yaw += dx * 0.012;
      viewRef.current.pitch = Math.min(
        1.35,
        Math.max(-1.35, viewRef.current.pitch + dy * 0.012),
      );
      requestDraw();
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    setHovered(
      rendererRef.current?.hitTest(
        event.clientX - rect.left,
        event.clientY - rect.top,
        viewRef.current,
      ) ?? null,
    );
    event.stopPropagation();
  }

  function finishCanvasDrag(event: ReactPointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    event.currentTarget.classList.remove("is-dragging");
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture?.(event.pointerId);
    }
    event.stopPropagation();
  }

  function onCanvasKeyDown(event: ReactKeyboardEvent<HTMLCanvasElement>) {
    event.stopPropagation();
    const view = viewRef.current;
    let handled = true;
    if (event.key === "ArrowLeft") view.yaw -= 0.12;
    else if (event.key === "ArrowRight") view.yaw += 0.12;
    else if (event.key === "ArrowUp") view.pitch = Math.max(-1.35, view.pitch - 0.1);
    else if (event.key === "ArrowDown") view.pitch = Math.min(1.35, view.pitch + 0.1);
    else if (event.key === "+" || event.key === "=") view.distance = Math.max(2.15, view.distance - 0.2);
    else if (event.key === "-") view.distance = Math.min(6.5, view.distance + 0.2);
    else handled = false;
    if (!handled) {
      if ([" ", "PageUp", "PageDown", "Home", "End"].includes(event.key)) {
        event.preventDefault();
      }
      return;
    }
    clearHover();
    requestDraw();
    event.preventDefault();
  }

  function resetView() {
    viewRef.current = { ...DEFAULT_3D_VIEW };
    clearHover();
    requestDraw();
  }

  return (
    <figure className="dyncmp-chart eval-deviation-3d">
      <figcaption>3D trajectory deviations: lateral × vertical</figcaption>
      <div
        ref={stageRef}
        className="eval-deviation-3d-stage"
        onPointerDown={(event) => event.stopPropagation()}
        onClick={(event) => event.stopPropagation()}
        onDoubleClick={(event) => event.stopPropagation()}
      >
        <canvas
          ref={canvasRef}
          role="img"
          aria-label="3D trajectory deviation view"
          aria-description="Drag to rotate the 3D view. Use the mouse wheel to zoom."
          data-renderer="webgl"
          data-point-count={points.length}
          tabIndex={0}
          onPointerDown={onCanvasPointerDown}
          onPointerMove={onCanvasPointerMove}
          onPointerUp={finishCanvasDrag}
          onPointerCancel={finishCanvasDrag}
          onPointerLeave={(event) => {
            if (!dragRef.current) clearHover();
            event.currentTarget.classList.remove("is-dragging");
          }}
          onKeyDown={onCanvasKeyDown}
        />
        <div className="eval-deviation-3d-axes" aria-hidden="true">
          <span className="is-lateral">L · lateral</span>
          <span className="is-vertical">V · vertical</span>
          <span className="is-flight">F · flights</span>
        </div>
        <button
          type="button"
          className="eval-deviation-3d-reset"
          onClick={(event) => {
            event.stopPropagation();
            resetView();
          }}
          aria-label="Reset 3D view"
        >
          Reset
        </button>
        {hovered ? (
          <div
            className="eval-deviation-3d-tooltip"
            style={{ left: hovered.screenX, top: hovered.screenY }}
          >
            <strong>{hovered.label}</strong>
            <span>L {formatNum(hovered.lateral, 2)} m</span>
            <span>V {formatNum(hovered.vertical, 2)} m</span>
          </div>
        ) : null}
        {webglAvailable === false ? (
          <p className="eval-deviation-3d-unavailable" role="status">
            WebGL is unavailable in this browser.
          </p>
        ) : null}
      </div>
      <p className="eval-chart-note">
        drag to orbit · wheel to zoom · hover a point for L/V · wireframe = gate
      </p>
    </figure>
  );
});

function rowWhy(row: EvaluationRow): string {
  return row.reason ?? row.violations.join("; ");
}

type FiniteDeviationRow = EvaluationRow & { lateral_m: number; vertical_m: number };
type DeviationStatus =
  | "measured"
  | "not established"
  | "not measured"
  | "invalid (non-finite)"
  | "not solved";

function hasFiniteDeviations(row: EvaluationRow): row is FiniteDeviationRow {
  return (
    typeof row.lateral_m === "number" &&
    Number.isFinite(row.lateral_m) &&
    typeof row.vertical_m === "number" &&
    Number.isFinite(row.vertical_m)
  );
}

function deviationStatus(row: EvaluationRow): DeviationStatus {
  if (!row.solved) return "not solved";
  if (hasFiniteDeviations(row)) return "measured";

  const values: unknown[] = [row.lateral_m, row.vertical_m];
  if (
    values.some(
      (value) => value != null && (typeof value !== "number" || !Number.isFinite(value)),
    )
  ) {
    return "invalid (non-finite)";
  }
  return row.event_status === "unavailable" ? "not measured" : "not measured";
}

function deviationStatusClass(status: DeviationStatus): string {
  if (status === "measured") return "is-measured";
  if (status === "invalid (non-finite)") return "is-invalid";
  return "is-unavailable";
}

export default function EvaluationReportWindow({ report, title, subtitle, onClose }: Props) {
  const solvedRows = useMemo(() => report.trajectories.filter((r) => r.solved), [report]);
  const measuredRows = useMemo(
    () => solvedRows.filter(hasFiniteDeviations),
    [solvedRows],
  );
  const deviationAvailability = useMemo(() => {
    let notMeasured = 0;
    let invalid = 0;
    for (const row of solvedRows) {
      const status = deviationStatus(row);
      if (status === "invalid (non-finite)") invalid += 1;
      else if (status !== "measured") notMeasured += 1;
    }
    return { notMeasured, invalid, excluded: notMeasured + invalid };
  }, [solvedRows]);
  const referenceRows = useMemo(
    () => solvedRows.filter((r) => r.reference?.flight_time_delta_s !== undefined),
    [solvedRows],
  );

  const commonNumber = (values: (number | null | undefined)[]): number | null => {
    const finite = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    if (!finite.length || finite.length !== values.length) return null;
    return finite.every((value) => Math.abs(value - finite[0]) < 1e-9) ? finite[0] : null;
  };
  const commonLateralBound = commonNumber(measuredRows.map((row) => row.bounds.lateral_m));
  const commonVerticalLower = commonNumber(measuredRows.map((row) => row.bounds.vertical_lower_m));
  const commonVerticalUpper = commonNumber(measuredRows.map((row) => row.bounds.vertical_upper_m));
  const lateralValues = useMemo(
    () =>
      measuredRows.map((r) => ({
        label: r.id,
        value: r.lateral_m,
        result: r.lateral_result,
      })),
    [measuredRows],
  );
  const verticalValues = useMemo(
    () =>
      measuredRows.map((r) => ({
        label: r.id,
        value: r.vertical_m,
        result: r.vertical_result,
      })),
    [measuredRows],
  );
  const timePoints = useMemo(
    () =>
      referenceRows.map((r) => ({
        label: r.id,
        observed: r.reference!.reference_flight_time_s ?? 0,
        optimized: r.final_time_s ?? 0,
        verdict: r.verdict,
      })),
    [referenceRows],
  );
  const deviation3DPoints = useMemo(
    () =>
      measuredRows.map((row) => ({
        label: row.id,
        lateral: row.lateral_m,
        vertical: row.vertical_m,
        verdict: row.verdict,
      })),
    [measuredRows],
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
  const isObserved = report.subject === "observed";
  const cards: { value: string; label: string }[] = [
    {
      value: String(report.total),
      label: isObserved ? "evaluated observed records" : "total trajectories",
    },
    observed
      ? {
          value: `${observed.event_estimated}/${observed.event_denominator}`,
          label: `threshold event estimated ${formatPct(observed.event_estimated_rate)}`,
        }
      : {
          value: `${report.solved}/${report.total}`,
          label: `solve rate ${formatPct(report.solve_rate)}`,
        },
    {
      value: `${report.successful}/${report.total}`,
      label: `pass rate ${formatPct(report.success_rate)}`,
    },
    { value: String(report.failed), label: "failed" },
    { value: String(report.indeterminate), label: "indeterminate" },
  ];
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
      aria-label={title}
      style={position ? { left: position.x, top: position.y } : undefined}
    >
      <header
        className="dyncmp-charts-header"
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
      >
        <div className="dyncmp-charts-titles">
          <h3>{title}</h3>
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

        {isLegacyEvaluationReport(report) ? (
          <p className="eval-report-deviation-warning" role="status">
            Pre-speed-gate report ({report.schema_version}): its verdicts grade lateral and
            vertical deviation only — the stall-anchored crossing-speed gate did not exist
            when this batch was evaluated. Rerun the optimizer batch to regenerate it under
            the current schema.
          </p>
        ) : null}

        <p className="eval-report-gates">
          Terminal bounds are shown in each row. Lateral is half the published runway
          width — did the crossing lie over the pavement — which is a landing-geometry
          claim, not a navigation-containment one. Vertical is benchmark specific: the
          published-TCH path and the 22 m RNAV/RNP terminal bound. The result grades
          terminal final-approach geometry, not touchdown or landing certification.
        </p>

        {deviationAvailability.excluded > 0 ? (
          <p className="eval-report-deviation-warning" role="status">
            {deviationAvailability.excluded} solved{" "}
            {deviationAvailability.excluded === 1 ? "flight" : "flights"}
            {" "}excluded from deviation charts: {deviationAvailability.notMeasured} not measured;{" "}
            {deviationAvailability.invalid} invalid/non-finite. They remain listed below with their
            deviation status and reason.
          </p>
        ) : null}

        <table className="dyncmp-final-table eval-report-aggregates">
          <caption>
            Aggregates (over {report.measured ?? measuredRows.length} measured threshold events)
          </caption>
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

        {measuredRows.length > 0 || timePoints.length > 0 ? (
          <div className="dyncmp-charts-grid">
            {measuredRows.length > 0 ? (
              <>
                <DeviationProfile
                  title="Final lateral deviation, worst → best"
                  points={lateralValues}
                  scale="log"
                  gate={commonLateralBound ?? undefined}
                />
                <DeviationProfile
                  title="Final vertical deviation, low → high"
                  points={verticalValues}
                  scale="symlog"
                  band={commonVerticalLower != null && commonVerticalUpper != null
                    ? [commonVerticalLower, commonVerticalUpper]
                    : undefined}
                />
              </>
            ) : null}
            {deviation3DPoints.length > 0 && commonLateralBound != null &&
            commonVerticalLower != null && commonVerticalUpper != null ? (
              <DeviationScatter3D
                points={deviation3DPoints}
                lateralGate={commonLateralBound}
                verticalBand={[commonVerticalLower, commonVerticalUpper]}
              />
            ) : null}
            {timePoints.length > 0 ? <TimeScatter points={timePoints} /> : null}
          </div>
        ) : (
          <p className="eval-report-empty">
            {solvedRows.length > 0
              ? "No finite deviation measurements to chart."
              : "No solved trajectories to chart."}
          </p>
        )}

        <div className="eval-report-rows">
          <table className="dyncmp-final-table eval-report-table">
            <caption>Per-trajectory verdicts</caption>
            <thead>
              <tr>
                <th scope="col">flight</th>
                <th scope="col">solved</th>
                <th scope="col">event</th>
                <th scope="col">lateral</th>
                <th scope="col">vertical</th>
                <th scope="col">overall</th>
                <th scope="col">deviation status</th>
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
                  className={!row.solved ? "eval-row-unsolved" : row.verdict === "fail" ? "eval-row-fail" : ""}
                >
                  <th scope="row">{row.id}</th>
                  <td>{row.solved ? "✓" : "✗"}</td>
                  <td>{row.event_status}</td>
                  <td>{row.lateral_result}</td>
                  <td>{row.vertical_result}</td>
                  <td>{row.verdict}</td>
                  <td className={`eval-deviation-status ${deviationStatusClass(deviationStatus(row))}`}>
                    {deviationStatus(row)}
                  </td>
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
