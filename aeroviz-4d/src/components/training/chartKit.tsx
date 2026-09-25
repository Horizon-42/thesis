/**
 * chartKit.tsx
 * ------------
 * The pieces the Training charts are drawn with (the read-back window, the prior's strips): one SVG unit is one pixel.
 * A chart frame whose caption is HTML (it wraps, where SVG text is clipped) and whose pointer reads the chart's own
 * coordinates; an axis; a vertical marker line; a line through points; a band's rectangle; a legend swatch.
 */

import type { MouseEvent, ReactNode } from "react";

/** The pointer's position in an SVG's own units (one per pixel): from the element's box, not the event target's — a
 *  child under the pointer would give `offsetX` relative to itself in some browsers. */
function pointerAt(event: MouseEvent<SVGSVGElement>): { x: number; y: number } {
  const box = event.currentTarget.getBoundingClientRect();
  return { x: event.clientX - box.left, y: event.clientY - box.top };
}

/** A chart: its caption above (indented to the plot's left edge), the SVG below; hovering calls ``onPointer``, a click
 *  ``onPick``. */
export function ChartFrame({ label, caption, captionIndent = 0, width, height, onPointer, onPick, children }: {
  label: string;
  caption?: ReactNode;
  captionIndent?: number;
  width: number;
  height: number;
  onPointer?: (x: number, y: number) => void;
  onPick?: (x: number, y: number) => void;
  children: ReactNode;
}) {
  return (
    <>
      {caption === undefined ? null : <p className="training-readback-caption" style={{ paddingLeft: captionIndent }}>{caption}</p>}
      <svg className="training-readback-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-label={label}
        onMouseMove={onPointer && ((event) => {
          const { x, y } = pointerAt(event);
          onPointer(x, y);
        })}
        onClick={onPick && ((event) => {
          const { x, y } = pointerAt(event);
          onPick(x, y);
        })}>
        {children}
      </svg>
    </>
  );
}

/** An axis along the plot's foot: its ticks and, at its right end, what it counts. */
export function Axis({ left, right, y, ticks, caption }: {
  left: number; right: number; y: number; ticks: Array<{ x: number; text: string }>; caption: string;
}) {
  return (
    <g>
      <line x1={left} x2={right} y1={y} y2={y} className="training-readback-axis" />
      {ticks.map((tick) => (
        <text key={tick.x} x={tick.x} y={y + 14} textAnchor="middle" className="training-readback-tick">{tick.text}</text>
      ))}
      <text x={right} y={y + 26} textAnchor="end" className="training-readback-tick">{caption}</text>
    </g>
  );
}

/** A vertical line across the plot: the cursor, the clearance. */
export function VLine({ x, top, bottom, className, stroke, dash, title }: {
  x: number; top: number; bottom: number; className?: string; stroke?: string; dash?: string; title?: string;
}) {
  return (
    <line x1={x} x2={x} y1={top} y2={bottom} className={className} stroke={stroke} strokeDasharray={dash}>
      {title === undefined ? null : <title>{title}</title>}
    </line>
  );
}

/** Points as an SVG `points` attribute. */
function pointsOf(xs: number[], ys: number[]): string {
  return xs.map((x, index) => `${x},${ys[index]}`).join(" ");
}

/** A line through points; its ends square unless ``round`` (a round cap would close a dash's gaps). */
export function Line({ xs, ys, stroke, width, className, dash, opacity, round, title }: {
  xs: number[]; ys: number[]; stroke: string; width: number; className?: string; dash?: string; opacity?: number;
  round?: boolean; title?: string;
}) {
  return (
    <polyline points={pointsOf(xs, ys)} fill="none" stroke={stroke} strokeWidth={width} strokeDasharray={dash}
      strokeOpacity={opacity} strokeLinecap={round ? "round" : undefined} className={className}>
      {title === undefined ? null : <title>{title}</title>}
    </polyline>
  );
}

/** A band's rectangle from x0 to x1 between two heights (y grows downward): never narrower than a pixel. */
export function bandRect(x0: number, x1: number, yTop: number, yBottom: number) {
  return { x: x0, width: Math.max(x1 - x0, 1), y: yTop, height: yBottom - yTop };
}

export type Swatch =
  | { kind: "line"; colour: string; dash?: string }
  | { kind: "area"; colour: string; opacity: number; dash?: string };

/** What a colour is drawn as, for a legend. */
export function SwatchIcon({ swatch }: { swatch: Swatch }) {
  return (
    <svg className="training-legend-swatch" width={22} height={10} aria-hidden="true">
      {swatch.kind === "line" ? (
        <line x1={1} x2={21} y1={5} y2={5} stroke={swatch.colour} strokeWidth={2.2} strokeDasharray={swatch.dash} />
      ) : (
        <rect x={1} y={1} width={20} height={8} fill={swatch.colour} fillOpacity={swatch.opacity} stroke={swatch.colour}
          strokeWidth={1.2} strokeDasharray={swatch.dash} />
      )}
    </svg>
  );
}
