/**
 * TrainingReadbackWindow.tsx
 * --------------------------
 * The interactive version of the vocabulary artefact's 300 hand-check pages
 * (`instruction_vocabulary.hand_check_figure`): the plan view in the runway's
 * frame, and the three signals the words are read from, each with the word in
 * force drawn as a step over it. Design: §3.2 of
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` (T4b).
 *
 * WHAT IT IS FOR: on paper a human marked each page 一致 / 漏读 / 误读 — agrees,
 * missed, misread. This window answers the same question with the numbers
 * attached, over the same flights (the export draws its sample from the very
 * pages that were checked), so a disagreement can be read off rather than
 * estimated off a PNG.
 *
 * THE HEADING CHART PLOTS THE UNWRAPPED COURSE, and the word's level is placed
 * on the same turn of the circle as the trace (`+ 360 × round(…)`, mirroring the
 * `period` branch of the figure). Without that a +90° word drawn against a trace
 * sitting at +450° looks like a 360° error, and every orbit would read as a
 * misread word.
 *
 * It renders through a PORTAL into `document.body` (AV7): `.flight-ops-panel`
 * carries a `backdrop-filter`, which makes a descendant with `position: fixed`
 * position against IT rather than the viewport.
 */

import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { FEET_TO_METERS, metresPerSecondToKnots } from "../utils/procedureGeoMath";
import {
  TRAINING_FLOWN_COLOR,
  TRAINING_KIND_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  altitudeCentreM,
  formatSeconds,
  headingCentreDeg,
  speedCentreMps,
  trainingWordLabel,
  type TrainingAbsorbed,
  type TrainingFlight,
  type TrainingGeometricColumn,
  type TrainingInstruction,
  type TrainingVocabulary,
} from "../data/trainingSample";

// ── geometry (pixels; the window measures itself) ───────────────────────────
const GUTTER = 62;
const PAD_R = 16;
const PLAN_H = 190;
const CHART_H = 116;
const AXIS_H = 26;
const DEFAULT_W = 980;
const MIN_W = 420;

/** An absorbed span can be a single 2 s row; the figure widens it by a row each
 *  side for the same reason — a zero-width mark is a mark nobody sees. */
const ABSORBED_PAD_S = 1;

/** The three signals the words are read from. The runway word names the frame
 *  they are all measured in, and the duration and terminal words are not
 *  signals at all, so neither gets a chart — they are the sentence bar's rows. */
const CHARTED_KINDS = ["heading", "altitude", "speed"] as const;
type ChartedKind = (typeof CHARTED_KINDS)[number];

/**
 * The flown sentence's own column per chart. The heading chart plots the
 * UNWRAPPED course for the observation, and the flown track carries the wrapped
 * one — so it is unwrapped here against its own previous value, or the line jumps
 * 360° every time the rule-follower passes the cut.
 */
const FLOWN_COLUMN: Record<ChartedKind, TrainingGeometricColumn> = {
  heading: "relCourseDeg",
  altitude: "heightM",
  speed: "groundSpeedMps",
};

function unwrapDegrees(values: number[]): number[] {
  const out: number[] = [];
  let previous = values[0];
  values.forEach((value, index) => {
    if (index === 0) {
      out.push(value);
      return;
    }
    const next = value + 360 * Math.round((previous - value) / 360);
    out.push(next);
    previous = next;
  });
  return out;
}

/** The flown track on a chart's axis, in that chart's display unit. */
export function flownTrace(flight: TrainingFlight, kind: ChartedKind): number[] {
  const column = flight.geometric[FLOWN_COLUMN[kind]];
  if (kind === "heading") return unwrapDegrees(column);
  if (kind === "altitude") return column.map((metres) => metres / FEET_TO_METERS);
  return column.map(metresPerSecondToKnots);
}

/** One column of a track, read at a time that falls between its rows. */
function at(tS: number[], values: number[], seconds: number): number {
  const row = rowAt(tS, seconds);
  if (row + 1 >= tS.length) return values[row];
  const span = tS[row + 1] - tS[row];
  return values[row] + ((values[row + 1] - values[row]) * (seconds - tS[row])) / span;
}

/**
 * How far apart the two tracks are at one moment, horizontally in the runway
 * frame. `null` once the flown sentence has stopped: there is nothing to compare
 * against then, and a number there would be measuring the stopping rule.
 *
 * BOTH TRACKS ARE READ AT THE SAME INSTANT. They are on different clocks — the
 * observation on 2 s rows, the flown sentence on 1 s steps — so flooring each to
 * its own row put the two positions up to a second apart, which is ~70 m of
 * flight (measured over the 40 exported flights: median 8.5 m of error, p95
 * 108 m, max 191 m). `meanGapM`, printed two lines below this on screen,
 * interpolates the flown track onto the observed times; this now agrees with it.
 */
export function gapAtS(flight: TrainingFlight, seconds: number): number | null {
  const flown = flight.geometric;
  if (seconds > flown.tS[flown.tS.length - 1]) return null;
  const observed = flight.observed;
  return Math.hypot(
    at(flown.tS, flown.toGoM, seconds) - at(observed.tS, observed.toGoM, seconds),
    at(flown.tS, flown.crossM, seconds) - at(observed.tS, observed.crossM, seconds),
  );
}

export interface ChartLevel {
  instruction: TrainingInstruction;
  /** Where the step is drawn, in the chart's display unit. */
  level: number;
  /** The word holds until the next instruction of its kind, or the track's end. */
  endS: number;
}

/**
 * The step levels for one kind, mirroring `hand_check_figure.steps`: each
 * instruction holds until the next of the same kind, and an unwrapped trace has
 * the level moved onto the trace's own turn of the circle, judged at the moment
 * the manoeuvre settled (the end of the track when it never did).
 */
export function chartLevels(
  flight: TrainingFlight,
  vocabulary: TrainingVocabulary,
  kind: ChartedKind,
): ChartLevel[] {
  const { tS } = flight.observed;
  const endOfTrack = tS[tS.length - 1];
  const issued = flight.instructions
    .filter((item) => item.kind === kind)
    .sort((a, b) => a.issuedS - b.issuedS);

  return issued.map((instruction, index) => {
    const endS = index + 1 < issued.length ? issued[index + 1].issuedS : endOfTrack;
    if (kind === "altitude") {
      return { instruction, endS, level: altitudeCentreM(vocabulary, instruction.word) / FEET_TO_METERS };
    }
    if (kind === "speed") {
      return { instruction, endS, level: metresPerSecondToKnots(speedCentreMps(vocabulary, instruction.word)) };
    }
    const centre = headingCentreDeg(vocabulary, instruction.word);
    const at = rowAt(tS, instruction.settledS ?? endOfTrack);
    const trace = flight.observed.courseUnwrappedDeg[at];
    return { instruction, endS, level: centre + 360 * Math.round((trace - centre) / 360) };
  });
}

/** The row in force at a time: the last row at or before it. */
export function rowAt(tS: number[], seconds: number): number {
  let low = 0;
  let high = tS.length - 1;
  if (seconds <= tS[0]) return 0;
  if (seconds >= tS[high]) return high;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (tS[mid] <= seconds) low = mid;
    else high = mid - 1;
  }
  return low;
}

function extent(values: number[]): [number, number] {
  let low = values[0];
  let high = values[0];
  for (const value of values) {
    if (value < low) low = value;
    if (value > high) high = value;
  }
  // A divide-by-zero guard, not a domain bound: `yFor` would produce NaN for
  // every row. It cannot fire on this artefact — the narrowest combined spread
  // over KRDU's 40 flights is 0.94° of heading — because what is constant for a
  // whole approach is the WORD, and `extent` always sees the trace too.
  if (high === low) return [low - 1, high + 1];
  const pad = (high - low) * 0.12;
  return [low - pad, high + pad];
}

/** One reading of an absorbed manoeuvre, used as both tooltip and accessible name. */
function absorbedName(item: TrainingAbsorbed): string {
  const change = `${item.change > 0 ? "+" : ""}${item.change.toFixed(1)}`;
  return `read but not worded: ${change} over ${formatSeconds(item.startS)}–${formatSeconds(item.endS)} s (${item.reason})`;
}

/**
 * The unbinned value the plateau settled at, in the unit its chart is drawn in.
 * `Instruction.target` is stored in SI — metres and metres per second — and the
 * charts are in feet and knots, so printing it raw put "190 kt … target 98.7"
 * in the one tooltip a person uses to decide whether the word was read right.
 */
function targetReading(kind: ChartedKind, target: number): string {
  if (kind === "altitude") return `${format(target / FEET_TO_METERS)} ft`;
  if (kind === "speed") return `${format(metresPerSecondToKnots(target))} kt`;
  return `${format(target, 1)}\u00b0`;
}

/** The same for one step: the word, where it came from, and its unbinned target. */
function levelName(kind: ChartedKind, item: ChartLevel, vocabulary: TrainingVocabulary): string {
  return (
    `${kind} ${trainingWordLabel(vocabulary, kind, item.instruction.word)} (word ${item.instruction.word}), ` +
    `issued ${formatSeconds(item.instruction.issuedS)} s, ` +
    `target ${targetReading(kind, item.instruction.target)}` +
    `${item.instruction.clamped ? " — clamped to the edge of the vocabulary" : ""}`
  );
}

function format(value: number, digits = 0): string {
  return value.toFixed(digits);
}

export interface TrainingReadbackWindowProps {
  flight: TrainingFlight;
  vocabulary: TrainingVocabulary;
  cursorS: number;
  onCursorChange: (seconds: number) => void;
  onClose: () => void;
}

export default function TrainingReadbackWindow({
  flight,
  vocabulary,
  cursorS,
  onCursorChange,
  onClose,
}: TrainingReadbackWindowProps) {
  const frameRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number>(DEFAULT_W);

  useLayoutEffect(() => {
    const node = frameRef.current;
    if (!node) return;
    const measure = () => setWidth(Math.max(node.clientWidth, MIN_W));
    measure();
    if (typeof ResizeObserver === "undefined") return;   // jsdom has no layout to observe
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const { observed, sentence } = flight;
  const { tS } = observed;
  const endOfTrack = tS[tS.length - 1];
  const plotW = width - GUTTER - PAD_R;
  const xFor = (seconds: number) => GUTTER + (seconds / endOfTrack) * plotW;
  // The ONE place a pixel becomes a time. The charts are drawn at one unit per
  // pixel, so the SVG's own offset is the x coordinate — no rect arithmetic, and
  // nothing to get wrong when the window is resized or scrolled. Measured in this
  // box's Chrome: over a child rect at user-space x = 250, `offsetX` reads 250,
  // because Chrome reports an SVG child's offset in the ROOT's user space. An
  // engine that reported it bbox-relative would jump the cursor whenever the
  // pointer crossed a band; this viewer is Cesium, so it is Chrome.
  const timeAtX = (x: number) =>
    Math.min(Math.max(((x - GUTTER) / plotW) * endOfTrack, 0), endOfTrack);

  const cursorRow = rowAt(tS, cursorS);
  const gapNow = gapAtS(flight, cursorS);
  const endedPastThePlane = flight.geometric.toGoM[flight.geometric.toGoM.length - 1] < 0;
  const lastEventS = sentence.eventTimesS[sentence.eventTimesS.length - 1];

  const traces: Record<ChartedKind, { values: number[]; unit: string; digits: number }> = {
    heading: { values: observed.courseUnwrappedDeg, unit: "° rel. course, unwrapped", digits: 1 },
    altitude: { values: observed.heightM.map((metres) => metres / FEET_TO_METERS), unit: "ft above the threshold", digits: 0 },
    speed: { values: observed.groundSpeedMps.map(metresPerSecondToKnots), unit: "kt ground speed", digits: 0 },
  };

  const chartTop = (index: number) => PLAN_H + 8 + index * CHART_H;
  const totalH = PLAN_H + 8 + CHARTED_KINDS.length * CHART_H + AXIS_H;

  // ── the plan view, at one scale on both axes so a turn looks like a turn ──
  const planX = observed.toGoM.map((metres) => -metres / 1000);
  const planY = observed.crossM.map((metres) => metres / 1000);
  const flownX = flight.geometric.toGoM.map((metres) => -metres / 1000);
  const flownY = flight.geometric.crossM.map((metres) => metres / 1000);
  // Both tracks decide the frame, or the flown one is drawn off the edge exactly
  // when it has gone somewhere the aircraft did not — which is what to look at.
  const [planXLow, planXHigh] = extent([...planX, ...flownX, 0]);
  const [planYLow, planYHigh] = extent([...planY, ...flownY, 0]);
  const planScale = Math.min(
    (plotW - 12) / (planXHigh - planXLow),
    (PLAN_H - 28) / (planYHigh - planYLow),
  );
  const planPx = (km: number) => GUTTER + 6 + (km - planXLow) * planScale;
  const planPy = (km: number) => 18 + (planYHigh - km) * planScale;

  const readout = (kind: ChartedKind) => {
    const trace = traces[kind];
    const levels = chartLevels(flight, vocabulary, kind);
    const inForce = levels.filter((item) => item.instruction.issuedS <= cursorS).pop() ?? null;
    return { trace, levels, inForce };
  };

  return createPortal(
    <div className="training-readback-backdrop">
      <div
        className="training-readback-window"
        role="dialog"
        aria-label="Read-back check"
        aria-modal="false"
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") onClose();
        }}
      >
        <header className="training-readback-head">
          <strong>Read-back check</strong>
          <span>{flight.callsign}</span>
          <span>runway {flight.runway}</span>
          <span>{flight.stratum}</span>
          <span>
            {flight.instructions.length} instructions · {flight.absorbed.length} absorbed
          </span>
          <span className="training-readback-cursor">
            t = {formatSeconds(cursorS)} s
            {gapNow === null
              ? " · the words have stopped"
              : ` · ${format(gapNow)} m apart`}
          </span>
          <button type="button" onClick={onClose} aria-label="Close the read-back check">
            ×
          </button>
        </header>

        {/* The measured element carries NO padding of its own: `clientWidth`
            includes padding, so measuring the window would draw an SVG wider
            than the box that holds it and put a scrollbar under every chart. */}
        <div className="training-readback-frame" ref={frameRef}>
        <svg
          className="training-readback-svg"
          width={width}
          height={totalH}
          viewBox={`0 0 ${width} ${totalH}`}
          onMouseMove={(event) => onCursorChange(timeAtX(event.nativeEvent.offsetX))}
          // A touch has no move before its tap, so the same reading is bound to
          // both; with a mouse the click is a no-op on a cursor already there.
          onClick={(event) => onCursorChange(timeAtX(event.nativeEvent.offsetX))}
        >
          <defs>
            <pattern id="training-readback-hatch" width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width={6} height={6} fill="rgba(226, 232, 240, 0.05)" />
              <line x1={0} y1={0} x2={0} y2={6} stroke="rgba(226, 232, 240, 0.4)" strokeWidth={1.5} />
            </pattern>
          </defs>

          {/* ── the plan view ─────────────────────────────────────────────── */}
          <text x={GUTTER} y={12} className="training-readback-title">
            plan view · along the course (km, threshold at 0) × cross-track (km, right +)
          </text>
          <line
            x1={planPx(planXLow)} x2={planPx(planXHigh)} y1={planPy(0)} y2={planPy(0)}
            className="training-readback-course"
          />
          <line
            x1={planPx(0)} x2={planPx(0)} y1={planPy(planYHigh)} y2={planPy(planYLow)}
            className="training-readback-threshold"
          />
          <polyline
            points={planX.map((km, index) => `${planPx(km)},${planPy(planY[index])}`).join(" ")}
            className="training-readback-trace"
          />
          <polyline
            points={flownX.map((km, index) => `${planPx(km)},${planPy(flownY[index])}`).join(" ")}
            className="training-readback-flown"
          />
          {/* Where each instruction was issued. The runway word is not a point on
              the track — it is the frame the others are measured in. */}
          {flight.instructions
            .filter((item) => item.kind !== "runway")
            .map((item, index) => {
              const row = rowAt(tS, item.issuedS);
              const name = `${item.kind} ${trainingWordLabel(vocabulary, item.kind, item.word)} issued at ${formatSeconds(item.issuedS)} s`;
              return (
                <circle
                  key={`issued-${index}`}
                  cx={planPx(planX[row])}
                  cy={planPy(planY[row])}
                  r={3}
                  fill={TRAINING_KIND_COLOR[item.kind]}
                  aria-label={name}
                >
                  <title>{name}</title>
                </circle>
              );
            })}
          <circle
            cx={planPx(planX[cursorRow])}
            cy={planPy(planY[cursorRow])}
            r={4.5}
            className="training-readback-cursor-dot"
          />

          {/* ── the three signals ─────────────────────────────────────────── */}
          {CHARTED_KINDS.map((kind, index) => {
            const { trace, levels, inForce } = readout(kind);
            const top = chartTop(index);
            const plotTop = top + 14;
            const plotH = CHART_H - 22;
            const flown = flownTrace(flight, kind);
            const [low, high] = extent([...trace.values, ...flown, ...levels.map((item) => item.level)]);
            const yFor = (value: number) => plotTop + ((high - value) / (high - low)) * plotH;

            return (
              <g key={kind} className="training-readback-chart">
                <text x={GUTTER} y={top + 9} className="training-readback-title">
                  {kind} — {trace.unit}
                  {inForce
                    ? ` · in force: ${trainingWordLabel(vocabulary, kind, inForce.instruction.word)}` +
                      ` (issued ${formatSeconds(inForce.instruction.issuedS)} s` +
                      `${inForce.instruction.settledS === null ? ", never settled" : `, settled ${formatSeconds(inForce.instruction.settledS)} s`})` +
                      ` · now ${format(trace.values[cursorRow], trace.digits)}`
                    : ""}
                </text>

                {/* absorbed first, so the trace and the steps read over it */}
                {flight.absorbed
                  .filter((item) => item.kind === kind)
                  .map((item, position) => (
                    <rect
                      key={`absorbed-${position}`}
                      x={xFor(Math.max(item.startS - ABSORBED_PAD_S, 0))}
                      y={plotTop}
                      width={Math.max(
                        xFor(Math.min(item.endS + ABSORBED_PAD_S, endOfTrack)) -
                          xFor(Math.max(item.startS - ABSORBED_PAD_S, 0)),
                        2,
                      )}
                      height={plotH}
                      fill="url(#training-readback-hatch)"
                      aria-label={absorbedName(item)}
                    >
                      <title>{absorbedName(item)}</title>
                    </rect>
                  ))}

                <polyline
                  points={tS.map((t, row) => `${xFor(t)},${yFor(trace.values[row])}`).join(" ")}
                  className="training-readback-trace"
                />
                {/* Truncated at the axis rather than clamped onto its edge: 39
                    of the 40 exported flights fly on past the observation, and a
                    clamp stacks all those rows on the right-hand pixel — a
                    spurious vertical line the moment one is still moving. */}
                <polyline
                  points={flight.geometric.tS
                    .map((t, step) => (t <= endOfTrack ? `${xFor(t)},${yFor(flown[step])}` : ""))
                    .filter(Boolean)
                    .join(" ")}
                  className="training-readback-flown"
                />

                {levels.map((item, position) => (
                  <g key={`level-${position}`}>
                    <line
                      x1={xFor(item.instruction.issuedS)}
                      x2={xFor(item.endS)}
                      y1={yFor(item.level)}
                      y2={yFor(item.level)}
                      stroke={TRAINING_WORD_COLOR}
                      strokeWidth={2}
                      aria-label={levelName(kind, item, vocabulary)}
                    >
                      <title>{levelName(kind, item, vocabulary)}</title>
                    </line>
                    <line
                      x1={xFor(item.instruction.issuedS)}
                      x2={xFor(item.instruction.issuedS)}
                      y1={plotTop}
                      y2={plotTop + plotH}
                      className="training-readback-issued"
                    />
                    {item.instruction.settledS === null ? null : (
                      <line
                        x1={xFor(item.instruction.settledS)}
                        x2={xFor(item.instruction.settledS)}
                        y1={plotTop}
                        y2={plotTop + plotH}
                        className="training-readback-settled"
                      />
                    )}
                  </g>
                ))}

                {/* where the sentence says it lands (D72) */}
                <line
                  x1={xFor(lastEventS)}
                  x2={xFor(lastEventS)}
                  y1={plotTop}
                  y2={plotTop + plotH}
                  className="training-readback-landed"
                />
                <line
                  x1={xFor(cursorS)}
                  x2={xFor(cursorS)}
                  y1={plotTop}
                  y2={plotTop + plotH}
                  className="training-readback-cursor-line"
                />
              </g>
            );
          })}

          {/* ── the shared time axis ──────────────────────────────────────── */}
          <line
            x1={GUTTER}
            x2={GUTTER + plotW}
            y1={totalH - AXIS_H}
            y2={totalH - AXIS_H}
            className="training-readback-axis"
          />
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
            <text
              key={`tick-${fraction}`}
              x={xFor(fraction * endOfTrack)}
              y={totalH - AXIS_H + 14}
              textAnchor="middle"
              className="training-readback-tick"
            >
              {formatSeconds(Math.round(fraction * endOfTrack))}
            </text>
          ))}
          <text x={GUTTER + plotW} y={totalH - 3} textAnchor="end" className="training-readback-tick">
            seconds from the start of the track
          </text>
        </svg>
        </div>

        <footer className="training-readback-legend">
          <span>
            Move the pointer across a chart to read it out; the plan view's dot and
            the sentence bar follow the same cursor.
          </span>
          <span>
            The second line is the sentence FLOWN BY RULE — a baseline and a
            diagnostic, never a model's answer. It starts at the aircraft's first
            row because the words say no starting point, so the two lines share a
            first point by construction: what grows between them is what the words
            did not say.{" "}
            {/* `finalGapM` is `hypot(toGo, cross)` — a distance FROM the
                threshold, never a shortfall. All 20 time-cap flights in KRDU's
                export end PAST the plane (3.8–8.9 km beyond it), so "m short"
                said the opposite of what happened. */}
            It ended {flight.geometric.endReason === "crossed-threshold"
              ? `across the threshold plane, ${format(flight.geometric.finalGapM)} m from the threshold`
              : `on the time cap, ${format(flight.geometric.finalGapM)} m from the threshold` +
                (endedPastThePlane ? ", having flown past the plane" : "")}
            , and the two were compared over {format(flight.geometric.comparedFraction * 100)}% of
            the approach (mean {format(flight.geometric.meanGapM)} m, p95{" "}
            {format(flight.geometric.gapP95M)} m).
          </span>
          <span>
            <b style={{ color: TRAINING_TRACE_COLOR }}>——</b> measured ·{" "}
            <b style={{ color: TRAINING_FLOWN_COLOR }}>——</b> flown by rule ·{" "}
            <b style={{ color: TRAINING_WORD_COLOR }}>——</b> the word in force · dotted =
            issued (yellow) and settled (green) · hatched = read but not worded ·
            dashed = where the sentence lands.
          </span>
          <span>
            Heading is plotted UNWRAPPED, and each word's step is placed on the
            trace's own turn of the circle — otherwise a flight that has turned
            through 360° reads as a misread word.
          </span>
          <span>
            The artefact stores metres and m/s; feet and knots here are the bins'
            own units, converted through the generated geokit constants.
          </span>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
