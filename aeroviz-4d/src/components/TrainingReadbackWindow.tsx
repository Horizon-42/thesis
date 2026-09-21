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
import {
  TRAINING_BAND_EDGE,
  TRAINING_BAND_FILL,
  TRAINING_FLOWN_COLOR,
  TRAINING_KIND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  formatSeconds,
  headingCentreDeg,
  speedCentreMps,
  speedToleranceMps,
  trainingWordBandLabel,
  trainingWordLabel,
  verticalCentreDeg,
  verticalToleranceDeg,
  type TrainingAbsorbed,
  type TrainingFlight,
  type TrainingGeometricColumn,
  type TrainingGeometry,
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
const CHARTED_KINDS = ["heading", "vertical", "speed"] as const;
type ChartedKind = (typeof CHARTED_KINDS)[number];

/**
 * The vertical chart plots HEIGHT, although the word is an ANGLE.
 *
 * The word is the slope of height against cumulative horizontal distance, so it
 * cannot be drawn as a level over a height trace — there is no height it names.
 * What is drawn instead is the height the word IMPLIES: from each segment's own
 * start, down at the commanded angle over the ground the aircraft actually
 * covered (`observed.pathM`, the axis the word was fitted on). Each segment
 * RE-ANCHORS, exactly as the labeller's fit does, so one segment's error does
 * not smear into the next.
 */
const VERTICAL_KIND = "vertical";

/**
 * The flown sentence's own column per chart. The heading chart plots the
 * UNWRAPPED course for the observation, and the flown track carries the wrapped
 * one — so it is unwrapped here against its own previous value, or the line jumps
 * 360° every time the rule-follower passes the cut.
 */
const FLOWN_COLUMN: Record<ChartedKind, TrainingGeometricColumn> = {
  heading: "relCourseDeg",
  vertical: "heightM",
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

/** The flown track on a chart's axis. Every chart is in SI, so only the heading
 *  needs work: the flown track stores the WRAPPED angle and the chart plots the
 *  unwrapped one, or the line drops 360° each time the rule-follower passes the
 *  cut. */
export function flownTrace(flight: TrainingFlight, kind: ChartedKind): number[] {
  const column = flight.geometric[FLOWN_COLUMN[kind]];
  if (kind === "heading") return unwrapDegrees(column);
  return column;
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

  if (kind === VERTICAL_KIND) {
    // The vertical word has no level to draw: it is a slope, and `verticalRamps`
    // is what draws it. Returning an empty list here would let a caller plot
    // nothing and see nothing wrong.
    throw new Error("the vertical word is an angle, not a level: use verticalRamps");
  }
  return issued.map((instruction, index) => {
    const endS = index + 1 < issued.length ? issued[index + 1].issuedS : endOfTrack;
    if (kind === "speed") {
      return { instruction, endS, level: speedCentreMps(vocabulary, instruction.word) };
    }
    const centre = headingCentreDeg(vocabulary, instruction.word);
    const at = rowAt(tS, instruction.settledS ?? endOfTrack);
    const trace = flight.observed.courseUnwrappedDeg[at];
    return { instruction, endS, level: centre + 360 * Math.round((trace - centre) / 360) };
  });
}

/**
 * One vertical segment drawn on the height chart: the height its word implies,
 * the two edges of the band that word allows, and the height its FITTED angle
 * implies — all sampled on the observed rows the segment spans.
 *
 * The three lines answer two different questions on one chart. Between `fitted`
 * and `centre` is the BINNING cost: what rounding the fitted angle to the
 * nearest mode is worth in metres. Between `lo` and `hi` is the DECODE slack:
 * how far an executor may sit from the commanded angle and still have obeyed.
 * They are not the same quantity, and drawing them together is the only way to
 * see which of the two is larger on a given approach (design §5.6).
 */
export interface VerticalRamp {
  instruction: TrainingInstruction;
  endS: number;
  tS: number[];
  /** The word's own angle. */
  centre: number[];
  /** The shallower edge (above) and the steeper edge (below). */
  lo: number[];
  hi: number[];
  /** The angle the labeller actually fitted, before it was rounded to a word. */
  fitted: number[];
}

/** Height after `run` metres of ground at `angleDeg`, descent POSITIVE. */
function drop(fromM: number, angleDeg: number, run: number): number {
  return fromM - Math.tan((angleDeg * Math.PI) / 180) * run;
}

/**
 * The vertical words as they are drawn: one ramp per segment, re-anchored at the
 * segment's own first row.
 *
 * The anchor is the OBSERVED height there, not the previous ramp's end. That is
 * what the labeller's piecewise fit does — the breakpoints are shared but the
 * fit is over the profile, not chained — and it is what makes each segment's
 * angle readable on its own instead of as the sum of everything before it.
 */
export function verticalRamps(
  flight: TrainingFlight,
  vocabulary: TrainingVocabulary,
): VerticalRamp[] {
  const { tS, heightM, pathM } = flight.observed;
  const endOfTrack = tS[tS.length - 1];
  const issued = flight.instructions
    .filter((item) => item.kind === VERTICAL_KIND)
    .sort((a, b) => a.issuedS - b.issuedS);

  return issued.map((instruction, index) => {
    const endS = instruction.settledS
      ?? (index + 1 < issued.length ? issued[index + 1].issuedS : endOfTrack);
    const centreDeg = verticalCentreDeg(vocabulary, instruction.word);
    const toleranceDeg = verticalToleranceDeg(vocabulary, instruction.word);
    const first = rowAt(tS, instruction.issuedS);
    const last = rowAt(tS, endS);
    const anchorH = heightM[first];
    const anchorX = pathM[first];

    const ramp: VerticalRamp = {
      instruction, endS, tS: [], centre: [], lo: [], hi: [], fitted: [],
    };
    for (let row = first; row <= last; row += 1) {
      const run = pathM[row] - anchorX;
      ramp.tS.push(tS[row]);
      ramp.centre.push(drop(anchorH, centreDeg, run));
      // Shallower is ABOVE: a smaller descent angle loses less height.
      ramp.lo.push(drop(anchorH, centreDeg - toleranceDeg, run));
      ramp.hi.push(drop(anchorH, centreDeg + toleranceDeg, run));
      ramp.fitted.push(drop(anchorH, instruction.target, run));
    }
    return ramp;
  });
}

/**
 * How much of the compared window the aircraft spent INSIDE the band of the word
 * in force, per kind — the readout the tolerance makes possible, because the
 * tolerance is precisely the rule for "did it obey".
 *
 * `null` for a kind with no tolerance: the heading word has none, so there is no
 * band to be inside and any number here would be invented (V36).
 *
 * IT MEASURES THE READING, NOT A MODEL. A low fraction says this flight's
 * plateaus or segments are coarse — the words were read off the very track being
 * judged — and nothing at all about anyone's prediction.
 */
export function insideBandFraction(
  flight: TrainingFlight,
  vocabulary: TrainingVocabulary,
  kind: ChartedKind,
): number | null {
  let inside = 0;
  let total = 0;
  if (kind === VERTICAL_KIND) {
    const { tS, heightM } = flight.observed;
    for (const ramp of verticalRamps(flight, vocabulary)) {
      ramp.tS.forEach((seconds, index) => {
        const height = heightM[rowAt(tS, seconds)];
        total += 1;
        if (height <= ramp.lo[index] && height >= ramp.hi[index]) inside += 1;
      });
    }
  } else if (kind === "speed") {
    const { tS, groundSpeedMps } = flight.observed;
    for (const level of chartLevels(flight, vocabulary, kind)) {
      const tolerance = speedToleranceMps(vocabulary, level.instruction.word);
      for (let row = rowAt(tS, level.instruction.issuedS); row <= rowAt(tS, level.endS); row += 1) {
        total += 1;
        if (Math.abs(groundSpeedMps[row] - level.level) <= tolerance) inside += 1;
      }
    }
  } else {
    return null;
  }
  return total === 0 ? null : inside / total;
}

/** The contiguous runs of `true` in a per-row flag, as [firstRow, lastRow] pairs.
 *  A run of one row still has to be drawn, so the pair is inclusive. */
export function runsOf(flags: boolean[]): Array<[number, number]> {
  const runs: Array<[number, number]> = [];
  let start: number | null = null;
  flags.forEach((flag, index) => {
    if (flag && start === null) start = index;
    if (!flag && start !== null) {
      runs.push([start, index - 1]);
      start = null;
    }
  });
  if (start !== null) runs.push([start, flags.length - 1]);
  return runs;
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
 * The unbinned value the plateau settled at, or the angle the segment was fitted
 * at, in the unit its chart is drawn in. Every chart is SI now, so this is a
 * formatter rather than a conversion — but it is still one place, because a
 * tooltip that printed the vertical target in metres would be answering a
 * different question from the one the reader asked.
 */
function targetReading(kind: ChartedKind, target: number): string {
  if (kind === "speed") return `${format(target, 1)} m/s`;
  return `${format(target, 2)}\u00b0`;
}

/** One instruction as a tooltip: the word WITH its band, where it came from, and
 *  the unbinned value it was read from. */
function instructionName(
  kind: ChartedKind,
  instruction: TrainingInstruction,
  vocabulary: TrainingVocabulary,
): string {
  return (
    `${kind} ${trainingWordBandLabel(vocabulary, kind, instruction.word)} (word ${instruction.word}), ` +
    `issued ${formatSeconds(instruction.issuedS)} s, ` +
    `read ${targetReading(kind, instruction.target)}` +
    `${instruction.clamped ? " — clamped to the edge of the vocabulary" : ""}`
  );
}

function format(value: number, digits = 0): string {
  return value.toFixed(digits);
}

export interface TrainingReadbackWindowProps {
  flight: TrainingFlight;
  vocabulary: TrainingVocabulary;
  /** The assumptions the flown line and its bands were drawn under. The legend
   *  quotes two of them: the height floor the fan closes on, and whether the two
   *  bands are the joint envelope (they are not). */
  geometry: TrainingGeometry;
  cursorS: number;
  onCursorChange: (seconds: number) => void;
  onClose: () => void;
}

export default function TrainingReadbackWindow({
  flight,
  vocabulary,
  geometry,
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

  // SI throughout, because the vocabulary is defined in SI (V30). The vertical
  // chart's unit names what is PLOTTED (height) beside what the word SAYS (an
  // angle), because they are not the same quantity.
  const traces: Record<ChartedKind, { values: number[]; unit: string; digits: number }> = {
    heading: { values: observed.courseUnwrappedDeg, unit: "° rel. course, unwrapped", digits: 1 },
    vertical: { values: observed.heightM, unit: "m above the threshold — the word is the ANGLE", digits: 0 },
    speed: { values: observed.groundSpeedMps, unit: "m/s ground speed", digits: 1 },
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
    // The vertical word is a slope, so it has ramps rather than levels; the
    // other two are values and keep their steps. Exactly one of the two lists is
    // ever non-empty.
    const levels = kind === VERTICAL_KIND ? [] : chartLevels(flight, vocabulary, kind);
    const ramps = kind === VERTICAL_KIND ? verticalRamps(flight, vocabulary) : [];
    const inForce = flight.instructions
      .filter((item) => item.kind === kind && item.issuedS <= cursorS)
      .pop() ?? null;
    const insideFraction = insideBandFraction(flight, vocabulary, kind);
    return { trace, levels, ramps, inForce, insideFraction };
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
          {/* The speed word's tolerance, flown: the same sentence with every
              speed word at each end of its ±3 %. In the PLAN it is nearly the
              nominal line — a wider turn radius and a few hundred metres of lead
              — because the speed tolerance's real cost is arrival TIME, which a
              static plan cannot show. It is drawn so that is visible rather than
              assumed; the number is in the sentence bar's header (design §5.6). */}
          {[flight.geometric.speedBand.low, flight.geometric.speedBand.high].map((edge, index) => (
            <polyline
              key={`speed-edge-${index}`}
              points={edge.toGoM
                .map((metres, row) => `${planPx(-metres / 1000)},${planPy(edge.crossM[row] / 1000)}`)
                .join(" ")}
              fill="none"
              stroke={TRAINING_BAND_EDGE}
              strokeWidth={1}
            />
          ))}
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
            const { trace, levels, ramps, inForce, insideFraction } = readout(kind);
            const top = chartTop(index);
            const plotTop = top + 14;
            const plotH = CHART_H - 22;
            const flown = flownTrace(flight, kind);
            const [low, high] = extent([
              ...trace.values,
              ...flown,
              ...levels.map((item) => item.level),
              ...ramps.flatMap((ramp) => [...ramp.lo, ...ramp.hi]),
            ]);
            const yFor = (value: number) => plotTop + ((high - value) / (high - low)) * plotH;

            // Where the aircraft sat OUTSIDE the band of the word in force. This
            // is the one mark on the chart that is a disagreement rather than a
            // drawing, so it is computed from the same bands that are drawn.
            const outside = trace.values.map(() => false);
            for (const ramp of ramps) {
              ramp.tS.forEach((seconds, position) => {
                const row = rowAt(tS, seconds);
                outside[row] = trace.values[row] > ramp.lo[position] || trace.values[row] < ramp.hi[position];
              });
            }
            for (const level of levels) {
              const tolerance = kind === "speed" ? speedToleranceMps(vocabulary, level.instruction.word) : null;
              if (tolerance === null) continue;
              for (let row = rowAt(tS, level.instruction.issuedS); row <= rowAt(tS, level.endS); row += 1) {
                outside[row] = Math.abs(trace.values[row] - level.level) > tolerance;
              }
            }

            return (
              <g key={kind} className="training-readback-chart">
                <text x={GUTTER} y={top + 9} className="training-readback-title">
                  {kind} — {trace.unit}
                  {inForce
                    ? ` · in force: ${trainingWordBandLabel(vocabulary, kind, inForce.word)}` +
                      ` (issued ${formatSeconds(inForce.issuedS)} s` +
                      `${inForce.settledS === null ? ", never settled" : `, settled ${formatSeconds(inForce.settledS)} s`})` +
                      ` · now ${format(trace.values[cursorRow], trace.digits)}`
                    : ""}
                  {insideFraction === null
                    ? " · no tolerance on this word"
                    : ` · inside the band ${(insideFraction * 100).toFixed(0)} % of the time`}
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

                {/* The band a word allows, drawn UNDER the traces: the level
                    kinds get a rectangle, the vertical word a fan that opens
                    with the ground covered (design §5.6). */}
                {levels.map((item, position) => {
                  const tolerance = kind === "speed" ? speedToleranceMps(vocabulary, item.instruction.word) : null;
                  if (tolerance === null) return null;
                  const x = xFor(item.instruction.issuedS);
                  return (
                    <rect
                      key={`band-${position}`}
                      x={x}
                      width={Math.max(xFor(item.endS) - x, 1)}
                      y={yFor(item.level + tolerance)}
                      height={Math.max(yFor(item.level - tolerance) - yFor(item.level + tolerance), 1)}
                      fill={TRAINING_BAND_FILL}
                    />
                  );
                })}
                {ramps.map((ramp, position) => (
                  <polygon
                    key={`fan-${position}`}
                    points={
                      ramp.tS.map((t, row) => `${xFor(t)},${yFor(ramp.lo[row])}`).join(" ") +
                      " " +
                      ramp.tS.map((t, row) => `${xFor(t)},${yFor(ramp.hi[row])}`).reverse().join(" ")
                    }
                    fill={TRAINING_BAND_FILL}
                    stroke={TRAINING_BAND_EDGE}
                    strokeWidth={0.75}
                  />
                ))}

                <polyline
                  points={tS.map((t, row) => `${xFor(t)},${yFor(trace.values[row])}`).join(" ")}
                  className="training-readback-trace"
                />
                {/* …and again, in red, over the stretches that left the band. */}
                {runsOf(outside).map(([first, last], position) =>
                  first === last ? (
                    <circle
                      key={`outside-${position}`}
                      cx={xFor(tS[first])}
                      cy={yFor(trace.values[first])}
                      r={1.8}
                      fill={TRAINING_OUTSIDE_COLOR}
                    />
                  ) : (
                    <polyline
                      key={`outside-${position}`}
                      points={tS.slice(first, last + 1)
                        .map((t, offset) => `${xFor(t)},${yFor(trace.values[first + offset])}`)
                        .join(" ")}
                      fill="none"
                      stroke={TRAINING_OUTSIDE_COLOR}
                      strokeWidth={1.6}
                    />
                  ),
                )}
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

                {/* What the vertical word SAYS, as a height: the commanded angle
                    from the segment's own anchor, and beside it the angle the
                    labeller actually fitted. The gap between the two is what
                    rounding to a word cost, in metres. */}
                {ramps.map((ramp, position) => (
                  <g key={`ramp-${position}`}>
                    <polyline
                      points={ramp.tS.map((t, row) => `${xFor(t)},${yFor(ramp.fitted[row])}`).join(" ")}
                      fill="none"
                      stroke={TRAINING_TRACE_COLOR}
                      strokeWidth={0.9}
                      strokeDasharray="2 3"
                      aria-label={`fitted ${targetReading(kind, ramp.instruction.target)} over ${formatSeconds(ramp.instruction.issuedS)}–${formatSeconds(ramp.endS)} s`}
                    >
                      <title>
                        {`fitted ${targetReading(kind, ramp.instruction.target)} — the angle before it was rounded to a word`}
                      </title>
                    </polyline>
                    <polyline
                      points={ramp.tS.map((t, row) => `${xFor(t)},${yFor(ramp.centre[row])}`).join(" ")}
                      fill="none"
                      stroke={TRAINING_WORD_COLOR}
                      strokeWidth={2}
                      aria-label={instructionName(kind, ramp.instruction, vocabulary)}
                    >
                      <title>{instructionName(kind, ramp.instruction, vocabulary)}</title>
                    </polyline>
                    <line
                      x1={xFor(ramp.instruction.issuedS)}
                      x2={xFor(ramp.instruction.issuedS)}
                      y1={plotTop}
                      y2={plotTop + plotH}
                      className="training-readback-issued"
                    />
                  </g>
                ))}

                {levels.map((item, position) => (
                  <g key={`level-${position}`}>
                    <line
                      x1={xFor(item.instruction.issuedS)}
                      x2={xFor(item.endS)}
                      y1={yFor(item.level)}
                      y2={yFor(item.level)}
                      stroke={TRAINING_WORD_COLOR}
                      strokeWidth={2}
                      aria-label={instructionName(kind, item.instruction, vocabulary)}
                    >
                      <title>{instructionName(kind, item.instruction, vocabulary)}</title>
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
            <b style={{ color: TRAINING_WORD_COLOR }}>——</b> the word in force ·{" "}
            <b style={{ color: TRAINING_BAND_EDGE }}>▩</b> the band that word allows ·{" "}
            <b style={{ color: TRAINING_OUTSIDE_COLOR }}>——</b> measured, outside it · dotted =
            issued (yellow) and settled (green) · hatched = read but not worded ·
            dashed = where the sentence lands.
          </span>
          <span>
            A WORD IS A BAND, not a point: the vertical and speed words each carry
            a tolerance, and an executor inside it has obeyed. The vertical band
            is a fan because its slack accumulates into HEIGHT with the ground
            covered; it closes at {format(geometry.heightFloorM)} m because the
            rule-follower levels at the threshold rather than flying through it —
            that is the executor's floor, not the vocabulary's. The two bands are
            {geometry.bandsAreJoint ? " " : " NOT "}
            the joint envelope: each is one kind's slack with the other at its
            centre, which is what makes them attributable.
          </span>
          <span>
            On the vertical chart the dashed grey line is the angle the labeller
            FITTED and the yellow one is the word it was rounded to: between them
            is what binning cost, in metres, while the fan's width is what
            decoding is allowed to cost. Each segment re-anchors on the observed
            height, as the fit does.
          </span>
          <span>
            Heading is plotted UNWRAPPED, and each word's step is placed on the
            trace's own turn of the circle — otherwise a flight that has turned
            through 360° reads as a misread word. It carries no tolerance, so it
            gets no band.
          </span>
          <span>
            Everything here is SI, because this vocabulary is: the speed words
            were fitted in m/s and rounded to 1 m/s, so knots would hide the
            grid the words sit on.
          </span>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
