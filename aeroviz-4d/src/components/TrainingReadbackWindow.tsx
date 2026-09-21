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
import type { TrainingLayers } from "../context/AppContext";
import {
  TRAINING_BAND_EDGE,
  TRAINING_MODEL_COLOR,
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
  type TrainingPrior,
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

/**
 * WHERE THE WORDS CUT THE FLOWN TRACK: one node per event, on the track's own
 * rows.
 *
 * The flown line is one integration with no seams in it, and everything it does
 * between two events it does because one set of words was in force. So the
 * nodes are what makes the line readable as a SENTENCE rather than as a curve —
 * a turn that begins nowhere near a node was not commanded there, it is the
 * previous word still being flown.
 *
 * The events, not the instructions: an event is the moment something changed,
 * and several kinds changing at once is one cut, not three at the same place.
 * The track starts at the first event, so the first node is its own start
 * point.
 */
export function segmentNodes(
  track: { tS: number[] },
  eventTimesS: number[],
): Array<{ eventS: number; row: number; event: number }> {
  const last = track.tS[track.tS.length - 1];
  return eventTimesS
    .map((eventS, event) => ({ eventS, event, row: rowAt(track.tS, eventS) }))
    // An event past the end of this track has no node ON it: the flown sentence
    // can stop before the aircraft did, and a node clamped to the last row would
    // claim the words cut the line at a place they never reached.
    .filter((node) => node.eventS <= last);
}

/** The MODEL's sentence on the same axis, when this set carries one. */
export function priorTrace(flight: TrainingFlight, kind: ChartedKind): number[] | null {
  const said = flight.prior;
  if (!said) return null;
  const column = said.geometric[FLOWN_COLUMN[kind]];
  return kind === "heading" ? unwrapDegrees(column) : column;
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
 * Which observed rows sat OUTSIDE the band of the word in force, and how many
 * were judged at all — one function, because the chart's red overlay and the
 * chart's percentage have to be the same reading.
 *
 * `hasTolerance: false` for a kind that carries no redundancy (the heading,
 * runway, duration and terminal words). That is NOT the same as having nothing
 * to judge: a kind whose instruction list is empty has a tolerance and no rows,
 * and saying "no tolerance on this word" there would be a false statement about
 * the vocabulary rather than about this flight (V36).
 *
 * IT MEASURES THE READING, NOT A MODEL. A low fraction says this flight's
 * plateaus or segments are coarse — the words were read off the very track being
 * judged — and nothing at all about anyone's prediction.
 *
 * Only rows a word actually covers are judged: between a plateau's `settledS`
 * and the next instruction nothing is in force to be inside of, and those rows
 * are left out of both counts rather than scored against a neighbour.
 *
 * EVERY ROW IS JUDGED AT MOST ONCE. Consecutive spans share their boundary row —
 * a vertical segment ends where the next begins, a plateau's span ends where the
 * next word is issued — so the verdicts are written into a per-row array and the
 * counts are read back off it, rather than accumulated as the spans are walked.
 * Accumulating double-counted those rows (the fixture: 134 judgements over 132
 * rows) and the overlay and the percentage then disagreed about the same flight.
 * A boundary row takes the LATER span's verdict, because that is the word in
 * force at that instant.
 */
export interface BandCoverage {
  hasTolerance: boolean;
  /** Per observed row: true where the measurement left its word's band. Rows no
   *  word covers are false — nothing to disagree with. */
  outside: boolean[];
  judged: number;
  inside: number;
}

export function bandCoverage(
  flight: TrainingFlight,
  vocabulary: TrainingVocabulary,
  kind: ChartedKind,
): BandCoverage {
  const { tS } = flight.observed;
  if (kind !== VERTICAL_KIND && kind !== "speed") {
    return { hasTolerance: false, outside: tS.map(() => false), judged: 0, inside: 0 };
  }
  // null = no word covers this row, so there is nothing to be inside of.
  const verdict: Array<boolean | null> = tS.map(() => null);
  if (kind === VERTICAL_KIND) {
    const { heightM } = flight.observed;
    for (const ramp of verticalRamps(flight, vocabulary)) {
      ramp.tS.forEach((seconds, index) => {
        const row = rowAt(tS, seconds);
        verdict[row] = heightM[row] <= ramp.lo[index] && heightM[row] >= ramp.hi[index];
      });
    }
  } else {
    const { groundSpeedMps } = flight.observed;
    for (const level of chartLevels(flight, vocabulary, kind)) {
      const tolerance = speedToleranceMps(vocabulary, level.instruction.word);
      for (let row = rowAt(tS, level.instruction.issuedS); row <= rowAt(tS, level.endS); row += 1) {
        verdict[row] = Math.abs(groundSpeedMps[row] - level.level) <= tolerance;
      }
    }
  }
  return {
    hasTolerance: true,
    outside: verdict.map((value) => value === false),
    judged: verdict.filter((value) => value !== null).length,
    inside: verdict.filter((value) => value === true).length,
  };
}

/** How the chart titles say it: the share inside, or WHY there is no share. */
export function bandCoverageReading(coverage: BandCoverage): string {
  if (!coverage.hasTolerance) return "no tolerance on this word";
  if (coverage.judged === 0) return "no worded stretch to measure";
  return `inside the band ${((coverage.inside / coverage.judged) * 100).toFixed(0)} % of the time`;
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

export function extent(values: number[]): [number, number] {
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
  /** Which of the two sentence-drawn lines to draw. The measured track has no
   *  switch: it is what everything else is read against. */
  layers: TrainingLayers;
  flight: TrainingFlight;
  vocabulary: TrainingVocabulary;
  /** The model that said `flight.prior`, when this set carries one. */
  prior?: TrainingPrior;
  /** The assumptions the flown line and its bands were drawn under. The legend
   *  quotes two of them: the height floor the fan closes on, and whether the two
   *  bands are the joint envelope (they are not). */
  geometry: TrainingGeometry;
  cursorS: number;
  onCursorChange: (seconds: number) => void;
  onClose: () => void;
}

export default function TrainingReadbackWindow({
  layers,
  flight,
  vocabulary,
  prior,
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
  const speedEdges = [flight.geometric.speedBand.low, flight.geometric.speedBand.high];
  // EVERY drawn track decides the frame, or one is drawn off the edge exactly
  // when it has gone somewhere the aircraft did not — which is what to look at.
  // The plan has no clip, so a line outside the frame draws over the charts.
  const edgeX = speedEdges.flatMap((edge) => edge.toGoM.map((metres) => -metres / 1000));
  const edgeY = speedEdges.flatMap((edge) => edge.crossM.map((metres) => metres / 1000));
  const modelX = flight.prior ? flight.prior.geometric.toGoM.map((metres) => -metres / 1000) : [];
  const modelY = flight.prior ? flight.prior.geometric.crossM.map((metres) => metres / 1000) : [];
  const [planXLow, planXHigh] = extent([...planX, ...flownX, ...edgeX, ...modelX, 0]);
  const [planYLow, planYHigh] = extent([...planY, ...flownY, ...edgeY, ...modelY, 0]);
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
    // Sorted, like `chartLevels` and `verticalRamps` are: array order is the
    // exporter's issue order today, and three call sites in one window must not
    // disagree about which word is in force because one of them assumed it.
    const inForce = flight.instructions
      .filter((item) => item.kind === kind && item.issuedS <= cursorS)
      .sort((a, b) => a.issuedS - b.issuedS)
      .pop() ?? null;
    const coverage = bandCoverage(flight, vocabulary, kind);
    return { trace, levels, ramps, inForce, coverage };
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
          {(layers.flown ? speedEdges : []).map((edge, index) => (
            <polyline
              key={`speed-edge-${index}`}
              className="training-readback-speed-edge"
              points={edge.toGoM
                .map((metres, row) => `${planPx(-metres / 1000)},${planPy(edge.crossM[row] / 1000)}`)
                .join(" ")}
              fill="none"
              stroke={TRAINING_BAND_EDGE}
              strokeWidth={1}
            />
          ))}
          {layers.flown ? (
            <polyline
              points={flownX.map((km, index) => `${planPx(km)},${planPy(flownY[index])}`).join(" ")}
              className="training-readback-flown"
            />
          ) : null}
          {/* WHAT THE MODEL SAID, flown by the same rules on the same event
              times — so the distance between this line and the orange one is the
              WORDS and nothing else. */}
          {flight.prior && layers.model ? (
            <polyline
              className="training-readback-model"
              points={flight.prior.geometric.toGoM
                .map((metres, row) => `${planPx(-metres / 1000)},${planPy(flight.prior!.geometric.crossM[row] / 1000)}`)
                .join(" ")}
              fill="none"
              stroke={TRAINING_MODEL_COLOR}
              strokeWidth={1.8}
            />
          ) : null}
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
                  className="training-readback-issued-mark"
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
          {/* The same events, marked ON THE FLOWN LINE: this is where the words
              cut the line the rules drew, which is what makes its shape readable
              as a sentence. */}
          {layers.flown
            ? segmentNodes(flight.geometric, sentence.eventTimesS).map((node) => {
                const name = `event ${node.event + 1} at ${formatSeconds(node.eventS)} s — the flown words change here`;
                return (
                  <circle
                    key={`flown-node-${node.event}`}
                    className="training-readback-node"
                    cx={planPx(flownX[node.row])}
                    cy={planPy(flownY[node.row])}
                    r={3.2}
                    fill="none"
                    stroke={TRAINING_FLOWN_COLOR}
                    strokeWidth={1.6}
                    aria-label={name}
                  >
                    <title>{name}</title>
                  </circle>
                );
              })
            : null}
          {flight.prior && layers.model
            ? segmentNodes(flight.prior.geometric, sentence.eventTimesS).map((node) => {
                const name = `event ${node.event + 1} at ${formatSeconds(node.eventS)} s — the model's words change here`;
                return (
                  <circle
                    key={`model-node-${node.event}`}
                    className="training-readback-node"
                    cx={planPx(-flight.prior!.geometric.toGoM[node.row] / 1000)}
                    cy={planPy(flight.prior!.geometric.crossM[node.row] / 1000)}
                    r={3.2}
                    fill="none"
                    stroke={TRAINING_MODEL_COLOR}
                    strokeWidth={1.6}
                    aria-label={name}
                  >
                    <title>{name}</title>
                  </circle>
                );
              })
            : null}
          <circle
            cx={planPx(planX[cursorRow])}
            cy={planPy(planY[cursorRow])}
            r={4.5}
            className="training-readback-cursor-dot"
          />

          {/* ── the three signals ─────────────────────────────────────────── */}
          {CHARTED_KINDS.map((kind, index) => {
            const { trace, levels, ramps, inForce, coverage } = readout(kind);
            const top = chartTop(index);
            const plotTop = top + 14;
            const plotH = CHART_H - 22;
            const flown = flownTrace(flight, kind);
            const model = priorTrace(flight, kind);
            const [low, high] = extent([
              ...trace.values,
              ...flown,
              ...(model ?? []),
              ...levels.map((item) => item.level),
              ...ramps.flatMap((ramp) => [...ramp.lo, ...ramp.hi]),
            ]);
            const yFor = (value: number) => plotTop + ((high - value) / (high - low)) * plotH;

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
                  {` · ${bandCoverageReading(coverage)}`}
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
                      className="training-readback-band"
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
                    className="training-readback-fan"
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
                {runsOf(coverage.outside).map(([first, last], position) =>
                  first === last ? (
                    <circle
                      key={`outside-${position}`}
                      className="training-readback-outside"
                      cx={xFor(tS[first])}
                      cy={yFor(trace.values[first])}
                      r={1.8}
                      fill={TRAINING_OUTSIDE_COLOR}
                    />
                  ) : (
                    <polyline
                      key={`outside-${position}`}
                      className="training-readback-outside"
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
                {layers.flown ? (
                  <polyline
                    points={flight.geometric.tS
                      .map((t, step) => (t <= endOfTrack ? `${xFor(t)},${yFor(flown[step])}` : ""))
                      .filter(Boolean)
                      .join(" ")}
                    className="training-readback-flown"
                  />
                ) : null}

                {/* What the vertical word SAYS, as a height: the commanded angle
                    from the segment's own anchor, and beside it the angle the
                    labeller actually fitted. The gap between the two is what
                    rounding to a word cost, in metres. */}
                {ramps.map((ramp, position) => (
                  <g key={`ramp-${position}`}>
                    <polyline
                      className="training-readback-fitted"
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
                      className="training-readback-word"
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

                {layers.flown
                  ? segmentNodes(flight.geometric, sentence.eventTimesS)
                      .filter((node) => node.eventS <= endOfTrack)
                      .map((node) => (
                        <circle
                          key={`flown-node-${node.event}`}
                          className="training-readback-node"
                          cx={xFor(node.eventS)}
                          cy={yFor(flown[node.row])}
                          r={2.6}
                          fill="none"
                          stroke={TRAINING_FLOWN_COLOR}
                          strokeWidth={1.4}
                        />
                      ))
                  : null}

                {model && layers.model ? (
                  <polyline
                    className="training-readback-model"
                    points={flight.prior!.geometric.tS
                      .map((t, step) => (t <= endOfTrack ? `${xFor(t)},${yFor(model[step])}` : ""))
                      .filter(Boolean)
                      .join(" ")}
                    fill="none"
                    stroke={TRAINING_MODEL_COLOR}
                    strokeWidth={1.6}
                  />
                ) : null}

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
            , and the two were compared over {format(flight.geometric.comparedS)} s —{" "}
            {format(flight.geometric.comparedFraction * 100)}% of the approach (mean{" "}
            {format(flight.geometric.meanGapM)} m, p95 {format(flight.geometric.gapP95M)} m).
          </span>
          {flight.prior && prior ? (
            <span>
              <b style={{ color: TRAINING_MODEL_COLOR }}>——</b> WHAT THE MODEL SAID, flown by the
              same rules on the same event times, so the distance to the orange line is the words
              and nothing else. It is <b>{prior.method}</b>: at every event the model saw the
              truth's own words and state up to that point and was asked for the next one — it did
              not generate this sentence, and a free run is a different experiment.{" "}
              {prior.trainedOnTheseFlights === 0
                ? "It was never fitted on any of these flights."
                : `It WAS fitted on ${prior.trainedOnTheseFlights} of these flights.`}
              {flight.prior.landedAtS === null
                ? ""
                : ` It first said "landed" at ${formatSeconds(flight.prior.landedAtS)} s.`}
            </span>
          ) : null}
          <span>
            <b style={{ color: TRAINING_TRACE_COLOR }}>——</b> measured ·{" "}
            <b style={{ color: TRAINING_FLOWN_COLOR }}>——</b> flown by rule ·{" "}
            <b style={{ color: TRAINING_WORD_COLOR }}>——</b> the word in force ·{" "}
            <b style={{ color: TRAINING_BAND_EDGE }}>▩</b> the band that word allows ·{" "}
            <b style={{ color: TRAINING_OUTSIDE_COLOR }}>——</b> measured, outside it · dotted =
            issued (yellow), and settled (green) on the plateau kinds — a vertical
            segment settles where the next one is issued, so it has no second
            line · hatched = read but not worded · dashed = where the sentence lands.
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
