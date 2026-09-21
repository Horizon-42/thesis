/**
 * TrainingReadbackWindow.tsx
 * --------------------------
 * The interactive read-back check: the plan view in the runway's frame, and the
 * three signals the boxes judge, each with the BOX in force drawn over it.
 * Design: §3.2 of `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` (T4b).
 *
 * WHAT IT IS FOR: under `box-v2-wedge` a sentence holds if the track is inside
 * its boxes. This window is that check, row by row — the box in force, the signal
 * it judges, and in red the rows that left it.
 *
 * THERE IS NO SECOND LINE ON THESE CHARTS. The retired reading drew the sentence
 * flown by rule beside the measurement; flying a box sentence needs a
 * height-tracking executor (the replay gate), which is not built. What is drawn
 * instead is the region the words allow, which is what a box vocabulary says.
 *
 * TWO LINES PER SIGNAL, AND THE FAINT ONE IS THE AIRCRAFT. The boxes were read
 * from a moving average, so the verdict is computed on the smoothed signal and
 * that is the bright line; the raw rows are behind it, because a chart showing
 * only the smoothed line would be showing a signal nobody flew.
 *
 * THE HEADING CHART IS WRAPPED. The heading box is an interval of the wrapped
 * relative course, so the chart has to be in the same coordinate the box is — an
 * unwrapped trace would leave a box at +170° looking nowhere near a trace at
 * +190°. A flight that turns through the cut therefore jumps on this chart. The
 * SMOOTHING behind it is done on the unwrapped signal (`box-v3` changed that;
 * `box-v2-wedge` averaged the wrapped one and turned +179° and −179° into 0°),
 * so the line no longer dives to zero at the wrap — only the plot does, and only
 * by one row.
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
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  altitudeTargetM,
  eventInForce,
  formatSeconds,
  headingBoxDeg,
  speedBoxMps,
  trainingContainment,
  trainingWordBandLabel,
  type TrainingEnvelope,
  type TrainingFlight,
  type TrainingObservedColumn,
  type TrainingPrior,
  type TrainingReadingRule,
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

/** The three signals the boxes judge. The runway word names the frame they are
 *  all measured in, and the duration and terminal words are not signals at all,
 *  so neither gets a chart — they are the sentence bar's rows. */
const CHARTED_KINDS = ["heading", "altitude", "speed"] as const;
type ChartedKind = (typeof CHARTED_KINDS)[number];

/** The smoothed column each chart's verdict is computed on, and the raw one
 *  drawn behind it. */
const READ_COLUMN: Record<ChartedKind, TrainingObservedColumn> = {
  heading: "readCourseDeg",
  altitude: "readHeightM",
  speed: "readSpeedMps",
};
const RAW_COLUMN: Record<ChartedKind, TrainingObservedColumn> = {
  heading: "relCourseDeg",
  altitude: "heightM",
  speed: "groundSpeedMps",
};

/**
 * One drawn box on a chart: the interval, and the stretch of time it stands for.
 *
 * Consecutive events carrying the same word are ONE span — only one column
 * changes at most events, so an unmerged row would be a picket fence of
 * identical rectangles and would read as repeated instructions.
 */
export interface BoxSpan {
  startS: number;
  endS: number;
  word: number;
  event: number;
  lo: number;
  hi: number;
}

/**
 * The heading or speed boxes of one sentence, merged. The ALTITUDE word has no
 * constant interval — its box narrows along its segment — so it is not here:
 * `altitudeSegments` draws it from the envelope's own per-row columns.
 */
export function boxSpans(
  flight: TrainingFlight,
  vocabulary: TrainingVocabulary,
  kind: "heading" | "speed",
  words: number[][],
): BoxSpan[] {
  const { eventTimesS, holdS } = flight.sentence;
  const column = kind === "heading" ? 0 : 2;
  const box = kind === "heading" ? headingBoxDeg : speedBoxMps;
  const spans: BoxSpan[] = [];
  eventTimesS.forEach((startS, event) => {
    const word = words[event][column];
    const endS = event + 1 < eventTimesS.length ? eventTimesS[event + 1] : startS + holdS[event];
    const open = spans[spans.length - 1];
    if (open && open.word === word) {
      open.endS = endS;
      return;
    }
    const [lo, hi] = box(vocabulary, word);
    spans.push({ startS, endS, word, event, lo, hi });
  });
  return spans;
}

/**
 * One altitude segment as it is drawn: the rows it covers, the wedge over them,
 * and the target the wedge closes onto.
 *
 * The rows come from the ENVELOPE the exporter wrote, not from a second
 * derivation here: the wedge depends on the remaining path to the segment's end,
 * and two implementations of that would be two answers on one chart.
 */
export interface AltitudeSegment {
  event: number;
  word: number;
  firstRow: number;
  lastRow: number;
  targetM: number;
}

export function altitudeSegments(
  flight: TrainingFlight,
  vocabulary: TrainingVocabulary,
  words: number[][],
): AltitudeSegment[] {
  const forced = eventInForce(flight.sentence.eventTimesS, flight.observed.tS);
  const segments: AltitudeSegment[] = [];
  forced.forEach((event, row) => {
    const word = words[event][1];
    const open = segments[segments.length - 1];
    if (open && open.word === word) {
      open.lastRow = row;
      return;
    }
    segments.push({
      event,
      word,
      firstRow: row,
      lastRow: row,
      targetM: altitudeTargetM(vocabulary, word),
    });
  });
  return segments;
}

/**
 * WHERE THE WORDS CUT THE TRACK: one node per event, on the track's own rows.
 *
 * Without them the track reads as a curve rather than as a sentence — and the
 * question "which word is in force here" has no answer on screen. The events,
 * not the words of one kind: an event is the moment something changed, and
 * several kinds changing at once is one cut, not three at the same place.
 */
export function segmentNodes(
  track: { tS: number[] },
  eventTimesS: number[],
): Array<{ eventS: number; row: number; event: number }> {
  const last = track.tS[track.tS.length - 1];
  return eventTimesS
    .map((eventS, event) => ({ eventS, event, row: rowAt(track.tS, eventS) }))
    .filter((node) => node.eventS <= last);
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
  // every row. It cannot fire on this artefact, because what is constant for a
  // whole approach is the WORD, and `extent` always sees the trace too.
  if (high === low) return [low - 1, high + 1];
  const pad = (high - low) * 0.12;
  return [low - pad, high + pad];
}

/** How a chart's title says the verdict, in this vocabulary's own terms. */
export function insideReading(count: { rows: number; outside: number }): string {
  if (count.outside === 0) return `every one of the ${count.rows} rows is inside its box`;
  return `${count.rows - count.outside} of ${count.rows} rows inside — ${count.outside} outside`;
}

function format(value: number, digits = 0): string {
  return value.toFixed(digits);
}

export interface TrainingReadbackWindowProps {
  /** Which of the two drawn sentences to show. The measured track has no switch:
   *  it is what everything else is read against. */
  layers: TrainingLayers;
  flight: TrainingFlight;
  vocabulary: TrainingVocabulary;
  /** The model that said `flight.prior`, when this set carries one. */
  prior?: TrainingPrior;
  /** How the signals the boxes judge were made. The legend quotes it, because a
   *  verdict whose signal is not stated is a number with no meaning. */
  reading: TrainingReadingRule;
  cursorS: number;
  onCursorChange: (seconds: number) => void;
  onClose: () => void;
}

export default function TrainingReadbackWindow({
  layers,
  flight,
  vocabulary,
  prior,
  reading,
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

  const { observed, sentence, envelope } = flight;
  const { tS } = observed;
  const endOfTrack = tS[tS.length - 1];
  const plotW = width - GUTTER - PAD_R;
  const xFor = (seconds: number) => GUTTER + (seconds / endOfTrack) * plotW;
  // The ONE place a pixel becomes a time. The charts are drawn at one unit per
  // pixel, so the SVG's own offset is the x coordinate — no rect arithmetic, and
  // nothing to get wrong when the window is resized or scrolled.
  const timeAtX = (x: number) =>
    Math.min(Math.max(((x - GUTTER) / plotW) * endOfTrack, 0), endOfTrack);

  const cursorRow = rowAt(tS, cursorS);
  const forced = eventInForce(sentence.eventTimesS, tS);
  const said = layers.model ? flight.prior : undefined;
  const inside = trainingContainment(flight, envelope, vocabulary, sentence.words);
  const modelInside = said
    ? trainingContainment(flight, said.envelope, vocabulary, said.words)
    : undefined;

  const UNIT: Record<ChartedKind, string> = {
    heading: "° of ground track relative to the course, wrapped",
    altitude: "m above the threshold — the word is a TARGET, the box is the wedge around it",
    speed: "m/s ground speed",
  };
  const DIGITS: Record<ChartedKind, number> = { heading: 1, altitude: 0, speed: 1 };

  const chartTop = (index: number) => PLAN_H + 8 + index * CHART_H;
  const totalH = PLAN_H + 8 + CHARTED_KINDS.length * CHART_H + AXIS_H;

  // ── the plan view, at one scale on both axes so a turn looks like a turn ──
  const planX = observed.toGoM.map((metres) => -metres / 1000);
  const planY = observed.crossM.map((metres) => metres / 1000);
  // The BOXES decide the frame as much as the track does: a footprint drawn off
  // the edge is exactly the one saying the words allow somewhere the aircraft
  // did not go. The plan has no clip, so a shape outside the frame draws over
  // the charts.
  const boxX = envelope.events.flatMap((box) => box.toGoM.map((metres) => -metres / 1000));
  const boxY = envelope.events.flatMap((box) => box.crossM.map((metres) => metres / 1000));
  const [planXLow, planXHigh] = extent([...planX, ...(layers.flown ? boxX : []), 0]);
  const [planYLow, planYHigh] = extent([...planY, ...(layers.flown ? boxY : []), 0]);
  const planScale = Math.min(
    (plotW - 12) / (planXHigh - planXLow),
    (PLAN_H - 28) / (planYHigh - planYLow),
  );
  const planPx = (km: number) => GUTTER + 6 + (km - planXLow) * planScale;
  const planPy = (km: number) => 18 + (planYHigh - km) * planScale;

  // The sector's outline, in the plan's own axes. The FIRST point is the apex —
  // the aircraft's position when that word opened — so the shape fans out from
  // the track rather than sitting beside it.
  const boxPolygon = (box: TrainingEnvelope["events"][number]) =>
    box.toGoM
      .map((metres, point) => `${planPx(-metres / 1000)},${planPy(box.crossM[point] / 1000)}`)
      .join(" ");

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
          <span>{sentence.eventTimesS.length} boxes</span>
          <span className="training-readback-cursor">
            t = {formatSeconds(cursorS)} s · box {forced[cursorRow] + 1}
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
          {/* THE CHAIN OF SECTORS, under the track: one footprint per word, the
              ground the heading and speed words allow while that word stands.
              Each one FANS OUT FROM THE AIRCRAFT — a pie slice of radius
              `hold × the speed box's upper edge`, spanning the heading box — and
              not a rectangle: the corners beside the apex are ground no heading
              inside the box can reach. They are thin (over a median 4 s hold a 2°
              box opens about 17 m at its far edge), and that thinness is the
              finding, not a drawing fault: horizontally one word says almost
              nothing, and it is the accumulation over a sentence that opens the
              funnel. */}
          {(layers.flown ? envelope.events : []).map((box, index) => {
            const name =
              `box ${index + 1} at ${formatSeconds(box.eventS)} s, held ${formatSeconds(box.holdS)} s: ` +
              `heading ${format(box.headingLoDeg, 1)}…${format(box.headingHiDeg, 1)}°, ` +
              `speed ${format(box.speedLoMps, 1)}…${format(box.speedHiMps, 1)} m/s, ` +
              `${format(box.altLoM)}…${format(box.altHiM)} m — the ground it reaches is a sector ` +
              `${format(box.holdS * box.speedHiMps)} m deep, fanning out from the aircraft`;
            return (
              <polygon
                key={`plan-box-${index}`}
                className="training-readback-plan-box"
                points={boxPolygon(box)}
                fill={TRAINING_BAND_FILL}
                stroke={TRAINING_BAND_EDGE}
                strokeWidth={0.6}
                aria-label={name}
              >
                <title>{name}</title>
              </polygon>
            );
          })}
          <polyline
            points={planX.map((km, index) => `${planPx(km)},${planPy(planY[index])}`).join(" ")}
            className="training-readback-trace"
          />
          {/* Where the words cut the track. This is the only track here, so the
              nodes go on it: the aircraft's own rows at each event. */}
          {segmentNodes(observed, sentence.eventTimesS).map((node) => {
            const name = `event ${node.event + 1} at ${formatSeconds(node.eventS)} s — a new box opens here`;
            return (
              <circle
                key={`node-${node.event}`}
                className="training-readback-node"
                cx={planPx(planX[node.row])}
                cy={planPy(planY[node.row])}
                r={3.2}
                fill="none"
                stroke={TRAINING_FLOWN_COLOR}
                strokeWidth={1.6}
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
            const read = observed[READ_COLUMN[kind]];
            const raw = observed[RAW_COLUMN[kind]];
            const top = chartTop(index);
            const plotTop = top + 14;
            const plotH = CHART_H - 22;
            const spans =
              kind === "altitude" ? [] : boxSpans(flight, vocabulary, kind, sentence.words);
            const segments =
              kind === "altitude" ? altitudeSegments(flight, vocabulary, sentence.words) : [];
            const modelSpans =
              said && kind !== "altitude" ? boxSpans(flight, vocabulary, kind, said.words) : [];
            const [low, high] = extent([
              ...read,
              ...raw,
              ...spans.flatMap((span) => [span.lo, span.hi]),
              ...modelSpans.flatMap((span) => [span.lo, span.hi]),
              ...(kind === "altitude" ? [...envelope.altLoM, ...envelope.altHiM] : []),
              ...(kind === "altitude" && said ? [...said.envelope.altLoM, ...said.envelope.altHiM] : []),
            ]);
            const yFor = (value: number) => plotTop + ((high - value) / (high - low)) * plotH;
            const inForce = sentence.words[forced[cursorRow]][
              kind === "heading" ? 0 : kind === "altitude" ? 1 : 2
            ];

            return (
              <g key={kind} className="training-readback-chart">
                <text x={GUTTER} y={top + 9} className="training-readback-title">
                  {kind} — {UNIT[kind]}
                  {` · in force: ${trainingWordBandLabel(vocabulary, kind, inForce)}`}
                  {` · now ${format(read[cursorRow], DIGITS[kind])}`}
                  {` · ${insideReading(inside[kind])}`}
                </text>

                {/* THE BOX, under the traces. The heading and speed words are a
                    rectangle each; the altitude word is a wedge, because it
                    narrows as the aircraft runs out of path to its segment's
                    end. */}
                {spans.map((span, position) => {
                  const x = xFor(span.startS);
                  const name =
                    `${kind} ${trainingWordBandLabel(vocabulary, kind, span.word)} (word ${span.word}), ` +
                    `${formatSeconds(span.startS)}–${formatSeconds(span.endS)} s`;
                  return (
                    <rect
                      key={`box-${position}`}
                      className="training-readback-band"
                      x={x}
                      width={Math.max(xFor(span.endS) - x, 1)}
                      y={yFor(span.hi)}
                      height={Math.max(yFor(span.lo) - yFor(span.hi), 1)}
                      fill={TRAINING_BAND_FILL}
                      stroke={TRAINING_BAND_EDGE}
                      strokeWidth={0.75}
                      aria-label={name}
                    >
                      <title>{name}</title>
                    </rect>
                  );
                })}
                {segments.map((segment, position) => {
                  const rows: number[] = [];
                  for (let row = segment.firstRow; row <= segment.lastRow; row += 1) rows.push(row);
                  const name =
                    `altitude target ${format(segment.targetM)} m (word ${segment.word}), ` +
                    `${formatSeconds(tS[segment.firstRow])}–${formatSeconds(tS[segment.lastRow])} s — ` +
                    `the wedge closes onto ±${(vocabulary.redundancyFraction * 100).toFixed(0)} % of it`;
                  return (
                    <g key={`wedge-${position}`}>
                      <polygon
                        className="training-readback-fan"
                        points={
                          rows.map((row) => `${xFor(tS[row])},${yFor(envelope.altHiM[row])}`).join(" ") +
                          " " +
                          rows.map((row) => `${xFor(tS[row])},${yFor(envelope.altLoM[row])}`).reverse().join(" ")
                        }
                        fill={TRAINING_BAND_FILL}
                        stroke={TRAINING_BAND_EDGE}
                        strokeWidth={0.75}
                        aria-label={name}
                      >
                        <title>{name}</title>
                      </polygon>
                      {/* the TARGET itself, at the end of its segment: what the
                          word names, as opposed to the room it leaves */}
                      <line
                        className="training-readback-word"
                        x1={xFor(tS[Math.max(segment.lastRow - 1, segment.firstRow)])}
                        x2={xFor(tS[segment.lastRow])}
                        y1={yFor(segment.targetM)}
                        y2={yFor(segment.targetM)}
                        stroke={TRAINING_WORD_COLOR}
                        strokeWidth={2}
                      />
                    </g>
                  );
                })}
                {/* WHAT THE MODEL SAID, as the boxes its words would have made:
                    outlined, never filled, so the truth's box stays readable
                    underneath. */}
                {modelSpans.map((span, position) => {
                  const x = xFor(span.startS);
                  const name =
                    `the model's ${kind} box here is ${trainingWordBandLabel(vocabulary, kind, span.word)} ` +
                    `(word ${span.word})`;
                  return (
                    <rect
                      key={`model-box-${position}`}
                      className="training-readback-model"
                      x={x}
                      width={Math.max(xFor(span.endS) - x, 1)}
                      y={yFor(span.hi)}
                      height={Math.max(yFor(span.lo) - yFor(span.hi), 1)}
                      fill="none"
                      stroke={TRAINING_MODEL_COLOR}
                      strokeWidth={1}
                      strokeDasharray="3 2"
                      aria-label={name}
                    >
                      <title>{name}</title>
                    </rect>
                  );
                })}
                {said && kind === "altitude" ? (
                  <polyline
                    className="training-readback-model"
                    points={
                      tS.map((t, row) => `${xFor(t)},${yFor(said.envelope.altHiM[row])}`).join(" ") +
                      " " +
                      tS.map((t, row) => `${xFor(t)},${yFor(said.envelope.altLoM[row])}`).reverse().join(" ")
                    }
                    fill="none"
                    stroke={TRAINING_MODEL_COLOR}
                    strokeWidth={1}
                    strokeDasharray="3 2"
                  />
                ) : null}

                {/* The raw rows, behind: what the aircraft did, as opposed to
                    what the boxes were read from. */}
                <polyline
                  points={tS.map((t, row) => `${xFor(t)},${yFor(raw[row])}`).join(" ")}
                  className="training-readback-raw"
                  fill="none"
                  stroke={TRAINING_TRACE_COLOR}
                  strokeOpacity={0.35}
                  strokeWidth={1}
                />
                <polyline
                  points={tS.map((t, row) => `${xFor(t)},${yFor(read[row])}`).join(" ")}
                  className="training-readback-trace"
                />
                {/* …and again, in red, over the stretches that left the box. */}
                {runsOf(inside[kind].inside.map((ok) => !ok)).map(([first, last], position) =>
                  first === last ? (
                    <circle
                      key={`outside-${position}`}
                      className="training-readback-outside"
                      cx={xFor(tS[first])}
                      cy={yFor(read[first])}
                      r={1.8}
                      fill={TRAINING_OUTSIDE_COLOR}
                    />
                  ) : (
                    <polyline
                      key={`outside-${position}`}
                      className="training-readback-outside"
                      points={tS.slice(first, last + 1)
                        .map((t, offset) => `${xFor(t)},${yFor(read[first + offset])}`)
                        .join(" ")}
                      fill="none"
                      stroke={TRAINING_OUTSIDE_COLOR}
                      strokeWidth={1.6}
                    />
                  ),
                )}

                {/* where each box opens */}
                {segmentNodes(observed, sentence.eventTimesS).map((node) => (
                  <circle
                    key={`node-${node.event}`}
                    className="training-readback-node"
                    cx={xFor(node.eventS)}
                    cy={yFor(read[node.row])}
                    r={2.2}
                    fill="none"
                    stroke={TRAINING_FLOWN_COLOR}
                    strokeWidth={1.2}
                  />
                ))}

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
            A WORD IS AN INTERVAL, and a sentence is a chain of boxes. The check is
            containment: every row of the track has to lie inside the boxes in
            force at its moment. On this flight{" "}
            {CHARTED_KINDS.map(
              (kind) => `${kind} ${inside[kind].rows - inside[kind].outside}/${inside[kind].rows}`,
            ).join(", ")}
            .
          </span>
          <span>
            The ALTITUDE box is a wedge, not a band: the word is a target height,
            and its box is the set that target is reachable from — opening
            backwards at {vocabulary.altitudeDownDeg}° above and{" "}
            {vocabulary.altitudeUpDeg}° below, and closing onto ±
            {(vocabulary.redundancyFraction * 100).toFixed(0)} % of the target at
            the segment's end. It is asymmetric because low is the dangerous side.
            The yellow tick at the end of each wedge is the target itself.
          </span>
          <span>
            There is NO sentence flown by rule here. Flying a box sentence needs a
            height-tracking executor — the replay gate, which is not built — so
            what is drawn is the region the words allow rather than one line
            through it.
          </span>
          <span>
            A word is a box in STATE space (heading × speed × altitude), and that
            is what the vocabulary calls a bounding box. The ground it reaches is
            NOT a box: it is a sector fanning out from the aircraft, as deep as
            the hold times the speed box's upper edge and as wide as the heading
            box. The speed box's LOWER edge does not bound it — at any instant
            before the hold is up the aircraft is nearer than that. Nothing in it
            limits how fast the heading may swing inside its box, because the word
            does not: that would be an executor's rule, and there is none here.
          </span>
          {flight.prior && prior ? (
            <span>
              <b style={{ color: TRAINING_MODEL_COLOR }}>┈┈</b> WHAT THE MODEL SAID, as the boxes
              its own words would have made over this same track. It is <b>{prior.method}</b>: at
              every event the model saw the truth's own words and state up to that point and was
              asked for the next one — it did not generate this sentence, and a free run is a
              different experiment.{" "}
              {modelInside
                ? `Its boxes hold ${CHARTED_KINDS.map(
                    (kind) => `${kind} ${modelInside[kind].rows - modelInside[kind].outside}/${modelInside[kind].rows}`,
                  ).join(", ")}.`
                : ""}{" "}
              {prior.trainedOnTheseFlights === 0
                ? "It was never fitted on any of these flights."
                : `It WAS fitted on ${prior.trainedOnTheseFlights} of these flights.`}
              {flight.prior.landedAtS === null
                ? ""
                : ` It first said "landed" at ${formatSeconds(flight.prior.landedAtS)} s.`}
            </span>
          ) : null}
          <span>
            <b style={{ color: TRAINING_TRACE_COLOR }}>——</b> the signal the boxes judge ·{" "}
            faint = the raw rows · <b style={{ color: TRAINING_BAND_EDGE }}>▩</b> the box in force ·{" "}
            <b style={{ color: TRAINING_WORD_COLOR }}>——</b> the altitude target ·{" "}
            <b style={{ color: TRAINING_OUTSIDE_COLOR }}>——</b> measured, outside its box ·{" "}
            <b style={{ color: TRAINING_FLOWN_COLOR }}>○</b> where a box opens.
          </span>
          <span>
            The verdict is computed on the smoothed signals the labeller read the
            boxes from ({vocabulary.courseSmoothingS} s on the course,{" "}
            {vocabulary.smoothingS} s on speed and height; {flight.courseWindowRows} and{" "}
            {flight.signalWindowRows} rows at this flight's {flight.dtS} s step).{" "}
            {reading.courseSignal}.
          </span>
          <span>
            Heading is plotted WRAPPED, because the box is an interval of the
            wrapped course, so a flight that turns through ±180° jumps on this
            chart. The average behind it is taken on the UNWRAPPED signal and
            wrapped afterwards — {reading.courseSignal} — which is what{" "}
            <b>box-v3</b> changed: the rule before it averaged the wrapped course
            and turned +179° and −179° into 0° at the cut.
          </span>
          <span>
            Everything here is SI, because this vocabulary is: the edges were
            fitted in metres and m/s, so knots would hide the grid the words sit
            on. The boxes were produced by <i>{reading.producedBy}</i>.
          </span>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
