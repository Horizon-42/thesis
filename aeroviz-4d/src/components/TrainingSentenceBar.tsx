/**
 * TrainingSentenceBar.tsx
 * -----------------------
 * The sentence as a picture: the six word kinds as six rows, drawn against REAL
 * TIME. Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §3.1 (T4a).
 *
 * THE HORIZONTAL AXIS IS TIME, NOT EVENT NUMBER. The sentence is an event
 * sequence (reading rule `plateau-v11`): the gaps are irregular — 26 s and 58 s
 * in the fixture, a median 44 s and a p95 of 128 s in the artefact — so evenly
 * spaced columns would draw a wrong picture of when anything was said. That is
 * the same bias the even 10 s grid was deleted for.
 *
 * TWO THINGS THE PICTURE MUST NOT HIDE, both measured on KRDU's export:
 *  • The track OUTLIVES the sentence. The last event sits a median 145 s before
 *    the end of a 326 s arrival, because the words in force there are held to the
 *    threshold. Every row's last band therefore runs to `durationS`, and the
 *    unworded tail is drawn as such on the gap row instead of being left blank.
 *  • `go-around` is a terminal class that EXISTS and is never observed here (the
 *    25 km arrival slice keeps only the final successful approach). The legend
 *    says so in those words — a reader must not take it for a word the model
 *    declines to use.
 *
 * The cursor is in FLIGHT TIME and is moved by clicking a band or an event: no
 * pixel→time arithmetic anywhere, so what it reports is the artefact's own number
 * rather than a rounded screen position. It does NOT drive `viewer.clock` yet —
 * Training loads no CZML (design V2), so there is nothing in the scene for a
 * clock to move, and the export carries no absolute epoch to anchor one to. Both
 * arrive together in T6, with the 3D tracks (V22).
 */

import { useLayoutEffect, useRef, useState } from "react";
import { useApp } from "../context/AppContext";
import TrainingReadbackWindow from "./TrainingReadbackWindow";
import { TRAINING_KIND_COLOR } from "../utils/trainingWordColors";
import {
  formatSeconds,
  TERMINAL_NEVER_OBSERVED,
  TERMINAL_WORDS,
  TRAINING_KINDS,
  TRAINING_KIND_COLUMN,
  trainingWordLabel,
  type TrainingAbsorbed,
  type TrainingFlight,
  type TrainingKind,
} from "../data/trainingSample";

// ── geometry ────────────────────────────────────────────────────────────────
// One SVG unit is one pixel: the bar is as wide as the dock and always exactly
// VIEW_H tall. A viewBox scaled to the width instead would make the bar taller on
// a wider window — tall enough here to cover the flight list it is read beside —
// and would scale the type with it.
const GUTTER = 104; // the row labels
const PAD_R = 22;
const HEAD_H = 20; // event numbers + cursor read-out
const ROW_H = 20;
const AXIS_H = 22;
const VIEW_H = HEAD_H + TRAINING_KINDS.length * ROW_H + AXIS_H;
/** Used until the element has been measured, and in jsdom, which has no layout. */
const DEFAULT_PLOT_W = 1074;
/** Below this the words cannot be read at all, so the bar scrolls instead. */
const MIN_PLOT_W = 320;

/** A band is wide enough for its word only above this many pixels; below it the
 *  word stays in the band's tooltip, which every band carries. */
const LABEL_MIN_W = 34;
/** Minimum spacing for an event number and for an axis tick label. */
const EVENT_LABEL_GAP = 13;
const TICK_LABEL_GAP = 32;

/**
 * The three kinds whose word is a TARGET the aircraft flies towards, the way a
 * clearance is. This is not decoration: KRDU's export has straight-in arrivals
 * that enter the slice at 977 m and carry the single altitude word 0 ft for the
 * whole approach — "descend to the threshold", not "it is at zero". A bar that
 * called that row "the altitude" would read as a flat track on the ground.
 */
const TARGET_KINDS: readonly TrainingKind[] = ["heading", "altitude", "speed"];

const ROW_LABEL: Record<TrainingKind, string> = {
  heading: "Heading (°)",
  altitude: "Altitude (ft)",
  speed: "Speed (kt)",
  runway: "Runway",
  duration: "Gap (s)",
  terminal: "Terminal",
};

interface Band {
  startS: number;
  endS: number;
  /** null on the gap row's unworded tail: after the last event the sentence says
   *  nothing more, and drawing a word there would invent one. */
  word: number | null;
}

/**
 * The bands of one row. Consecutive events carrying the SAME word are one band —
 * only one column changes at most events, so an unmerged row would be chopped
 * into identical pieces and read as repeated instructions.
 */
export function rowBands(flight: TrainingFlight, kind: TrainingKind): Band[] {
  const { eventTimesS, words } = flight.sentence;
  const column = TRAINING_KIND_COLUMN[kind];

  if (kind === "duration") {
    // The gap word measures the interval BEFORE its event, so it is drawn over
    // that interval. The first event has nothing before it, and the stretch after
    // the last event has no gap word at all.
    const bands: Band[] = eventTimesS.slice(1).map((time, index) => ({
      startS: eventTimesS[index],
      endS: time,
      word: words[index + 1][column],
    }));
    const lastEventS = eventTimesS[eventTimesS.length - 1];
    if (flight.durationS > lastEventS) {
      bands.push({ startS: lastEventS, endS: flight.durationS, word: null });
    }
    return bands;
  }

  const bands: Band[] = [];
  eventTimesS.forEach((time, index) => {
    const word = words[index][column];
    const endS = index + 1 < eventTimesS.length ? eventTimesS[index + 1] : flight.durationS;
    const open = bands[bands.length - 1];
    if (open && open.word === word) open.endS = endS;
    else bands.push({ startS: time, endS, word });
  });
  return bands;
}

/**
 * Which of these ascending x positions may carry a text label: greedy from the
 * left, keeping one only when it clears the last kept by `minGap`. The LAST
 * position is always kept (it is the end of the axis), evicting the previous
 * keeper if they would collide.
 *
 * This earns its keep on real data: five of the forty flights in KRDU's export
 * have events 2 s apart — 3.7 px at the default width — so their numbers and
 * tick labels would print on top of each other into a smudge. The label is
 * dropped, never the event: its hit area and its tooltip are untouched, exactly
 * as `LABEL_MIN_W` does for a band too narrow to write in.
 */
export function spacedLabels(xs: number[], minGap: number, keepLast = false): boolean[] {
  const keep = xs.map(() => false);
  let lastKept = -Infinity;
  xs.forEach((x, index) => {
    if (x - lastKept < minGap) return;
    keep[index] = true;
    lastKept = x;
  });
  if (keepLast && xs.length) {
    const end = xs.length - 1;
    if (!keep[end]) {
      for (let i = end - 1; i >= 0; i -= 1) {
        if (keep[i]) {
          if (xs[end] - xs[i] < minGap) keep[i] = false;
          break;
        }
      }
      keep[end] = true;
    }
  }
  return keep;
}

/** What an absorbed manoeuvre says, for the tooltip and the accessible name:
 *  the change the labeller measured and the rule that kept it out of the words. */
function absorbedLabel(item: TrainingAbsorbed): string {
  const change = `${item.change > 0 ? "+" : ""}${item.change.toFixed(1)}`;
  return `absorbed ${item.kind}: ${change} over ${formatSeconds(item.startS)}–${formatSeconds(item.endS)} s, not worded (${item.reason})`;
}

export default function TrainingSentenceBar() {
  const { trainingSelection } = useApp();
  const frameRef = useRef<HTMLDivElement>(null);
  const [plotW, setPlotW] = useState<number>(DEFAULT_PLOT_W);
  const [readbackOpen, setReadbackOpen] = useState<boolean>(false);
  const [notesOpen, setNotesOpen] = useState<boolean>(false);
  const [cursor, setCursor] = useState<{ flightKey: string | null; atS: number }>({
    flightKey: null,
    atS: 0,
  });

  const flightKey = trainingSelection?.flight.flightKey ?? null;

  // A new flight starts at its own beginning, and the reset happens DURING the
  // render that changes flight — in an effect it would happen after it, letting
  // one frame paint the new flight's bands under the old flight's cursor and read
  // out a time the new flight may not even have.
  if (cursor.flightKey !== flightKey) setCursor({ flightKey, atS: 0 });
  const cursorS = cursor.flightKey === flightKey ? cursor.atS : 0;
  const setCursorS = (atS: number) => setCursor({ flightKey, atS });

  // Measured before paint, so the frame in which the bar appears is already drawn
  // at the real width instead of at DEFAULT_PLOT_W.
  useLayoutEffect(() => {
    const node = frameRef.current;
    if (!node) return;
    const measure = () => setPlotW(Math.max(node.clientWidth - GUTTER - PAD_R, MIN_PLOT_W));
    measure();
    if (typeof ResizeObserver === "undefined") return;   // jsdom has no layout to observe
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [flightKey]);

  if (!trainingSelection) return null;
  const { flight, vocabulary } = trainingSelection;
  const { eventTimesS } = flight.sentence;
  const lastEventS = eventTimesS[eventTimesS.length - 1];
  const tailS = flight.durationS - lastEventS;

  const xFor = (seconds: number) => GUTTER + (seconds / flight.durationS) * plotW;
  const label = (kind: TrainingKind, word: number) => trainingWordLabel(vocabulary, kind, word);

  const terminalWords = Array.from({ length: TERMINAL_WORDS }, (_, word) =>
    label("terminal", word),
  );
  const neverObserved = TERMINAL_NEVER_OBSERVED.map((word) => label("terminal", word));

  const eventNumberShown = spacedLabels(eventTimesS.map(xFor), EVENT_LABEL_GAP);
  // The axis is ticked at every event, plus the end of the track — which is kept
  // whatever else has to go, because it is where the last band stops.
  const tickTimesS = [...eventTimesS, flight.durationS];
  const tickShown = spacedLabels(tickTimesS.map(xFor), TICK_LABEL_GAP, true);

  return (
    <section className="training-sentence-bar" aria-label="Sentence bar">
      <header className="training-sentence-head">
        <strong>{flight.callsign}</strong>
        <span>runway {flight.runway}</span>
        <span>{flight.stratum}</span>
        <span>
          {eventTimesS.length} events · {flight.instructions.length} instructions ·{" "}
          {flight.absorbed.length} absorbed
        </span>
        <span>{formatSeconds(flight.durationS)} s of track</span>
        <span className="training-sentence-cursor-readout">t = {formatSeconds(cursorS)} s</span>
        {/* The window shares THIS cursor — it is the same moment of the same
            flight, so it is one number, held here and passed down. */}
        <button
          type="button"
          className="training-sentence-readback-button"
          onClick={() => setReadbackOpen((open) => !open)}
        >
          {readbackOpen ? "Close read-back check" : "Read-back check"}
        </button>
      </header>

      <div className="training-sentence-frame" ref={frameRef}>
      <svg
        className="training-sentence-svg"
        width={GUTTER + plotW + PAD_R}
        height={VIEW_H}
        viewBox={`0 0 ${GUTTER + plotW + PAD_R} ${VIEW_H}`}
        role="img"
        aria-label={`The sentence of ${flight.callsign} on runway ${flight.runway}`}
      >
        <defs>
          {/* An absorbed manoeuvre is a real thing the words do not carry, so it
              is drawn ON its kind's row rather than in a separate legend. */}
          <pattern
            id="training-absorbed-hatch"
            width={6}
            height={6}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect width={6} height={6} fill="rgba(226, 232, 240, 0.05)" />
            <line x1={0} y1={0} x2={0} y2={6} stroke="rgba(226, 232, 240, 0.45)" strokeWidth={1.5} />
          </pattern>
        </defs>

        {/* the event lines, behind everything */}
        {eventTimesS.map((time, index) => (
          <line
            key={`event-line-${index}`}
            x1={xFor(time)}
            x2={xFor(time)}
            y1={HEAD_H - 8}
            y2={HEAD_H + TRAINING_KINDS.length * ROW_H}
            className="training-sentence-event-line"
          />
        ))}

        {/* The event numbers, each a button that moves the cursor to its own
            time. The hit areas run from midpoint to midpoint, so they cannot
            overlap however close two events are — a fixed-width box would hand a
            click on the 52 s event to the one at 54 s. */}
        {eventTimesS.map((time, index) => {
          const left = index === 0 ? GUTTER : xFor((eventTimesS[index - 1] + time) / 2);
          const right =
            index === eventTimesS.length - 1
              ? GUTTER + plotW
              : xFor((time + eventTimesS[index + 1]) / 2);
          const name = `Event ${index + 1} at ${formatSeconds(time)} s`;
          return (
            <g
              key={`event-${index}`}
              role="button"
              tabIndex={0}
              aria-label={name}
              className="training-sentence-event"
              onClick={() => setCursorS(time)}
              onKeyDown={(keyEvent) => {
                if (keyEvent.key === "Enter" || keyEvent.key === " ") setCursorS(time);
              }}
            >
              <title>{name}</title>
              <rect x={left} y={2} width={Math.max(right - left, 1)} height={HEAD_H - 6} fill="transparent" />
              {eventNumberShown[index] ? (
                <text x={xFor(time)} y={HEAD_H - 9} textAnchor="middle" className="training-sentence-event-number">
                  {index + 1}
                </text>
              ) : null}
            </g>
          );
        })}

        {TRAINING_KINDS.map((kind, row) => {
          const y = HEAD_H + row * ROW_H;
          return (
            <g key={kind}>
              <rect
                x={GUTTER}
                y={y}
                width={plotW}
                height={ROW_H}
                className={`training-sentence-row-bg${row % 2 ? " odd" : ""}`}
              />
              <text x={GUTTER - 8} y={y + ROW_H / 2 + 4} textAnchor="end" className="training-sentence-row-label">
                {ROW_LABEL[kind]}
              </text>

              {rowBands(flight, kind).map((band, index) => {
                const x = xFor(band.startS);
                const width = xFor(band.endS) - x;
                const text =
                  band.word === null ? "no word" : label(kind, band.word);
                const title =
                  band.word === null
                    ? `${formatSeconds(band.startS)}–${formatSeconds(band.endS)} s: no gap word — the sentence ends at its last event and these words are held to the threshold`
                    : `${kind}${TARGET_KINDS.includes(kind) ? " target" : ""} ${text} (word ${band.word}), ${formatSeconds(band.startS)}–${formatSeconds(band.endS)} s`;
                return (
                  <g
                    key={`${kind}-${index}`}
                    role="button"
                    tabIndex={0}
                    aria-label={title}
                    className={`training-sentence-band${band.word === null ? " unworded" : ""}`}
                    onClick={() => setCursorS(band.startS)}
                    onKeyDown={(keyEvent) => {
                      if (keyEvent.key === "Enter" || keyEvent.key === " ") setCursorS(band.startS);
                    }}
                  >
                    <title>{title}</title>
                    <rect
                      x={x + 1}
                      y={y + 4}
                      width={Math.max(width - 2, 1)}
                      height={ROW_H - 8}
                      rx={3}
                      fill={TRAINING_KIND_COLOR[kind]}
                      fillOpacity={band.word === null ? 0.06 : 0.18}
                      stroke={TRAINING_KIND_COLOR[kind]}
                      strokeOpacity={band.word === null ? 0.3 : 0.6}
                      strokeDasharray={band.word === null ? "3 3" : undefined}
                    />
                    {width >= LABEL_MIN_W ? (
                      <text
                        x={x + width / 2}
                        y={y + ROW_H / 2 + 4}
                        textAnchor="middle"
                        className="training-sentence-word"
                        fill={TRAINING_KIND_COLOR[kind]}
                      >
                        {text}
                      </text>
                    ) : null}
                  </g>
                );
              })}

              {/* the manoeuvres this kind's words did not carry */}
              {flight.absorbed
                .filter((item) => item.kind === kind)
                .map((item, index) => (
                  <g
                    key={`absorbed-${kind}-${index}`}
                    className="training-sentence-absorbed"
                    aria-label={absorbedLabel(item)}
                  >
                    <title>{absorbedLabel(item)}</title>
                    <rect
                      x={xFor(item.startS)}
                      y={y + 4}
                      width={Math.max(xFor(item.endS) - xFor(item.startS), 2)}
                      height={ROW_H - 8}
                      fill="url(#training-absorbed-hatch)"
                    />
                  </g>
                ))}
            </g>
          );
        })}

        {/* the axis: every event time, which is what the reader needs to line the
            bands up against the gap words */}
        <line
          x1={GUTTER}
          x2={GUTTER + plotW}
          y1={HEAD_H + TRAINING_KINDS.length * ROW_H}
          y2={HEAD_H + TRAINING_KINDS.length * ROW_H}
          className="training-sentence-axis"
        />
        {/* The last tick carries the unit, so the axis needs no caption beside it —
            a caption anchored to the same end simply overprinted it. */}
        {tickTimesS.map((time, index) =>
          tickShown[index] ? (
            <text
              key={`tick-${index}`}
              x={xFor(time)}
              y={HEAD_H + TRAINING_KINDS.length * ROW_H + 15}
              textAnchor={index === tickTimesS.length - 1 ? "end" : "middle"}
              className="training-sentence-tick"
            >
              {index === tickTimesS.length - 1 ? `${formatSeconds(time)} s` : formatSeconds(time)}
            </text>
          ) : null,
        )}
        <line
          x1={xFor(cursorS)}
          x2={xFor(cursorS)}
          y1={HEAD_H - 8}
          y2={HEAD_H + TRAINING_KINDS.length * ROW_H + 4}
          className="training-sentence-cursor"
        />
      </svg>
      </div>

      <footer className="training-sentence-legend">
        {/* This line stays out: without it the rows read as the measured state,
            and a straight-in carrying one altitude word "0 ft" for the whole
            approach reads as a track on the ground. The rest folds away — the
            bar is docked over the flight list, and height is what it costs. */}
        <span>
          A band is the TARGET in force — what the words send the aircraft
          towards, as a clearance does — not the measured state.
          <button
            type="button"
            className="training-sentence-notes-toggle"
            aria-expanded={notesOpen}
            onClick={() => setNotesOpen((open) => !open)}
          >
            {notesOpen ? "Fewer notes" : "More notes"}
          </button>
        </span>
        {notesOpen ? (
          <>
        <span>
          Heading is relative to the final approach course, altitude is above the
          threshold, speed is ground speed. The measured signals beside them are
          the read-back window's job.
        </span>
        <span>
          Runway words: {vocabulary.runwayIdents.join(", ")} — the runways the
          arrival manifest covers, not the airport's full runway list.
        </span>
        <span>
          {/* Both lists come from the vocabulary's own constants: a class the
              data never shows is still a class, and spelling either list out by
              hand is how one of them silently stops matching the other. */}
          Terminal words: {terminalWords.join(" / ")};{" "}
          {neverObserved.join(" and ")}{" "}
          {neverObserved.length > 1 ? "are" : "is"} never observed in this data —
          the 25 km arrival slice keeps only the final successful approach.
        </span>
        <span>
          The sentence's last event is at {formatSeconds(lastEventS)} s; the
          remaining {formatSeconds(tailS)} s carry no further word, the words in
          force being held to the threshold.
        </span>
        {flight.sentence.durationClamped > 0 ? (
          <span>
            {flight.sentence.durationClamped === 1
              ? "1 gap hit"
              : `${flight.sentence.durationClamped} gaps hit`}{" "}
            the {vocabulary.durationMaxS} s duration ceiling and was clamped: the
            gap word understates it.
          </span>
        ) : null}
        <span className="training-sentence-absorbed-key">
          Hatched = a manoeuvre read but not worded (same word / small change /
          short tail).
        </span>
          </>
        ) : null}
      </footer>

      {readbackOpen ? (
        <TrainingReadbackWindow
          flight={flight}
          vocabulary={vocabulary}
          cursorS={cursorS}
          onCursorChange={setCursorS}
          onClose={() => setReadbackOpen(false)}
        />
      ) : null}
    </section>
  );
}
