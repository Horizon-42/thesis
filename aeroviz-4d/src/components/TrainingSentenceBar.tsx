/**
 * TrainingSentenceBar.tsx
 * -----------------------
 * The sentence as a picture: the six word kinds as six rows, drawn against REAL
 * TIME. Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §3.1 (T4a).
 *
 * THE HORIZONTAL AXIS IS TIME, NOT EVENT NUMBER. The sentence is an event
 * sequence: the gaps are irregular — a median hold of 4 s and a p95 of 20 s in
 * the artefact — so evenly spaced columns would draw a wrong picture of when
 * anything was said.
 *
 * THE BOXES TILE THE TRACK. Under `box-v2-wedge` each box carries its own hold
 * (the duration word describes the row it sits on), and the next box opens where
 * it closes — so every row of the track has exactly one box in force and there is
 * no unworded tail. That is a change from the retired rule, where the gap word
 * measured the interval BEHIND its event and the last 145 s of a median arrival
 * carried no word at all.
 *
 * WHAT THE HEADER READS OUT IS CONTAINMENT, because that is this vocabulary's
 * criterion: a sentence holds if every row of the track is inside the boxes in
 * force at its moment. It is recomputed here from the columns rather than read
 * off the file — the parser has already refused a file whose own verdict differs.
 *
 * `go-around` is a terminal class that EXISTS and is never observed here (the
 * 25 km arrival slice keeps only the final successful approach). The legend says
 * so in those words — a reader must not take it for a word the model declines to
 * use.
 *
 * The cursor is in FLIGHT TIME and is moved by clicking a band or an event: no
 * pixel→time arithmetic anywhere, so what it reports is the artefact's own number
 * rather than a rounded screen position. It does NOT drive `viewer.clock`:
 * Training loads no CZML (design V2), and the export carries no absolute epoch to
 * anchor one to.
 */

import { useLayoutEffect, useRef, useState } from "react";
import { useApp } from "../context/AppContext";
import TrainingReadbackWindow from "./TrainingReadbackWindow";
import { TRAINING_KIND_COLOR, TRAINING_MODEL_COLOR } from "../utils/trainingWordColors";
import {
  formatSeconds,
  TERMINAL_NEVER_OBSERVED,
  TERMINAL_WORDS,
  TRAINING_BOX_KINDS,
  TRAINING_KINDS,
  TRAINING_KIND_COLUMN,
  isTrainingBoxKind,
  trainingContainment,
  trainingWordBandLabel,
  trainingWordLabel,
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
/** And wide enough for the word AND its box only above this many. */
const BAND_LABEL_MIN_W = 64;
/** Minimum spacing for an event number and for an axis tick label. */
const EVENT_LABEL_GAP = 13;
const TICK_LABEL_GAP = 32;

const ROW_LABEL: Record<TrainingKind, string> = {
  heading: "Heading (°)",
  // A target HEIGHT above the threshold, with a wedge around it — not the
  // retired vertical word, which was a flight path angle.
  altitude: "Altitude (m)",
  speed: "Speed (m/s)",
  runway: "Runway",
  duration: "Hold (s)",
  terminal: "Terminal",
};

interface Band {
  startS: number;
  endS: number;
  word: number;
}

/**
 * The bands of one row. Consecutive events carrying the SAME word are one band —
 * only one column changes at most events, so an unmerged row would be chopped
 * into identical pieces and read as repeated instructions.
 *
 * THE DURATION ROW IS NOT MERGED. Its word describes the row it sits on ("this
 * box is held for T"), so two consecutive 4 s holds are two boxes, not one 8 s
 * one, and merging them would draw a hold the sentence never says.
 */
export function rowBands(flight: TrainingFlight, kind: TrainingKind): Band[] {
  const { eventTimesS, holdS, words } = flight.sentence;
  const column = TRAINING_KIND_COLUMN[kind];
  const bands: Band[] = [];
  eventTimesS.forEach((time, index) => {
    const word = words[index][column];
    const endS = index + 1 < eventTimesS.length ? eventTimesS[index + 1] : time + holdS[index];
    const open = bands[bands.length - 1];
    if (kind !== "duration" && open && open.word === word) open.endS = endS;
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
 * This earns its keep on real data: a median hold is 4 s — a few pixels at the
 * default width — so the numbers and tick labels would print on top of each
 * other into a smudge. The label is dropped, never the event: its hit area and
 * its tooltip are untouched, exactly as `LABEL_MIN_W` does for a narrow band.
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

export default function TrainingSentenceBar() {
  const { trainingSelection, trainingLayers } = useApp();
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
  const { flight, vocabulary, reading } = trainingSelection;
  const { eventTimesS } = flight.sentence;

  const xFor = (seconds: number) => GUTTER + (seconds / flight.durationS) * plotW;
  const label = (kind: TrainingKind, word: number) => trainingWordLabel(vocabulary, kind, word);
  const bandLabel = (kind: TrainingKind, word: number) =>
    trainingWordBandLabel(vocabulary, kind, word);

  /**
   * WHAT THE MODEL SAID, drawn only where it DIFFERS from the truth.
   *
   * Agreement is the common case, so drawing every said word would paint the
   * whole bar and hide the answer. The purple strips are the disagreements; their
   * absence is agreement, and the header counts both so the eye is not left to
   * estimate it.
   */
  const said = trainingLayers.model ? flight.prior : undefined;
  const disagreements = said
    ? TRAINING_KINDS.flatMap((kind) => {
        const column = TRAINING_KIND_COLUMN[kind];
        return eventTimesS.flatMap((startS, event) => {
          if (event < said.givenEvents) return [];
          if (said.words[event][column] === flight.sentence.words[event][column]) return [];
          return [{
            kind,
            startS,
            endS: event + 1 < eventTimesS.length
              ? eventTimesS[event + 1]
              : startS + flight.sentence.holdS[event],
            word: said.words[event][column],
            truth: flight.sentence.words[event][column],
            confidence: said.confidence[event][column],
          }];
        });
      })
    : [];
  const saidWords = said
    ? (eventTimesS.length - said.givenEvents) * TRAINING_KINDS.length
    : 0;

  /**
   * THE VERDICT: how many of the track's rows sit inside the boxes in force.
   *
   * It is the criterion of this vocabulary, so it is in the header rather than
   * behind a toggle — and it is recomputed from the columns, not read off the
   * file, so what is on screen is what this code measures.
   */
  const inside = trainingContainment(flight, flight.envelope, vocabulary, flight.sentence.words);
  const modelInside = said
    ? trainingContainment(flight, said.envelope, vocabulary, said.words)
    : undefined;
  const insideReadout = TRAINING_BOX_KINDS
    .map((kind) => `${kind} ${inside[kind].rows - inside[kind].outside}/${inside[kind].rows}`)
    .join(" · ");

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
        <span>{eventTimesS.length} boxes</span>
        <span>{formatSeconds(flight.durationS)} s of track</span>
        <span
          className="training-sentence-arrival"
          title={
            "Containment is this vocabulary's criterion: a sentence holds if every row of the " +
            "track is inside the boxes in force at its moment. Measured on the smoothed signals " +
            `the boxes were read from — ${reading.courseSignal}.`
          }
        >
          inside: {insideReadout}
        </span>
        {said ? (
          <span
            className="training-sentence-model-readout"
            style={{ color: TRAINING_MODEL_COLOR }}
            title={
              "What the model said at each of this flight's events, asked from the truth's own " +
              "history and state (teacher-forced). Purple marks a word it got wrong; the model " +
              "did not generate this sentence."
            }
          >
            model: {saidWords - disagreements.length}/{saidWords} words
            {modelInside
              ? `, its boxes hold ${TRAINING_BOX_KINDS.map(
                  (kind) => `${modelInside[kind].rows - modelInside[kind].outside}/${modelInside[kind].rows}`,
                ).join("/")}`
              : ""}
            {said.landedAtS === null
              ? ""
              : `, said landed at ${formatSeconds(said.landedAtS)} s`}
          </span>
        ) : null}
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
                // The tooltip always carries the box; the drawn label drops the
                // tolerance when the band is too narrow, and the word after that.
                const text =
                  width >= BAND_LABEL_MIN_W ? bandLabel(kind, band.word) : label(kind, band.word);
                const title =
                  `${kind}${isTrainingBoxKind(kind) ? " box" : ""} ${bandLabel(kind, band.word)} ` +
                  `(word ${band.word}), ${formatSeconds(band.startS)}–${formatSeconds(band.endS)} s`;
                return (
                  <g
                    key={`${kind}-${index}`}
                    role="button"
                    tabIndex={0}
                    aria-label={title}
                    className="training-sentence-band"
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
                      fillOpacity={0.18}
                      stroke={TRAINING_KIND_COLOR[kind]}
                      strokeOpacity={0.6}
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

              {/* WHERE THE MODEL SAID SOMETHING ELSE. A strip along the band's
                  bottom edge rather than a row of its own: the bar is docked
                  over the flight list and its height is what that costs. */}
              {disagreements
                .filter((item) => item.kind === kind)
                .map((item, index) => {
                  const name =
                    `the model said ${trainingWordLabel(vocabulary, kind, item.word)} here ` +
                    `(p ${item.confidence.toFixed(2)}); the words say ${trainingWordLabel(vocabulary, kind, item.truth)}`;
                  return (
                    <g key={`said-${kind}-${index}`} aria-label={name}>
                      <title>{name}</title>
                      <rect
                        x={xFor(item.startS) + 1}
                        y={y + ROW_H - 7}
                        width={Math.max(xFor(item.endS) - xFor(item.startS) - 2, 1)}
                        height={3}
                        fill={TRAINING_MODEL_COLOR}
                      />
                    </g>
                  );
                })}
            </g>
          );
        })}

        {/* the axis: every event time, which is what the reader needs to line the
            bands up against the holds */}
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
        {/* This line stays out: without it the rows read as the measured state
            rather than as the interval the aircraft was told to stay inside. The
            rest folds away: the bar is docked over the flight list, and height is
            what it costs. */}
        <span>
          A band is the BOX in force — the interval the words allow, not the
          measured state.
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
          Heading is a range of ground track relative to the final approach
          course; speed a range of ground speed; the altitude word is a TARGET
          height above the threshold, and its box is the wedge that target is
          reachable from — wide at the start of a segment, closing onto
          ±{(vocabulary.redundancyFraction * 100).toFixed(0)} % at its end. The
          runway word is the frame the other three are measured in, the hold is
          how long this box stands, and the terminal word says whether the
          sentence ends here.
        </span>
        <span>
          The boxes TILE the track: each one is held for exactly its hold and the
          next opens where it closes, so every row has one box in force and there
          is no unworded tail.
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
          The verdict is measured on the smoothed signals the boxes were read
          from ({vocabulary.courseSmoothingS} s on the course,{" "}
          {vocabulary.smoothingS} s on speed and height): {reading.heightSignal}.
        </span>
          </>
        ) : null}
      </footer>

      {readbackOpen ? (
        <TrainingReadbackWindow
          flight={flight}
          vocabulary={vocabulary}
          prior={trainingSelection.prior}
          layers={trainingLayers}
          reading={reading}
          cursorS={cursorS}
          onCursorChange={setCursorS}
          onClose={() => setReadbackOpen(false)}
        />
      ) : null}
    </section>
  );
}
