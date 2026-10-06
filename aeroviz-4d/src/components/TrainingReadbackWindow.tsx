/**
 * TrainingReadbackWindow.tsx
 * --------------------------
 * The read-back check: one flight's sentence against its track, envelope by envelope. The charts are
 * `training/Readback*.tsx` over one shared model (`training/readbackModel.ts`):
 *
 *  • PLAN VIEW (the airport frame, one scale on both axes): the runways, the observed track, the flown path of a
 *    closed-loop reading, the words it added, the DA point, the live segment; the rows a heading band judged outside, red.
 *  • HEADING against flight time: each heading word's BAND — its target ± the heading tolerance over the rows it is
 *    judged on — with its rows outside red.
 *  • ALTITUDE against flight time: each altitude word's tube (a level above the airport elevation E, in MSL), the angle
 *    words, the DA point.
 *  • SPEED against flight time: each word's band and the "unspecified" spans.
 *
 * WHAT IS SELECTED STANDS OUT, THE REST RECEDES: yellow is the selected word alone (`column`, and its word in force at the
 * cursor) — its envelope's edge, and the rows it is in force over the track and over the one chart that plots its
 * signal; every other word's envelope fades. Hovering moves the cursor only; a click on a chart selects its column.
 *
 * EVERY SHAPE IS THE EXPORTER'S: bands, row verdicts and tubes are numbers computed in Python, every verdict the judge's
 * or the labeller's; this window draws them. A closed-loop reading judges the FLOWN path (teal) with the observed track
 * (white) beside it; the labelled reading judges the observed track.
 *
 * THE FLOWN FLIGHT's end is written below the charts — the judge's outcome, the threshold crossing and the DA check's
 * values — and THE LIVE SEGMENT (solid, blue or red by how it ended) says what it is in one line, only when it is there.
 */

import type { TrainingLayers } from "../context/AppContext";
import {
  TRAINING_CORRECTION_COLOR,
  TRAINING_DECISION_FAIL_COLOR,
  TRAINING_DECISION_PASS_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_SPEED_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
  trainingOutcomeColour,
} from "../utils/trainingWordColors";
import { closedCycleTimeS, sentenceWordAt, trainingBandLabel, wordsOutside, type TrainingColumn, type TrainingReading, type TrainingSelection } from "../data/trainingSample";
import { autopilotColour, type TrainingAutopilotSegment } from "../data/trainingAutopilot";
import { checkMark, crossingText, replayText, segmentEndText, TRAINING_OUTCOME_TAG } from "../data/trainingText";
import useMeasuredWidth from "../hooks/useMeasuredWidth";
import TrainingWindow from "./training/TrainingWindow";
import { SwatchIcon, type Swatch } from "./training/chartKit";
import NotesToggle, { NotesList } from "./training/NotesToggle";
import { readbackModel, type ReadbackModel } from "./training/readbackModel";
import { flownSentenceKind, sentenceName, TRAINING_SENTENCE_KIND_TEXT } from "../data/trainingSentenceKind";
import ReadbackPlan from "./training/ReadbackPlan";
import ReadbackHeading from "./training/ReadbackHeading";
import ReadbackAltitude from "./training/ReadbackAltitude";
import ReadbackSpeed from "./training/ReadbackSpeed";

const DEFAULT_W = 980;
const MIN_W = 420;

export interface TrainingReadbackWindowProps {
  selection: TrainingSelection;
  reading: TrainingReading;
  layers: TrainingLayers;
  cursorS: number;
  onCursorChange: (seconds: number) => void;
  /** The selected word class; its word in force at the cursor is the one drawn yellow. */
  column: TrainingColumn | null;
  onColumnChange: (column: TrainingColumn) => void;
  onClose: () => void;
  /** The picked word's segment, flown live (`trainingAutopilot`, ready); null otherwise. */
  autopilot: TrainingAutopilotSegment | null;
}

/** The colours the charts use, in one row; what each is, in its title. */
function footerSwatches(m: ReadbackModel): Array<{ key: string; swatch: Swatch; text: string; title: string }> {
  const { vocabulary } = m;
  return [
    { key: "trace", swatch: { kind: "line", colour: TRAINING_TRACE_COLOR }, text: "observed track",
      title: "the observed flight, on the 2 s rows of the data" },
    ...(m.flown ? [{ key: "flown", swatch: { kind: "line", colour: m.flownColour } as Swatch, text: "flown path",
      title: `${TRAINING_SENTENCE_KIND_TEXT[flownSentenceKind(m.flight)]}, from the first predicted step; the envelopes judge it` }] : []),
    { key: "heading", swatch: { kind: "area", colour: TRAINING_HEADING_BAND_COLOR, opacity: 0.3 }, text: "heading band",
      title: `a heading word's band: its target track ± ${vocabulary.headingToleranceDeg}° over the rows it is judged on, from ` +
        `${vocabulary.headingLeadS} s after it is said` },
    { key: "tube", swatch: { kind: "area", colour: TRAINING_TUBE_COLOR, opacity: 0.3 }, text: "altitude tube",
      title: "an altitude word's tube, re-anchored at every angle word; heights are MSL, the level is above the airport elevation E" },
    { key: "speed", swatch: { kind: "area", colour: TRAINING_SPEED_COLOR, opacity: 0.3 }, text: "speed band",
      title: "a speed word: the transition to it (dashed), then its band; grey: speed left to the pilot" },
    { key: "outside", swatch: { kind: "line", colour: TRAINING_OUTSIDE_COLOR }, text: "outside",
      title: "rows the judge counted outside a band, or an envelope whose check failed" },
    { key: "selected", swatch: { kind: "line", colour: TRAINING_WORD_COLOR }, text: "selected word",
      title: "the class chosen in the sentence bar, or by clicking a chart, at the cursor" },
    ...(m.closed ? [
      { key: "correction", swatch: { kind: "area", colour: TRAINING_CORRECTION_COLOR, opacity: 0.6 } as Swatch, text: "correction",
        title: "a word the closed-loop reading added to the labelled sentence to bring the flown path back to the observed one" },
      { key: "da", swatch: { kind: "point", colour: "none", ring: TRAINING_DECISION_PASS_COLOR } as Swatch, text: "DA point",
        title: "the decision-altitude check of the threshold crossing: green passed, red failed" },
    ] : []),
    ...(m.live ? [{ key: "autopilot", swatch: { kind: "line", colour: m.liveColour } as Swatch, text: "autopilot",
      title: "the clicked word's segment, flown live" }] : []),
  ];
}

export default function TrainingReadbackWindow(props: TrainingReadbackWindowProps) {
  const { selection, reading, cursorS, onCursorChange, onColumnChange, onClose, autopilot } = props;
  const { flight } = selection;
  const [frame, width] = useMeasuredWidth(MIN_W, DEFAULT_W);
  const m = readbackModel({ ...props, width });
  const live = autopilot;
  const liveWord = live === null ? null : sentenceWordAt(reading, live.segment.column, live.segment.row)!.event;
  const liveTitle = "The clicked word's segment, flown by the executor when it was clicked: the closed-loop flight flown again " +
    "from the first predicted step, its track, altitude and ground speed from the word on (flight time). It lies " +
    `${live === null ? 0 : live.stored.horizontalM.toFixed(3)} m horizontally and ${live === null ? 0 : live.stored.verticalM.toFixed(3)} m ` +
    "vertically from the exported flown states on the 2 s rows both have.";
  const swatches = footerSwatches(m);
  const replay = m.closed === null ? null : m.closed.replay;

  return (
    <TrainingWindow title="Read-back check" closeLabel="Close the read-back check" cursorS={cursorS} cursorRow={m.cursorRow}
      onClose={onClose}
      chips={<>
        <span>{flight.callsign}</span>
        <span>runway {flight.runway}</span>
        <span>{flight.stratum}</span>
        <span>{sentenceName(flight, reading.intervalS)}</span>
      </>}>
      <div className="training-readback-frame" ref={frame}>
        <ReadbackPlan m={m} />
        <ReadbackHeading m={m} onCursorChange={onCursorChange} onColumnChange={onColumnChange} />
        <ReadbackAltitude m={m} onCursorChange={onCursorChange} onColumnChange={onColumnChange} />
        <ReadbackSpeed m={m} onCursorChange={onCursorChange} onColumnChange={onColumnChange} />

        {replay !== null || live !== null ? (
          <div className="training-readback-slots">
            {replay !== null ? (
              <p className="training-readback-slot" aria-label="The flown flight" title={replayText(replay)}>
                <strong style={{ color: trainingOutcomeColour(replay.outcome) }}>Flown flight</strong> —{" "}
                {TRAINING_OUTCOME_TAG[replay.outcome]} at {closedCycleTimeS(m.closed!, replay.endCycle)} s
                {replay.crossing === null ? " · no threshold crossing" : ` · crossing ${crossingText(replay.crossing)}`}
                {replay.crossing === null ? null
                  : replay.crossing.decision === null ? " · no DA check" : (
                    <>
                      {" "}· DA check{" "}
                      <strong style={{ color: replay.crossing.decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR }}>
                        {replay.crossing.decision.passed ? "passed" : "failed"}
                      </strong>
                      : {Math.abs(replay.crossing.decision.rightM).toFixed(1)} m {replay.crossing.decision.rightM >= 0 ? "right" : "left"} of the
                      centreline, cone half width {replay.crossing.decision.coneHalfWidthM.toFixed(1)} m{" "}
                      {checkMark(replay.crossing.decision.lateralOk)}; {Math.abs(replay.crossing.decision.aboveGlidepathM).toFixed(1)} m{" "}
                      {replay.crossing.decision.aboveGlidepathM >= 0 ? "above" : "below"} the glidepath {checkMark(replay.crossing.decision.verticalOk)};
                      at {replay.crossing.decision.heightMslM.toFixed(0)} m MSL
                    </>
                  )}
                {/* landed but not as said: the words that left their envelopes (the outcome says the rest) */}
                {replay.outcome === "landed" && !replay.flewTheSentence && replay.envelopes !== null
                  ? ` · landed, but ${wordsOutside(replay.envelopes)} words left their envelopes` : ""}
                {replay.notReached > 0 ? ` · ${replay.notReached} words said after the landing` : ""}
              </p>
            ) : null}
            {live !== null ? (
              <p className="training-readback-slot" aria-label="The autopilot, live" title={liveTitle}>
                <strong style={{ color: autopilotColour(live) }}>Autopilot</strong> — {live.segment.column} {trainingBandLabel(liveWord!.says)} from Δ
                row {live.segment.row}{live.segment.correction ? " (a correction)" : ""} · {segmentEndText(live)} ·{" "}
                {live.stored.horizontalM.toFixed(3)} m from the exported flight
              </p>
            ) : null}
          </div>
        ) : null}
      </div>

      <footer className="training-readback-legend">
        {swatches.map((item) => (
          <span key={item.key} title={item.title}>
            <SwatchIcon swatch={item.swatch} /> {item.text}
          </span>
        ))}
        <span className="training-readback-hint">hover to read · click a chart to select its column</span>
        <NotesToggle label="What the lines and colours are">
          <NotesList items={[
            ...swatches.map((item) => ({ key: item.key, name: <><SwatchIcon swatch={item.swatch} /> {item.text}</>, text: item.title })),
            ...(live !== null ? [{ key: "autopilot-slot", name: "Autopilot", text: liveTitle }] : []),
          ]} />
        </NotesToggle>
      </footer>
    </TrainingWindow>
  );
}
