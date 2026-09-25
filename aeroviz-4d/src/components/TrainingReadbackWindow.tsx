/**
 * TrainingReadbackWindow.tsx
 * --------------------------
 * The read-back check: one flight's sentence against its track, envelope by envelope. Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`. The charts are `training/Readback*.tsx` over one shared model
 * (`training/readbackModel.ts`):
 *
 *  • PLAN VIEW (the airport frame, one scale on both axes): the runways, the capture corridor and turn, the track, the
 *    words' issues, the clearance, the capture, the end of the sentence; the rows a heading band judged outside, red.
 *  • HEADING against time: each heading word's BAND — its target ± the heading tolerance over the rows it is judged on
 *    — with its rows outside red; the capture turn onto the course; the course band after the capture.
 *  • ALTITUDE against the horizontal distance flown — the axis the tubes are defined on: the tubes, the angle words that
 *    re-anchor them, the runway's elevation.
 *  • SPEED against time: each word's transition and band, the "unspecified" spans.
 *
 * WHAT IS SELECTED STANDS OUT, THE REST RECEDES: yellow is the selected word alone (`column`, and its word in force at the
 * cursor) — its envelope's edge, and the rows it is in force over the track and over the one chart that plots its
 * signal; every other word's envelope fades. Hovering moves the cursor only; a click on a chart selects its column.
 *
 * EVERY SHAPE IS THE EXPORTER'S: bands, row verdicts and tubes are numbers computed in Python, every verdict the
 * labeller's; this window draws them. The smoothed signal is the bright line (what the labeller read), the raw rows
 * faint behind it, red the rows counted outside.
 *
 * THE EXECUTOR'S REPLAY (`executor`, dashed teal) runs on its OWN clock and distance flown — it flies at its own pace,
 * so the axes hold both and nothing is aligned; its heading bands are drawn from where IT was told each word. THE LIVE
 * SEGMENT (`autopilot`, solid, blue or red by its verdict) starts where the observed aircraft was when its word was said,
 * on the flight's clock, so its lines begin on the observed ones and part from them. Each says what it is in one line
 * below the charts, only when it is there.
 */

import type { TrainingLayers } from "../context/AppContext";
import {
  TRAINING_CAPTURE_TURN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_RAW_COLOR,
  TRAINING_SPEED_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  TRAINING_COLUMN_INDEX,
  type TrainingCandidate,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingVocabulary,
} from "../data/trainingSample";
import type { TrainingExecutorFlight } from "../data/trainingOverlays";
import { autopilotColour, type TrainingAutopilotSegment } from "../data/trainingAutopilot";
import { TRAINING_OUTCOME_TAG, TRAINING_VERDICT_TEXT } from "../data/trainingText";
import useMeasuredWidth from "../hooks/useMeasuredWidth";
import TrainingWindow from "./training/TrainingWindow";
import { SwatchIcon, type Swatch } from "./training/chartKit";
import NotesToggle, { NotesList } from "./training/NotesToggle";
import { readbackModel, type ReadbackModel } from "./training/readbackModel";
import ReadbackPlan from "./training/ReadbackPlan";
import ReadbackHeading from "./training/ReadbackHeading";
import ReadbackAltitude from "./training/ReadbackAltitude";
import ReadbackSpeed from "./training/ReadbackSpeed";

const DEFAULT_W = 980;
const MIN_W = 420;

export interface TrainingReadbackWindowProps {
  flight: TrainingFlight;
  vocabulary: TrainingVocabulary;
  candidates: TrainingCandidate[];
  layers: TrainingLayers;
  cursorS: number;
  onCursorChange: (seconds: number) => void;
  /** The selected word class; its word in force at the cursor is the one drawn yellow. */
  column: TrainingColumn | null;
  onColumnChange: (column: TrainingColumn) => void;
  onClose: () => void;
  /** The executor's replay of this flight, when its overlay is on; null otherwise. */
  executor: TrainingExecutorFlight | null;
  /** The picked word's segment, flown live (`trainingAutopilot`, ready); null otherwise. */
  autopilot: TrainingAutopilotSegment | null;
}

/** The executor's replay, in one line (null when it is off); what its lines are, in the title. */
function replaySlot(m: ReadbackModel, executor: TrainingExecutorFlight): { text: string; title: string } {
  if (!executor.flown) return { text: `not flown: ${executor.group}.`, title: "the replay does not fly this flight" };
  const outcome = `${TRAINING_OUTCOME_TAG[executor.outcome]} on ${executor.group}`;
  if (m.flownTrack === null) return { text: `${outcome} within its first step: no flown track to draw.`, title: outcome };
  return {
    text: `${outcome}; dashed teal, on its own clock`,
    title: "Its flown track, and its heading, altitude and ground speed on its own clock and its own distance flown: it " +
      "flies at its own pace from row 0, each word said where the observed aircraft heard it, so its lines do not line up " +
      "with the observed ones. Its heading words' bands are the teal outlines, from where IT was told each word plus the " +
      "lead, its rows outside them red (dashed: on the flown track as its judge read it); its other words are judged on " +
      "envelopes re-drawn from where it was told them — the sentence bar's dots.",
  };
}

/** The colours the charts use, in one row; what each is, in its title. */
function footerSwatches(m: ReadbackModel): Array<{ key: string; swatch: Swatch; text: string; title: string }> {
  const { vocabulary } = m;
  return [
    { key: "trace", swatch: { kind: "line", colour: TRAINING_TRACE_COLOR }, text: "smoothed signal",
      title: `what the labeller read: track ${vocabulary.smoothingS.track} s, altitude ${vocabulary.smoothingS.altitude} s, ` +
        `speed ${vocabulary.smoothingS.speed} s; the raw rows are the faint line behind it` },
    { key: "raw", swatch: { kind: "line", colour: TRAINING_RAW_COLOR }, text: "raw", title: "the raw rows" },
    { key: "heading", swatch: { kind: "area", colour: TRAINING_HEADING_BAND_COLOR, opacity: 0.3 }, text: "heading band",
      title: "a heading word's band: its target ± the tolerance over the rows it is judged on (the panel's ⓘ has the numbers)" },
    { key: "capture", swatch: { kind: "area", colour: TRAINING_CAPTURE_TURN_COLOR, opacity: 0.2, dash: "3 2" }, text: "capture turn",
      title: "from the clearance onto the course" },
    { key: "corridor", swatch: { kind: "area", colour: TRAINING_CORRIDOR_COLOR, opacity: 0.3 }, text: "corridor",
      title: "the capture corridor, and the course band after the capture" },
    { key: "tube", swatch: { kind: "area", colour: TRAINING_TUBE_COLOR, opacity: 0.3 }, text: "altitude tube",
      title: "an altitude word's tube, re-anchored at every angle word" },
    { key: "speed", swatch: { kind: "area", colour: TRAINING_SPEED_COLOR, opacity: 0.3 }, text: "speed band",
      title: "a speed word: its transition, then its band; grey: \"unspecified\", the pilot's own speed" },
    { key: "outside", swatch: { kind: "line", colour: TRAINING_OUTSIDE_COLOR }, text: "outside",
      title: "rows the labeller counted outside, or an envelope whose check failed" },
    { key: "selected", swatch: { kind: "line", colour: TRAINING_WORD_COLOR }, text: "selected word",
      title: "the class chosen in the sentence bar, or by clicking a chart, at the cursor" },
    ...(m.flownTrack ? [{ key: "executor", swatch: { kind: "line", colour: TRAINING_EXECUTOR_COLOR, dash: "5 3" } as Swatch,
      text: "executor replay", title: "the executor's replay, on its own clock" }] : []),
    ...(m.live ? [{ key: "autopilot", swatch: { kind: "line", colour: m.liveColour } as Swatch, text: "autopilot",
      title: "the picked word's segment, flown live" }] : []),
  ];
}

export default function TrainingReadbackWindow(props: TrainingReadbackWindowProps) {
  const { flight, cursorS, onCursorChange, onColumnChange, onClose, executor } = props;
  const [frame, width] = useMeasuredWidth(MIN_W, DEFAULT_W);
  const m = readbackModel({ ...props, width });
  const replay = executor === null ? null : replaySlot(m, executor);
  // the live answer is named whether or not it has a line to draw (`m.live`)
  const live = props.autopilot;
  const liveWord = live === null ? null
    : m.label(live.segment.column, flight.words.inForce[TRAINING_COLUMN_INDEX[live.segment.column]][live.segment.row]);
  const liveTitle = "The picked word's segment, flown by the executor when it was picked, from the observed state where the " +
    "word was said — its track, altitude (against the distance flown, from the observed aircraft's there) and ground speed " +
    "on the flight's own clock." + (m.liveBand ? " Its heading band is the outline on the heading chart, as its judge read " +
    "the flown segment, its rows outside red." : "");
  const swatches = footerSwatches(m);

  return (
    <TrainingWindow title="Read-back check" closeLabel="Close the read-back check" cursorS={cursorS} cursorRow={m.cursorRow}
      onClose={onClose}
      chips={<>
        <span>{flight.callsign}</span>
        <span>runway {flight.runway}</span>
        <span>{flight.stratum}</span>
      </>}>
      <div className="training-readback-frame" ref={frame}>
        <ReadbackPlan m={m} />
        <ReadbackHeading m={m} onCursorChange={onCursorChange} onColumnChange={onColumnChange} />
        <ReadbackAltitude m={m} onCursorChange={onCursorChange} onColumnChange={onColumnChange} />
        <ReadbackSpeed m={m} onCursorChange={onCursorChange} onColumnChange={onColumnChange} />

        {replay !== null || live !== null ? (
          <div className="training-readback-slots">
            {replay !== null ? (
              <p className="training-readback-slot" aria-label="Executor replay" title={replay.title}>
                <strong style={{ color: TRAINING_EXECUTOR_COLOR }}>Executor replay</strong> — {replay.text}
              </p>
            ) : null}
            {live !== null ? (
              <p className="training-readback-slot" aria-label="The autopilot, live" title={liveTitle}>
                <strong style={{ color: autopilotColour(live) }}>Autopilot</strong> — {live.segment.column} {liveWord} from step{" "}
                {live.segment.row} · {TRAINING_VERDICT_TEXT[live.word.status]} ·{" "}
                {m.live === null ? "no line to draw: it ended in its first cycle"
                  : `solid ${live.word.status === "outside" ? "red" : "blue"}`}
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
            ...(replay !== null ? [{ key: "replay-slot", name: "Executor replay", text: replay.title }] : []),
            ...(live !== null ? [{ key: "autopilot-slot", name: "Autopilot", text: liveTitle }] : []),
          ]} />
        </NotesToggle>
      </footer>
    </TrainingWindow>
  );
}
