/**
 * TrainingTrafficStrip.tsx
 * ------------------------
 * THE WINDOW'S CLOCK, above the sentence bar while an aircraft of a multi-aircraft window is on screen (the Training module
 * §4.11): its time across, a row per commanded aircraft — its span as the sentence read has it (the record, or a model's
 * sample: the model's colour), its landing (▼) and, where the judge ended it for a loss of separation, a red ✕ — and one
 * row for every other aircraft of the window, replayed as recorded. A pair under its minimum is a red stretch on the
 * commanded rows it involves (VISUAL filled — the reading that ends an aircraft — IFR outlined, where only IFR has it),
 * two commanded aircraft joined. The aircraft on screen is bracketed: the sentence bar under it draws that stretch.
 *
 * The cursor is the window's time (`trainingSceneS`): click or drag across the strip to move it, click a row to put that
 * aircraft on screen, ▶ to play the window at 10× (or 1×, 30×). The judge's numbers are the exporter's; nothing is judged
 * here.
 */

import { useEffect, useLayoutEffect, useRef, useState, type PointerEvent } from "react";
import { useApp, useTrainingCursor } from "../context/AppContext";
import useMeasuredWidth from "../hooks/useMeasuredWidth";
import {
  episodesAt,
  windowOnScreen,
  windowReading,
  windowOpening,
  windowSpanS,
  type TrainingLossEpisode,
  type TrainingWindowReading,
  type TrainingWindowView,
} from "../data/trainingTraffic";
import { sentenceAxisEndS, trainingModelLabel } from "../data/trainingOverlays";
import {
  TRAINING_LOSS_COLOR,
  TRAINING_OTHER_AIRCRAFT_COLOR,
  TRAINING_TRACE_COLOR,
  trainingModelColour,
} from "../utils/trainingWordColors";

const GUTTER = 70;
const PAD_R = 12;
const ROW_H = 12;
const HEAD_H = 14;
/** A loss of separation: VISUAL filled near-opaque, IFR outlined dashed — both in the failure red, heavy enough to be
 *  seen at a glance (the user, 2026-09-30: the pale dashed outline was not). */
const LOSS_FILL_OPACITY = 0.85;
const LOSS_STROKE_W = 1.5;
const IFR_DASH = "3 2";
/** Playback speeds: × real time. */
const SPEEDS = [1, 10, 30] as const;
/** The cursor moves at most this often while playing (ms): each move re-renders the bar and the 3D points. */
const PLAY_TICK_MS = 50;

/** Seconds as "m:ss" on the window's clock. */
function clockText(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** The play button: advances the window's time from where it is, at the speed chosen, until the window's end. */
function Playback({ startS, endS, atS, onTime }: { startS: number; endS: number; atS: number; onTime: (atS: number) => void }) {
  const [playing, setPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(10);
  // the time it is at and what moves it, as of the last render: the timer reads them, and outlives the renders
  const at = useRef(atS);
  const move = useRef(onTime);
  useLayoutEffect(() => {
    at.current = atS;
    move.current = onTime;
  });
  // one timer for as long as it plays at one speed: each tick adds the time since the last, whatever the renders between
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const timer = setInterval(() => {
      const now = performance.now();
      const next = Math.min(at.current + ((now - last) / 1000) * speed, endS);
      last = now;
      at.current = next;
      move.current(next);
      if (next >= endS) setPlaying(false);
    }, PLAY_TICK_MS);
    return () => clearInterval(timer);
  }, [playing, speed, endS]);
  return (
    <span className="training-traffic-play">
      <button type="button" aria-pressed={playing} title={playing ? "pause the window" : `play the window at ${speed}×`}
        onClick={() => {
          if (!playing && at.current >= endS) onTime(startS);
          setPlaying(!playing);
        }}>
        {playing ? "❚❚" : "▶"}
      </button>
      <select aria-label="playback speed" value={speed} onChange={(event) => setSpeed(Number(event.target.value) as (typeof SPEEDS)[number])}>
        {SPEEDS.map((value) => <option key={value} value={value}>{value}×</option>)}
      </select>
    </span>
  );
}

/** A loss's text: which reading, with whom, how close against its minimum, when. */
function lossTitle(episode: TrainingLossEpisode, reading: "VISUAL" | "IFR", callsign: (id: string) => string): string {
  return `${reading}: ${episode.pair.map(callsign).join(" – ")} under the minimum from ${clockText(episode.fromS)} to ` +
    `${clockText(episode.toS)} (${episode.kinds.join(", ")}; ${episode.relation}), closest ${Math.round(episode.closestM)} m ` +
    `of ${Math.round(episode.requiredM)} m` + (episode.ended.length > 0 ? `; ended ${episode.ended.map(callsign).join(", ")}` : "");
}

function Strip({ view, reading, onScreen }: { view: TrainingWindowView; reading: TrainingWindowReading; onScreen: string }) {
  const { trainingSceneS: atS, setTrainingSceneS } = useTrainingCursor();
  const [frame, frameW] = useMeasuredWidth(400, 900);
  const { window: current, set } = view;
  const plotW = frameW - GUTTER - PAD_R;
  const spanS = windowSpanS(current, reading.tracks);
  // kept on the axis: an other aircraft's span can run past it either way
  const xFor = (seconds: number) => GUTTER + (Math.min(Math.max(seconds, spanS[0]), spanS[1]) - spanS[0]) / (spanS[1] - spanS[0]) * plotW;
  const timeAt = (x: number) => spanS[0] + Math.min(Math.max((x - GUTTER) / plotW, 0), 1) * (spanS[1] - spanS[0]);
  const rows = current.commanded.length + 1;
  const height = HEAD_H + rows * ROW_H + 4;
  const stepS = set.vocabulary.stepS;
  const callsign = (id: string) => current.commanded.find((one) => one.flight.datasetId === id)?.flight.callsign
    ?? current.others.find((other) => other.datasetId === id)?.callsign ?? id;
  const rowOf = (id: string) => current.commanded.findIndex((one) => one.flight.datasetId === id);
  const modelCss = reading.model === null ? TRAINING_TRACE_COLOR : trainingModelColour(reading.model.overlay.model);
  const focused = current.commanded.find((one) => one.flight.datasetId === onScreen)!;
  const focusedEndS = focused.rowZeroS + sentenceAxisEndS(focused.flight, stepS,
    reading.model === null ? null : reading.model.sample.aircraft[current.commanded.indexOf(focused)]);
  // every IFR stretch outlined, under the VISUAL ones filled (a pair's IFR stretch outlasts its VISUAL one: IFR's minima are
  // the larger)
  const losses = [
    ...reading.losses.ifr.episodes.map((episode) => ({ episode, reading: "IFR" as const })),
    ...reading.losses.visual.episodes.map((episode) => ({ episode, reading: "VISUAL" as const })),
  ];
  const underNow = episodesAt(reading.losses.visual, atS, stepS).length;
  // the time follows a press on the plot, never one on the callsigns (those put an aircraft on screen)
  const onPlot = (event: PointerEvent<SVGSVGElement>) => event.clientX - event.currentTarget.getBoundingClientRect().left >= GUTTER;
  const move = (event: PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    setTrainingSceneS(timeAt(event.clientX - box.left));
  };

  return (
    <div className="training-traffic-strip" ref={frame}>
      <div className="training-traffic-head">
        <Playback startS={spanS[0]} endS={spanS[1]} atS={atS} onTime={setTrainingSceneS} />
        <span title={`the window opens ${current.opensUtc}; its clock runs from its opening`}>
          Window {windowOpening(current)} · t = {clockText(atS)}
        </span>
        <span className="training-traffic-read" style={{ color: modelCss }}>
          {reading.model === null ? "as recorded" : `${trainingModelLabel(reading.model.overlay.model)} · sample ${reading.model.sample.sample + 1}`}
        </span>
        {underNow > 0 ? (
          <span className="training-traffic-alert" style={{ color: TRAINING_LOSS_COLOR }} role="status">
            {underNow} {underNow === 1 ? "pair" : "pairs"} under the minimum
          </span>
        ) : null}
      </div>
      <svg width={frameW} height={height} role="img" aria-label="The window's clock"
        onPointerDown={(event) => {
          if (!onPlot(event)) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          move(event);
        }}
        onPointerMove={(event) => {
          if (event.buttons === 1 && event.currentTarget.hasPointerCapture(event.pointerId)) move(event);
        }}>
        {/* the aircraft on screen: the stretch the sentence bar draws */}
        <rect x={xFor(focused.rowZeroS)} y={1} width={Math.max(xFor(focusedEndS) - xFor(focused.rowZeroS), 1)} height={HEAD_H - 4}
          className="training-traffic-bracket"><title>{focused.flight.callsign}: the stretch the sentence bar draws</title></rect>
        {current.commanded.map((one, at) => {
          const track = reading.tracks[at];
          const y = HEAD_H + at * ROW_H;
          const landing = reading.landings.find((item) => item.datasetId === one.flight.datasetId);
          return (
            // a press on a row — its bar or its callsign — puts it on screen: on PRESS, since a press on the plot captures the
            // pointer and the click after it goes to the capture's target, never the row
            <g key={one.flight.datasetId} className={`training-traffic-row${one === focused ? " active" : ""}`}
              onPointerDown={() => view.focus(one.flight.datasetId)}>
              <text x={GUTTER - 6} y={y + ROW_H - 3} textAnchor="end" className="training-traffic-callsign">
                {one === focused ? `▶ ${one.flight.callsign}` : one.flight.callsign}</text>
              <rect x={xFor(track.tS[0])} y={y + 3} width={Math.max(xFor(track.tS[track.tS.length - 1]) - xFor(track.tS[0]), 1)}
                height={ROW_H - 6} rx={2} fill={modelCss} opacity={one === focused ? 1 : 0.7}>
                <title>{one.flight.callsign}: from {clockText(track.tS[0])} to {clockText(track.tS[track.tS.length - 1])} — click to put it on screen</title>
              </rect>
              {landing ? (
                <text x={xFor(landing.atS)} y={y + ROW_H - 2} textAnchor="middle" className="training-traffic-landing">▼
                  <title>landed at {clockText(landing.atS)}</title></text>
              ) : null}
            </g>
          );
        })}
        {/* every other aircraft of the window, replayed as recorded, on one row */}
        <text x={GUTTER - 6} y={HEAD_H + current.commanded.length * ROW_H + ROW_H - 3} textAnchor="end" className="training-traffic-others">
          others
        </text>
        {current.others.map((other) => (
          <rect key={other.datasetId} x={xFor(other.track.tS[0])} y={HEAD_H + current.commanded.length * ROW_H + 4}
            width={Math.max(xFor(other.track.tS[other.track.tS.length - 1]) - xFor(other.track.tS[0]), 1)} height={ROW_H - 8}
            fill={TRAINING_OTHER_AIRCRAFT_COLOR[other.role]} opacity={0.5}>
            <title>{other.callsign} ({other.role}{other.category === null ? "" : `, ${other.category}`}): {clockText(other.track.tS[0])}–
              {clockText(other.track.tS[other.track.tS.length - 1])}</title>
          </rect>
        ))}
        {/* the losses: on the commanded rows each involves; two commanded aircraft joined */}
        {losses.flatMap(({ episode, reading: name }) => {
          const x0 = xFor(episode.fromS);
          const w = Math.max(xFor(episode.toS + stepS) - x0, 2);
          const rowsIn = episode.pair.map(rowOf).filter((row) => row >= 0);
          const title = lossTitle(episode, name, callsign);
          const key = `${name}-${episode.pair.join("|")}-${episode.fromS}`;
          return [
            ...rowsIn.map((row) => (
              <rect key={`${key}-${row}`} x={x0} y={HEAD_H + row * ROW_H + 1} width={w} height={ROW_H - 2}
                fill={name === "VISUAL" ? TRAINING_LOSS_COLOR : "none"} fillOpacity={LOSS_FILL_OPACITY} stroke={TRAINING_LOSS_COLOR}
                strokeWidth={LOSS_STROKE_W} strokeDasharray={name === "VISUAL" ? undefined : IFR_DASH}><title>{title}</title></rect>
            )),
            ...(rowsIn.length === 2 ? [
              <line key={`${key}-join`} x1={x0 + w / 2} x2={x0 + w / 2} y1={HEAD_H + Math.min(...rowsIn) * ROW_H + ROW_H / 2}
                y2={HEAD_H + Math.max(...rowsIn) * ROW_H + ROW_H / 2} stroke={TRAINING_LOSS_COLOR} strokeWidth={LOSS_STROKE_W}
                strokeDasharray={name === "VISUAL" ? undefined : IFR_DASH}><title>{title}</title></line>,
            ] : []),
          ];
        })}
        {/* where the judge ended an aircraft (always a commanded one): over the losses that ended it */}
        {reading.losses.visual.ended.map((ended) => (
          <text key={`ended-${ended.datasetId}`} x={xFor(ended.atS)} y={HEAD_H + (rowOf(ended.datasetId) + 1) * ROW_H}
            textAnchor="middle" fill={TRAINING_LOSS_COLOR} className="training-traffic-ended">✕
            <title>{callsign(ended.datasetId)} ended at {clockText(ended.atS)}: lost separation with {callsign(ended.with)}
              {" "}({ended.kind}, {ended.relation})</title></text>
        ))}
        <line x1={xFor(atS)} x2={xFor(atS)}
          y1={0} y2={height} className="training-sentence-cursor" />
      </svg>
    </div>
  );
}

export default function TrainingTrafficStrip() {
  const { trainingWindow, trainingSelection, trainingSource } = useApp();
  const view = windowOnScreen(trainingWindow, trainingSelection);
  if (view === null) return null;
  // a strip (and its playback) per window
  return <Strip key={trainingSelection!.clock.scope} view={view} reading={windowReading(view, trainingSource)}
    onScreen={trainingSelection!.flight.datasetId} />;
}

