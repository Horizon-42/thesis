/**
 * CursorSlider.tsx
 * ----------------
 * THE CURSOR SLIDER (frontend §6.1, D155): one line of the left panel, under the item's readouts, the same in every
 * stage. It moves the view's one cursor (`useTrainingCursor`) along its axis — by a drag of its thumb, a click on its
 * track, or the keys (← → one row, with Shift ten, Home and End the first and the last) — and PLAYS it: the button at its
 * left moves the cursor on at the speed chosen at its right (`useCursorPlayback`; the user, 2026-10-12). Every reader of
 * the cursor follows it: the aircraft in the scene, the word and the runway in force, the traffic, the bar's cursor line,
 * the read-back charts. It drives no Cesium clock and flies nothing.
 *
 * Its axis is the session's (`TrainingSliderAxis`): by default the sentence bar's axis (`sentenceAxis`, stages A, B, C);
 * stage D's session gives the window scene's (`sceneAxis`). Its rows and marks are `data/trainingSlider.ts`'s, on the
 * axis's clock; the session gives the marks, the slider computes none. With no sentence on screen it is disabled.
 *
 * A LEAF: it reads the Training cursor, which moves on every chart hover.
 */

import { useRef, type PointerEvent } from "react";
import { useApp, useTrainingCursor } from "../../context/AppContext";
import useCursorPlayback, { PLAYBACK_SPEEDS } from "../../hooks/useCursorPlayback";
import { formatSeconds, trainingReadingOf, trainingSelectionKey } from "../../data/trainingSample";
import {
  nearestStop,
  sentenceAxis,
  stopAt,
  stopForKey,
  type TrainingSliderAxis,
  type TrainingSliderMark,
} from "../../data/trainingSlider";

/** The track's drawing width (the SVG stretches it to the line). */
const WIDE = 1000;

export default function CursorSlider({ marks, axis: given }: { marks: TrainingSliderMark[]; axis?: TrainingSliderAxis | null }) {
  const { trainingSelection: selection, trainingIntervalS } = useApp();
  const { trainingCursorS, setTrainingCursorS } = useTrainingCursor();
  const axis = given !== undefined ? given : (selection === null ? null : sentenceAxis(trainingSelectionKey(selection)!,
    trainingReadingOf(selection.flight, selection.vocabulary.stepS, trainingIntervalS), selection.vocabulary.stepS));
  const playback = useCursorPlayback(axis);
  const dragging = useRef(false);

  if (selection === null || axis === null) {
    return (
      <div className="training-cursor-slider" aria-label="Cursor">
        <button type="button" className="training-cursor-slider-play" disabled aria-label="Play">▶</button>
        <span className="training-cursor-slider-time">t –</span>
        <div className="training-cursor-slider-track" role="slider" aria-label="Cursor" aria-disabled="true"
          aria-valuetext="choose a sentence">choose a sentence</div>
      </div>
    );
  }
  const { stops, endS, offsetS, clock } = axis;
  // the cursor is the flight time of the aircraft on screen; the axis's time is that less its offset
  const atS = trainingCursorS - offsetS;
  const place = stopAt(stops, atS);
  const x = (seconds: number) => (Math.min(Math.max(seconds, 0), endS) / endS) * WIDE;
  const moveTo = (stop: number) => setTrainingCursorS(stops[stop] + offsetS);

  const putAt = (event: PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const share = box.width > 0 ? (event.clientX - box.left) / box.width : 0;
    moveTo(nearestStop(stops, share * endS));
  };

  return (
    <div className="training-cursor-slider" aria-label="Cursor">
      <button type="button" className="training-cursor-slider-play" onClick={playback.toggle}
        aria-label={playback.playing ? "Pause" : "Play"} aria-pressed={playback.playing}
        title={playback.playing ? "Pause" : `Play the tracks from the cursor at ${playback.speed}× (no flight is computed)`}>
        {playback.playing ? "❚❚" : "▶"}
      </button>
      <span className="training-cursor-slider-time">{clock} {formatSeconds(atS)} s</span>
      <div className="training-cursor-slider-track" role="slider" tabIndex={0} aria-label="Cursor"
        aria-valuemin={0} aria-valuemax={stops.length - 1} aria-valuenow={place}
        aria-valuetext={`${clock} = ${formatSeconds(atS)} s, row ${place} of ${stops.length - 1}`}
        onPointerDown={(event) => {
          playback.stop();
          dragging.current = true;
          event.currentTarget.setPointerCapture?.(event.pointerId);
          putAt(event);
        }}
        onPointerMove={(event) => {
          if (dragging.current) putAt(event);
        }}
        onPointerUp={() => {
          dragging.current = false;
        }}
        onPointerCancel={() => {
          dragging.current = false;
        }}
        onKeyDown={(event) => {
          const next = stopForKey(stops, place, event.key, event.shiftKey);
          if (next === null) return;
          event.preventDefault();
          playback.stop();
          moveTo(next);
        }}>
        <svg viewBox={`0 0 ${WIDE} 12`} preserveAspectRatio="none" aria-hidden="true">
          <line x1={0} x2={WIDE} y1={11.5} y2={11.5} className="training-cursor-slider-rail" vectorEffect="non-scaling-stroke" />
          {marks.map((mark) => (mark.kind === "tick" ? (
            <line key={mark.key} x1={x(mark.atS)} x2={x(mark.atS)} y1={2} y2={12} stroke={mark.colour} strokeWidth={1.5}
              vectorEffect="non-scaling-stroke" data-mark={mark.key}>
              <title>{mark.title}</title>
            </line>
          ) : mark.kind === "band" ? (
            <rect key={mark.key} x={x(mark.fromS)} width={Math.max(0, x(mark.toS) - x(mark.fromS))} y={4} height={6}
              fill={mark.colour} fillOpacity={0.45} data-mark={mark.key}>
              <title>{mark.title}</title>
            </rect>
          ) : (
            <polyline key={mark.key} points={mark.points.map(([markS, value]) => `${x(markS)},${11 - value * 10}`).join(" ")} fill="none"
              stroke={mark.colour} strokeWidth={1} vectorEffect="non-scaling-stroke" data-mark={mark.key}>
              <title>{mark.title}</title>
            </polyline>
          )))}
          <line x1={x(atS)} x2={x(atS)} y1={0} y2={12} className="training-cursor-slider-thumb"
            vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
      <select className="training-cursor-slider-speed" aria-label="Playback speed" title="playback speed, × real time"
        value={playback.speed} onChange={(event) => playback.setSpeed(Number(event.target.value))}>
        {PLAYBACK_SPEEDS.map((speed) => <option key={speed} value={speed}>{speed}×</option>)}
      </select>
    </div>
  );
}
