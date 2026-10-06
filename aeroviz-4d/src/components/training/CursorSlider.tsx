/**
 * CursorSlider.tsx
 * ----------------
 * THE CURSOR SLIDER (frontend §6.1, D155): one line of the left panel, under the item's readouts, the same in every
 * stage. It moves the view's one cursor (`useTrainingCursor`) over the rows of the sentence on screen — by a drag of its
 * thumb, a click on its track, or the keys (← → one row, with Shift ten, Home and End the first and the last). Every
 * reader of the cursor follows it: the aircraft in the scene, the word and the runway in force, the traffic, the bar's
 * cursor line, the read-back charts. It does not drive the Cesium clock and has no playback.
 *
 * Its track is the sentence bar's axis (`readingAxisEndS`); its rows and the marks on it are `data/trainingSlider.ts`'s.
 * The session of the stage on screen gives the marks; the slider computes none. With no sentence on screen it is disabled.
 *
 * A LEAF: it reads the Training cursor, which moves on every chart hover.
 */

import { useRef, type PointerEvent } from "react";
import { useApp, useTrainingCursor } from "../../context/AppContext";
import { formatSeconds, readingAxisEndS, trainingReadingOf } from "../../data/trainingSample";
import { nearestStop, sliderStops, stopAt, stopForKey, type TrainingSliderMark } from "../../data/trainingSlider";

/** The track's drawing width (the SVG stretches it to the line). */
const WIDE = 1000;

export default function CursorSlider({ marks }: { marks: TrainingSliderMark[] }) {
  const { trainingSelection: selection, trainingIntervalS } = useApp();
  const { trainingCursorS, setTrainingCursorS } = useTrainingCursor();
  const dragging = useRef(false);

  if (selection === null) {
    return (
      <div className="training-cursor-slider" aria-label="Cursor">
        <span className="training-cursor-slider-time">t –</span>
        <div className="training-cursor-slider-track" role="slider" aria-label="Cursor" aria-disabled="true"
          aria-valuetext="choose a sentence">choose a sentence</div>
      </div>
    );
  }
  const stepS = selection.vocabulary.stepS;
  const reading = trainingReadingOf(selection.flight, stepS, trainingIntervalS);
  const stops = sliderStops(reading, stepS);
  const endS = readingAxisEndS(reading);
  const place = stopAt(stops, trainingCursorS);
  const x = (seconds: number) => (Math.min(Math.max(seconds, 0), endS) / endS) * WIDE;

  const putAt = (event: PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const share = box.width > 0 ? (event.clientX - box.left) / box.width : 0;
    setTrainingCursorS(stops[nearestStop(stops, share * endS)]);
  };

  return (
    <div className="training-cursor-slider" aria-label="Cursor">
      <span className="training-cursor-slider-time">t {formatSeconds(trainingCursorS)} s</span>
      <div className="training-cursor-slider-track" role="slider" tabIndex={0} aria-label="Cursor"
        aria-valuemin={0} aria-valuemax={stops.length - 1} aria-valuenow={place}
        aria-valuetext={`t = ${formatSeconds(trainingCursorS)} s, row ${place} of ${stops.length - 1}`}
        onPointerDown={(event) => {
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
          setTrainingCursorS(stops[next]);
        }}>
        <svg viewBox={`0 0 ${WIDE} 12`} preserveAspectRatio="none" aria-hidden="true">
          <line x1={0} x2={WIDE} y1={11.5} y2={11.5} className="training-cursor-slider-rail" vectorEffect="non-scaling-stroke" />
          {marks.map((mark) => (mark.kind === "tick" ? (
            <line key={mark.key} x1={x(mark.atS)} x2={x(mark.atS)} y1={2} y2={12} stroke={mark.colour} strokeWidth={1.5}
              vectorEffect="non-scaling-stroke" data-mark={mark.key}>
              <title>{mark.title}</title>
            </line>
          ) : (
            <polyline key={mark.key} points={mark.points.map(([atS, value]) => `${x(atS)},${11 - value * 10}`).join(" ")} fill="none"
              stroke={mark.colour} strokeWidth={1} vectorEffect="non-scaling-stroke" data-mark={mark.key}>
              <title>{mark.title}</title>
            </polyline>
          )))}
          <line x1={x(trainingCursorS)} x2={x(trainingCursorS)} y1={0} y2={12} className="training-cursor-slider-thumb"
            vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
    </div>
  );
}
