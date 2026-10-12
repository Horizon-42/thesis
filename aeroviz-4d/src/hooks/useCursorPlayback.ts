/**
 * useCursorPlayback.ts
 * --------------------
 * THE CURSOR SLIDER'S PLAYBACK (the user, 2026-10-12): Play moves the Training cursor on along the slider's axis at the
 * chosen speed, so every reader of the cursor plays the exported tracks — the aircraft in the scene, the other aircraft,
 * the sentence bar and the read-back charts. It flies nothing (the live executor is another thing) and drives no Cesium
 * clock (Training runs none).
 *
 * Play at the axis's end starts again from its 0. The playback stops at the end, when the axis changes (another flight,
 * another window: its `key`; stage D's aircraft and rounds keep the window's scene, so they keep playing), when the user
 * takes the slider (`stop`), and outside the Training task (the panel stays mounted there, hidden).
 *
 * A LEAF's hook: it reads the Training cursor, which moves on every chart hover.
 */

import { useEffect, useRef, useState } from "react";
import { useApp, useTrainingCursor } from "../context/AppContext";
import type { TrainingSliderAxis } from "../data/trainingSlider";

/** The speeds offered, × real time. */
export const PLAYBACK_SPEEDS = [1, 2, 5, 10, 20, 50];
export const DEFAULT_PLAYBACK_SPEED = 10;
/** The least real time between two moves of the cursor (ms): each move re-renders the bar, the charts and the scene. */
const FRAME_MS = 33;
/** The most real time one move plays (ms): a tab hidden for a while resumes where it was, not at the end. */
const LONGEST_FRAME_MS = 250;

export interface CursorPlayback {
  playing: boolean;
  speed: number;
  setSpeed: (speed: number) => void;
  /** Play, or pause. */
  toggle: () => void;
  stop: () => void;
}

export default function useCursorPlayback(axis: TrainingSliderAxis | null): CursorPlayback {
  const { mode } = useApp();
  const { trainingCursorS, setTrainingCursorS } = useTrainingCursor();
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const [speed, setSpeed] = useState(DEFAULT_PLAYBACK_SPEED);
  const startS = axis === null ? 0 : axis.offsetS;
  const endS = axis === null ? 0 : axis.offsetS + axis.endS;
  const playing = axis !== null && playingKey === axis.key && mode === "training";
  // a playback that stopped by itself (another axis, another task) does not come back unasked
  useEffect(() => {
    if (!playing) setPlayingKey(null);
  }, [playing]);
  // THE END is judged when the CURSOR moves, never on a change of the axis alone: stage D's session renders a newly
  // selected aircraft's axis once with the last aircraft's cursor, before the new flight and its cursor are on screen
  const atEnd = useRef(false);
  atEnd.current = trainingCursorS >= endS - 1e-9;
  useEffect(() => {
    if (atEnd.current) setPlayingKey(null);
  }, [trainingCursorS]);

  const latest = useRef({ speed, endS });
  latest.current = { speed, endS };
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      if (now - last >= FRAME_MS) {
        const stepS = (Math.min(now - last, LONGEST_FRAME_MS) / 1000) * latest.current.speed;
        const stopS = latest.current.endS;
        last = now;
        setTrainingCursorS((currentS) => Math.min(currentS + stepS, stopS));
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, setTrainingCursorS]);

  const toggle = () => {
    if (axis === null) return;
    if (playing) {
      setPlayingKey(null);
      return;
    }
    if (trainingCursorS >= endS - 1e-9 || trainingCursorS < startS) setTrainingCursorS(startS);
    setPlayingKey(axis.key);
  };
  return { playing, speed, setSpeed, toggle, stop: () => setPlayingKey(null) };
}
