/**
 * makeReadoutEmitter: the live-readout sampler shared by the trajectory and
 * comparison playbacks. The regression at stake: Cesium's LOOP_STOP wrap sets
 * currentTime = startTime + overshoot (Clock.js) — it never dwells at the stop
 * time — so the exact-end sample must come from the clock's onStop event (the
 * `stop` handler), not from any elapsed-based tick heuristic.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import * as Cesium from "cesium";
import { makeReadoutEmitter } from "../useOptimizedTrajectoryPlayback";

const START = Cesium.JulianDate.fromIso8601("2026-04-01T08:00:00Z");
const STOP_S = 324.0;

function makeViewer(): Cesium.Viewer {
  return {
    clock: {
      currentTime: Cesium.JulianDate.clone(START),
      stopTime: Cesium.JulianDate.addSeconds(START, STOP_S, new Cesium.JulianDate()),
      multiplier: 8,
    },
  } as unknown as Cesium.Viewer;
}

function setElapsed(viewer: Cesium.Viewer, elapsedS: number): void {
  viewer.clock.currentTime = Cesium.JulianDate.addSeconds(
    START, elapsedS, new Cesium.JulianDate(),
  );
}

describe("makeReadoutEmitter", () => {
  let nowMs = 0;

  beforeEach(() => {
    nowMs = 0;
    vi.spyOn(performance, "now").mockImplementation(() => nowMs);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function setup() {
    const viewer = makeViewer();
    const emitted: number[] = [];
    const emitter = makeReadoutEmitter(
      viewer,
      { current: START },
      (t) => emitted.push(t),
    );
    return { viewer, emitted, emitter };
  }

  it("throttles tick emissions by wall clock", () => {
    const { viewer, emitted, emitter } = setup();
    nowMs = 1000;
    setElapsed(viewer, 10.0);
    emitter.tick();
    setElapsed(viewer, 10.5);
    emitter.tick(); // within the throttle window -> suppressed
    nowMs = 1100;
    setElapsed(viewer, 11.0);
    emitter.tick();
    expect(emitted).toEqual([10.0, 11.0]);
  });

  it("stop emits the EXACT stop-time sample, bypassing the throttle", () => {
    const { viewer, emitted, emitter } = setup();
    nowMs = 1000;
    setElapsed(viewer, 323.5);
    emitter.tick();
    // LOOP_STOP wrap: Cesium raises onStop, then ticks resume from the
    // OVERSHOOT (~0.13 s at 8x / 60 fps), never dwelling at 324.0.
    emitter.stop(); // same wall instant -> must not be throttled away
    expect(emitted).toEqual([323.5, STOP_S]);
    nowMs = 1100;
    setElapsed(viewer, 0.13);
    emitter.tick();
    expect(emitted).toHaveLength(3);
    expect(emitted[2]).toBeCloseTo(0.13, 6);
    // ...and the next loop's wrap emits the terminal sample again.
    nowMs = 2000;
    emitter.stop();
    expect(emitted).toHaveLength(4);
    expect(emitted[3]).toBe(STOP_S);
  });

  it("parked at a CLAMPED end, repeated onStop raises do not re-emit", () => {
    const { emitted, emitter } = setup();
    nowMs = 1000;
    emitter.stop();
    emitter.stop();
    emitter.stop();
    expect(emitted).toEqual([STOP_S]);
  });

  it("does nothing without a start time", () => {
    const viewer = makeViewer();
    const emitted: number[] = [];
    const emitter = makeReadoutEmitter(viewer, { current: null }, (t) => emitted.push(t));
    emitter.tick();
    emitter.stop();
    expect(emitted).toEqual([]);
  });
});
