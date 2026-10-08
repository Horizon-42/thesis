/**
 * The cursor of a window: it starts at the window's row 0 when a window comes on screen (D129: the other aircraft show from
 * the start), stays where the user then puts it, and starts again at the next window's row 0; a change of the round keeps
 * the window's instant, or puts it at the nearer end of the new sentence (frontend §6.1); so does a change of the selected
 * aircraft (stage D), through each aircraft's flight clock at the window's row 0 (``clockS``). With the real provider.
 */
import { describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({ defaultAirport: "KXXX", airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }] }),
}));

import { AppProvider, useApp, useTrainingCursor } from "../../context/AppContext";
import { WindowCursor } from "../training/TrainingWindowSession";
import { stageCSample, stageCSelection } from "../../data/__tests__/stageC";
import type { TrainingSelection } from "../../data/trainingSample";

/** The leaf, the app and the cursor in the real provider. */
function harness() {
  let app!: ReturnType<typeof useApp>;
  let cursor!: ReturnType<typeof useTrainingCursor>;
  function Harness({ leaf }: { leaf: { windowKey: string; flightKey: string; labelled?: boolean; clockS: number; endS: number } | null }) {
    app = useApp();
    cursor = useTrainingCursor();
    return leaf === null ? null : <WindowCursor labelled={false} {...leaf} />;
  }
  const view = render(<AppProvider><Harness leaf={null} /></AppProvider>);
  return {
    show: (leaf: { windowKey: string; flightKey: string; labelled?: boolean; clockS: number; endS: number } | null) =>
      view.rerender(<AppProvider><Harness leaf={leaf} /></AppProvider>),
    publish: (selection: TrainingSelection) => act(() => app.setTrainingSelection(selection)),
    cursor: () => cursor,
  };
}

/** Another round of a window: the same window, another derived flight (the fixture flies one round). */
function otherRound(selection: TrainingSelection, round: number): TrainingSelection {
  return { ...selection, flight: { ...selection.flight, flightKey: selection.flight.flightKey.replace(/round-.*$/, `round-${round}`) } };
}

describe("the cursor of a window", () => {
  it("starts at the window's row 0, then is the user's, and starts again with another window", () => {
    const sample = stageCSample();
    // the synthetic flight's window starts at its own row 0 (clockS 0): the window's row 0 is given here (a real flight's
    // closed-loop sentence starts after its row 0)
    const first = stageCSelection(sample, 0);
    const third = stageCSelection(sample, 2);
    const { show, publish, cursor } = harness();
    publish(first);
    expect(cursor().trainingCursorS).toBe(0);                          // a flight on screen: its row 0
    show({ windowKey: "set/0", flightKey: first.flight.flightKey, clockS: 6, endS: 500 });
    expect(cursor().trainingCursorS).toBe(6);
    act(() => cursor().setTrainingCursorS(99));
    show({ windowKey: "set/0", flightKey: first.flight.flightKey, clockS: 6, endS: 500 });
    expect(cursor().trainingCursorS).toBe(99);                         // the user's, not set again
    // the next window's leaf before the session has put that window on screen: nothing written on the flight before
    show({ windowKey: "set/2", flightKey: third.flight.flightKey, clockS: 10, endS: 500 });
    expect(cursor().trainingCursorS).toBe(99);
    publish(third);
    show({ windowKey: "set/2", flightKey: third.flight.flightKey, clockS: 10, endS: 500 });
    expect(cursor().trainingCursorS).toBe(10);
  });

  it("keeps the window's instant on a change of the round, or puts it at the nearer end of the new sentence", () => {
    const start = stageCSelection(stageCSample(), 0);
    const r1 = otherRound(start, 1);
    const r2 = otherRound(start, 2);
    const { show, publish, cursor } = harness();
    publish(start);
    show({ windowKey: "set/0", flightKey: start.flight.flightKey, clockS: 6, endS: 500 });
    act(() => cursor().setTrainingCursorS(128));
    show({ windowKey: "set/0", flightKey: start.flight.flightKey, clockS: 6, endS: 500 });
    // round 1: a new flight on screen (the provider puts its cursor at 0), then the leaf puts back the instant
    show({ windowKey: "set/0", flightKey: r1.flight.flightKey, clockS: 6, endS: 500 });
    publish(r1);
    show({ windowKey: "set/0", flightKey: r1.flight.flightKey, clockS: 6, endS: 500 });
    expect(cursor().trainingCursorS).toBe(128);
    act(() => cursor().setTrainingCursorS(300));
    show({ windowKey: "set/0", flightKey: r1.flight.flightKey, clockS: 6, endS: 500 });
    // round 2 ends before the instant: its sentence's end
    show({ windowKey: "set/0", flightKey: r2.flight.flightKey, clockS: 6, endS: 240 });
    publish(r2);
    show({ windowKey: "set/0", flightKey: r2.flight.flightKey, clockS: 6, endS: 240 });
    expect(cursor().trainingCursorS).toBe(240);
  });

  it("keeps the window's instant across a change of the selected aircraft, through each one's clock", () => {
    const anchor = stageCSelection(stageCSample(), 0);
    const other = otherRound(anchor, 7);                                 // another aircraft's flight: another derived flight
    const third = otherRound(anchor, 8);
    const { show, publish, cursor } = harness();
    publish(anchor);
    show({ windowKey: "set/0", flightKey: anchor.flight.flightKey, clockS: 6, endS: 500 });
    act(() => cursor().setTrainingCursorS(128));                         // the window's 122 s
    show({ windowKey: "set/0", flightKey: anchor.flight.flightKey, clockS: 6, endS: 500 });
    // an aircraft that joined 32 s later: its flight time at the window's row 0 is 32 s less
    show({ windowKey: "set/0", flightKey: other.flight.flightKey, clockS: -26, endS: 500 });
    publish(other);
    show({ windowKey: "set/0", flightKey: other.flight.flightKey, clockS: -26, endS: 500 });
    expect(cursor().trainingCursorS).toBe(96);                           // 122 − 26 on its clock
    // one whose sentence starts after that instant: the nearer end, its 0
    act(() => cursor().setTrainingCursorS(10));                          // the window's 36 s
    show({ windowKey: "set/0", flightKey: other.flight.flightKey, clockS: -26, endS: 500 });
    show({ windowKey: "set/0", flightKey: third.flight.flightKey, clockS: -60, endS: 500 });
    publish(third);
    show({ windowKey: "set/0", flightKey: third.flight.flightKey, clockS: -60, endS: 500 });
    expect(cursor().trainingCursorS).toBe(0);
    // another window opened on an aircraft that joined late: its row 0 is before its flight's, so at the axis's 0
    const elsewhere = otherRound(anchor, 9);
    show({ windowKey: "set/1", flightKey: elsewhere.flight.flightKey, clockS: -32, endS: 500 });
    publish(elsewhere);
    show({ windowKey: "set/1", flightKey: elsewhere.flight.flightKey, clockS: -32, endS: 500 });
    expect(cursor().trainingCursorS).toBe(0);
  });

  it("clamps the cursor to the sentence's axis between Labelled and the round it is read on (one flight)", () => {
    const start = stageCSelection(stageCSample(), 0);
    const { show, publish, cursor } = harness();
    publish(start);
    show({ windowKey: "set/0", flightKey: start.flight.flightKey, labelled: true, clockS: 6, endS: 700 });
    act(() => cursor().setTrainingCursorS(600));                         // late on the recorded track
    show({ windowKey: "set/0", flightKey: start.flight.flightKey, labelled: true, clockS: 6, endS: 700 });
    show({ windowKey: "set/0", flightKey: start.flight.flightKey, labelled: false, clockS: 6, endS: 400 });
    expect(cursor().trainingCursorS).toBe(400);                         // the round's sentence ends at 400 s
    act(() => cursor().setTrainingCursorS(100));
    show({ windowKey: "set/0", flightKey: start.flight.flightKey, labelled: false, clockS: 6, endS: 400 });
    show({ windowKey: "set/0", flightKey: start.flight.flightKey, labelled: true, clockS: 6, endS: 700 });
    expect(cursor().trainingCursorS).toBe(100);                         // within the labelled axis: kept
  });
});
