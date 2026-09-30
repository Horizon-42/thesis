/**
 * The window's clock above the sentence bar (TrainingTrafficStrip): a row per commanded aircraft as the sentence read has
 * it, the pairs under their minimum, the cursor on the window's time — moved by a press on the plot, never by one on a
 * callsign, which puts that aircraft on screen.
 */
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

const { appState, setSceneS, focus } = vi.hoisted(() => ({
  appState: { trainingWindow: null as unknown, trainingSelection: null as unknown, trainingSource: null as unknown },
  setSceneS: vi.fn(),
  focus: vi.fn(),
}));

vi.mock("../../context/AppContext", async () => {
  const { useState } = await import("react");
  return {
    useApp: () => appState,
    useTrainingCursor: () => {
      const [trainingSceneS, set] = useState(121);
      return { trainingSceneS, setTrainingSceneS: (at: number) => { setSceneS(at); set(at); } };
    },
  };
});

import TrainingTrafficStrip from "../TrainingTrafficStrip";
import { TRAINING_FAILURE_COLOR } from "../../utils/trainingWordColors";
import { parseTrainingTrafficSet, parseTrainingWindowGenerationOverlay, trainingWindowSelection } from "../../data/trainingTraffic";
import { parseTrainingOverlays } from "../../data/trainingOverlays";
import { trainingSelectionOf } from "../../data/trainingSample";
import {
  STRAIGHT_ID, WINDOW_MODEL_ID, mockTrafficSet, mockWindowOverlay, mockWindowOverlays,
} from "../../data/__tests__/trainingTraffic.fixture";

function setUp(source: unknown) {
  const set = parseTrainingTrafficSet(mockTrafficSet());
  const entries = parseTrainingOverlays(mockWindowOverlays());
  if (!set.ok || !entries.ok) throw new Error("the fixture should parse");
  const overlay = parseTrainingWindowGenerationOverlay(mockWindowOverlay(), entries.value.overlays[0], set.value);
  if (!overlay.ok) throw new Error(overlay.problem);
  const [window] = set.value.windows;
  appState.trainingWindow = { set: set.value, window, overlays: [overlay.value], focus };
  appState.trainingSelection = trainingWindowSelection(set.value, window, window.commanded[0]);
  appState.trainingSource = source;
  return { set: set.value, window };
}

describe("TrainingTrafficStrip", () => {
  beforeAll(() => {
    // jsdom has no pointer events nor pointer capture: a mouse event carrying a pointer id stands in
    class PointerEventStandIn extends MouseEvent {
      pointerId: number;
      constructor(type: string, init: PointerEventInit = {}) {
        super(type, init);
        this.pointerId = init.pointerId ?? 0;
      }
    }
    vi.stubGlobal("PointerEvent", PointerEventStandIn);
    Element.prototype.setPointerCapture = () => undefined;
    Element.prototype.hasPointerCapture = () => true;
  });

  beforeEach(() => {
    setSceneS.mockClear();
    focus.mockClear();
  });

  it("draws a row per commanded aircraft and one for the others, as recorded under the truth", () => {
    setUp(null);
    render(<TrainingTrafficStrip />);
    expect(screen.getByText("as recorded")).toBeTruthy();
    expect(screen.getByText("▶ TST1")).toBeTruthy();             // the aircraft on screen
    expect(screen.getByText("TST2")).toBeTruthy();
    expect(screen.getByText("others")).toBeTruthy();
    expect(screen.getByText(/t = 2:01/)).toBeTruthy();
    // as recorded only IFR has a pair under its minimum (at 150 s): none now
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("says a model's sample, and the pairs under their minimum at the cursor", () => {
    setUp({ overlayId: WINDOW_MODEL_ID, sample: 0 });
    render(<TrainingTrafficStrip />);
    expect(screen.getByText("base · sample 1")).toBeTruthy();
    expect(screen.getByRole("status").textContent).toBe("1 pair under the minimum");
    expect(screen.getAllByText("✕").map((mark) => mark.getAttribute("fill"))).toEqual([TRAINING_FAILURE_COLOR]);
  });

  it("moves the window's time on a press on the plot, and puts an aircraft on screen on one on its callsign", () => {
    setUp(null);
    const { container } = render(<TrainingTrafficStrip />);
    const svg = container.querySelector("svg")!;
    fireEvent.pointerDown(svg, { clientX: 30, pointerId: 1, buttons: 1 });
    expect(setSceneS).not.toHaveBeenCalled();
    fireEvent.pointerDown(screen.getByText("TST2"), { clientX: 30, pointerId: 1, buttons: 1 });
    expect(focus).toHaveBeenLastCalledWith(STRAIGHT_ID);
    expect(setSceneS).not.toHaveBeenCalled();
    fireEvent.pointerDown(svg, { clientX: 70, pointerId: 1, buttons: 1 });
    expect(setSceneS).toHaveBeenLastCalledWith(100);                // the plot's left edge: the window span's start
  });

  it("puts an aircraft on screen on a PRESS on its bar — the press the plot captures — and moves the time there too", () => {
    setUp(null);
    const { container } = render(<TrainingTrafficStrip />);
    const bar = [...container.querySelectorAll("rect")].find((rect) => rect.querySelector("title")?.textContent?.startsWith("TST1: from"))!;
    fireEvent.pointerDown(bar, { clientX: 400, pointerId: 1, buttons: 1 });
    expect(focus).toHaveBeenLastCalledWith(expect.stringContaining("TST1"));
    expect(setSceneS).toHaveBeenCalledTimes(1);
  });

  it("plays the window at the speed chosen, whatever the renders between its ticks, and stops at its end", () => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval", "performance"] });
    try {
      setUp(null);
      render(<TrainingTrafficStrip />);
      fireEvent.click(screen.getByRole("button", { name: "▶" }));
      act(() => vi.advanceTimersByTime(1000));
      // 10× for a second, from 121 s
      expect(setSceneS.mock.calls[setSceneS.mock.calls.length - 1][0]).toBeCloseTo(131, 0);
      act(() => vi.advanceTimersByTime(60_000));
      expect(setSceneS.mock.calls[setSceneS.mock.calls.length - 1][0]).toBe(228);   // the span's end
      expect(screen.getByRole("button", { name: "▶" })).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
  });

  it("is drawn only while an aircraft of the window is on screen on its clock", () => {
    const { set } = setUp(null);
    appState.trainingSelection = trainingSelectionOf({ ...set, cohort: { split: "val", perStratum: 1, seed: 1, drawnFrom: "", pool: 1, read: 1 } },
      set.flights[0]);
    const { container } = render(<TrainingTrafficStrip />);
    expect(container.innerHTML).toBe("");
  });
});
