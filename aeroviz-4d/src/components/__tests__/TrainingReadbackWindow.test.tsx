/**
 * The read-back check: every envelope the exporter sent is drawn where it belongs — the lateral
 * ones in the plan view and on the heading chart, the tubes against distance flown, the speed
 * transitions and bands against time — the switches reach every chart, the rows the labeller
 * counted outside are red, and the two empty slots say why they are empty.
 */
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import TrainingReadbackWindow, { extent, rowAtDistance, runsOf } from "../TrainingReadbackWindow";
import { parseTrainingSample } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import type { TrainingLayers } from "../../context/AppContext";

const ALL: TrainingLayers = { lateral: true, vertical: true, candidates: true };

function open(layers: TrainingLayers = ALL, position = 0, cursorS = 0) {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  const onCursorChange = vi.fn();
  render(
    <TrainingReadbackWindow
      flight={parsed.value.flights[position]}
      vocabulary={parsed.value.vocabulary}
      candidates={parsed.value.candidates}
      layers={layers}
      cursorS={cursorS}
      onCursorChange={onCursorChange}
      onClose={() => undefined}
    />,
  );
  return { onCursorChange };
}

/** The window renders through a portal into `document.body`. */
const count = (selector: string) => document.body.querySelectorAll(selector).length;

describe("TrainingReadbackWindow", () => {
  it("draws the plan view's envelopes: one turn region per turn, a funnel per hold, the corridor, the capture turn", () => {
    open();
    expect(count(".training-readback-turn")).toBe(1);
    // where the turn may end: its own four-cornered polygon, and the fastest and slowest turns
    expect(count(".training-readback-turn-end")).toBe(1);
    expect(document.body.querySelector(".training-readback-turn-end")!.getAttribute("points")!.split(" ")).toHaveLength(4);
    expect(count(".training-readback-turn-path")).toBe(2);
    expect(count(".training-readback-funnel")).toBe(2);
    expect(count(".training-readback-corridor")).toBe(1);
    expect(count(".training-readback-capture-turn")).toBe(1);
    expect(count(".training-readback-candidate")).toBe(2);
  });

  it("keeps the designated runway when the other candidates are switched off", () => {
    open({ ...ALL, candidates: false });
    const drawn = [...document.body.querySelectorAll(".training-readback-candidate")].map((g) => g.getAttribute("aria-label"));
    expect(drawn).toEqual([expect.stringMatching(/^the designated runway 09/)]);
  });

  it("marks a hold with rows outside its funnel, and a hold the labeller does not judge", () => {
    open();
    const funnels = [...document.body.querySelectorAll(".training-readback-funnel")];
    expect(funnels.map((funnel) => funnel.textContent)).toEqual([
      expect.stringMatching(/11 of 11 hold rows inside/), expect.stringMatching(/4 of 5 hold rows inside/),
    ]);
    const raw: any = mockSample();
    raw.flights[0].envelopes.heading[0].holdCheck = null;
    const parsed = parseTrainingSample(raw);
    if (!parsed.ok) throw new Error(parsed.problem);
    render(
      <TrainingReadbackWindow flight={parsed.value.flights[0]} vocabulary={parsed.value.vocabulary}
        candidates={parsed.value.candidates} layers={ALL} cursorS={0} onCursorChange={() => undefined}
        onClose={() => undefined} />,
    );
    expect(screen.getAllByText(/not judged by the labeller/).length).toBeGreaterThan(0);
  });

  it("switches the lateral envelopes off in the plan view AND on the heading chart", () => {
    open({ ...ALL, lateral: false });
    for (const selector of [".training-readback-turn", ".training-readback-funnel", ".training-readback-corridor",
                            ".training-readback-turn-band", ".training-readback-hold-band", ".training-readback-course-band"]) {
      expect(count(selector)).toBe(0);
    }
  });

  it("draws the heading chart's bands from the exporter's numbers", () => {
    open();
    expect(count(".training-readback-turn-band")).toBe(1);
    expect(count(".training-readback-hold-band")).toBe(2);
    expect(count(".training-readback-capture-band")).toBe(1);
    expect(count(".training-readback-course-band")).toBe(1);
  });

  it("draws one tube per altitude word, and the row the labeller counted outside in red", () => {
    open();
    expect(count(".training-readback-tube")).toBe(2);
    expect(count(".training-readback-outside")).toBe(1);
    expect(screen.getByLabelText(/angle word descent 3 \(3\.06°\) at step 20/)).toBeTruthy();
  });

  it("draws each speed word's transition and band, and 'unspecified' as the pilot's own", () => {
    open();
    expect(count(".training-readback-transition")).toBe(1);
    expect(count(".training-readback-speed-band")).toBe(1);
    expect(count(".training-readback-unspecified")).toBe(1);
  });

  it("switches the tubes and the speed envelopes off with the vertical layer", () => {
    open({ ...ALL, vertical: false });
    expect(count(".training-readback-tube") + count(".training-readback-speed-band") + count(".training-readback-transition")).toBe(0);
  });

  it("names what is in force at the cursor, with the turn's verdict", () => {
    open(ALL, 0, 22);
    expect(screen.getByText(/in force: 180° · its turn: ✓ monotone, ✓ rate \(mean 2\.40°\/s, max 3\.10°\/s, max bank 24\.9°\) · its hold: 4\/5 rows in the funnel/)).toBeTruthy();
    expect(screen.getByText(/in force: 1110 m, level/)).toBeTruthy();
  });

  it("says the corridor holds once the flight is captured", () => {
    open(ALL, 0, 60);
    expect(screen.getByText(/captured: the corridor holds/)).toBeTruthy();
  });

  it("keeps the two slots empty, saying why", () => {
    open();
    expect(screen.getByLabelText("Executor replay slot").textContent).toMatch(/not built yet/);
    expect(screen.getByLabelText("Prior sentence slot").textContent).toMatch(/no prior is trained/);
  });

  it("reads the cursor off a chart", () => {
    const { onCursorChange } = open();
    fireEvent.click(screen.getByLabelText("Speed chart"));
    expect(onCursorChange).toHaveBeenCalled();
  });
});

describe("helpers", () => {
  it("pads an extent and never returns a zero-wide one", () => {
    expect(extent([0, 100])).toEqual([-8, 108]);
    expect(extent([5, 5])).toEqual([4, 6]);
  });

  it("finds the runs of a flag", () => {
    expect(runsOf([false, true, true, false, true])).toEqual([[1, 2], [4, 4]]);
  });

  it("finds the row at a distance flown", () => {
    expect(rowAtDistance([0, 100, 200, 300], 250)).toBe(2);
  });
});
