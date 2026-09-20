/**
 * The read-back window's invariants (T4b). The one that matters most is the
 * heading step's placement: on an unwrapped trace the word has to be drawn on
 * the trace's own turn of the circle, or a flight that has turned through 360°
 * reads as a misread word.
 */
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import TrainingReadbackWindow, {
  chartLevels,
  flownTrace,
  gapAtS,
  rowAt,
} from "../TrainingReadbackWindow";
import { parseTrainingSample } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import { FEET_TO_METERS, metresPerSecondToKnots } from "../../utils/procedureGeoMath";

function selection() {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return { vocabulary: parsed.value.vocabulary, flight: parsed.value.flights[0] };
}

function renderWindow(overrides: Partial<React.ComponentProps<typeof TrainingReadbackWindow>> = {}) {
  const { vocabulary, flight } = selection();
  const onCursorChange = vi.fn();
  const onClose = vi.fn();
  render(
    <TrainingReadbackWindow
      flight={flight}
      vocabulary={vocabulary}
      cursorS={0}
      onCursorChange={onCursorChange}
      onClose={onClose}
      {...overrides}
    />,
  );
  return { onCursorChange, onClose, flight, vocabulary };
}

describe("rowAt", () => {
  // The figure uses `np.searchsorted(t, x)`, which is the FIRST row at or after;
  // this is the last row at or before. The two agree on every instruction time,
  // because `read_instructions` builds each one out of the row times themselves —
  // and they differ only for the cursor, which lands between rows and wants the
  // row it is standing on.
  it("finds the last row at or before a time", () => {
    const tS = [0, 2, 4, 6, 8];
    expect(rowAt(tS, 0)).toBe(0);
    expect(rowAt(tS, 3)).toBe(1);
    expect(rowAt(tS, 4)).toBe(2);
    expect(rowAt(tS, 7.9)).toBe(3);
    expect(rowAt(tS, 100)).toBe(4);
    expect(rowAt(tS, -5)).toBe(0);
  });
});

describe("chartLevels", () => {
  it("holds each word until the next of its kind, then to the end of the track", () => {
    const { vocabulary, flight } = selection();
    const speed = chartLevels(flight, vocabulary, "speed");
    expect(speed.map((item) => [item.instruction.issuedS, item.endS])).toEqual([
      [0, 26],
      [26, 108],
      [108, 188],
      [188, flight.observed.tS[flight.observed.tS.length - 1]],
    ]);
  });

  it("puts altitude in feet and speed in knots, off the bin centres", () => {
    const { vocabulary, flight } = selection();
    expect(chartLevels(flight, vocabulary, "altitude")[0].level).toBeCloseTo(
      (10 * vocabulary.altitudeBinM) / FEET_TO_METERS,
      6,
    );
    expect(chartLevels(flight, vocabulary, "speed")[0].level).toBeCloseTo(
      metresPerSecondToKnots(vocabulary.speedMinMps + 21 * vocabulary.speedBinMps),
      6,
    );
  });

  // THE test: the fixture's trace runs +90 → 0, so the levels sit where the word
  // centres are. Lift the whole trace by one turn and every level must follow,
  // because the trace is unwrapped and the word is not.
  it("places a heading step on the trace's own turn of the circle", () => {
    const { vocabulary, flight } = selection();
    const plain = chartLevels(flight, vocabulary, "heading").map((item) => item.level);
    expect(plain).toEqual([90, 50, 20, 0]);

    const orbited = {
      ...flight,
      observed: {
        ...flight.observed,
        courseUnwrappedDeg: flight.observed.courseUnwrappedDeg.map((value) => value + 360),
      },
    };
    expect(chartLevels(orbited, vocabulary, "heading").map((item) => item.level)).toEqual([
      450, 410, 380, 360,
    ]);
  });

  // A word whose manoeuvre never settled is judged at the END of the track — the
  // same fallback the figure uses. It has to be pinned on a HEADING instruction:
  // the altitude and speed branches return before `settledS` is read at all, so a
  // fallback of 0 would pass unnoticed there.
  it("judges a never-settled turn at the end of the track, not at its start", () => {
    const { vocabulary, flight } = selection();
    const turned = {
      ...flight,
      // the trace crosses onto the next turn of the circle part-way through
      observed: {
        ...flight.observed,
        courseUnwrappedDeg: flight.observed.courseUnwrappedDeg.map((value, row) =>
          flight.observed.tS[row] >= 200 ? value + 360 : value,
        ),
      },
      instructions: flight.instructions.map((item) =>
        item.kind === "heading" && item.issuedS === 188 ? { ...item, settledS: null } : item,
      ),
    };
    const last = chartLevels(turned, vocabulary, "heading").pop();
    expect(last?.instruction.settledS).toBeNull();
    // judged at the end (trace ≈ 360) the level is 360; judged at the start it
    // would be 0, which is the bug this pins
    expect(last?.level).toBe(360);
  });

  it("still judges a never-settled descent by its word alone", () => {
    const { vocabulary, flight } = selection();
    const last = flight.instructions[flight.instructions.length - 1];
    expect(last.settledS).toBeNull();
    expect(chartLevels(flight, vocabulary, "altitude").pop()?.level).toBe(0);
  });
});

describe("TrainingReadbackWindow", () => {
  it("renders into document.body, not into its parent (AV7)", () => {
    const { container } = render(
      <div className="flight-ops-panel">
        <TrainingReadbackWindow
          {...selection()}
          cursorS={0}
          onCursorChange={vi.fn()}
          onClose={vi.fn()}
        />
      </div>,
    );
    expect(container.querySelector(".training-readback-window")).toBeNull();
    expect(document.body.querySelector(".training-readback-window")).toBeTruthy();
  });

  it("has a chart for each signal the words are read from, and none for the rest", () => {
    renderWindow();
    expect(screen.getByText(/^heading —/)).toBeTruthy();
    expect(screen.getByText(/^altitude —/)).toBeTruthy();
    expect(screen.getByText(/^speed —/)).toBeTruthy();
    expect(screen.queryByText(/^runway —/)).toBeNull();
    expect(screen.queryByText(/^duration —/)).toBeNull();
    expect(screen.queryByText(/^terminal —/)).toBeNull();
  });

  // The readout is taken from the arrays at the cursor, never from a pixel.
  it("reads out the word in force and the measured value at the cursor", () => {
    renderWindow({ cursorS: 140 });
    // at 140 s the third speed instruction (issued 108 s) is in force
    expect(screen.getByText(/speed —.*in force: 230 kt \(issued 108 s, settled 126 s\)/)).toBeTruthy();
    // and the altitude word issued at 130 s, whose manoeuvre settled at 162 s
    expect(screen.getByText(/altitude —.*in force: 4000 ft \(issued 130 s, settled 162 s\)/)).toBeTruthy();
  });

  it("says when an instruction never settled inside the track", () => {
    renderWindow({ cursorS: 240 });
    expect(screen.getByText(/altitude —.*in force: 0 ft \(issued 222 s, never settled\)/)).toBeTruthy();
  });

  // The runway word names the frame the others are measured in; it is not a
  // place the aircraft was, so the plan view must not mark one.
  it("marks where each instruction was issued, except the runway", () => {
    const { flight, vocabulary } = renderWindow();
    const marks = document.body.querySelectorAll(".training-readback-svg circle");
    const issued = flight.instructions.filter((item) => item.kind !== "runway").length;
    expect(marks.length).toBe(issued + 1); // + the cursor's dot
    expect(screen.getByLabelText(/heading \+50° issued at 70 s/)).toBeTruthy();
    expect(vocabulary.runwayIdents).toContain(flight.runway);
    expect(document.body.textContent).not.toContain("runway 05L issued");
  });

  // The target is stored in SI and the charts are in feet and knots, so a raw
  // target read "190 kt … target 98.7" in the one tooltip a person judges by.
  it("reads a target in the unit of the chart it sits on", () => {
    renderWindow();
    expect(screen.getByLabelText(/speed 310 kt \(word 21\), issued 0 s, target 309 kt/)).toBeTruthy();
    expect(screen.getByLabelText(/altitude 10000 ft \(word 10\), issued 0 s, target 9912 ft/)).toBeTruthy();
    expect(screen.getByLabelText(/heading \+90° \(word 9\), issued 0 s, target 88\.4°/)).toBeTruthy();
  });

  it("explains an absorbed span rather than only shading it", () => {
    renderWindow();
    expect(
      screen.getByLabelText(/read but not worded: -2\.4 over 150–168 s \(small change\)/),
    ).toBeTruthy();
  });

  it("closes on its own button and on Escape", () => {
    const { onClose } = renderWindow();
    fireEvent.click(screen.getByLabelText("Close the read-back check"));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});

// ── the sentence flown by rule, beside the aircraft (T6) ─────────────────────

describe("the flown sentence on the charts", () => {
  // Every row, not just the first: an implementation that repeated element 0
  // would have passed the version of this test that only looked at [0].
  it("converts every row of the flown track into each chart's own unit", () => {
    const { flight } = selection();
    const feet = flownTrace(flight, "altitude");
    const knots = flownTrace(flight, "speed");
    expect(feet).toHaveLength(flight.geometric.heightM.length);
    flight.geometric.heightM.forEach((metres, row) => {
      expect(feet[row]).toBeCloseTo(metres / FEET_TO_METERS, 6);
    });
    flight.geometric.groundSpeedMps.forEach((mps, row) => {
      expect(knots[row]).toBeCloseTo(metresPerSecondToKnots(mps), 6);
    });
    expect(new Set(feet).size).toBeGreaterThan(1);
  });

  // The observed heading chart plots the UNWRAPPED course and the flown track
  // carries the wrapped one, so the flown line has to be unwrapped against itself
  // or it drops 360° the moment the rule-follower passes the cut.
  it("unwraps the flown heading so it does not jump at the cut", () => {
    const { vocabulary, flight } = selection();
    const wrapping = {
      ...flight,
      geometric: { ...flight.geometric, relCourseDeg: [170, 175, -179, -174, -170] },
    };
    expect(flownTrace(wrapping, "heading")).toEqual([170, 175, 181, 186, 190]);
    expect(vocabulary.readingRule).toBe("plateau-v11");
  });

  // The two tracks are on DIFFERENT clocks — 2 s rows against 1 s steps — so this
  // is pinned by VALUE at a time only one of them has a row at. Flooring each to
  // its own row instead put the positions up to a second apart, ~70 m of flight
  // (measured on the real export: p95 108 m, max 191 m), while the mean printed
  // beside it on screen interpolates. The three loose assertions this replaces
  // all passed with the wrong track indexed.
  it("reads both tracks at the SAME instant, between their rows", () => {
    const { flight } = selection();
    expect(gapAtS(flight, 0)).toBeCloseTo(0, 6);   // they share a first point by construction

    const seconds = 141;                            // odd: the observed rows are even
    const lerp = (tS: number[], v: number[]) => {
      let row = 0;
      while (row + 1 < tS.length && tS[row + 1] <= seconds) row += 1;
      const span = tS[row + 1] - tS[row];
      return v[row] + ((v[row + 1] - v[row]) * (seconds - tS[row])) / span;
    };
    const expected = Math.hypot(
      lerp(flight.geometric.tS, flight.geometric.toGoM) - lerp(flight.observed.tS, flight.observed.toGoM),
      lerp(flight.geometric.tS, flight.geometric.crossM) - lerp(flight.observed.tS, flight.observed.crossM),
    );
    expect(gapAtS(flight, seconds)).toBeCloseTo(expected, 6);
    expect(expected).toBeGreaterThan(0);
  });

  // After the flown sentence stops there is nothing to compare against, and a
  // number there would be measuring the stopping rule instead of the words.
  it("has no gap to report once the words have stopped", () => {
    const { flight } = selection();
    const stopped = flight.geometric.tS[flight.geometric.tS.length - 1];
    expect(gapAtS(flight, stopped + 1)).toBeNull();
  });

  it("says where the flown sentence ended and over how much it was compared", () => {
    renderWindow();
    expect(screen.getAllByText(/flown by rule/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/across the threshold plane, 1481 m from the threshold/)).toBeTruthy();
    expect(screen.getByText(/compared over 100% of/)).toBeTruthy();
  });

  it("tells the reader the second line is a baseline, not a model's answer", () => {
    renderWindow();
    expect(screen.getByText(/never a model's answer/)).toBeTruthy();
    expect(screen.getByText(/what the words did not say/)).toBeTruthy();
  });
});
