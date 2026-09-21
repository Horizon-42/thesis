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
  bandCoverage,
  bandCoverageReading,
  extent,
  rowAt,
  runsOf,
  segmentNodes,
  verticalRamps,
} from "../TrainingReadbackWindow";
import {
  parseTrainingSample,
  speedCentreMps,
  speedToleranceMps,
} from "../../data/trainingSample";
import {
  DESCEND_24,
  mockPriorSample,
  mockSample,
} from "../../data/__tests__/trainingSample.fixture";

function selection() {
  const parsed = parseTrainingSample(mockSample(), "vocabulary-readback");
  if (!parsed.ok) throw new Error(parsed.problem);
  return {
    vocabulary: parsed.value.vocabulary,
    flight: parsed.value.flights[0],
    geometry: parsed.value.geometry,
  };
}

function renderWindow(overrides: Partial<React.ComponentProps<typeof TrainingReadbackWindow>> = {}) {
  const { vocabulary, flight, geometry } = selection();
  const onCursorChange = vi.fn();
  const onClose = vi.fn();
  render(
    <TrainingReadbackWindow
      layers={{ flown: true, model: true }}
      flight={flight}
      vocabulary={vocabulary}
      geometry={geometry}
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

  it("draws the speed step at the word's own centre, in m/s", () => {
    const { vocabulary, flight } = selection();
    expect(chartLevels(flight, vocabulary, "speed")[0].level).toBeCloseTo(
      speedCentreMps(vocabulary, 11),
      6,
    );
  });

  // The vertical word is a SLOPE. A caller that asked for levels would get a
  // list of horizontal lines at heights the word never names, and the chart
  // would look entirely plausible — so it throws instead.
  it("refuses to give the vertical word a level at all", () => {
    const { vocabulary, flight } = selection();
    expect(() => chartLevels(flight, vocabulary, "vertical")).toThrow(/angle, not a level/);
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
  // the speed branch returns before `settledS` is read at all, so a fallback of
  // 0 would pass unnoticed there.
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

  it("still judges a never-settled deceleration by its word alone", () => {
    const { vocabulary, flight } = selection();
    const last = flight.instructions[flight.instructions.length - 1];
    expect(last.settledS).toBeNull();
    expect(chartLevels(flight, vocabulary, "speed").pop()?.level).toBe(
      speedCentreMps(vocabulary, 5),
    );
  });
});

describe("verticalRamps", () => {
  // The word is the slope of height against GROUND COVERED, so the ramp has to
  // fall by tan(angle) × distance — not by anything per second. At these speeds
  // the two differ by tens of metres over a segment.
  it("draws the height the word implies, over the ground the aircraft covered", () => {
    const { vocabulary, flight } = selection();
    const ramp = verticalRamps(flight, vocabulary)[1];
    expect(ramp.instruction.word).toBe(DESCEND_24);
    const { tS, heightM, pathM } = flight.observed;
    const first = rowAt(tS, ramp.instruction.issuedS);
    const last = ramp.tS.length - 1;
    const run = pathM[rowAt(tS, ramp.tS[last])] - pathM[first];
    expect(ramp.centre[0]).toBeCloseTo(heightM[first], 6);
    expect(ramp.centre[last]).toBeCloseTo(
      heightM[first] - Math.tan((2.4 * Math.PI) / 180) * run,
      6,
    );
  });

  // Every segment starts again from the OBSERVED height, the way the labeller's
  // piecewise fit does. Chaining them instead would carry one segment's error
  // into the next and make a late word look misread because an early one was.
  it("re-anchors each segment on the observed height", () => {
    const { vocabulary, flight } = selection();
    const { tS, heightM } = flight.observed;
    for (const ramp of verticalRamps(flight, vocabulary)) {
      expect(ramp.centre[0]).toBeCloseTo(heightM[rowAt(tS, ramp.tS[0])], 6);
    }
  });

  // The shallower edge loses less height, so it stays ABOVE — for the climb mode
  // too, where "shallower" means a steeper climb. Swapping them keeps the fan
  // exactly as wide and draws it inside out.
  it("opens the fan with distance, shallow edge above", () => {
    const { vocabulary, flight } = selection();
    const ramp = verticalRamps(flight, vocabulary)[2];
    const last = ramp.tS.length - 1;
    expect(ramp.lo[0]).toBeCloseTo(ramp.hi[0], 6);
    expect(ramp.lo[last]).toBeGreaterThan(ramp.centre[last]);
    expect(ramp.hi[last]).toBeLessThan(ramp.centre[last]);
    expect(ramp.lo[last] - ramp.hi[last]).toBeGreaterThan(ramp.lo[1] - ramp.hi[1]);
  });

  // The two errors the chart is there to separate: the fitted angle's line is
  // NOT the word's line, and the distance between them is what binning cost.
  it("draws the fitted angle apart from the word it was rounded to", () => {
    const { vocabulary, flight } = selection();
    const ramp = verticalRamps(flight, vocabulary)[1];
    const last = ramp.tS.length - 1;
    expect(ramp.instruction.target).not.toBe(2.4);
    expect(ramp.fitted[last]).not.toBeCloseTo(ramp.centre[last], 3);
  });
});

describe("bandCoverage", () => {
  // The fixture's profile is built FROM the words, so the aircraft sits inside
  // the vertical band nearly all the time; what is pinned is that the number is
  // a share of the rows, not that it is 1.
  it("reports the share of rows inside the band of the word in force", () => {
    const { vocabulary, flight } = selection();
    const vertical = bandCoverage(flight, vocabulary, "vertical");
    expect(vertical.hasTolerance).toBe(true);
    expect(vertical.judged).toBeGreaterThan(0);
    expect(vertical.inside / vertical.judged).toBeGreaterThan(0.5);
    expect(vertical.inside).toBeLessThanOrEqual(vertical.judged);
  });

  it("falls when the aircraft leaves the band, and says so on the rows", () => {
    const { vocabulary, flight } = selection();
    const before = bandCoverage(flight, vocabulary, "speed");
    const drifted = {
      ...flight,
      observed: {
        ...flight.observed,
        groundSpeedMps: flight.observed.groundSpeedMps.map((mps) => mps + 12),
      },
    };
    const after = bandCoverage(drifted, vocabulary, "speed");
    expect(after.inside / after.judged).toBeLessThan(before.inside / before.judged);
    // and the rows it names are exactly the ones the red overlay draws
    expect(after.outside.filter(Boolean).length).toBe(after.judged - after.inside);
  });

  // THE distinction this type exists for. "No tolerance" is a statement about
  // the VOCABULARY; "nothing to measure" is a statement about this flight. A
  // kind whose instructions are missing used to report the first, which is a
  // false claim about the word (V36).
  it("keeps 'no tolerance' apart from 'nothing to measure'", () => {
    const { vocabulary, flight } = selection();
    const heading = bandCoverage(flight, vocabulary, "heading");
    expect(heading.hasTolerance).toBe(false);
    expect(bandCoverageReading(heading)).toBe("no tolerance on this word");

    const unworded = {
      ...flight,
      instructions: flight.instructions.filter((item) => item.kind !== "vertical"),
    };
    const vertical = bandCoverage(unworded, vocabulary, "vertical");
    expect(vertical.hasTolerance).toBe(true);
    expect(vertical.judged).toBe(0);
    expect(bandCoverageReading(vertical)).toBe("no worded stretch to measure");
  });

  // The band's edges are INCLUSIVE: a measurement exactly on the tolerance has
  // obeyed the word. Exclusive edges would report a flight holding the word
  // exactly as disobeying it.
  it("counts a row exactly on the edge of the band as inside", () => {
    const { vocabulary, flight } = selection();
    const level = chartLevels(flight, vocabulary, "speed")[0];
    const edge = level.level + speedToleranceMps(vocabulary, level.instruction.word);
    const onTheEdge = {
      ...flight,
      observed: { ...flight.observed, groundSpeedMps: flight.observed.groundSpeedMps.map(() => edge) },
    };
    const coverage = bandCoverage(onTheEdge, vocabulary, "speed");
    expect(coverage.outside[rowAt(flight.observed.tS, level.instruction.issuedS)]).toBe(false);
  });

  // Rows no word covers are judged by nothing, and must not be drawn as a
  // disagreement: after a plateau settles and before the next instruction there
  // is no band to be outside of.
  it("judges only the rows a word covers", () => {
    const { vocabulary, flight } = selection();
    const coverage = bandCoverage(flight, vocabulary, "vertical");
    expect(coverage.judged).toBeLessThanOrEqual(flight.observed.tS.length);
    expect(coverage.outside).toHaveLength(flight.observed.tS.length);
  });
});

describe("runsOf", () => {
  it("returns every contiguous run, single rows included", () => {
    expect(runsOf([false, true, true, false, true, false])).toEqual([[1, 2], [4, 4]]);
    expect(runsOf([true, true])).toEqual([[0, 1]]);
    expect(runsOf([false, false])).toEqual([]);
  });
});

describe("TrainingReadbackWindow", () => {
  it("renders into document.body, not into its parent (AV7)", () => {
    const { container } = render(
      <div className="flight-ops-panel">
        <TrainingReadbackWindow
          layers={{ flown: true, model: true }}
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
    expect(screen.getByText(/^vertical —/)).toBeTruthy();
    expect(screen.getByText(/^speed —/)).toBeTruthy();
    expect(screen.queryByText(/^runway —/)).toBeNull();
    expect(screen.queryByText(/^duration —/)).toBeNull();
    expect(screen.queryByText(/^terminal —/)).toBeNull();
  });

  // The readout is taken from the arrays at the cursor, never from a pixel.
  it("reads out the word in force and the measured value at the cursor", () => {
    renderWindow({ cursorS: 140 });
    // at 140 s the third speed instruction (issued 108 s) is in force, and the
    // readout gives the BAND, because that is what the word says
    expect(screen.getByText(/speed —.*in force: 93 m\/s±2\.8 \(issued 108 s, settled 126 s\)/)).toBeTruthy();
    // and the vertical segment issued at 130 s, which runs to the end of the track
    expect(screen.getByText(/vertical —.*in force: ↓3\.1°±0\.22 \(issued 130 s, settled 262 s\)/)).toBeTruthy();
  });

  it("says when an instruction never settled inside the track", () => {
    renderWindow({ cursorS: 240 });
    expect(screen.getByText(/speed —.*in force: 79 m\/s±2\.4 \(issued 188 s, never settled\)/)).toBeTruthy();
  });

  // The tolerance is the rule for "did it obey", so it gives a readout nothing
  // else could: how much of the flight sat inside the band it was told to hold.
  // The heading word has no tolerance, and the chart says so rather than
  // printing a number it would have had to invent.
  it("gives the share of time inside the band, and none where there is no band", () => {
    renderWindow();
    expect(screen.getByText(/vertical —.*inside the band \d+ % of the time/)).toBeTruthy();
    expect(screen.getByText(/speed —.*inside the band \d+ % of the time/)).toBeTruthy();
    expect(screen.getByText(/heading —.*no tolerance on this word/)).toBeTruthy();
    expect(screen.queryByText(/heading —.*inside the band/)).toBeNull();
  });

  // The runway word names the frame the others are measured in; it is not a
  // place the aircraft was, so the plan view must not mark one.
  it("marks where each instruction was issued, except the runway", () => {
    const { flight, vocabulary } = renderWindow();
    const marks = document.body.querySelectorAll(".training-readback-issued-mark");
    const issued = flight.instructions.filter((item) => item.kind !== "runway").length;
    expect(marks.length).toBe(issued);
    expect(screen.getByLabelText(/heading \+50° issued at 70 s/)).toBeTruthy();
    // the speed band's two edges are polylines in the plan view, not marks
    expect(vocabulary.runwayIdents).toContain(flight.runway);
    expect(document.body.textContent).not.toContain("runway 05L issued");
  });

  // The tooltip a person judges a word by: the word WITH its band, and the
  // unbinned value it was read from, in that chart's own unit.
  it("reads a word and the value it was read from, in the unit of its chart", () => {
    renderWindow();
    expect(screen.getByLabelText(/speed 121 m\/s±3\.6 \(word 11\), issued 0 s, read 120\.4 m\/s/)).toBeTruthy();
    expect(screen.getByLabelText(/vertical ↓2\.4°±0\.17 \(word 3\), issued 70 s, read 2\.31°/)).toBeTruthy();
    expect(screen.getByLabelText(/heading \+90° \(word 18\), issued 0 s, read 88\.40°/)).toBeTruthy();
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
  it("plots every row of the flown track on each chart's own axis", () => {
    const { flight } = selection();
    const height = flownTrace(flight, "vertical");
    const speed = flownTrace(flight, "speed");
    expect(height).toEqual(flight.geometric.heightM);
    expect(speed).toEqual(flight.geometric.groundSpeedMps);
    expect(new Set(height).size).toBeGreaterThan(1);
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
    expect(vocabulary.readingRule).toBe("segment-v13");
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
    expect(screen.getByText(/compared over 262 s — 100% of/)).toBeTruthy();
  });

  it("tells the reader the second line is a baseline, not a model's answer", () => {
    renderWindow();
    expect(screen.getByText(/never a model's answer/)).toBeTruthy();
    expect(screen.getByText(/what the words did not say/)).toBeTruthy();
  });
});

// ── what is actually DRAWN (T11's review: every one of these mutations used to
//    survive, because the geometry was tested and the pixels were not) ────────

describe("the corridor on the charts", () => {
  function svg(): HTMLElement {
    const node = document.body.querySelector(".training-readback-svg");
    if (!node) throw new Error("no chart svg");
    return node as unknown as HTMLElement;
  }

  // The commit's headline drawing. Deleting it left every test green.
  it("draws the two speed edges in the plan view", () => {
    const { flight } = renderWindow();
    const edges = svg().querySelectorAll(".training-readback-speed-edge");
    expect(edges).toHaveLength(2);
    const points = (edges[0].getAttribute("points") ?? "").trim().split(/\s+/);
    expect(points).toHaveLength(flight.geometric.speedBand.low.tS.length);
  });

  // A fan is a closed polygon: the lower edge forward, the upper edge back. Drop
  // the reversal and each one becomes a bow-tie that still renders.
  it("closes each vertical fan around its own rows", () => {
    const { flight, vocabulary } = renderWindow();
    const ramps = verticalRamps(flight, vocabulary);
    const fans = svg().querySelectorAll(".training-readback-fan");
    expect(fans).toHaveLength(ramps.length);
    fans.forEach((fan, index) => {
      const points = (fan.getAttribute("points") ?? "").trim().split(/\s+/);
      expect(points).toHaveLength(ramps[index].tS.length * 2);
      // the first point of the return leg is the LAST row's steeper edge: with
      // the reversal dropped it would be the first row's instead
      const half = ramps[index].tS.length;
      expect(points[half].split(",")[0]).toBe(points[half - 1].split(",")[0]);
    });
  });

  // The other half of "two errors on one chart": without this line the binning
  // cost is not drawn at all.
  it("draws the fitted angle as a line of its own, apart from the word's", () => {
    const { flight, vocabulary } = renderWindow();
    const ramps = verticalRamps(flight, vocabulary);
    const fitted = svg().querySelectorAll(".training-readback-fitted");
    const word = svg().querySelectorAll(".training-readback-word");
    expect(fitted).toHaveLength(ramps.length);
    expect(word).toHaveLength(ramps.length);
    // the fitted angle is not the word's angle on this fixture, so the two
    // polylines must not be the same string
    expect(fitted[1].getAttribute("points")).not.toBe(word[1].getAttribute("points"));
  });

  // Swapping y and height gives a negative height, which SVG drops silently.
  it("draws a speed band with a positive height", () => {
    const { flight, vocabulary } = renderWindow();
    const bands = svg().querySelectorAll(".training-readback-band");
    expect(bands).toHaveLength(chartLevels(flight, vocabulary, "speed").length);
    bands.forEach((band) => {
      expect(Number(band.getAttribute("height"))).toBeGreaterThan(0);
    });
  });

  // The red overlay is the only rendered output of the tolerance contract; a
  // sign flip in it was invisible to the whole suite.
  it("marks the measured trace red exactly where it left the band", () => {
    const { vocabulary, flight, geometry } = selection();
    const drifted = {
      ...flight,
      observed: {
        ...flight.observed,
        groundSpeedMps: flight.observed.groundSpeedMps.map((mps) => mps + 12),
      },
    };
    render(
      <TrainingReadbackWindow
        layers={{ flown: true, model: true }}
        flight={drifted}
        vocabulary={vocabulary}
        geometry={geometry}
        cursorS={0}
        onCursorChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    const marks = document.body.querySelectorAll(".training-readback-outside");
    expect(marks.length).toBeGreaterThan(0);
    // and the runs it draws are the runs the reading names
    const coverage = bandCoverage(drifted, vocabulary, "speed");
    expect(runsOf(coverage.outside).length).toBeGreaterThan(0);
  });

  it("draws no red where the measurement stayed inside", () => {
    const { vocabulary, flight, geometry } = selection();
    const held = {
      ...flight,
      instructions: flight.instructions.filter((item) => item.kind === "speed"),
      observed: {
        ...flight.observed,
        groundSpeedMps: flight.observed.groundSpeedMps.map(() => 121),
      },
    };
    render(
      <TrainingReadbackWindow
        layers={{ flown: true, model: true }}
        flight={held}
        vocabulary={vocabulary}
        geometry={geometry}
        cursorS={0}
        onCursorChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    // the first speed word IS 121 m/s, so the opening stretch is inside it
    const coverage = bandCoverage(held, vocabulary, "speed");
    expect(coverage.outside[0]).toBe(false);
  });

  // The legend quotes the artefact, not this file: both numbers are claims the
  // design makes (V32 / V34) and both were hardcoded past the block.
  it("quotes the height floor and the joint-envelope flag from the artefact", () => {
    const { vocabulary, flight, geometry } = selection();
    render(
      <TrainingReadbackWindow
        layers={{ flown: true, model: true }}
        flight={flight}
        vocabulary={vocabulary}
        geometry={{ ...geometry, heightFloorM: 17, bandsAreJoint: true }}
        cursorS={0}
        onCursorChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByText(/closes at 17 m/)).toBeTruthy();
    expect(screen.getByText(/The two bands are the joint envelope/)).toBeTruthy();
  });
});

describe("extent", () => {
  // The guard is not decoration now: a level flight holding a level word can
  // make trace, flown line and band all constant, and `yFor` would return NaN
  // for every row of the chart.
  it("pads a range, and widens a constant one instead of dividing by zero", () => {
    const [low, high] = extent([10, 20]);
    expect(low).toBeLessThan(10);
    expect(high).toBeGreaterThan(20);
    const [flatLow, flatHigh] = extent([5, 5, 5]);
    expect(flatHigh - flatLow).toBeGreaterThan(0);
  });
});

describe("verticalRamps without a settled time", () => {
  // `settledS` is null for a manoeuvre that never settles. A vertical segment
  // normally settles where the next is issued, so this branch is only reachable
  // on an export that wrote one — and it must fall back to the next segment's
  // start, not to zero.
  it("runs to the next segment when its own end is missing", () => {
    const { vocabulary, flight } = selection();
    const loosened = {
      ...flight,
      instructions: flight.instructions.map((item) =>
        item.kind === "vertical" && item.issuedS === 70 ? { ...item, settledS: null } : item,
      ),
    };
    const ramp = verticalRamps(loosened, vocabulary)[1];
    expect(ramp.instruction.settledS).toBeNull();
    expect(ramp.endS).toBe(130);
    expect(ramp.tS[ramp.tS.length - 1]).toBe(130);
  });
});

describe("gapAtS at the moment the words stop", () => {
  // At the flown track's LAST step both tracks still exist, so the gap is a
  // number; one step later there is nothing to compare against. `>=` here would
  // report "the words have stopped" a second before they did.
  it("reads a gap at the last step and none after it", () => {
    const { flight } = selection();
    const stopped = flight.geometric.tS[flight.geometric.tS.length - 1];
    expect(gapAtS(flight, stopped)).not.toBeNull();
    expect(gapAtS(flight, stopped + 0.5)).toBeNull();
  });
});

describe("the model's line in the read-back window", () => {
  function priorProps() {
    const parsed = parseTrainingSample(mockPriorSample(), "prior-generated");
    if (!parsed.ok) throw new Error(parsed.problem);
    return {
      flight: parsed.value.flights[0],
      vocabulary: parsed.value.vocabulary,
      geometry: parsed.value.geometry,
      prior: parsed.value.prior,
      cursorS: 0,
      onCursorChange: vi.fn(),
      onClose: vi.fn(),
    };
  }

  // One line in the plan view and one on each of the three charts — the same
  // rules and the same event times as the orange line, so the distance between
  // them is the WORDS.
  it("draws the model's sentence on the plan view and every chart", () => {
    render(<TrainingReadbackWindow layers={{ flown: true, model: true }} {...priorProps()} />);
    expect(document.body.querySelectorAll(".training-readback-model")).toHaveLength(4);
  });

  it("draws none of it when the set carries no model", () => {
    renderWindow();
    expect(document.body.querySelectorAll(".training-readback-model")).toHaveLength(0);
  });

  // THE sentence that stops the whole window being read as free generation.
  it("says how the model was asked, and whether it had seen these flights", () => {
    render(<TrainingReadbackWindow layers={{ flown: true, model: true }} {...priorProps()} />);
    expect(screen.getByText(/teacher-forced-next-word/)).toBeTruthy();
    expect(screen.getByText(/did not generate this sentence/)).toBeTruthy();
    expect(screen.getByText(/never fitted on any of these flights/)).toBeTruthy();
    expect(screen.getByText(/first said "landed" at 188 s/)).toBeTruthy();
  });
});

describe("where the words cut the flown track", () => {
  // The flown line is one integration with no seams in it. The nodes are what
  // make it readable as a SENTENCE: a turn that begins nowhere near one was not
  // commanded there, it is the previous word still being flown.
  it("marks every event on the flown track, in the plan view and on each chart", () => {
    const { flight } = renderWindow();
    const events = flight.sentence.eventTimesS.length;
    // the plan view plus three charts
    expect(document.body.querySelectorAll(".training-readback-node")).toHaveLength(events * 4);
  });

  // An event is one cut, however many kinds changed at it — three instructions
  // issued at the same instant are not three places on the line.
  it("cuts once per event, not once per instruction", () => {
    const { flight } = renderWindow();
    const nodes = segmentNodes(flight.geometric, flight.sentence.eventTimesS);
    expect(nodes).toHaveLength(flight.sentence.eventTimesS.length);
    expect(nodes.length).toBeLessThan(flight.instructions.length);
    expect(nodes[0].eventS).toBe(0);
  });

  // The flown sentence can stop before the aircraft did. A node clamped to the
  // last row would claim the words cut the line where they never reached.
  it("drops an event the flown track never got to", () => {
    const { flight } = renderWindow();
    const short = { ...flight.geometric, tS: flight.geometric.tS.slice(0, 100) };
    const nodes = segmentNodes(short, flight.sentence.eventTimesS);
    expect(nodes.length).toBeLessThan(flight.sentence.eventTimesS.length);
    expect(nodes.every((node) => node.eventS <= 99)).toBe(true);
  });

  it("marks nothing when the line it would cut is switched off", () => {
    const { vocabulary, flight, geometry } = selection();
    render(
      <TrainingReadbackWindow
        layers={{ flown: false, model: false }}
        flight={flight}
        vocabulary={vocabulary}
        geometry={geometry}
        cursorS={0}
        onCursorChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    expect(document.body.querySelectorAll(".training-readback-node")).toHaveLength(0);
  });
});
