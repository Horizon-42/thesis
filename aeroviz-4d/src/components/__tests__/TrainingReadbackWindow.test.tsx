/**
 * The read-back check under `box-v2-wedge` (design §3.2, T4b / T15).
 *
 * What it has to get right is the BOX: the right interval over the right stretch
 * of time, the wedge narrowing to its target, the verdict computed on the
 * smoothed signal rather than the raw one, and the red overlay covering exactly
 * the rows the count says are outside.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import TrainingReadbackWindow, {
  altitudeSegments,
  boxSpans,
  extent,
  insideReading,
  rowAt,
  runsOf,
  segmentNodes,
} from "../TrainingReadbackWindow";
import {
  headingBoxDeg,
  parseTrainingSample,
  speedBoxMps,
  trainingContainment,
  type TrainingFlight,
  type TrainingPrior,
  type TrainingReadingRule,
  type TrainingVocabulary,
} from "../../data/trainingSample";
import {
  MOCK_EVENT_TIMES_S,
  MOCK_SPEC,
  mockPriorSample,
  mockSample,
} from "../../data/__tests__/trainingSample.fixture";

function read(raw: unknown = mockSample(), kind: "vocabulary-readback" | "prior-generated" = "vocabulary-readback") {
  const parsed = parseTrainingSample(raw, kind);
  if (!parsed.ok) throw new Error(parsed.problem);
  return {
    flight: parsed.value.flights[0] as TrainingFlight,
    vocabulary: parsed.value.vocabulary as TrainingVocabulary,
    reading: parsed.value.reading as TrainingReadingRule,
    prior: parsed.value.prior as TrainingPrior | undefined,
  };
}

/** The window renders through a PORTAL into `document.body` (AV7), so nothing it
 *  draws is under the render's own container — every query below goes to the
 *  document. */
function open(raw?: unknown, kind?: "vocabulary-readback" | "prior-generated") {
  const { flight, vocabulary, reading, prior } = read(raw, kind);
  render(
    <TrainingReadbackWindow
      layers={{ flown: true, model: true }}
      flight={flight}
      vocabulary={vocabulary}
      reading={reading}
      prior={prior}
      cursorS={0}
      onCursorChange={vi.fn()}
      onClose={vi.fn()}
    />,
  );
  return document.body;
}

describe("the boxes a chart draws", () => {
  it("merges consecutive events carrying the same word, and keeps the interval", () => {
    const { flight, vocabulary } = read();
    const spans = boxSpans(flight, vocabulary, "heading", flight.sentence.words);
    expect(spans.map((span) => span.word)).toEqual([9, 8, 7, 6, 5]);
    expect(spans[0]).toMatchObject({ startS: 0, endS: 30, word: 9 });
    // the span carries the box itself, not a centre: there is no centre here
    expect([spans[0].lo, spans[0].hi]).toEqual(headingBoxDeg(vocabulary, 9));
    expect(spans[spans.length - 1].endS).toBe(flight.durationS);
  });

  it("a speed span is its word's own edges", () => {
    const { flight, vocabulary } = read();
    for (const span of boxSpans(flight, vocabulary, "speed", flight.sentence.words)) {
      expect([span.lo, span.hi]).toEqual(speedBoxMps(vocabulary, span.word));
    }
  });

  it("cuts the altitude into segments on its own word, and names each one's target", () => {
    const { flight, vocabulary } = read();
    const segments = altitudeSegments(flight, vocabulary, flight.sentence.words);
    expect(segments.map((segment) => segment.word)).toEqual([5, 4, 1]);
    expect(segments.map((segment) => segment.targetM)).toEqual(
      [5, 4, 1].map((word) => MOCK_SPEC.altitudeTargetsM[word]),
    );
    // they cover every row exactly once, in order
    expect(segments[0].firstRow).toBe(0);
    expect(segments[segments.length - 1].lastRow).toBe(flight.observed.tS.length - 1);
    for (let i = 1; i < segments.length; i += 1) {
      expect(segments[i].firstRow).toBe(segments[i - 1].lastRow + 1);
    }
  });

  it("the wedge narrows to its target: the box at a segment's end is the tightest it gets", () => {
    const { flight, vocabulary } = read();
    for (const segment of altitudeSegments(flight, vocabulary, flight.sentence.words)) {
      const atStart = flight.envelope.altHiM[segment.firstRow] - flight.envelope.altLoM[segment.firstRow];
      const atEnd = flight.envelope.altHiM[segment.lastRow] - flight.envelope.altLoM[segment.lastRow];
      expect(atEnd).toBeLessThan(atStart);
      // and it closes ON the target, not beside it
      expect(flight.envelope.altLoM[segment.lastRow]).toBeLessThanOrEqual(segment.targetM);
      expect(flight.envelope.altHiM[segment.lastRow]).toBeGreaterThanOrEqual(segment.targetM);
    }
  });
});

describe("the window", () => {
  it("names the three charts, and says the altitude word is a TARGET", () => {
    open();
    expect(screen.getByText(/^heading — ° of ground track relative to the course, wrapped/)).toBeTruthy();
    expect(screen.getByText(/^altitude —.*the word is a TARGET, the box is the wedge/)).toBeTruthy();
    expect(screen.getByText(/^speed — m\/s ground speed/)).toBeTruthy();
  });

  it("reads out containment on every chart, because that is the criterion", () => {
    const { flight } = read();
    open();
    const rows = flight.observed.tS.length;
    for (const kind of ["heading", "altitude", "speed"]) {
      expect(
        screen.getByText(new RegExp(`^${kind} —.*every one of the ${rows} rows is inside its box`)),
      ).toBeTruthy();
    }
  });

  it("draws the raw rows as well as the signal the boxes judge", () => {
    const container = open();
    // Two lines per chart: a window that showed only the smoothed one would be
    // showing a signal nobody flew.
    expect(container.querySelectorAll("polyline.training-readback-raw")).toHaveLength(3);
    expect(container.querySelectorAll("polyline.training-readback-trace").length).toBeGreaterThanOrEqual(3);
  });

  it("draws one footprint per word in the plan, and one wedge per altitude segment", () => {
    const { flight, vocabulary } = read();
    const container = open();
    expect(container.querySelectorAll("polygon.training-readback-plan-box")).toHaveLength(
      flight.sentence.eventTimesS.length,
    );
    expect(container.querySelectorAll("polygon.training-readback-fan")).toHaveLength(
      altitudeSegments(flight, vocabulary, flight.sentence.words).length,
    );
  });

  it("the envelope switch takes the boxes out of the plan with it", () => {
    const { flight, vocabulary, reading } = read();
    render(
      <TrainingReadbackWindow
        layers={{ flown: false, model: false }}
        flight={flight}
        vocabulary={vocabulary}
        reading={reading}
        cursorS={0}
        onCursorChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );
    const container = document.body;
    expect(container.querySelectorAll("polygon.training-readback-plan-box")).toHaveLength(0);
    // the wedge is the chart's subject rather than a layer, so it stays
    expect(container.querySelectorAll("polygon.training-readback-fan").length).toBeGreaterThan(0);
  });

  it("marks where each box opens, on the track and on every chart", () => {
    const { flight } = read();
    const container = open();
    const events = flight.sentence.eventTimesS.length;
    // one per event in the plan, and one per event on each of three charts
    expect(container.querySelectorAll("circle.training-readback-node")).toHaveLength(events * 4);
  });

  it("says there is no sentence flown by rule, and why", () => {
    open();
    expect(screen.getByText(/There is NO sentence flown by rule here/)).toBeTruthy();
    expect(screen.getByText(/replay gate, which is not built/)).toBeTruthy();
  });

  it("states the smoothing the verdict was computed on", () => {
    const { flight } = read();
    open();
    expect(
      screen.getByText(
        new RegExp(`${flight.courseWindowRows} and ${flight.signalWindowRows} rows at this flight's ${flight.dtS} s step`),
      ),
    ).toBeTruthy();
  });

  it("draws the MODEL's boxes outlined, and says how well they hold the aircraft", () => {
    const container = open(mockPriorSample(), "prior-generated");
    expect(container.querySelectorAll("rect.training-readback-model").length).toBeGreaterThan(0);
    expect(screen.getByText(/Its boxes hold heading \d+\/\d+, altitude \d+\/\d+, speed \d+\/\d+\./)).toBeTruthy();
    expect(screen.getByText(/teacher-forced-next-word/)).toBeTruthy();
  });
});

describe("the red overlay and the count are one reading", () => {
  it("covers exactly the rows the verdict calls outside", () => {
    const { flight, vocabulary } = read();
    const moved: TrainingFlight = {
      ...flight,
      observed: {
        ...flight.observed,
        readSpeedMps: flight.observed.readSpeedMps.map((value, row) =>
          row >= 10 && row <= 13 ? value + 40 : value),
      },
    };
    const measured = trainingContainment(moved, flight.envelope, vocabulary, flight.sentence.words);
    expect(measured.speed.outside).toBe(4);
    // `runsOf` is what the overlay is drawn from: one run of four rows, not four
    // marks and not five.
    expect(runsOf(measured.speed.inside.map((ok) => !ok))).toEqual([[10, 13]]);
  });

  it("a single outside row is still a run", () => {
    expect(runsOf([false, true, false])).toEqual([[1, 1]]);
    expect(runsOf([true, true])).toEqual([[0, 1]]);
    expect(runsOf([false, false])).toEqual([]);
  });

  it("says the verdict in this vocabulary's own terms", () => {
    expect(insideReading({ rows: 61, outside: 0 })).toBe("every one of the 61 rows is inside its box");
    expect(insideReading({ rows: 61, outside: 3 })).toBe("58 of 61 rows inside — 3 outside");
  });
});

describe("the small readers", () => {
  it("rowAt is the last row at or before a time, and clamps at both ends", () => {
    const tS = [0, 2, 4, 6];
    expect(rowAt(tS, -1)).toBe(0);
    expect(rowAt(tS, 3)).toBe(1);
    expect(rowAt(tS, 4)).toBe(2);
    expect(rowAt(tS, 99)).toBe(3);
  });

  it("extent pads, and never divides by zero on a constant column", () => {
    expect(extent([0, 10])).toEqual([-1.2, 11.2]);
    expect(extent([5, 5])).toEqual([4, 6]);
  });

  it("segmentNodes puts one node per event on the track's own rows", () => {
    const { flight } = read();
    const nodes = segmentNodes(flight.observed, flight.sentence.eventTimesS);
    expect(nodes.map((node) => node.eventS)).toEqual(MOCK_EVENT_TIMES_S);
    expect(nodes.every((node) => flight.observed.tS[node.row] <= node.eventS)).toBe(true);
    // an event past the end of the track gets no node
    expect(segmentNodes({ tS: [0, 2] }, [0, 2, 4])).toHaveLength(2);
  });
});
