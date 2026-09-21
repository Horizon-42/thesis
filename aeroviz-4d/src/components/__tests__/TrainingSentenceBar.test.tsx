/**
 * The sentence bar's invariants, on the T2 fixture (design §7, T4a).
 *
 * The three that matter are the ones an even-grid drawing would pass anyway:
 * the axis is real time (so 26 s and 58 s are not the same width), the bands run
 * to the end of the TRACK rather than to the last event, and clicking an event
 * puts the cursor on the artefact's own number.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState } = vi.hoisted(() => ({
  appState: { trainingSelection: null as unknown },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState }),
}));

import TrainingSentenceBar, { rowBands, spacedLabels } from "../TrainingSentenceBar";
import { parseTrainingSample, TRAINING_KINDS } from "../../data/trainingSample";
import {
  MOCK_EVENT_TIMES_S,
  MOCK_FLIGHT,
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

const TRACK_S = MOCK_FLIGHT.durationS;

function bandRect(name: RegExp | string): SVGRectElement {
  const group = screen.getByLabelText(name);
  const rect = group.querySelector("rect");
  if (!rect) throw new Error(`no rect under ${name}`);
  return rect as unknown as SVGRectElement;
}

describe("TrainingSentenceBar", () => {
  beforeEach(() => {
    appState.trainingSelection = selection();
  });

  it("draws nothing until Training publishes a flight", () => {
    appState.trainingSelection = null;
    const { container } = render(<TrainingSentenceBar />);
    expect(container.firstChild).toBeNull();
  });

  it("has one row per word kind, labelled", () => {
    render(<TrainingSentenceBar />);
    for (const label of ["Heading (°)", "Vertical (°)", "Speed (m/s)", "Runway", "Gap (s)", "Terminal"]) {
      expect(screen.getByText(label)).toBeTruthy();
    }
    expect(TRAINING_KINDS).toHaveLength(6);
  });

  // The bands merge consecutive events carrying the same word: at most events
  // only one column changes, so an unmerged row reads as repeated instructions.
  it("merges a word held across several events into one band", () => {
    const { flight } = selection();
    expect(rowBands(flight, "heading")).toEqual([
      { startS: 0, endS: 70, word: 18 },
      { startS: 70, endS: 130, word: 10 },
      { startS: 130, endS: 188, word: 4 },
      { startS: 188, endS: TRACK_S, word: 0 },
    ]);
  });

  // The gap word measures the interval BEFORE its event, which is the one place
  // in the drawing an off-by-one is invisible: shifting it by one row still
  // produces a full, plausible row of durations. So it is pinned by value.
  it("draws each gap word over the interval it measures", () => {
    const { flight } = selection();
    expect(rowBands(flight, "duration")).toEqual([
      { startS: 0, endS: 26, word: 13 }, // 26 s / 2 s bins
      { startS: 26, endS: 70, word: 22 },
      { startS: 70, endS: 108, word: 19 },
      { startS: 108, endS: 130, word: 11 },
      { startS: 130, endS: 188, word: 29 },
      { startS: 188, endS: 222, word: 17 },
      { startS: 222, endS: TRACK_S, word: null }, // no gap word after the last event
    ]);
  });

  it("labels those gaps in seconds", () => {
    render(<TrainingSentenceBar />);
    expect(screen.getByLabelText(/duration 26 s \(word 13\), 0–26 s/)).toBeTruthy();
    expect(screen.getByLabelText(/duration 58 s \(word 29\), 130–188 s/)).toBeTruthy();
  });

  // A two-event sentence is the shortest the artefact contains, and it has no
  // interior band at all — the merge loop must still tile the whole track.
  it("tiles the whole track for the shortest sentence in the export", () => {
    const { flight } = selection();
    const short = {
      ...flight,
      sentence: {
        ...flight.sentence,
        eventTimesS: [0, 26],
        words: flight.sentence.words.slice(0, 2),
      },
    };
    for (const kind of TRAINING_KINDS) {
      const bands = rowBands(short, kind);
      expect(bands[0].startS).toBe(0);
      expect(bands[bands.length - 1].endS).toBe(TRACK_S);
      bands.forEach((band, index) => {
        if (index) expect(band.startS).toBe(bands[index - 1].endS);
        expect(band.endS).toBeGreaterThan(band.startS);
      });
    }
  });

  // THE axis test: the gaps are 26 s and 58 s, and the picture must say so. An
  // even-column bar — the retired grid's shape — would draw these equal.
  it("scales the bands by real time, not by event number", () => {
    render(<TrainingSentenceBar />);
    // The speed row's four bands start at 0, 26, 108 and 188 s. Their x positions
    // are compared rather than their widths, so the assertion is about the time
    // axis and not about the band inset.
    const xOf = (name: RegExp) => Number(bandRect(name).getAttribute("x"));
    const starts = [
      /speed target 121 m\/s/, /speed target 107 m\/s/,
      /speed target 93 m\/s/, /speed target 79 m\/s/,
    ].map(xOf);
    expect(starts[1] - starts[0]).toBeGreaterThan(0);
    expect((starts[2] - starts[1]) / (starts[1] - starts[0])).toBeCloseTo(82 / 26, 6);
    expect((starts[3] - starts[2]) / (starts[1] - starts[0])).toBeCloseTo(80 / 26, 6);
  });

  // The track outlives the sentence (a median 145 s in the real export): every
  // row's last band runs to durationS, and the gap row says so in words.
  it("runs the last band to the end of the track and names the unworded tail", () => {
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: "More notes" }));
    const lastEventS = MOCK_EVENT_TIMES_S[MOCK_EVENT_TIMES_S.length - 1];
    const landed = bandRect(/terminal landed/);
    const gapTail = bandRect(/no gap word/);
    // the landed band and the unworded tail both span the same closing stretch
    expect(Number(landed.getAttribute("width"))).toBeCloseTo(
      Number(gapTail.getAttribute("width")),
      6,
    );
    expect(
      screen.getByText(new RegExp(`last event is at ${lastEventS} s`)),
    ).toBeTruthy();
    expect(screen.getByText(new RegExp(`remaining ${TRACK_S - lastEventS} s carry no further word`))).toBeTruthy();
  });

  // The acceptance test of T4a: the cursor lands on the artefact's own number,
  // because no pixel→time arithmetic is involved anywhere.
  it("moves the cursor to an event's own time when that event is clicked", () => {
    render(<TrainingSentenceBar />);
    expect(screen.getByText("t = 0 s")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Event 3 at 70 s/ }));
    expect(screen.getByText("t = 70 s")).toBeTruthy();
  });

  it("moves the cursor to a band's start when the band is clicked", () => {
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(/heading target \+20°/));
    expect(screen.getByText("t = 130 s")).toBeTruthy();
  });

  // Five of the forty flights in KRDU's export have events 2 s apart. A
  // fixed-width hit box would hand a click on the earlier one to the later.
  it("keeps the hit areas apart when two events are seconds apart", () => {
    const { vocabulary, flight } = selection();
    appState.trainingSelection = {
      vocabulary,
      geometry: selection().geometry,
      flight: {
        ...flight,
        sentence: {
          ...flight.sentence,
          eventTimesS: [0, 26, 70, 108, 130, 132, 222], // 130 and 132: 2 s apart
        },
      },
    };
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: /Event 5 at 130 s/ }));
    expect(screen.getByText("t = 130 s")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Event 6 at 132 s/ }));
    expect(screen.getByText("t = 132 s")).toBeTruthy();
  });

  // The label is dropped, never the event: the hit area and the tooltip stay.
  it("drops a label that would overprint, and keeps the end of the axis", () => {
    expect(spacedLabels([0, 4, 100], 13)).toEqual([true, false, true]);
    expect(spacedLabels([0, 100, 120], 13, true)).toEqual([true, true, true]);
    // the end of the axis is kept whatever else has to go: here it evicts the
    // keeper 4 px before it, which a plain greedy pass would have preferred
    expect(spacedLabels([0, 100, 104], 13)).toEqual([true, true, false]);
    expect(spacedLabels([0, 100, 104], 13, true)).toEqual([true, false, true]);
  });

  it("clears the cursor when the flight changes, in the same render", () => {
    const { rerender } = render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: /Event 3 at 70 s/ }));
    expect(screen.getByText("t = 70 s")).toBeTruthy();

    const other = selection();
    other.flight = { ...other.flight, flightKey: "AAL456_23R_b2c3d4_1700000000" };
    appState.trainingSelection = other;
    rerender(<TrainingSentenceBar />);
    expect(screen.getByText("t = 0 s")).toBeTruthy();
  });

  // One flight, one moment, one number: the window is handed the bar's cursor
  // rather than keeping its own, so the two can never disagree about "now".
  it("opens the read-back check on the same cursor", () => {
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: /Event 3 at 70 s/ }));
    fireEvent.click(screen.getByRole("button", { name: "Read-back check" }));

    const window_ = screen.getByRole("dialog", { name: "Read-back check" });
    expect(window_.textContent).toContain("t = 70 s");   // the window
    expect(screen.getByText("t = 70 s")).toBeTruthy();   // and the bar, same number

    fireEvent.click(screen.getByRole("button", { name: "Close read-back check" }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  // A manoeuvre the words do not carry is drawn ON its kind's row with its
  // reason, so "the sentence misses this turn" is visible rather than argued.
  it("draws the absorbed manoeuvres with their reason", () => {
    render(<TrainingSentenceBar />);
    expect(screen.getByLabelText(/absorbed heading: \+4\.2 over 40–48 s, not worded \(short tail\)/)).toBeTruthy();
    expect(screen.getByLabelText(/absorbed speed: -2\.4 over 150–168 s, not worded \(small change\)/)).toBeTruthy();
  });

  // V16 / design §4.4-3: the class exists and is never observed here. Saying
  // only "never used" would read as the model declining to use it.
  // The word is a clearance, not a measurement — and for the two kinds that
  // carry one, the tolerance is half of what it says, so the label carries it.
  it("says a band is the target in force, not the measured state", () => {
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: "More notes" }));
    expect(screen.getByText(/TARGET in force/)).toBeTruthy();
    expect(screen.getByLabelText(/vertical target ↓2\.4°±0\.17 \(word 3\)/)).toBeTruthy();
    expect(screen.getByLabelText(/speed target 121 m\/s±3\.6 \(word 11\)/)).toBeTruthy();
    // and the heading word, which has no tolerance, carries no band
    expect(screen.getByLabelText(/heading target \+90° \(word 18\)/)).toBeTruthy();
  });

  // The speed tolerance's cost is arrival TIME, and the flown sentence usually
  // runs past the right-hand edge of this bar, so the window is a header readout
  // rather than a bracket that would be clipped exactly when it had something to
  // say (V39).
  it("reads out the arrival window the speed tolerance opens", () => {
    render(<TrainingSentenceBar />);
    const { flight } = selection();
    const [fast, slow] = flight.geometric.speedBand.arrivalWindowS as [number, number];
    expect(screen.getByText(`arrives ${fast}–${slow} s`)).toBeTruthy();
  });

  it("says why there is no window when an edge never reached the runway", () => {
    const { vocabulary, flight, geometry } = selection();
    appState.trainingSelection = {
      vocabulary,
      geometry,
      flight: {
        ...flight,
        geometric: {
          ...flight.geometric,
          speedBand: {
            ...flight.geometric.speedBand,
            low: { ...flight.geometric.speedBand.low, endReason: "time-cap" as const },
            arrivalWindowS: null,
          },
        },
      },
    };
    render(<TrainingSentenceBar />);
    expect(screen.getByText(/no arrival window: an edge ended on time-cap/)).toBeTruthy();
  });

  it("lists go-around and says it is never observed in this data", () => {
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: "More notes" }));
    expect(screen.getByText(/continue \/ landed \/ go-around/)).toBeTruthy();
    expect(screen.getByText(/go-around is never observed in this data/)).toBeTruthy();
  });

  // Stated, never silent: a clamped gap word understates the gap it names.
  it("says when a gap hit the duration ceiling", () => {
    const { vocabulary, flight } = selection();
    appState.trainingSelection = {
      vocabulary,
      geometry: selection().geometry,
      flight: { ...flight, sentence: { ...flight.sentence, durationClamped: 1 } },
    };
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: "More notes" }));
    expect(screen.getByText(/1 gap hit the 300 s duration ceiling/)).toBeTruthy();
  });

  // V17: KRDU's vocabulary covers four runways, the airport has six thresholds.
  it("takes the runway classes from the vocabulary, and says what they are", () => {
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: "More notes" }));
    expect(screen.getByText(/05L, 05R, 23L, 23R/)).toBeTruthy();
    expect(screen.getByText(/not the airport's full runway list/)).toBeTruthy();
  });
});

// The band's VISIBLE text, not only its tooltip: the tolerance is half of what
// the word says, and dropping it from the drawn label left every test green.
describe("what a band actually prints", () => {
  it("writes the tolerance on the band, not only in its tooltip", () => {
    render(<TrainingSentenceBar />);
    // the opening band is too narrow for any label at this width; the next is not
    expect(screen.getByText("107 m/s±3.2")).toBeTruthy();
    expect(screen.getByText("↓3.1°±0.22")).toBeTruthy();
    // and the kinds without a tolerance print the bare word
    expect(screen.getByText("+90°")).toBeTruthy();
  });
});

// ── what the model said, on the bar (T16) ────────────────────────────────────

describe("the model's words on the bar", () => {
  function priorSelection() {
    const parsed = parseTrainingSample(mockPriorSample(), "prior-generated");
    if (!parsed.ok) throw new Error(parsed.problem);
    return {
      vocabulary: parsed.value.vocabulary,
      flight: parsed.value.flights[0],
      geometry: parsed.value.geometry,
      prior: parsed.value.prior,
    };
  }

  // Agreement is the common case (the real prior gets the heading right 86 % of
  // the time), so drawing every said word would paint the bar and hide the
  // answer. Only the disagreements are drawn, and the header counts both.
  it("marks only where the model differs, and says how often it did not", () => {
    appState.trainingSelection = priorSelection();
    render(<TrainingSentenceBar />);
    // the fixture disagrees on five durations and one vertical word, out of
    // six events × six kinds after the given opening
    expect(screen.getByText(/model: 30\/36 words/)).toBeTruthy();
    expect(screen.getByLabelText(/the model said ↓3\.1° here \(p 0\.84\); the words say ↓2\.4°/))
      .toBeTruthy();
  });

  // The model stopping early is a real answer, not an error to hide.
  it("says when the model called the landing early", () => {
    appState.trainingSelection = priorSelection();
    render(<TrainingSentenceBar />);
    expect(screen.getByText(/said landed at 188 s/)).toBeTruthy();
  });

  // A set without a model must not sprout an empty readout.
  it("says nothing about a model when the set carries none", () => {
    appState.trainingSelection = selection();
    render(<TrainingSentenceBar />);
    expect(screen.queryByText(/model:/)).toBeNull();
  });
});
