/**
 * The sentence bar's invariants under `box-v2-wedge` (design §7, T4a / T15).
 *
 * The ones that matter are the ones a drawing of the retired rule would pass
 * anyway: the axis is real time, the duration row is NOT merged (its word
 * describes the row it sits on, so two 4 s holds are two boxes), the bands run to
 * the end of the TRACK because the boxes tile it, and the header reads out
 * containment rather than an arrival window.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState } = vi.hoisted(() => ({
  appState: {
    trainingSelection: null as unknown,
    // both on, which is the default: a test that silently drew neither would
    // pass every assertion about the bands and none about the model
    trainingLayers: { flown: true, model: true },
  },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState }),
}));

import TrainingSentenceBar, { rowBands, spacedLabels } from "../TrainingSentenceBar";
import { parseTrainingSample, TRAINING_KINDS } from "../../data/trainingSample";
import {
  MOCK_EVENT_TIMES_S,
  MOCK_FLIGHT,
  MOCK_HOLDS_S,
  mockPriorSample,
  mockSample,
} from "../../data/__tests__/trainingSample.fixture";

function selection(raw: unknown = mockSample(), kind: "vocabulary-readback" | "prior-generated" = "vocabulary-readback") {
  const parsed = parseTrainingSample(raw, kind);
  if (!parsed.ok) throw new Error(parsed.problem);
  return {
    vocabulary: parsed.value.vocabulary,
    flight: parsed.value.flights[0],
    reading: parsed.value.reading,
    ...(parsed.value.prior ? { prior: parsed.value.prior } : {}),
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

  it("has one row per word kind, labelled — and the second is an ALTITUDE", () => {
    render(<TrainingSentenceBar />);
    for (const label of ["Heading (°)", "Altitude (m)", "Speed (m/s)", "Runway", "Hold (s)", "Terminal"]) {
      expect(screen.getByText(label)).toBeTruthy();
    }
    expect(TRAINING_KINDS).toHaveLength(6);
  });

  it("merges a word held across several events into one band", () => {
    const { flight } = selection();
    // heading words are 9 9 8 8 7 6 5 over events at 0 10 30 46 60 84 104
    expect(rowBands(flight, "heading")).toEqual([
      { startS: 0, endS: 30, word: 9 },
      { startS: 30, endS: 60, word: 8 },
      { startS: 60, endS: 84, word: 7 },
      { startS: 84, endS: 104, word: 6 },
      { startS: 104, endS: TRACK_S, word: 5 },
    ]);
  });

  it("does NOT merge the duration row: its word describes the box it sits on", () => {
    const { flight } = selection();
    const holds = rowBands(flight, "duration");
    // Two consecutive equal holds are two boxes, not one of twice the length —
    // merging them would draw a hold the sentence never says.
    expect(holds).toHaveLength(MOCK_EVENT_TIMES_S.length);
    holds.forEach((band, event) => {
      expect(band.startS).toBe(MOCK_EVENT_TIMES_S[event]);
      expect(band.endS - band.startS).toBe(MOCK_HOLDS_S[event]);
    });
  });

  it("the last band runs to the end of the track, because the boxes tile it", () => {
    const { flight } = selection();
    for (const kind of TRAINING_KINDS) {
      const bands = rowBands(flight, kind);
      expect(bands[0].startS).toBe(0);
      expect(bands[bands.length - 1].endS).toBe(TRACK_S);
    }
  });

  it("the axis is real time: a 24 s box is wider than a 10 s one", () => {
    render(<TrainingSentenceBar />);
    // events at 0 (held 10 s) and 60 (held 24 s), on the hold row
    const short = bandRect(/^duration 10 s \(word 5\), 0–10 s$/);
    const long = bandRect(/^duration 24 s \(word 12\), 60–84 s$/);
    expect(Number(long.getAttribute("width"))).toBeGreaterThan(
      Number(short.getAttribute("width")) * 2,
    );
  });

  it("reads out CONTAINMENT, the criterion of this vocabulary", () => {
    render(<TrainingSentenceBar />);
    const rows = MOCK_FLIGHT.observed.tS.length;
    expect(screen.getByText(new RegExp(`inside: heading ${rows}/${rows}`))).toBeTruthy();
  });

  it("clicking an event moves the cursor to the artefact's own number", () => {
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(`Event 4 at ${MOCK_EVENT_TIMES_S[3]} s`));
    expect(screen.getByText(`t = ${MOCK_EVENT_TIMES_S[3]} s`)).toBeTruthy();
  });

  it("names every box with its interval, so a narrow one is still readable", () => {
    render(<TrainingSentenceBar />);
    // the heading label IS the interval; the altitude one is the target with the
    // box it closes onto
    expect(screen.getByLabelText(/^heading box \+30…\+90°/)).toBeTruthy();
    expect(screen.getByLabelText(/^altitude box 170 m±11\.0/)).toBeTruthy();
  });

  it("marks where the MODEL said something else, and counts the agreement", () => {
    appState.trainingSelection = selection(mockPriorSample(), "prior-generated");
    render(<TrainingSentenceBar />);
    // 6 events answered × 6 kinds = 36 words; the fixture disagrees on 7 of them
    expect(
      screen.getByTitle(/teacher-forced/).textContent,
    ).toMatch(/model: 29\/36 words/);
    expect(screen.getByLabelText(/the model said \+3…\+10° here \(p 0\.84\); the words say \+10…\+30°/)).toBeTruthy();
  });

  it("says how well the MODEL's own boxes would have held the aircraft", () => {
    appState.trainingSelection = selection(mockPriorSample(), "prior-generated");
    render(<TrainingSentenceBar />);
    // The model's sentence is a different sentence, so it makes a different
    // envelope — and the number that matters is whether THAT one contains the
    // track, not only how many words matched.
    expect(screen.getByTitle(/teacher-forced/).textContent).toMatch(/its boxes hold \d+\/\d+\/\d+\/\d+/);
  });

  it("drops a label that would collide, never the event itself", () => {
    // greedy from the left, and the last position always survives
    expect(spacedLabels([0, 4, 40, 80], 13)).toEqual([true, false, true, true]);
    expect(spacedLabels([0, 10, 18], 13, true)).toEqual([true, false, true]);
  });
});
