/**
 * The cursor slider (frontend §6.1, D155): its rows (the observed 2 s rows before the sentence opens, then the sentence's
 * rows), the drag, the click and the keys moving the one cursor by rows, its aria values, every reader of the cursor
 * following it, disabled with no sentence on screen; and each stage's marks on its track. With the real provider, on the
 * fixtures the exports write.
 */
import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({ defaultAirport: "KXXX", airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }] }),
}));

import { AppProvider, useApp, useTrainingCursor } from "../../context/AppContext";
import CursorSlider from "../training/CursorSlider";
import { readingAxisEndS, readingRowTimeS, trainingReadingOf } from "../../data/trainingSample";
import {
  correctionMarks,
  firstStepMark,
  goAroundLine,
  lossMark,
  nearestStop,
  sliderStops,
  stopAt,
  stopForKey,
  type TrainingSliderMark,
} from "../../data/trainingSlider";
import { stageASample, stageASelection } from "../../data/__tests__/stageA";
import { stageBSample } from "../../data/__tests__/stageB";
import { trainingPriorFlightView } from "../../data/trainingPriorSample";
import { stageCSample } from "../../data/__tests__/stageC";
import { onAircraftClock, otherOf, trainingWindowFlightView } from "../../data/trainingWindowSample";

const STEP_S = 2;

describe("the slider's rows", () => {
  it("are the observed 2 s rows before a closed-loop sentence opens, then its Δ rows to its last", () => {
    const flight = stageASample().flights[0];
    const reading = trainingReadingOf(flight, STEP_S, 4);
    const stops = sliderStops(reading, STEP_S);
    const before = Math.ceil(reading.originS / STEP_S);
    expect(stops.slice(0, before)).toEqual(Array.from({ length: before }, (_, k) => k * STEP_S));
    expect(stops.slice(before)).toEqual(Array.from({ length: reading.rows }, (_, row) => readingRowTimeS(reading, row)));
    expect(stops[before]).toBe(reading.originS);
    expect(stops[before + 1] - stops[before]).toBe(4);
  });

  it("are the 2 s rows of a labelled sentence", () => {
    const reading = trainingReadingOf(stageASample().flights[0], STEP_S, null);
    const stops = sliderStops(reading, STEP_S);
    expect(stops.length).toBe(reading.rows);
    expect(stops.every((atS, place) => atS === reading.originS + place * STEP_S)).toBe(true);
  });

  it("place a time at the row in force, the nearest row, and a key's row", () => {
    const stops = [0, 2, 4, 8, 12, 16];
    expect(stopAt(stops, -1)).toBe(0);
    expect(stopAt(stops, 7.9)).toBe(2);
    expect(stopAt(stops, 8)).toBe(3);
    expect(stopAt(stops, 99)).toBe(5);
    expect(nearestStop(stops, 9.9)).toBe(3);
    expect(nearestStop(stops, 10.1)).toBe(4);
    expect(stopForKey(stops, 2, "ArrowRight", false)).toBe(3);
    expect(stopForKey(stops, 2, "ArrowLeft", false)).toBe(1);
    expect(stopForKey(stops, 2, "ArrowRight", true)).toBe(5);
    expect(stopForKey(stops, 2, "ArrowLeft", true)).toBe(0);
    expect(stopForKey(stops, 2, "Home", false)).toBe(0);
    expect(stopForKey(stops, 2, "End", false)).toBe(5);
    expect(stopForKey(stops, 2, "a", false)).toBeNull();
  });
});

/** The slider with the real provider, stage A's flight on screen at Δ ``intervalS``, and a reader of the cursor. */
function mount(marks: TrainingSliderMark[] = [], intervalS: number | null = 4) {
  let app!: ReturnType<typeof useApp>;
  let cursor!: ReturnType<typeof useTrainingCursor>;
  function Harness() {
    app = useApp();
    cursor = useTrainingCursor();
    return <CursorSlider marks={marks} />;
  }
  render(<AppProvider><Harness /></AppProvider>);
  const selection = stageASelection();
  act(() => {
    app.setTrainingSelection(selection);
    app.setTrainingIntervalS(intervalS);
  });
  const reading = trainingReadingOf(selection.flight, STEP_S, intervalS);
  return {
    reading, stops: sliderStops(reading, STEP_S), cursorS: () => cursor.trainingCursorS,
    setCursorS: (atS: number) => cursor.setTrainingCursorS(atS),
  };
}

/** A track 1000 px wide from x = 100. */
function placeTrack(track: HTMLElement) {
  track.getBoundingClientRect = () => ({ left: 100, width: 1000, top: 0, height: 12, right: 1100, bottom: 12, x: 100, y: 0, toJSON: () => ({}) });
}

describe("the slider", () => {
  it("is disabled and says so with no sentence on screen", () => {
    render(<AppProvider><CursorSlider marks={[]} /></AppProvider>);
    const slider = screen.getByRole("slider", { name: "Cursor" });
    expect(slider.getAttribute("aria-disabled")).toBe("true");
    expect(slider.textContent).toBe("choose a sentence");
  });

  it("moves the cursor by rows with the keys, and says the row in its aria values", () => {
    const { stops, cursorS } = mount();
    const slider = screen.getByRole("slider", { name: "Cursor" });
    expect(slider.getAttribute("aria-valuenow")).toBe("0");
    expect(slider.getAttribute("aria-valuemax")).toBe(String(stops.length - 1));
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    expect(cursorS()).toBe(stops[1]);
    fireEvent.keyDown(slider, { key: "ArrowRight", shiftKey: true });
    expect(cursorS()).toBe(stops[11]);
    expect(slider.getAttribute("aria-valuenow")).toBe("11");
    expect(slider.getAttribute("aria-valuetext")).toBe(`t = ${stops[11]} s, row 11 of ${stops.length - 1}`);
    fireEvent.keyDown(slider, { key: "ArrowLeft" });
    expect(cursorS()).toBe(stops[10]);
    fireEvent.keyDown(slider, { key: "End" });
    expect(cursorS()).toBe(stops[stops.length - 1]);
    fireEvent.keyDown(slider, { key: "Home" });
    expect(cursorS()).toBe(0);
    expect(screen.getByText("t 0 s")).toBeTruthy();
  });

  it("puts the cursor at the row nearest a click on its track, and follows a drag until it is let go", () => {
    const { stops, reading, cursorS } = mount();
    const slider = screen.getByRole("slider", { name: "Cursor" });
    placeTrack(slider);
    const endS = readingAxisEndS(reading);
    const xOf = (atS: number) => 100 + (atS / endS) * 1000;
    fireEvent(slider, new MouseEvent("pointerdown", { clientX: xOf(stops[40] + 0.4), bubbles: true }));
    expect(cursorS()).toBe(stops[40]);
    fireEvent(slider, new MouseEvent("pointermove", { clientX: xOf(stops[60]), bubbles: true }));
    expect(cursorS()).toBe(stops[60]);
    fireEvent(slider, new MouseEvent("pointerup", { clientX: xOf(stops[60]), bubbles: true }));
    fireEvent(slider, new MouseEvent("pointermove", { clientX: xOf(stops[80]), bubbles: true }));
    expect(cursorS()).toBe(stops[60]);                                    // let go: a move is no drag
  });

  it("reads and writes the one cursor: a time set elsewhere (a chart's hover) moves its thumb to the row in force", () => {
    const { stops, reading, cursorS, setCursorS } = mount();
    const slider = screen.getByRole("slider", { name: "Cursor" });
    act(() => setCursorS(stops[20] + 1));                                // between two rows
    expect(slider.getAttribute("aria-valuenow")).toBe("20");
    const thumb = slider.querySelector(".training-cursor-slider-thumb")!;
    expect(Number(thumb.getAttribute("x1"))).toBeCloseTo(((stops[20] + 1) / readingAxisEndS(reading)) * 1000);
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    expect(cursorS()).toBe(stops[21]);                                  // the other reader sees the slider's row
  });

  it("draws the marks it is given, a tick at its time and a line of its values", () => {
    const { reading } = mount([firstStepMark(16), { kind: "line", key: "go-around", colour: "#fb923c", title: "p", points: [[16, 0], [20, 1]] }]);
    const endS = readingAxisEndS(reading);
    const tick = document.querySelector('[data-mark="first-step"]')!;
    expect(Number(tick.getAttribute("x1"))).toBeCloseTo((16 / endS) * 1000);
    expect(tick.querySelector("title")!.textContent).toContain("first predicted step");
    const line = document.querySelector('[data-mark="go-around"]')!;
    expect(line.getAttribute("points")).toBe(`${(16 / endS) * 1000},11 ${(20 / endS) * 1000},1`);
  });
});

describe("each stage's marks", () => {
  it("A: a tick at each row of a closed-loop sentence with a word the reading added", () => {
    const flight = stageASample().flights[0];
    const reading = trainingReadingOf(flight, STEP_S, 4);
    const rows = [...new Set(reading.events.filter((event) => event.correction).map((event) => event.row))];
    const marks = correctionMarks(reading);
    expect(rows.length).toBeGreaterThan(0);
    expect(marks.map((mark) => (mark.kind === "tick" ? mark.atS : NaN))).toEqual(rows.sort((a, b) => a - b).map((row) => readingRowTimeS(reading, row)));
  });

  it("B: the probability of go-around at each row of a sample, on the rows' times", () => {
    const sample = stageBSample();
    const flight = sample.flights[0];
    const sentence = flight.sentences[0];
    const reading = trainingReadingOf(trainingPriorFlightView(sample, flight, sentence.sample), STEP_S, sample.model.rowIntervalS);
    const line = goAroundLine(reading, sentence.goAroundProbability);
    expect(line.kind).toBe("line");
    if (line.kind !== "line") return;
    expect(line.points.length).toBe(sentence.goAroundProbability.length);
    expect(line.points[3]).toEqual([readingRowTimeS(reading, 3), sentence.goAroundProbability[3]]);
  });

  it("C: a red tick at the round's loss of separation, on the aircraft's flight clock", () => {
    const sample = stageCSample();
    const window = sample.windows.find((item) => item.rounds[0].losses.length > 0)!;
    const [aircraft] = window.commanded;
    const [loss] = window.rounds[0].losses;
    const atS = onAircraftClock(aircraft, loss.timeS);
    const mark = lossMark(atS, otherOf(loss, aircraft.datasetId));
    expect(mark.kind === "tick" && mark.atS).toBe(atS);
    // the loss falls on the round's sentence's axis
    const view = trainingWindowFlightView(sample, window, aircraft, window.rounds[0].round);
    expect(atS).toBeLessThanOrEqual(readingAxisEndS(trainingReadingOf(view, STEP_S, sample.model.rowIntervalS)));
  });
});
