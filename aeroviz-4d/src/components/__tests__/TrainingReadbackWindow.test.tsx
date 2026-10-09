/**
 * The read-back check: the four charts draw what the exporter sent — the heading bands with their rows outside, the altitude
 * tubes, the speed bands, the DA point — on one flight clock, the flown path of a closed-loop sentence beside the observed
 * track, the correction words on the axis, the flown flight's outcome and DA values below, the live segment when there is one;
 * the switches reach every chart and the selected column's word is the yellow one.
 */
import { describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import TrainingReadbackWindow from "../TrainingReadbackWindow";
import { extent, readbackModel } from "../training/readbackModel";
import { parseTrainingAutopilot, type TrainingAutopilotSegment } from "../../data/trainingAutopilot";
import { trainingReadingOf, type TrainingColumn } from "../../data/trainingSample";
import type { TrainingLayers } from "../../context/AppContext";
import { TRAINING_SENTENCE_COLOR, TRAINING_WORD_COLOR } from "../../utils/trainingWordColors";
import { stageBSample, stageBSelection } from "../../data/__tests__/stageB";
import { stageCSample, stageCSelection } from "../../data/__tests__/stageC";
import { requestOf, stageAAnswers, stageASampleFile, stageASelection } from "../../data/__tests__/stageA";
import { parseTrainingSample, TRAINING_SPLITS, trainingSelectionOf } from "../../data/trainingSample";

const ALL: TrainingLayers = { observed: true, headingBands: true, vertical: true, candidates: true };
const selection = stageASelection();

function liveSegment(which = 0): TrainingAutopilotSegment {
  const raw = stageAAnswers()[which];
  const parsed = parseTrainingAutopilot(raw, requestOf(raw), selection);
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

function open(intervalS: number | null = 2, layers: TrainingLayers = ALL, cursorS = 100, column: TrainingColumn | null = null,
  autopilot: TrainingAutopilotSegment | null = null) {
  const onCursorChange = vi.fn();
  const onColumnChange = vi.fn();
  const reading = trainingReadingOf(selection.flight, selection.vocabulary.stepS, intervalS);
  const view = render(
    <TrainingReadbackWindow selection={selection} reading={reading} layers={layers} cursorS={cursorS} onCursorChange={onCursorChange}
      column={column} onColumnChange={onColumnChange} onClose={vi.fn()} autopilot={autopilot} />,
  );
  return { ...view, onCursorChange, onColumnChange };
}

const chart = (name: string) => document.querySelector(`svg[aria-label="${name}"]`) as SVGSVGElement;

describe("TrainingReadbackWindow", () => {
  it("is a dialog with the four charts, named by the sentence it reads", () => {
    open(4);
    expect(screen.getByRole("dialog", { name: "Read-back check" })).toBeTruthy();
    for (const name of ["Plan view", "Heading chart", "Altitude chart", "Speed chart"]) expect(chart(name)).toBeTruthy();
    expect(screen.getByText("closed loop · Δ 4 s")).toBeTruthy();
    cleanup();
    open(null);
    expect(screen.getByText("labelled sentence")).toBeTruthy();
  });

  it("draws the flown path beside the observed track in every chart and the plan, only for a closed-loop sentence", () => {
    open(2);
    expect(document.querySelector("svg[aria-label='Plan view'] polyline.training-readback-executor")).toBeTruthy();
    for (const name of ["Heading chart", "Altitude chart", "Speed chart"]) {
      expect(chart(name).querySelector("polyline.training-readback-executor")).toBeTruthy();
      expect(chart(name).querySelector("polyline.training-readback-trace")).toBeTruthy();
    }
    cleanup();
    open(null);
    for (const name of ["Plan view", "Heading chart", "Altitude chart", "Speed chart"]) {
      expect(chart(name).querySelector("polyline.training-readback-executor")).toBeNull();
    }
  });

  it("draws a band per judged heading word, a tube per altitude word and a band per speed word, and the switches reach them", () => {
    const reading = trainingReadingOf(selection.flight, selection.vocabulary.stepS, 2);
    const judged = reading.envelopes!.heading.filter((band) => band.stopRow > band.firstRow).length;
    const { rerender, onCursorChange } = open(2);
    expect(chart("Heading chart").querySelectorAll("rect.training-readback-heading-band")).toHaveLength(judged);
    expect(chart("Altitude chart").querySelectorAll("polygon.training-readback-tube")).toHaveLength(reading.envelopes!.altitude.length);
    expect(chart("Speed chart").querySelectorAll("rect.training-readback-speed-band")).toHaveLength(reading.envelopes!.speed.length);
    rerender(<TrainingReadbackWindow selection={selection} reading={reading} layers={{ ...ALL, headingBands: false, vertical: false }}
      cursorS={100} onCursorChange={onCursorChange} column={null} onColumnChange={vi.fn()} onClose={vi.fn()} autopilot={null} />);
    expect(chart("Heading chart").querySelectorAll("rect.training-readback-heading-band")).toHaveLength(0);
    expect(chart("Altitude chart").querySelectorAll("polygon.training-readback-tube")).toHaveLength(0);
    expect(chart("Speed chart").querySelectorAll("rect.training-readback-speed-band")).toHaveLength(0);
  });

  it("the observed track's switch hides its line on every chart; with the labelled sentence, the envelopes that judge it too", () => {
    const charts = ["Plan view", "Heading chart", "Altitude chart", "Speed chart"];
    open(2, { ...ALL, observed: false });
    for (const name of charts) {
      expect(chart(name).querySelector("polyline.training-readback-trace")).toBeNull();
      expect(chart(name).querySelector("polyline.training-readback-executor")).toBeTruthy();       // the flown path stays
    }
    // a closed-loop reading judges the flown path: its envelopes stay
    expect(chart("Heading chart").querySelectorAll("rect.training-readback-heading-band").length).toBeGreaterThan(0);
    cleanup();
    open(null, { ...ALL, observed: false });
    expect(chart("Heading chart").querySelectorAll("rect.training-readback-heading-band")).toHaveLength(0);
    expect(chart("Altitude chart").querySelectorAll("polygon.training-readback-tube")).toHaveLength(0);
    expect(chart("Speed chart").querySelectorAll("rect.training-readback-speed-band")).toHaveLength(0);
    cleanup();
    open(null, { ...ALL, observed: false }, 100, "heading");              // nor the selected word's rows on it
    expect(document.querySelector(".training-readback-focus")).toBeNull();
    cleanup();
    open(null, ALL, 100, "heading");
    expect(document.querySelector(".training-readback-focus")).toBeTruthy();
    expect(chart("Heading chart").querySelectorAll("rect.training-readback-heading-band").length).toBeGreaterThan(0);
  });

  it("marks the correction words on the axis of their columns' charts and the plan", () => {
    open(2);
    const reading = trainingReadingOf(selection.flight, selection.vocabulary.stepS, 2);
    const heading = reading.events.filter((event) => event.correction && event.says.column === "heading").length;
    expect(chart("Heading chart").querySelectorAll("polygon.training-readback-correction")).toHaveLength(heading);
    expect(chart("Plan view").querySelectorAll(".training-readback-correction")).toHaveLength(9);
    expect(chart("Speed chart").querySelectorAll("polygon.training-readback-correction")).toHaveLength(
      reading.events.filter((event) => event.correction && event.says.column === "speed").length);
  });

  it("writes the flown flight's outcome, the crossing and the DA check's values below the charts", () => {
    open(2);
    const slot = screen.getByLabelText("The flown flight");
    expect(slot.textContent).toContain("unstable at minimums at 511 s");
    expect(slot.textContent).toContain("crossing 15.5 m right of the centreline, 41.8 m above the threshold");
    expect(slot.textContent).toContain("DA check failed");
    expect(slot.textContent).toContain("15.5 m right of the centreline, cone half width 116.3 m ✓");
    expect(slot.textContent).toContain("26.8 m above the glidepath ✗");
    expect(slot.textContent).toContain("at 157 m MSL");
    // the DA point is drawn in the plan and on the altitude chart
    expect(chart("Plan view").querySelector('g[aria-label="the decision-altitude point"]')).toBeTruthy();
    expect(chart("Altitude chart").querySelector("circle title")!.textContent).toContain("DA check failed");
    cleanup();
    open(null);
    expect(screen.queryByLabelText("The flown flight")).toBeNull();
    expect(chart("Plan view").querySelector('g[aria-label="the decision-altitude point"]')).toBeNull();
  });

  it("draws the live segment on every chart and names it", () => {
    open(2, ALL, 100, null, liveSegment());
    for (const name of ["Plan view", "Heading chart", "Altitude chart", "Speed chart"]) {
      expect(chart(name).querySelector("polyline.training-readback-autopilot")).toBeTruthy();
    }
    const slot = screen.getByLabelText("The autopilot, live");
    expect(slot.textContent).toContain("heading");
    expect(slot.textContent).toContain("from Δ row 52");
    expect(slot.textContent).toContain("reached the point where the next heading word is said");
    cleanup();
    open(2, ALL, 100, null, liveSegment(1));
    expect(screen.getByLabelText("The autopilot, live").textContent).toContain("(a correction)");
  });

  it("hovers move the cursor in flight time and a click selects the chart's column", () => {
    const { onCursorChange, onColumnChange } = open(2);
    const heading = chart("Heading chart");
    heading.getBoundingClientRect = () => ({ left: 0, top: 0, right: 980, bottom: 132, width: 980, height: 132, x: 0, y: 0, toJSON: () => ({}) });
    fireEvent.mouseMove(heading, { clientX: 64, clientY: 20 });
    expect(onCursorChange).toHaveBeenLastCalledWith(0);
    fireEvent.click(heading, { clientX: 64 + 450, clientY: 20 });
    expect(onColumnChange).toHaveBeenLastCalledWith("heading");
    expect(onCursorChange.mock.calls[onCursorChange.mock.calls.length - 1][0]).toBeGreaterThan(100);
  });

  it("draws the selected word alone in yellow, on its own column's chart", () => {
    const reading = trainingReadingOf(selection.flight, selection.vocabulary.stepS, 2);
    // the heading word in force 100 s in
    open(2, ALL, 100, "heading");
    expect(chart("Heading chart").querySelector(`polyline.training-readback-focus[stroke="${TRAINING_WORD_COLOR}"]`)).toBeTruthy();
    expect(chart("Speed chart").querySelector("polyline.training-readback-focus")).toBeNull();
    expect(chart("Plan view").querySelector("polyline.training-readback-focus")).toBeTruthy();
    const m = readbackModel({ selection, reading, layers: ALL, cursorS: 100, column: "heading", autopilot: null, width: 980 });
    expect(m.focusIndex).not.toBeNull();
    expect(m.recede(false)).toBe(0.3);
    expect(m.recede(true)).toBe(1);
  });

  it("draws a flight that did not cross the threshold and has no envelopes: lines only, outcome without a DA check", () => {
    const file = stageASampleFile();
    file.flights[0].closedLoop["2"].replay.crossing = null;
    file.flights[0].closedLoop["2"].replay.envelopes = null;
    file.flights[0].closedLoop["2"].replay.outcome = "ground_contact";
    const parsed = parseTrainingSample(file, TRAINING_SPLITS);
    if (!parsed.ok) throw new Error(parsed.problem);
    const bare = trainingSelectionOf(parsed.value, parsed.value.flights[0]);
    const reading = trainingReadingOf(bare.flight, bare.vocabulary.stepS, 2);
    render(<TrainingReadbackWindow selection={bare} reading={reading} layers={ALL} cursorS={100} onCursorChange={vi.fn()}
      column="heading" onColumnChange={vi.fn()} onClose={vi.fn()} autopilot={null} />);
    expect(chart("Heading chart").querySelectorAll("rect.training-readback-heading-band")).toHaveLength(0);
    expect(chart("Heading chart").querySelector("polyline.training-readback-executor")).toBeTruthy();
    expect(chart("Plan view").querySelector('g[aria-label="the decision-altitude point"]')).toBeNull();
    const slot = screen.getByLabelText("The flown flight");
    expect(slot.textContent).toContain("ground contact");
    expect(slot.textContent).toContain("no threshold crossing");
  });

  it("scales a chart over what it draws, padded, never zero-wide", () => {
    expect(extent([10, 20])).toEqual([9.2, 20.8]);
    expect(extent([5, 5])).toEqual([4, 6]);
  });
});

describe("the flown line in its kind's colour (frontend §3 item 11, D159)", () => {
  it("a sample of the base is magenta in every chart and in the swatch below; the closed loop teal", () => {
    cleanup();
    const b = stageBSample();
    const sampled = stageBSelection(b, 0);
    render(<TrainingReadbackWindow selection={sampled} reading={trainingReadingOf(sampled.flight, sampled.vocabulary.stepS, 4)}
      layers={ALL} cursorS={100} onCursorChange={vi.fn()} column={null} onColumnChange={vi.fn()} onClose={vi.fn()} autopilot={null} />);
    for (const name of ["Plan view", "Heading chart", "Altitude chart", "Speed chart"]) {
      expect(chart(name).querySelector(`polyline[stroke="${TRAINING_SENTENCE_COLOR.base}"]`), name).not.toBeNull();
      expect(chart(name).querySelector(`polyline[stroke="${TRAINING_SENTENCE_COLOR.closedLoop}"]`), name).toBeNull();
    }
    // the title names the sentence it reads: the prior's sample, not the closed loop (stage D's browser check, 2026-10-07)
    expect(screen.getByText("sample 0 · Δ 4 s")).toBeTruthy();
    expect(screen.queryByText("closed loop · Δ 4 s")).toBeNull();
    cleanup();
    const c = stageCSample();
    const round = stageCSelection(c, 0);
    render(<TrainingReadbackWindow selection={round} reading={trainingReadingOf(round.flight, round.vocabulary.stepS, 4)}
      layers={ALL} cursorS={100} onCursorChange={vi.fn()} column={null} onColumnChange={vi.fn()} onClose={vi.fn()} autopilot={null} />);
    expect(screen.getByText("start (base) · Δ 4 s")).toBeTruthy();
    cleanup();
    open(4);
    expect(chart("Heading chart").querySelector(`polyline[stroke="${TRAINING_SENTENCE_COLOR.closedLoop}"]`)).not.toBeNull();
  });
});
