/**
 * The read-back check: every envelope the exporter sent is drawn where it belongs — the heading words'
 * bands on the heading chart with their rows outside in red (and in the plan view), the capture turn,
 * the corridor, the tubes against distance flown, the speed transitions and bands against time — the
 * switches reach every chart, only the selected column's word is yellow, and the executor's replay, when
 * it is on, is drawn on its own clock beside the observed track with its own heading bands.
 */
import { describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import TrainingReadbackWindow, { extent } from "../TrainingReadbackWindow";
import { parseTrainingSample, type TrainingColumn } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import { EXECUTOR_ID, mockExecutorOverlay, mockOverlayEntry } from "../../data/__tests__/trainingOverlays.fixture";
import { parseTrainingExecutorOverlay, type TrainingExecutorFlight } from "../../data/trainingOverlays";
import type { TrainingLayers } from "../../context/AppContext";
import { parseTrainingAutopilot, type TrainingAutopilotSegment } from "../../data/trainingAutopilot";
import { VECTORED_KEY } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest } from "../../data/__tests__/trainingAutopilot.fixture";
import { TRAINING_WORD_COLOR } from "../../utils/trainingWordColors";

const ALL: TrainingLayers = { headingBands: true, corridor: true, vertical: true, candidates: true };

function executorFlight(position = 0): TrainingExecutorFlight {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  const overlay = parseTrainingExecutorOverlay(mockExecutorOverlay(), mockOverlayEntry(EXECUTOR_ID), parsed.value);
  if (!overlay.ok) throw new Error(overlay.problem);
  return overlay.value.flights[position];
}

function autopilotSegment(): TrainingAutopilotSegment {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  const request = mockAutopilotRequest(parsed.value, VECTORED_KEY, "heading", 8);
  const answer = parseTrainingAutopilot(mockAutopilotAnswer(parsed.value, request), request, parsed.value);
  if (!answer.ok) throw new Error(answer.problem);
  return answer.value;
}

function open(layers: TrainingLayers = ALL, position = 0, cursorS = 0, column: TrainingColumn | null = null,
  executor: TrainingExecutorFlight | null = null, autopilot: TrainingAutopilotSegment | null = null) {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  const onCursorChange = vi.fn();
  const onColumnChange = vi.fn();
  render(
    <TrainingReadbackWindow
      flight={parsed.value.flights[position]}
      vocabulary={parsed.value.vocabulary}
      candidates={parsed.value.candidates}
      layers={layers}
      cursorS={cursorS}
      onCursorChange={onCursorChange}
      column={column}
      onColumnChange={onColumnChange}
      onClose={() => undefined}
      executor={executor}
      autopilot={autopilot}
    />,
  );
  return { onCursorChange, onColumnChange };
}

/** The elements stroked in the selected word's colour. */
const yellow = () => [...document.body.querySelectorAll(`[stroke="${TRAINING_WORD_COLOR}"]`)];

/** The window renders through a portal into `document.body`. */
const count = (selector: string) => document.body.querySelectorAll(selector).length;
const chart = (name: string) => screen.getByLabelText(name);

describe("TrainingReadbackWindow", () => {
  it("draws the plan view's envelopes: the corridor, the capture turn's rows, every candidate — and no turn region or funnel", () => {
    open();
    expect(count(".training-readback-corridor")).toBe(1);
    expect(count(".training-readback-capture-turn")).toBe(1);
    expect(count(".training-readback-candidate")).toBe(2);
    for (const gone of [".training-readback-turn", ".training-readback-turn-end", ".training-readback-turn-path",
                        ".training-readback-funnel", ".training-readback-hold-band", ".training-readback-turn-band"]) {
      expect(count(gone), gone).toBe(0);
    }
  });

  it("keeps the designated runway when the other candidates are switched off", () => {
    open({ ...ALL, candidates: false });
    const drawn = [...document.body.querySelectorAll(".training-readback-candidate")].map((g) => g.getAttribute("aria-label"));
    expect(drawn).toEqual([expect.stringMatching(/^the designated runway 09/)]);
  });

  it("draws each heading word's band over the rows it is judged on, from its first judged step to the next word's", () => {
    open();
    const bands = [...chart("Heading chart").querySelectorAll(".training-readback-heading-band")];
    expect(bands).toHaveLength(3);
    expect(bands.map((band) => band.textContent)).toEqual([
      expect.stringMatching(/^270° ±4\.5°, said at step 0 and judged at steps 2–9 \(4 s later, to the next heading word's\): 8 of 8 rows inside$/),
      expect.stringMatching(/^225° ±4\.5°, said at step 8 and judged at steps 10–11 .*: 2 of 2 rows inside$/),
      expect.stringMatching(/^180° ±4\.5°, said at step 10 and judged at steps 12–19 .*: 7 of 8 rows inside$/),
    ]);
    // one band's width is its rows: steps 12 to 20 (the clearance) — eight steps of 2 s
    const widths = bands.map((band) => Number(band.getAttribute("width")));
    expect(widths[2] / widths[0]).toBeCloseTo(1);
    expect(widths[1] / widths[0]).toBeCloseTo(0.25);
    // a band with a row outside is edged in red; the row itself is red on the track, in the chart and in the plan
    expect(bands.map((band) => band.getAttribute("stroke"))).toEqual(["#38bdf8", "#38bdf8", "#f87171"]);
    expect(chart("Heading chart").querySelectorAll(".training-readback-outside")).toHaveLength(1);
    expect(chart("Plan view").querySelectorAll(".training-readback-outside")).toHaveLength(1);
  });

  it("draws no band for a word the lead carries to the clearance", () => {
    open(ALL, 1);
    expect(count(".training-readback-heading-band")).toBe(0);
    expect(count(".training-readback-capture-band") + count(".training-readback-capture-turn")).toBe(0);
    expect(screen.getByText(/in force: 090° · no row of its own: the 4 s lead carries it to the clearance/)).toBeTruthy();
  });

  it("switches the heading bands off everywhere, and the capture with the corridor", () => {
    open({ ...ALL, headingBands: false });
    expect(count(".training-readback-heading-band")).toBe(0);
    expect(chart("Heading chart").querySelectorAll(".training-readback-outside")).toHaveLength(0);
    expect(chart("Plan view").querySelectorAll(".training-readback-outside")).toHaveLength(0);
    cleanup();
    open({ ...ALL, corridor: false });
    for (const selector of [".training-readback-corridor", ".training-readback-capture-turn", ".training-readback-capture-band",
                            ".training-readback-capture-course", ".training-readback-course-band"]) {
      expect(count(selector), selector).toBe(0);
    }
    expect(count(".training-readback-heading-band")).toBe(3);
  });

  it("draws the capture turn against time, from the clearance to the capture, and the course band after it", () => {
    open();
    expect(count(".training-readback-capture-band")).toBe(1);
    expect(count(".training-readback-capture-course")).toBe(1);
    expect(count(".training-readback-course-band")).toBe(1);
    expect(document.body.querySelector(".training-readback-capture-band")!.textContent)
      .toMatch(/the capture turn, steps 20–25: from the clearance onto the course, monotone ✓, rate and bank ✓/);
  });

  it("draws one tube per altitude word, and the row the labeller counted outside in red", () => {
    open();
    expect(count(".training-readback-tube")).toBe(2);
    // one row outside is a segment on to the next row: a one-point polyline would draw nothing
    const outside = [...chart("Altitude chart").querySelectorAll(".training-readback-outside")];
    expect(outside).toHaveLength(1);
    expect(outside[0].getAttribute("points")!.split(" ")).toHaveLength(2);
    expect(screen.getByLabelText(/angle word descent 3 \(3\.06°\) at step 20/)).toBeTruthy();
    // switched off with the tubes it belongs to
    cleanup();
    open({ ...ALL, vertical: false });
    expect(chart("Altitude chart").querySelectorAll(".training-readback-outside")).toHaveLength(0);
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

  it("names what is in force at the cursor: the heading word's band, and the capture turn's verdict inside it", () => {
    open(ALL, 0, 22);
    expect(screen.getByText(/in force: 180° · its band: steps 12–19 \(4 s after it was said\), 7\/8 rows within ±4\.5°/)).toBeTruthy();
    expect(screen.getByText(/in force: 1110 m, level/)).toBeTruthy();
    cleanup();
    open(ALL, 0, 44);
    expect(screen.getByText(/the capture turn: ✓ monotone, ✓ rate \(mean 2\.40°\/s, max 3\.10°\/s, max bank 24\.9°\)/)).toBeTruthy();
  });

  it("says the corridor holds once the flight is captured", () => {
    open(ALL, 0, 60);
    expect(screen.getByText(/captured: the corridor holds/)).toBeTruthy();
  });

  it("draws no executor without its replay, and says so", () => {
    open();
    expect(screen.getByLabelText("Executor replay").textContent).toMatch(/off, or not published for this set/);
    expect(document.body.querySelectorAll(".training-readback-executor")).toHaveLength(0);
  });

  it("draws the executor's flown track in plan, its three signals on its own clock, and its own heading bands", () => {
    open(ALL, 0, 0, null, executorFlight(0));
    // the plan view's track, the heading, the altitude and the ground speed
    expect(document.body.querySelectorAll(".training-readback-executor")).toHaveLength(4);
    expect(screen.getByText("executor: landed")).toBeTruthy();
    expect(screen.getByLabelText("Executor replay").textContent).toMatch(/landed on own dynamics; dashed teal/);
    // its bands from where IT was told each word; the one it flew a row late is edged red, its row outside red too
    const bands = [...document.body.querySelectorAll(".training-readback-executor-band")];
    expect(bands.map((band) => band.getAttribute("stroke"))).toEqual(["#14b8a6", "#14b8a6", "#f87171"]);
    expect(chart("Heading chart").querySelectorAll(".training-readback-executor-outside")).toHaveLength(1);
    expect(chart("Plan view").querySelectorAll(".training-readback-executor-outside")).toHaveLength(1);
    cleanup();
    open({ ...ALL, headingBands: false }, 0, 0, null, executorFlight(0));
    expect(count(".training-readback-executor-band") + count(".training-readback-executor-outside")).toBe(0);
  });

  it("says so when the executor's flight ended within its first step", () => {
    const flown = executorFlight(0);
    if (!flown.flown) throw new Error("the fixture's first flight is flown");
    const point = Object.fromEntries(Object.entries(flown.track).map(([key, values]) => [key, (values as number[]).slice(0, 1)]));
    open(ALL, 0, 0, null, { ...flown, outcome: "dynamics_failure", track: point as unknown as typeof flown.track });
    expect(screen.getByLabelText("Executor replay").textContent).toMatch(/dynamics failure on own dynamics within its first step/);
    expect(document.body.querySelectorAll(".training-readback-executor")).toHaveLength(0);
    expect(count(".training-readback-executor-band")).toBe(0);
  });

  it("names why a flight the replay does not fly has no executor line", () => {
    open(ALL, 1, 0, null, executorFlight(1));
    expect(screen.getByLabelText("Executor replay").textContent).toMatch(/not flown: no identified type/);
    expect(document.body.querySelectorAll(".training-readback-executor")).toHaveLength(0);
  });

  it("reads the cursor off a chart, and a click selects that chart's column", () => {
    const { onCursorChange, onColumnChange } = open();
    fireEvent.mouseMove(screen.getByLabelText("Heading chart"));
    expect(onCursorChange).toHaveBeenCalled();
    expect(onColumnChange).not.toHaveBeenCalled();
    fireEvent.click(screen.getByLabelText("Speed chart"));
    expect(onColumnChange).toHaveBeenCalledWith("speed");
  });

  it("draws nothing yellow until a column is selected", () => {
    open(ALL, 0, 22);
    expect(yellow()).toHaveLength(0);
    expect(count(".training-readback-focus")).toBe(0);
  });

  it("draws ONLY the selected column's word yellow — not the other words in force at the same step", () => {
    // at 22 s heading 180° (from step 10) and altitude 1110 m (from step 0) are both in force
    open(ALL, 0, 22, "heading");
    const lit = yellow().map((element) => element.getAttribute("class"));
    expect(lit.sort()).toEqual(["training-readback-focus", "training-readback-focus", "training-readback-heading-band"]);
    // its rows: over the plan track and over the heading chart, not the altitude or speed chart
    expect(chart("Altitude chart").querySelector(".training-readback-focus")).toBeNull();
    // every other word's envelope recedes; with nothing selected nothing does
    const opacities = [...document.body.querySelectorAll(".training-readback-heading-band")].map((band) => band.getAttribute("opacity"));
    expect(opacities).toEqual(["0.3", "0.3", "1"]);
    expect(document.body.querySelector(".training-readback-corridor")!.getAttribute("opacity")).toBe("0.3");
    cleanup();
    open(ALL, 0, 22);
    expect(document.body.querySelector(".training-readback-corridor")!.getAttribute("opacity")).toBe("1");
  });

  it("gives the clearance the corridor and the capture turn, and an angle word its rows on the altitude chart", () => {
    open(ALL, 0, 60, "approach");
    const lit = yellow().map((element) => element.getAttribute("class"));
    expect(lit).toEqual(expect.arrayContaining(["training-readback-corridor", "training-readback-capture-turn",
      "training-readback-capture-band", "training-readback-capture-course", "training-readback-course-band"]));
    expect(lit).not.toContain("training-readback-heading-band");
    cleanup();
    open(ALL, 0, 60, "angle");
    expect(screen.getByLabelText("Altitude chart").querySelector(".training-readback-focus")).not.toBeNull();
  });
});

describe("helpers", () => {
  it("pads an extent and never returns a zero-wide one", () => {
    expect(extent([0, 100])).toEqual([-8, 108]);
    expect(extent([5, 5])).toEqual([4, 6]);
  });
});

describe("the live executor in the read-back check", () => {
  it("draws its segment on the plan, heading, altitude and speed charts, with the selected heading word's band", () => {
    cleanup();
    open(ALL, 0, 16, "heading", null, autopilotSegment());
    // plan, heading, altitude, speed
    expect(count("polyline.training-readback-autopilot")).toBe(4);
    expect(count("rect.training-readback-autopilot-band")).toBe(1);
    expect(screen.getByLabelText("The autopilot, live").textContent).toMatch(/solid blue: the selected heading word's segment/);
  });

  it("says how to get one when there is none", () => {
    cleanup();
    open(ALL, 0, 16, "heading");
    expect(count("polyline.training-readback-autopilot")).toBe(0);
    expect(screen.getByLabelText("The autopilot, live").textContent).toMatch(/select a word/);
  });
});
