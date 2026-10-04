/**
 * The sentence bar: five rows in the vocabulary's order (no approach row), row 0 complete, a short header whose chips carry
 * their full reading in their tooltips, a tab per sentence (the labelled one and each Δ), the words the closed-loop reading
 * added marked, the flown flight's outcome and DA check in chips, the cursor moved by the artefact's own rows, ONE column's word
 * selected at a time, and the live executor's pick made by a click on a closed-loop word.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, DEFAULT_LAYERS, setTrainingPick, setTrainingIntervalS } = vi.hoisted(() => {
  const DEFAULT_LAYERS = { headingBands: true, vertical: true, candidates: true };
  return {
    DEFAULT_LAYERS,
    setTrainingPick: vi.fn(),
    setTrainingIntervalS: vi.fn(),
    appState: {
      mode: "training", trainingSelection: null as unknown, trainingLayers: { ...DEFAULT_LAYERS },
      trainingIntervalS: 2 as number | null, trainingAutopilot: null as unknown, trainingPick: null as unknown,
    },
  };
});

vi.mock("../../context/AppContext", async () => {
  const { useState } = await import("react");
  return {
    useApp: () => {
      const [trainingColumn, setTrainingColumn] = useState<string | null>(null);
      return { ...appState, trainingColumn, setTrainingColumn, setTrainingPick, setTrainingIntervalS };
    },
    useTrainingCursor: () => {
      const [trainingCursorS, setTrainingCursorS] = useState(0);
      return { trainingCursorS, setTrainingCursorS };
    },
  };
});

import TrainingSentenceBar, { spacedLabels } from "../TrainingSentenceBar";
import TrainingLegend from "../TrainingLegend";
import { parseTrainingAutopilot } from "../../data/trainingAutopilot";
import { TRAINING_COLUMNS } from "../../data/trainingSample";
import { TRAINING_CORRECTION_COLOR } from "../../utils/trainingWordColors";
import { requestOf, stageAAnswers, stageASelection } from "../../data/__tests__/stageA";

function open(intervalS: number | null = 2) {
  appState.trainingSelection = stageASelection();
  appState.trainingIntervalS = intervalS;
  return render(<TrainingSentenceBar />);
}

function band(container: HTMLElement, column: string, row: number): SVGGElement {
  const rowGroup = container.querySelector(`g[aria-label="${column} row"]`)!;
  const bands = [...rowGroup.querySelectorAll<SVGGElement>("g.training-sentence-band")];
  const found = bands.find((item) => item.getAttribute("aria-label")!.includes(`said at row ${row} `));
  if (!found) throw new Error(`no ${column} band at row ${row}`);
  return found;
}

describe("TrainingSentenceBar", () => {
  beforeEach(() => {
    appState.mode = "training";
    appState.trainingSelection = null;
    appState.trainingLayers = { ...DEFAULT_LAYERS };
    appState.trainingIntervalS = 2;
    appState.trainingAutopilot = null;
    appState.trainingPick = null;
    setTrainingPick.mockClear();
    setTrainingIntervalS.mockClear();
  });

  it("draws only in Training and only with a flight", () => {
    const { container, rerender } = render(<TrainingSentenceBar />);
    expect(container.firstChild).toBeNull();
    appState.trainingSelection = stageASelection();
    appState.mode = "evaluation";
    rerender(<TrainingSentenceBar />);
    expect(container.firstChild).toBeNull();
  });

  it("has the five rows in the vocabulary's order and no approach row", () => {
    const { container } = open();
    const labels = [...container.querySelectorAll("text.training-sentence-row-label")].map((node) => node.textContent);
    expect(labels).toEqual(["Runway", "Heading", "Altitude", "Angle", "Speed"]);
    expect([...container.querySelectorAll('g[aria-label$=" row"]')].map((node) => node.getAttribute("aria-label")))
      .toEqual(TRAINING_COLUMNS.map((column) => `${column} row`));
    // row 0 says every column: each row opens with a band
    for (const column of TRAINING_COLUMNS) expect(band(container, column, 0)).toBeTruthy();
  });

  it("has a tab for the labelled sentence and one per Δ, and the pressed one is the interval read", () => {
    const { rerender } = open(4);
    const tabs = screen.getAllByRole("button").filter((item) => item.classList.contains("training-source-tab"));
    expect(tabs.map((tab) => tab.textContent)).toEqual(["Labelled", "Δ 2 s", "Δ 4 s", "Δ 8 s"]);
    expect(tabs.map((tab) => tab.getAttribute("aria-pressed"))).toEqual(["false", "false", "true", "false"]);
    fireEvent.click(tabs[3]);
    expect(setTrainingIntervalS).toHaveBeenLastCalledWith(8);
    fireEvent.click(tabs[0]);
    expect(setTrainingIntervalS).toHaveBeenLastCalledWith(null);
    appState.trainingIntervalS = null;
    rerender(<TrainingSentenceBar />);
    expect(screen.getAllByRole("button").filter((item) => item.classList.contains("training-source-tab"))
      .map((tab) => tab.getAttribute("aria-pressed"))).toEqual(["true", "false", "false", "false"]);
  });

  it("marks a correction word three ways: dashed, orange and with a mark; the labelled sentence has none", () => {
    const { container, unmount } = open(2);
    const corrections = [...container.querySelectorAll("g.training-sentence-band.correction")];
    expect(corrections.length).toBe(9);
    const first = corrections[0];
    expect(first.getAttribute("aria-label")).toContain("a CORRECTION: added by the closed-loop reading");
    const fill = first.querySelector("rect.training-sentence-band-fill")!;
    expect(fill.getAttribute("stroke-dasharray")).toBe("3 2");
    expect(fill.getAttribute("fill")).toBe(TRAINING_CORRECTION_COLOR);
    expect(first.querySelector("polygon.training-sentence-correction-mark")).toBeTruthy();
    expect(screen.getByText("9 corrections")).toBeTruthy();
    // an ordinary word is none of the three
    const ordinary = band(container, "heading", 0);
    expect(ordinary.classList.contains("correction")).toBe(false);
    expect(ordinary.querySelector("polygon.training-sentence-correction-mark")).toBeNull();
    unmount();
    const labelled = open(null);
    expect(labelled.container.querySelectorAll("g.training-sentence-band.correction")).toHaveLength(0);
    expect(screen.queryByText(/correction/)).toBeNull();
  });

  it("says how the flown flight ended and its DA check, with the values in the tooltips", () => {
    open(2);
    const outcome = screen.getByText("unstable at minimums");
    expect(outcome.title).toContain("15.5 m right of the centreline, 41.8 m above the threshold");
    const da = screen.getByText(/DA failed/);
    expect(da.title).toBe("DA check failed · 15.5 m right of the centreline (cone half width 116.3 m) ✓ · 26.8 m above the glidepath ✗");
    expect(screen.getByText("runway 09")).toBeTruthy();
  });

  it("shades the rows before the first predicted step as observed only, and ends the axis at the flown flight", () => {
    const { container } = open(4);
    expect(container.querySelector("g.training-sentence-observed title")!.textContent).toContain("observed only");
    // the axis ends where the sentence does (16 s + 124 rows of 4 s); the flown flight ended at cycle 494 of the executor's 1 s
    const ticks = [...container.querySelectorAll("text.training-sentence-tick")].map((node) => node.textContent);
    expect(ticks[ticks.length - 1]).toBe("512 s");
    const end = [...container.querySelectorAll("g[aria-label^='the flown flight ended']")];
    expect(end).toHaveLength(1);
    expect(end[0].getAttribute("aria-label")).toContain("ended at 510 s: ");
    expect(end[0].querySelector("line.training-sentence-marker")).toBeTruthy();
    // and its x is the flown end's place on the axis: 510 of 512 s
    const x = (node: Element, attr: string) => Number(node.getAttribute(attr));
    const axis = container.querySelector("line.training-sentence-axis")!;
    const atEnd = (x(end[0].querySelector("line")!, "x1") - x(axis, "x1")) / (x(axis, "x2") - x(axis, "x1"));
    expect(atEnd).toBeCloseTo(510 / 512, 4);
    const { container: labelled } = open(null);
    expect(labelled.querySelectorAll("g.training-sentence-observed")).toHaveLength(0);
  });

  it("moves the cursor by the artefact's own rows and selects ONE column's word, again to clear", () => {
    const { container } = open(2);
    expect(screen.getByText(/t = 0 s · before the sentence/)).toBeTruthy();
    fireEvent.click(band(container, "heading", 0));
    // the sentence opens at 16 s (state row 8 of a Δ of 2 s)
    expect(screen.getByText(/t = 16 s · row 0/)).toBeTruthy();
    expect(setTrainingPick).toHaveBeenLastCalledWith({ rowIntervalS: 2, column: "heading", row: 0, attempt: 0 });
    const heading = band(container, "heading", 0);
    expect(heading.getAttribute("aria-pressed")).toBe("true");
    expect(band(container, "speed", 0).getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(heading);
    expect(heading.getAttribute("aria-pressed")).toBe("false");
    expect(setTrainingPick).toHaveBeenLastCalledWith(null);
  });

  it("picks for the live executor by the Fly button and by a band click, in a closed-loop sentence only", () => {
    const { container, unmount } = open(2);
    const fly = screen.getByRole("button", { name: /Fly/ }) as HTMLButtonElement;
    expect(fly.disabled).toBe(true);
    fireEvent.click(band(container, "altitude", 0));
    expect(fly.disabled).toBe(false);
    setTrainingPick.mockClear();
    fireEvent.click(fly);
    expect(setTrainingPick).toHaveBeenCalledWith({ rowIntervalS: 2, column: "altitude", row: 0, attempt: 0 });
    unmount();
    // the labelled sentence is not flown
    const labelled = open(null);
    setTrainingPick.mockClear();
    fireEvent.click(band(labelled.container, "heading", 0));
    expect(setTrainingPick).not.toHaveBeenCalled();
    expect((screen.getByRole("button", { name: /Fly/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole("button", { name: /Fly/ }).title).toContain("choose a Δ");
  });

  it("carries the judge's verdict on a word with an envelope as a dot, and the row's tally at its end", () => {
    const { container } = open(2);
    expect(container.querySelectorAll("circle.training-sentence-verdict").length).toBeGreaterThan(0);
    const tally = container.querySelector('text[aria-label^="heading:"]')!;
    expect(tally.textContent).toMatch(/\d+\/\d+$/);
    expect(container.querySelector('g[aria-label="runway row"] text.training-sentence-tally')).toBeNull();
  });

  it("shows the live executor's line for the picked word and nothing for another interval", () => {
    const [raw] = stageAAnswers();
    const selection = stageASelection();
    const request = requestOf(raw);
    const answer = parseTrainingAutopilot(raw, request, selection);
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: answer.value, playedAt: Date.now() };
    const { container, unmount } = open(2);
    expect(screen.getByText(/flown to the next word/)).toBeTruthy();
    expect(container.querySelector("rect.training-sentence-autopilot-cursor")).toBeTruthy();
    unmount();
    open(4);
    expect(screen.queryByText(/flown to the next word/)).toBeNull();
  });

  it("opens the read-back check and the notes", () => {
    open(2);
    fireEvent.click(screen.getByRole("button", { name: "Read-back" }));
    expect(screen.getByRole("dialog", { name: "Read-back check" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "How to read the bar" }));
    expect(screen.getByText(/is a correction — a word the reading added/)).toBeTruthy();
  });
});

describe("spacedLabels", () => {
  it("keeps a label only where it clears the last kept, and the last one if asked", () => {
    expect(spacedLabels([0, 5, 20, 22, 40], 14)).toEqual([true, false, true, false, true]);
    expect(spacedLabels([0, 10, 12], 14, true)).toEqual([false, false, true]);
    expect(spacedLabels([], 14, true)).toEqual([]);
  });
});

describe("TrainingLegend", () => {
  it("lists the flown path, the DA point and the correction words only for a closed-loop sentence that has them", () => {
    const vocabulary = stageASelection().vocabulary;
    const layers = { ...DEFAULT_LAYERS };
    const { rerender } = render(<TrainingLegend layers={layers} vocabulary={vocabulary} closed corrections autopilotColour={null} />);
    fireEvent.click(screen.getByRole("button", { name: /Legend/ }));
    expect(screen.getByText("the flown path")).toBeTruthy();
    expect(screen.getByText("DA point")).toBeTruthy();
    expect(screen.getByText("correction word")).toBeTruthy();
    rerender(<TrainingLegend layers={layers} vocabulary={vocabulary} closed={false} corrections={false} autopilotColour={null} />);
    expect(screen.queryByText("the flown path")).toBeNull();
    expect(screen.queryByText("correction word")).toBeNull();
    rerender(<TrainingLegend layers={{ ...layers, headingBands: false }} vocabulary={vocabulary} closed={false} corrections={false}
      autopilotColour="#2563eb" />);
    expect(screen.queryByText("heading word: judged rows")).toBeNull();
    expect(screen.getByText("autopilot segment")).toBeTruthy();
  });
});
