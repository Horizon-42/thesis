/**
 * The sentence bar: five rows in the vocabulary's order (no approach row), row 0 complete, a short header whose chips carry
 * their full reading in their tooltips, a tab per sentence (the labelled one and each Δ), the words the closed-loop reading
 * added marked, the flown flight's outcome and DA check in chips, the cursor moved by the artefact's own rows, ONE column's word
 * selected at a time, and the live executor's pick made by a click on a closed-loop word.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

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
import { publishTrainingTabs, useTrainingTabs } from "../../data/trainingTabs";

/** The tabs a stage-A session gives the bar (outline §6.2 item 2): Labelled and one per Δ, the outcome of each. */
function publishStageATabs(chosen: string) {
  const selection = stageASelection() as any;
  publishTrainingTabs({ scope: "test", chosen, tabs: [
    { id: "labelled", label: "Labelled", title: "the labelled sentence", outcome: null },
    ...selection.vocabulary.rowIntervalsS.map((interval: number) => ({
      id: `interval-${interval}`, label: `Δ ${interval} s`, title: `Δ ${interval} s`,
      outcome: selection.flight.closedLoop[String(interval)].replay.outcome })),
  ] });
}

function open(intervalS: number | null = 2) {
  appState.trainingSelection = stageASelection();
  appState.trainingIntervalS = intervalS;
  publishStageATabs(intervalS === null ? "labelled" : `interval-${intervalS}`);
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
    // no backend in the tests: the set's intent is named as absent
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("no backend in the tests"); }));
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

  it("shows the tabs the session gives, a dot in each flown sentence's outcome colour, and chooses by click and by key", () => {
    let chosen: string | null = null;
    function Spy() {
      chosen = useTrainingTabs()?.chosen ?? null;
      return null;
    }
    open(4);
    render(<Spy />);
    const tabsNow = () => screen.getAllByRole("button").filter((item) => item.classList.contains("training-source-tab"));
    expect(tabsNow().map((tab) => tab.textContent)).toEqual(["Labelled", "Δ 2 s", "Δ 4 s", "Δ 8 s"]);
    expect(tabsNow().map((tab) => tab.getAttribute("aria-pressed"))).toEqual(["false", "false", "true", "false"]);
    expect(tabsNow()[0].querySelector(".training-source-tab-dot")).toBeNull();               // not flown: no dot
    expect(tabsNow()[1].querySelector(".training-source-tab-dot")).not.toBeNull();
    fireEvent.click(tabsNow()[3]);
    expect(chosen).toBe("interval-8");
    const group = screen.getByRole("group", { name: "Which sentence is read" });
    fireEvent.keyDown(group, { key: "ArrowRight" });                                        // wraps to the first
    expect(chosen).toBe("labelled");
    fireEvent.keyDown(group, { key: "End" });
    expect(chosen).toBe("interval-8");
    fireEvent.keyDown(group, { key: "Home" });
    expect(chosen).toBe("labelled");
    fireEvent.keyDown(group, { key: "ArrowLeft" });
    expect(chosen).toBe("interval-8");
    expect(tabsNow().map((tab) => tab.getAttribute("aria-pressed"))).toEqual(["false", "false", "false", "true"]);
    expect(setTrainingIntervalS).not.toHaveBeenCalled();                                     // the session maps the choice
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

describe("the bar's chips and notes follow the kind of the sentence on screen (outline §6.2 item 2)", () => {
  beforeEach(() => {
    appState.mode = "training";
    appState.trainingLayers = { ...DEFAULT_LAYERS };
    appState.trainingAutopilot = null;
    appState.trainingPick = null;
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("no backend in the tests"); }));
  });

  const chips = () => [...document.querySelectorAll(".training-sentence-head .training-chip")].map((node) => node.textContent ?? "");

  it("a prior sample: the outcome, the DA check and the go-arounds — no corrections", async () => {
    const { stageBSelection, stageBSample } = await import("../../data/__tests__/stageB");
    const sample = stageBSample();
    appState.trainingSelection = stageBSelection(sample, 0);
    appState.trainingIntervalS = 4;
    render(<TrainingSentenceBar />);
    const said = sample.flights[0].sentences[0];
    expect(chips().some((text) => text === `${said.goArounds} go-around${said.goArounds === 1 ? "" : "s"}`)).toBe(true);
    expect(chips().some((text) => /correction/.test(text))).toBe(false);
  });

  it("stage B's closed loop: the outcome, the DA check and the corrections — no go-arounds", async () => {
    const { stageBSelection } = await import("../../data/__tests__/stageB");
    appState.trainingSelection = stageBSelection(undefined, "closedLoop");
    appState.trainingIntervalS = 4;
    render(<TrainingSentenceBar />);
    expect(chips().some((text) => /correction/.test(text))).toBe(true);
  });

  it("a round: the outcome, the reward and the loss of separation — no DA check, no corrections", async () => {
    const { stageCSample, stageCSelection } = await import("../../data/__tests__/stageC");
    const { stageASample } = await import("../../data/__tests__/stageA");
    const sample = stageCSample();
    // a round that crossed the threshold with a DA check (the fixture's rounds did not cross): the bar still shows no DA chip
    const crossing = stageASample().flights[0].closedLoop["2"].replay.crossing!;
    expect(crossing.decision).not.toBeNull();
    sample.windows[0].rounds[0].crossing = crossing;
    appState.trainingSelection = stageCSelection(sample, 0, "start");
    appState.trainingIntervalS = sample.model.rowIntervalS;
    render(<TrainingSentenceBar />);
    const end = sample.windows[0].rounds[0].end;
    expect(chips()).toContain(`reward ${end.reward.toFixed(2)}`);
    expect(chips().some((text) => (end.loss === null ? text === "no loss of separation" : text.startsWith("loss of separation")))).toBe(true);
    expect(chips().some((text) => /correction|DA /.test(text))).toBe(false);
  });

  it("a labelled sentence: no chip of a flown flight", () => {
    open(null);
    expect(chips().some((text) => /correction|go-around|reward|DA /.test(text))).toBe(false);
  });

  it("the notes start with the set and its intent line, and open the details page on the set and the experiment", async () => {
    const { useDetailsPage } = await import("../training/PanelParts");
    let shown: string | null = null;
    function Page() {
      shown = useDetailsPage(false).shown?.section ?? null;
      return null;
    }
    open(2);
    render(<Page />);
    fireEvent.click(screen.getByRole("button", { name: "How to read the bar" }));
    const notes = document.querySelector(".training-sentence-legend")!;
    expect(notes.firstElementChild!.textContent).toContain("Set fixture_set:");
    await waitFor(() => expect(document.querySelector(".training-sentence-legend")!.firstElementChild!.textContent)
      .toContain("No intent for fixture_set"));
    expect(notes.textContent).toContain("added");                                   // a closed-loop sentence's notes
    fireEvent.click(screen.getByRole("button", { name: "The set and the experiment ›" }));
    expect(shown).toBe("experiment");
  });
});
