/**
 * The sentence bar: six rows in the vocabulary's order, step 0 complete, the silent steps counted,
 * the moments that are not words marked, the cursor moved by the artefact's own steps, ONE
 * column's word selected at a time, and — when their overlays are on — the executor's verdict on
 * each word and the way into the prior's predictions.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, DEFAULT_LAYERS } = vi.hoisted(() => {
  const DEFAULT_LAYERS = { turnPaths: true, turnRegions: false, holdFunnels: false, corridor: true, vertical: true, candidates: true };
  return {
    DEFAULT_LAYERS,
    appState: {
      trainingSelection: null as unknown, trainingLayers: { ...DEFAULT_LAYERS },
      trainingExecutor: null as unknown, trainingPrior: null as unknown,
    },
  };
});

vi.mock("../../context/AppContext", async () => {
  const { useState } = await import("react");
  return {
    useApp: () => {
      const [trainingCursorS, setTrainingCursorS] = useState(0);
      const [trainingColumn, setTrainingColumn] = useState<string | null>(null);
      return { ...appState, trainingCursorS, setTrainingCursorS, trainingColumn, setTrainingColumn };
    },
  };
});

import TrainingSentenceBar, { spacedLabels } from "../TrainingSentenceBar";
import { parseTrainingSample, TRAINING_COLUMNS } from "../../data/trainingSample";
import { MOCK_ROWS, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockExecutorOverlay, mockPriorOverlay } from "../../data/__tests__/trainingOverlays.fixture";
import { parseTrainingExecutorOverlay, parseTrainingPriorOverlay } from "../../data/trainingOverlays";

/** Publish the fixture's overlays for the selected flight, as the panel does. */
function overlays(position = 0) {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  const executor = parseTrainingExecutorOverlay(mockExecutorOverlay(), parsed.value);
  const prior = parseTrainingPriorOverlay(mockPriorOverlay(), parsed.value);
  if (!executor.ok || !prior.ok) throw new Error("the overlay fixtures do not read");
  appState.trainingExecutor = { overlay: executor.value, flight: executor.value.flights[position] };
  appState.trainingPrior = { overlay: prior.value, flight: prior.value.flights[position] };
}

function select(position = 0) {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  appState.trainingSelection = {
    vocabulary: parsed.value.vocabulary, candidates: parsed.value.candidates, flight: parsed.value.flights[position],
  };
}

describe("TrainingSentenceBar", () => {
  beforeEach(() => {
    appState.trainingSelection = null;
    appState.trainingLayers = { ...DEFAULT_LAYERS };
    appState.trainingExecutor = null;
    appState.trainingPrior = null;
  });

  it("marks each word with the executor's verdict, and none where a word has no check of its own", () => {
    select();
    overlays();
    render(<TrainingSentenceBar />);
    expect(document.querySelectorAll(".training-sentence-verdict-inside")).toHaveLength(5);
    expect(document.querySelectorAll(".training-sentence-verdict-outside")).toHaveLength(1);
    expect(document.querySelectorAll(".training-sentence-verdict")).toHaveLength(6);
    expect(screen.getByLabelText(/^approach cleared .* issued at step 10 /).textContent).toMatch(
      /the executor: outside, told at its step 10 — capture turn monotone ✓, corridor held to the landing ✗ \(30\/35\)/);
    expect(screen.getByText(/the executor \(own dynamics\): landed 1\.5 m right of the centreline, 20\.8 m above the threshold/)).toBeTruthy();
    expect(screen.getByText(/5\/6 words inside their envelopes · evaluation pass \(observed pass\)/)).toBeTruthy();
  });

  it("says why a flight the replay does not fly has no verdicts", () => {
    select(1);
    overlays(1);
    render(<TrainingSentenceBar />);
    expect(document.querySelectorAll(".training-sentence-verdict")).toHaveLength(0);
    expect(screen.getByText("the executor: not flown — no identified type")).toBeTruthy();
  });

  it("opens the prior's predictions from its own button", () => {
    select();
    overlays();
    render(<TrainingSentenceBar />);
    expect(screen.getByText(/the prior: 0\.250 nats per step here \(val 0\.1778\)/)).toBeTruthy();
    fireEvent.click(screen.getByText("Prior predictions"));
    expect(screen.getByRole("dialog", { name: "Prior predictions" })).toBeTruthy();
  });

  it("ignores an overlay published for another flight", () => {
    select(0);
    overlays(1);
    render(<TrainingSentenceBar />);
    expect(screen.queryByText(/the executor/)).toBeNull();
    expect(screen.queryByText("Prior predictions")).toBeNull();
  });

  it("draws nothing until a flight is selected", () => {
    const { container } = render(<TrainingSentenceBar />);
    expect(container.innerHTML).toBe("");
  });

  it("draws the six columns in the vocabulary's order", () => {
    select();
    render(<TrainingSentenceBar />);
    const rows = [...document.querySelectorAll("g[aria-label$=' row']")].map((row) => row.getAttribute("aria-label"));
    expect(rows).toEqual(TRAINING_COLUMNS.map((column) => `${column} row`));
  });

  it("opens every column at step 0 — step 0 gives all six words", () => {
    select();
    render(<TrainingSentenceBar />);
    for (const column of TRAINING_COLUMNS) {
      expect(screen.getAllByLabelText(new RegExp(`^${column} .* issued at step 0 `))).toHaveLength(1);
    }
  });

  it("labels the words from the vocabulary, the heading turn by why it was issued", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByLabelText(/^heading 180° — a turn, issued at step 10 \(20 s\), in force to 120 s$/)).toBeTruthy();
    expect(screen.getByLabelText(/^altitude descend to land — a new target, issued at step 20/)).toBeTruthy();
    expect(screen.getByLabelText(/^speed unspecified — speed left to the pilot, issued at step 30/)).toBeTruthy();
    expect(screen.getByLabelText(/^approach cleared — cleared to join the final, issued at step 10/)).toBeTruthy();
  });

  it("counts the words after step 0 and the silent steps", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByText(`${MOCK_ROWS} steps · 5 words after step 0 · ${MOCK_ROWS - 4} of ${MOCK_ROWS - 1} later steps silent`)).toBeTruthy();
  });

  it("reads out the labeller's verdicts", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByText(/turns 1\/1 monotone, 1\/1 rate · holds 1\/2 in their funnel \(1 weak: the funnel starts over 2 km wide\) · capture turn ✓ ✓ · altitude 1\/2 · speed 1\/1/)).toBeTruthy();
    select(1);
    render(<TrainingSentenceBar />);
    expect(screen.getByText(/capture turn none \(on the final at entry\)/)).toBeTruthy();
  });

  it("marks the clearance, the capture and the unspecified speed", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByLabelText(/^cleared to join the final at 20 s$/)).toBeTruthy();
    expect(screen.getByLabelText(/^the final captured at 50 s, 12\.5 km before the threshold$/)).toBeTruthy();
    expect(screen.getByLabelText(/^speed left to the pilot from 60 s$/)).toBeTruthy();
  });

  it("moves the cursor to a band's issue and to a numbered step", () => {
    select();
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(/^heading 180° — a turn/));
    expect(screen.getByText("t = 20 s · step 10")).toBeTruthy();
    fireEvent.keyDown(screen.getByLabelText(/^Step 20 at 40 s: altitude descend to land, angle descent 3/), { key: "Enter" });
    expect(screen.getByText("t = 40 s · step 20")).toBeTruthy();
  });

  it("selects one column's word: the other words in force at that step are not highlighted", () => {
    select();
    render(<TrainingSentenceBar />);
    const pressed = () => [...document.querySelectorAll("[aria-pressed='true']")].map((band) => band.getAttribute("aria-label"));
    expect(pressed()).toEqual([]);
    fireEvent.click(screen.getByLabelText(/^heading 180° — a turn/));
    // step 10 also has approach "cleared", and every other column's step-0 word is in force there
    expect(pressed()).toEqual([expect.stringMatching(/^heading 180°/)]);
    // the step numbers move the cursor and keep the column: the heading word in force there
    fireEvent.click(screen.getByLabelText(/^Step 0 at 0 s/));
    expect(pressed()).toEqual([expect.stringMatching(/^heading 270°/)]);
    fireEvent.click(screen.getByLabelText(/^altitude descend to land/));
    expect(pressed()).toEqual([expect.stringMatching(/^altitude descend to land/)]);
    // the selected word, clicked again, clears the selection
    fireEvent.click(screen.getByLabelText(/^altitude descend to land/));
    expect(pressed()).toEqual([]);
  });

  it("carries a legend of what is drawn: only the switches that are on, and it folds away", () => {
    select();
    render(<TrainingSentenceBar />);
    const legend = screen.getByLabelText("What the 3D scene shows");
    expect(legend.textContent).toMatch(/where the turn may end: between the two turns' ends/);
    expect(legend.textContent).not.toMatch(/hold funnel/);          // off by default
    expect(legend.textContent).not.toMatch(/turn region/);
    fireEvent.click(screen.getByText("Legend ▾"));
    expect(legend.textContent).not.toMatch(/where the turn may end/);
    appState.trainingLayers = { ...appState.trainingLayers, holdFunnels: true };
    render(<TrainingSentenceBar />);
    expect(screen.getAllByLabelText("What the 3D scene shows")[1].textContent)
      .toMatch(/a judged hold whose funnel starts over 2 km wide: a weak check/);
  });

  it("opens the read-back check", () => {
    select();
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByText("Read-back check"));
    expect(screen.getByRole("dialog", { name: "Read-back check" })).toBeTruthy();
  });
});

describe("spacedLabels", () => {
  it("keeps a label only when it clears the last kept one", () => {
    expect(spacedLabels([0, 5, 20, 22, 40], 10)).toEqual([true, false, true, false, true]);
  });

  it("always keeps the last with keepLast, evicting a neighbour that would collide", () => {
    expect(spacedLabels([0, 20, 25], 10, true)).toEqual([true, false, true]);
  });
});
