/**
 * The sentence bar: six rows in the vocabulary's order, step 0 complete, the silent steps counted,
 * the moments that are not words marked, and the cursor moved by the artefact's own steps.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState } = vi.hoisted(() => ({
  appState: {
    trainingSelection: null as unknown,
    trainingLayers: { lateral: true, vertical: true, candidates: true },
  },
}));

vi.mock("../../context/AppContext", async () => {
  const { useState } = await import("react");
  return {
    useApp: () => {
      const [trainingCursorS, setTrainingCursorS] = useState(0);
      return { ...appState, trainingCursorS, setTrainingCursorS };
    },
  };
});

import TrainingSentenceBar, { spacedLabels } from "../TrainingSentenceBar";
import { parseTrainingSample, TRAINING_COLUMNS } from "../../data/trainingSample";
import { MOCK_ROWS, mockSample } from "../../data/__tests__/trainingSample.fixture";

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
    expect(screen.getByText(/turns 1\/1 monotone, 1\/1 rate · holds 1\/2 in their funnel · capture turn ✓ ✓ · altitude 1\/2 · speed 1\/1/)).toBeTruthy();
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
