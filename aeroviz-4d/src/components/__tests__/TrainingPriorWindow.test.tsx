/**
 * The prior's window: at the cursor, per column, the truth, its probability and the prior's ranked words; along
 * the flight, a strip per column with a tick at every word the truth says after step 0, teal where the prior's most
 * likely word is that word and red where it is another.
 */
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";

import TrainingPriorWindow from "../TrainingPriorWindow";
import { parseTrainingSample, type TrainingColumn } from "../../data/trainingSample";
import { parseTrainingPriorOverlay } from "../../data/trainingOverlays";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockPriorOverlay } from "../../data/__tests__/trainingOverlays.fixture";
import { TRAINING_EXECUTOR_COLOR, TRAINING_OUTSIDE_COLOR } from "../../utils/trainingWordColors";

function open(cursorS = 0, column: TrainingColumn | null = null) {
  const sample = parseTrainingSample(mockSample());
  if (!sample.ok) throw new Error(sample.problem);
  const prior = parseTrainingPriorOverlay(mockPriorOverlay(), sample.value);
  if (!prior.ok) throw new Error(prior.problem);
  const onCursorChange = vi.fn();
  const onColumnChange = vi.fn();
  render(
    <TrainingPriorWindow
      flight={sample.value.flights[0]}
      vocabulary={sample.value.vocabulary}
      candidates={sample.value.candidates}
      prior={{ overlay: prior.value, flight: prior.value.flights[0] }}
      cursorS={cursorS}
      onCursorChange={onCursorChange}
      column={column}
      onColumnChange={onColumnChange}
      onClose={() => undefined}
    />,
  );
  return { onCursorChange, onColumnChange };
}

describe("TrainingPriorWindow", () => {
  it("reads the step at the cursor: the truth, its probability, and the prior's ranked words", () => {
    open(20);   // step 10: the heading word 180°
    const table = screen.getByRole("table", { name: "The prior at step 10" });
    const heading = within(table).getByText("heading").closest("tr")!;
    expect(heading.textContent).toMatch(/says 180°/);
    expect(heading.textContent).toMatch(/0\.180/);                  // P(truth) = 0.6 × 0.3
    expect(heading.textContent).toMatch(/185° 0\.500 · 180° 0\.300 · 190° 0\.100/);
    const speed = within(table).getByText("speed").closest("tr")!;
    expect(speed.textContent).toMatch(/unchanged \(in force: 110 m\/s\)/);
  });

  it("ticks each word the truth says after step 0, teal where the prior ranked it first and red where not", () => {
    open();
    const ticks = [...document.body.querySelectorAll(".training-prior-tick line")];
    // the vectored flight's words after step 0: heading at 8 and 10, the clearance, altitude and angle at 20, speed at 30
    expect(ticks).toHaveLength(6);
    const red = ticks.filter((line) => line.getAttribute("stroke") === TRAINING_OUTSIDE_COLOR);
    expect(red).toHaveLength(1);
    expect(red[0].closest("g")!.getAttribute("aria-label")).toBe("heading 180° at step 10");
    expect(ticks.filter((line) => line.getAttribute("stroke") === TRAINING_EXECUTOR_COLOR)).toHaveLength(5);
  });

  it("states the flight's likelihood beside the readout's", () => {
    open();
    expect(screen.getByText(/this flight 0\.250 nats per step · val: prior 0\.1778, repeat 0\.3444, previous word 0\.3260/)).toBeTruthy();
  });

  it("selects a column from its row", () => {
    const { onColumnChange } = open();
    fireEvent.click(within(screen.getByRole("table")).getByText("altitude"));
    expect(onColumnChange).toHaveBeenCalledWith("altitude");
  });
});
