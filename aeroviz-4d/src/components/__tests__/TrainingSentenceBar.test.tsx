/**
 * The sentence bar: six rows in the vocabulary's order, step 0 complete, the silent steps counted,
 * the moments that are not words marked, the cursor moved by the artefact's own steps, ONE
 * column's word selected at a time, and — when their overlays are on — the executor's verdict on
 * each word and the way into the prior's predictions.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, DEFAULT_LAYERS, setTrainingPick } = vi.hoisted(() => {
  const DEFAULT_LAYERS = { headingBands: true, corridor: true, vertical: true, candidates: true };
  return {
    DEFAULT_LAYERS,
    setTrainingPick: vi.fn(),
    appState: {
      trainingSelection: null as unknown, trainingLayers: { ...DEFAULT_LAYERS },
      trainingExecutor: null as unknown, trainingPrior: null as unknown, trainingAutopilot: null as unknown,
      trainingPick: null as unknown, trainingAutopilotAuto: true,
    },
  };
});

vi.mock("../../context/AppContext", async () => {
  const { useState } = await import("react");
  return {
    useApp: () => {
      const [trainingCursorS, setTrainingCursorS] = useState(0);
      const [trainingColumn, setTrainingColumn] = useState<string | null>(null);
      return { ...appState, trainingCursorS, setTrainingCursorS, trainingColumn, setTrainingColumn, setTrainingPick };
    },
  };
});

import TrainingSentenceBar, { spacedLabels } from "../TrainingSentenceBar";
import { parseTrainingSample, TRAINING_COLUMNS } from "../../data/trainingSample";
import { MOCK_ROWS, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { EXECUTOR_ID, PRIOR_ID, mockExecutorOverlay, mockOverlayEntry, mockPriorOverlay } from "../../data/__tests__/trainingOverlays.fixture";
import { parseTrainingExecutorOverlay, parseTrainingPriorOverlay } from "../../data/trainingOverlays";
import { parseTrainingAutopilot } from "../../data/trainingAutopilot";
import { VECTORED_KEY } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest } from "../../data/__tests__/trainingAutopilot.fixture";

/** Publish the fixture's overlays for the selected flight, as the panel does. */
function overlays(position = 0) {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  const executor = parseTrainingExecutorOverlay(mockExecutorOverlay(), mockOverlayEntry(EXECUTOR_ID), parsed.value);
  const prior = parseTrainingPriorOverlay(mockPriorOverlay(), mockOverlayEntry(PRIOR_ID), parsed.value);
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
    appState.trainingAutopilot = null;
    appState.trainingPick = null;
    appState.trainingAutopilotAuto = true;
    setTrainingPick.mockClear();
  });

  it("picks a band's word for the live executor when the band is clicked, and clears it on a second click", () => {
    select();
    render(<TrainingSentenceBar />);
    const band = screen.getByLabelText(/^heading 225° .* issued at step 8 /);
    fireEvent.click(band);
    expect(setTrainingPick).toHaveBeenLastCalledWith({ flightKey: VECTORED_KEY, column: "heading", row: 8, attempt: 0 });
    fireEvent.click(band);
    expect(setTrainingPick).toHaveBeenLastCalledWith(null);
  });

  it("flies the selected word from its Fly button — the only way with the panel's switch off", () => {
    select();
    appState.trainingAutopilotAuto = false;
    render(<TrainingSentenceBar />);
    const fly = screen.getByRole("button", { name: "▶ Fly a segment — select a word first" }) as HTMLButtonElement;
    expect(fly.disabled).toBe(true);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    expect(setTrainingPick).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "▶ Fly this segment" }));
    expect(setTrainingPick).toHaveBeenLastCalledWith({ flightKey: VECTORED_KEY, column: "heading", row: 8, attempt: 0 });
  });

  it("offers to fly the same segment again once it has flown, and waits while it flies", () => {
    select();
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const request = mockAutopilotRequest(parsed.value, VECTORED_KEY, "heading", 8);
    appState.trainingPick = { flightKey: VECTORED_KEY, column: "heading", row: 8, attempt: 2 };
    appState.trainingAutopilot = { status: "flying", request };
    const { unmount } = render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    expect((screen.getByRole("button", { name: "Flying …" }) as HTMLButtonElement).disabled).toBe(true);
    unmount();
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(parsed.value, request), request, parsed.value);
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: answer.value, playedAt: 0, roundTripS: 0.5 };
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    setTrainingPick.mockClear();
    fireEvent.click(screen.getByRole("button", { name: "↻ Fly again" }));
    expect(setTrainingPick).toHaveBeenLastCalledWith({ flightKey: VECTORED_KEY, column: "heading", row: 8, attempt: 3 });
  });

  it("reads out the live executor's segment in one short line: the word, its verdict, the two times", () => {
    select();
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const request = mockAutopilotRequest(parsed.value, VECTORED_KEY, "heading", 8);
    appState.trainingAutopilot = { status: "flying", request };
    const { unmount } = render(<TrainingSentenceBar />);
    expect(screen.getByText("Autopilot · flying heading 225° …")).toBeTruthy();
    unmount();
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(parsed.value, request), request, parsed.value);
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: answer.value, playedAt: 0, roundTripS: 0.5 };
    const { unmount: done } = render(<TrainingSentenceBar />);
    const line = document.querySelector(".training-sentence-autopilot")!;
    expect(line.textContent).toBe("Autopilot · heading 225° · ✓ inside its envelope · 8.00 s flown in 1.24 s");
    expect((line.querySelector("strong") as HTMLElement).style.color).toBe("rgb(37, 99, 235)");
    done();
    // outside its envelope: the verdict in the loud red
    const outside = mockAutopilotAnswer(parsed.value, request);
    outside.word.status = "outside";
    outside.word.checks[0].ok = false;
    outside.word.checks[0].inside = 1;
    outside.word.heading.inside[1] = 0;
    const flown = parseTrainingAutopilot(outside, request, parsed.value);
    if (!flown.ok) throw new Error(flown.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: flown.value, playedAt: 0, roundTripS: 0.5 };
    render(<TrainingSentenceBar />);
    const red = document.querySelector(".training-sentence-autopilot strong") as HTMLElement;
    expect(red.textContent).toBe("✗ outside its envelope");
    expect(red.style.color).toBe("rgb(255, 45, 45)");
  });

  it("marks each word with the executor's verdict, and none where a word has no check of its own", () => {
    select();
    overlays();
    render(<TrainingSentenceBar />);
    expect(document.querySelectorAll(".training-sentence-verdict-inside")).toHaveLength(5);
    expect(document.querySelectorAll(".training-sentence-verdict-outside")).toHaveLength(2);
    expect(document.querySelectorAll(".training-sentence-verdict")).toHaveLength(7);
    expect(screen.getByLabelText(/^approach cleared .* issued at step 20 /).textContent).toMatch(
      /the executor: outside, told at its step 20 — capture turn monotone ✓, corridor held to the landing ✗ \(30\/35\)/);
    // a heading word's verdict: its band's rows on the flown track, from where it was told plus the lead
    expect(screen.getByLabelText(/^heading 180° .* issued at step 10 /).textContent).toMatch(
      /the executor: outside, told at its step 10 — track within ±4\.5° of the word, 4 s after it was told, to the next word's ✗ \(7\/8\)/);
    expect(screen.getByText(/the executor \(own dynamics\): landed 1\.5 m right of the centreline, 20\.8 m above the threshold/)).toBeTruthy();
    expect(screen.getByText(/5\/7 words inside their envelopes · evaluation pass \(observed pass\)/)).toBeTruthy();
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

  it("labels the words from the vocabulary, a heading word by why it was issued", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByLabelText(/^heading 225° — the heading the track reaches a lead later, issued at step 8 \(16 s\), in force to 20 s$/)).toBeTruthy();
    expect(screen.getByLabelText(/^heading 180° — the heading the track reaches a lead later, issued at step 10 \(20 s\), in force to 120 s$/)).toBeTruthy();
    expect(screen.getByLabelText(/^altitude descend to land — a new target, issued at step 20/)).toBeTruthy();
    expect(screen.getByLabelText(/^speed unspecified — speed left to the pilot, issued at step 30/)).toBeTruthy();
    expect(screen.getByLabelText(/^approach cleared — cleared to join the final, issued at step 20/)).toBeTruthy();
  });

  it("counts the words after step 0 and the silent steps", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByText(`${MOCK_ROWS} steps · 6 words after step 0 · ${MOCK_ROWS - 5} of ${MOCK_ROWS - 1} later steps silent`)).toBeTruthy();
  });

  it("reads out the labeller's verdicts", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByText(/heading 2\/3 in their bands · capture turn ✓ ✓ · altitude 1\/2 · speed 1\/1/)).toBeTruthy();
    select(1);
    render(<TrainingSentenceBar />);
    expect(screen.getByText(/heading 0\/0 in their bands \(1 with no row of their own: the lead reaches the clearance\) · capture turn none \(on the final at entry\)/)).toBeTruthy();
  });

  it("marks the clearance, the capture and the unspecified speed", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByLabelText(/^cleared to join the final at 40 s$/)).toBeTruthy();
    expect(screen.getByLabelText(/^the final captured at 50 s, 12\.5 km before the threshold$/)).toBeTruthy();
    expect(screen.getByLabelText(/^speed left to the pilot from 60 s$/)).toBeTruthy();
  });

  it("moves the cursor to a band's issue and to a numbered step", () => {
    select();
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(/^heading 180° — the heading/));
    expect(screen.getByText("t = 20 s · step 10")).toBeTruthy();
    fireEvent.keyDown(screen.getByLabelText(/^Step 20 at 40 s: approach cleared, altitude descend to land, angle descent 3/), { key: "Enter" });
    expect(screen.getByText("t = 40 s · step 20")).toBeTruthy();
  });

  it("selects one column's word: the other words in force at that step are not highlighted", () => {
    select();
    render(<TrainingSentenceBar />);
    const pressed = () => [...document.querySelectorAll("[aria-pressed='true']")].map((band) => band.getAttribute("aria-label"));
    expect(pressed()).toEqual([]);
    fireEvent.click(screen.getByLabelText(/^heading 180° — the heading/));
    // every other column's step-0 word is in force at step 10 too
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
    expect(legend.textContent).toMatch(/a heading word's judged rows, on the ground: from 4 s after it is said to the next word's, where the track must stay within ±4\.5° of it/);
    expect(legend.textContent).toMatch(/the capture turn's rows on the ground/);
    expect(legend.textContent).not.toMatch(/funnel|turn region|where the turn may end/);
    fireEvent.click(screen.getByText("Legend ▾"));
    expect(legend.textContent).not.toMatch(/judged rows/);
    appState.trainingLayers = { ...appState.trainingLayers, headingBands: false };
    render(<TrainingSentenceBar />);
    expect(screen.getAllByLabelText("What the 3D scene shows")[1].textContent).not.toMatch(/judged rows/);
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
