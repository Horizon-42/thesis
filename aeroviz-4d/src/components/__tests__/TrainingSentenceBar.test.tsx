/**
 * The sentence bar: six rows in the vocabulary's order, step 0 complete, a short header whose chips carry their full
 * reading in their tooltips, the moments that are not words marked, the cursor moved by the artefact's own steps, ONE
 * column's word selected at a time, and — when their overlays are on — the executor's verdict on each word and the way
 * into the prior's predictions; one window open at a time.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, DEFAULT_LAYERS, setTrainingPick, setTrainingSource } = vi.hoisted(() => {
  const DEFAULT_LAYERS = { headingBands: true, corridor: true, vertical: true, candidates: true };
  return {
    DEFAULT_LAYERS,
    setTrainingPick: vi.fn(),
    setTrainingSource: vi.fn(),
    appState: {
      mode: "training", trainingSelection: null as unknown, trainingLayers: { ...DEFAULT_LAYERS },
      trainingExecutor: null as unknown, trainingPrior: null as unknown, trainingAutopilot: null as unknown,
      trainingPick: null as unknown, trainingAutopilotAuto: true,
      trainingGenerations: [] as unknown[], trainingSource: null as unknown,
    },
  };
});

vi.mock("../../context/AppContext", async () => {
  const { useState } = await import("react");
  return {
    useApp: () => {
      const [trainingColumn, setTrainingColumn] = useState<string | null>(null);
      return { ...appState, trainingColumn, setTrainingColumn, setTrainingPick, setTrainingSource };
    },
    useTrainingCursor: () => {
      const [trainingCursorS, setTrainingCursorS] = useState(0);
      return { trainingCursorS, setTrainingCursorS };
    },
  };
});

import TrainingSentenceBar, { spacedLabels } from "../TrainingSentenceBar";
import { parseTrainingSample, trainingSelectionOf, TRAINING_COLUMNS } from "../../data/trainingSample";
import { MOCK_ROWS, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { EXECUTOR_ID, PRIOR_ID, mockExecutorOverlay, mockOverlayEntry, mockPriorOverlay } from "../../data/__tests__/trainingOverlays.fixture";
import { parseTrainingExecutorOverlay, parseTrainingGenerationOverlay, parseTrainingPriorOverlay } from "../../data/trainingOverlays";
import {
  BASE_MODEL_ID, POST_TRAINED_ID, mockGenerationEntry, mockGenerationOverlay,
} from "../../data/__tests__/trainingOverlays.fixture";
import { parseTrainingAutopilot } from "../../data/trainingAutopilot";
import { VECTORED_KEY } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest, mockSelection } from "../../data/__tests__/trainingAutopilot.fixture";

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

/** Publish both models' own sentences over the selected flight, as the panel does (``change``: of each raw payload). */
function generations(position = 0, change: (raw: any) => void = () => undefined) {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  appState.trainingGenerations = [[BASE_MODEL_ID, false], [POST_TRAINED_ID, true]].map(([id, post]) => {
    const raw = mockGenerationOverlay(id as string, post as boolean);
    change(raw);
    const overlay = parseTrainingGenerationOverlay(raw, mockGenerationEntry(id as string), parsed.value);
    if (!overlay.ok) throw new Error(overlay.problem);
    return { overlay: overlay.value, flight: overlay.value.flights[position] };
  });
}

function select(position = 0) {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  appState.trainingSelection = trainingSelectionOf(parsed.value, parsed.value.flights[position]);
}

describe("TrainingSentenceBar", () => {
  beforeEach(() => {
    appState.mode = "training";
    appState.trainingSelection = null;
    appState.trainingLayers = { ...DEFAULT_LAYERS };
    appState.trainingExecutor = null;
    appState.trainingPrior = null;
    appState.trainingAutopilot = null;
    appState.trainingPick = null;
    appState.trainingAutopilotAuto = true;
    appState.trainingGenerations = [];
    appState.trainingSource = null;
    setTrainingPick.mockClear();
    setTrainingSource.mockClear();
  });

  it("offers the models' own sentences as tabs beside the truth, each choosing the sentence read", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.queryByRole("group", { name: "Which sentence is read" })).toBeNull();      // none published: no tabs
    generations();
    render(<TrainingSentenceBar />);
    const groups = screen.getAllByRole("group", { name: "Which sentence is read" });
    const tabs = groups[groups.length - 1];
    const names = [...tabs.querySelectorAll("button")].map((button) => button.textContent);
    expect(names).toEqual(["Truth", "base model", "post-trained"]);
    expect(tabs.querySelector("button")!.getAttribute("aria-pressed")).toBe("true");
    const post = screen.getAllByRole("button", { name: "post-trained" });
    fireEvent.click(post[post.length - 1]);
    expect(setTrainingSource).toHaveBeenCalledWith({ overlayId: POST_TRAINED_ID, sample: 0 });
  });

  it("starts another model at its first sample, whichever sample the last one was read at", () => {
    select();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 1 };
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: "post-trained" }));
    expect(setTrainingSource).toHaveBeenLastCalledWith({ overlayId: POST_TRAINED_ID, sample: 0 });
  });

  it("offers the truth's windows only on its tab: they read the truth, not the model's words", () => {
    select();
    overlays();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    expect(screen.queryByRole("button", { name: "Read-back" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Prior" })).toBeNull();
  });

  it("shades the rows a model spoke on after its flight ended — not the part of its last step past the end", () => {
    select();
    // the second sample crossed the threshold without the capture at 150 s — the executor flies on after it — and the
    // model spoke on for 15 steps
    generations(0, (raw) => {
      Object.assign(raw.flights[0].samples[1], { outcome: "crossed_without_capture", crossing: { crossM: 80, heightM: 30, atS: 149.6 },
        rows: 90 });
    });
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 1 };
    const { unmount } = render(<TrainingSentenceBar />);
    expect(screen.getByLabelText("after 150 s: the flight had ended; the model spoke on to where the executor stopped")).toBeTruthy();
    unmount();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    expect(screen.queryByLabelText(/^after .* the flight had ended/)).toBeNull();          // 110 s, its step ends at 112 s
  });

  it("draws a model's sample hatched, over the span it only observed, with the truth's words ticked under each row", () => {
    select();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    // the model's words, said by it — never the labeller's kinds
    expect(screen.getByLabelText(/^heading 225° — said by base model at step 12 \(24 s\), in force to 32 s$/)).toBeTruthy();
    expect(screen.queryByLabelText(/— the heading the track reaches a lead later/)).toBeNull();
    expect(document.querySelectorAll(".training-sentence-hatch").length).toBe(document.querySelectorAll(".training-sentence-band.model").length);
    expect(document.querySelectorAll(".training-sentence-band.model").length).toBe(12);          // its twelve words
    expect(screen.getByLabelText("steps 0–3: observed only, the model speaks from step 4")).toBeTruthy();
    // a tick under a row wherever the truth says a word of that column after step 0
    const truth = (appState.trainingSelection as { flight: { words: { events: Array<{ row: number }> } } }).flight.words.events;
    expect(document.querySelectorAll(".training-sentence-truth-tick").length).toBe(truth.filter((event) => event.row > 0).length);
    // its own moments: its clearance, its end, and where the observed flight's sentence ends
    expect(screen.getByLabelText("base model cleared the flight to join the final at 48 s")).toBeTruthy();
    expect(screen.getByLabelText("base model's flight landed at 110 s")).toBeTruthy();
    expect(screen.getByLabelText("the observed flight's sentence ends at 120 s")).toBeTruthy();
    // how the sample ended, and no truth-only chip or button
    expect(screen.getByText("#1: landed on 09 at 110 s (observed 120 s)")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "▶ Fly" })).toBeNull();
    expect(screen.queryByText(/^heading \d+\/\d+ · capture/)).toBeNull();
  });

  it("numbers a model's samples, marking the ones whose flight did not land, and reads the one chosen", () => {
    select();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 1 };
    render(<TrainingSentenceBar />);
    const samples = screen.getByRole("group", { name: "base model's samples" });
    expect([...samples.querySelectorAll("button")].map((button) => [button.textContent, button.getAttribute("aria-pressed")]))
      .toEqual([["1", "false"], ["2 ✗", "true"]]);
    expect(screen.getByText("#2: timed out, pointing at 27 at 150 s (observed 120 s)")).toBeTruthy();
    // its rows run past the observed flight's: the axis runs on to its end
    expect(screen.getByText("152 s")).toBeTruthy();
    fireEvent.click(samples.querySelector("button")!);
    expect(setTrainingSource).toHaveBeenCalledWith({ overlayId: BASE_MODEL_ID, sample: 0 });
  });

  it("keeps the truth's own axis under its tab, never stretched by a model's sample it does not show", () => {
    select();
    generations();
    render(<TrainingSentenceBar />);
    expect(screen.getByText("120 s")).toBeTruthy();
    expect(screen.queryByText("152 s")).toBeNull();
  });

  it("says so, and reads the truth, for a flight the chosen model's sentences do not fly", () => {
    select(1);
    generations(1);
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    expect(screen.getByText("base model: not flown (stand-in dynamics) — the truth is shown")).toBeTruthy();
    // the truth is drawn with its whole header: the labeller's verdicts, and the Fly button for its words
    expect(screen.getByText(/^heading \d+\/\d+ · capture/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "▶ Fly" })).toBeTruthy();
    expect((document.querySelector(".training-sentence-bar") as HTMLElement).style.borderColor).toBe("");
    // the straight-in flight's opening words: cleared at entry, the other five in force
    expect(screen.getAllByLabelText(/— in force at entry \(step 0\), issued at step 0/).length).toBe(5);
    expect(screen.getByLabelText(/^approach cleared — cleared to join the final, issued at step 0/)).toBeTruthy();
    expect(document.querySelectorAll(".training-sentence-band.model")).toHaveLength(0);
  });

  it("picks a band's word for the live executor when the band is clicked, and clears it on a second click", () => {
    select();
    render(<TrainingSentenceBar />);
    const band = screen.getByLabelText(/^heading 225° .* issued at step 8 /);
    fireEvent.click(band);
    expect(setTrainingPick).toHaveBeenLastCalledWith({ column: "heading", row: 8, attempt: 0 });
    fireEvent.click(band);
    expect(setTrainingPick).toHaveBeenLastCalledWith(null);
  });

  it("flies the selected word from its Fly button — the only way with the panel's switch off", () => {
    select();
    appState.trainingAutopilotAuto = false;
    render(<TrainingSentenceBar />);
    const fly = screen.getByRole("button", { name: "▶ Fly" }) as HTMLButtonElement;
    expect(fly.disabled).toBe(true);
    expect(fly.title).toMatch(/^Select a word/);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    expect(setTrainingPick).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "▶ Fly" }));
    expect(setTrainingPick).toHaveBeenLastCalledWith({ column: "heading", row: 8, attempt: 0 });
  });

  it("offers to fly the same segment again once it has flown, and waits while it flies", () => {
    select();
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const request = mockAutopilotRequest(parsed.value, VECTORED_KEY, "heading", 8);
    appState.trainingPick = { column: "heading", row: 8, attempt: 2 };
    appState.trainingAutopilot = { status: "flying", request };
    const { unmount } = render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    expect((screen.getByRole("button", { name: "Flying …" }) as HTMLButtonElement).disabled).toBe(true);
    unmount();
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(parsed.value, request), request, mockSelection(parsed.value, request));
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: answer.value, playedAt: 0, roundTripS: 0.5 };
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    setTrainingPick.mockClear();
    fireEvent.click(screen.getByRole("button", { name: "↻ Fly again" }));
    expect(setTrainingPick).toHaveBeenLastCalledWith({ column: "heading", row: 8, attempt: 3 });
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
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(parsed.value, request), request, mockSelection(parsed.value, request));
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
    const flown = parseTrainingAutopilot(outside, request, mockSelection(parsed.value, request));
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
      /the executor: outside, told at its step 20 — ✓ capture turn monotone, ✗ corridor held to the landing \(30\/35 rows\)/);
    // a heading word's verdict: its band's rows on the flown track, from where it was told plus the lead
    expect(screen.getByLabelText(/^heading 180° .* issued at step 10 /).textContent).toMatch(
      /the executor: outside, told at its step 10 — ✗ track within ±4\.5° of the word, 4 s after it was told, to the next word's \(7\/8 rows\)/);
    // the replay in one chip, its full reading in the tooltip
    const chip = screen.getByText("replay: landed · 5/7");
    expect(chip.getAttribute("title")).toMatch(/The executor \(own dynamics\) flew this sentence from row 0.*: landed, 1\.5 m right of the centreline, 20\.8 m above the threshold/);
    expect(chip.getAttribute("title")).toMatch(/5\/7 words inside their envelopes.* · evaluation pass \(observed pass\)/);
  });

  it("says why a flight the replay does not fly has no verdicts", () => {
    select(1);
    overlays(1);
    render(<TrainingSentenceBar />);
    expect(document.querySelectorAll(".training-sentence-verdict")).toHaveLength(0);
    expect(screen.getByText("replay: not flown").getAttribute("title")).toBe("the executor's replay does not fly it: no identified type");
  });

  it("opens the prior's predictions from its own button", () => {
    select();
    overlays();
    render(<TrainingSentenceBar />);
    const button = screen.getByRole("button", { name: "Prior" });
    expect(button.getAttribute("title")).toMatch(/this flight 0\.250 nats per step, val 0\.1778/);
    fireEvent.click(button);
    expect(screen.getByRole("dialog", { name: "Prior predictions" })).toBeTruthy();
    // one window at a time: the read-back check replaces it
    fireEvent.click(screen.getByRole("button", { name: "Read-back" }));
    expect(screen.queryByRole("dialog", { name: "Prior predictions" })).toBeNull();
    expect(screen.getByRole("dialog", { name: "Read-back check" })).toBeTruthy();
  });

  it("ignores an overlay published for another flight", () => {
    select(0);
    overlays(1);
    render(<TrainingSentenceBar />);
    expect(screen.queryByText(/replay:/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Prior" })).toBeNull();
  });

  it("draws nothing until a flight is selected", () => {
    const { container } = render(<TrainingSentenceBar />);
    expect(container.innerHTML).toBe("");
  });

  it("draws nothing outside Training, though the flight stays selected (the session outlives a task switch)", () => {
    select();
    appState.mode = "observe";
    const { container, rerender } = render(<TrainingSentenceBar />);
    expect(container.innerHTML).toBe("");
    appState.mode = "training";
    rerender(<TrainingSentenceBar />);
    expect(screen.getByText("Heading")).toBeTruthy();
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

  it("counts the words after step 0 and the silent steps, in the callsign's tooltip", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByText("TST1").getAttribute("title"))
      .toBe(`A320 · vectored · ${MOCK_ROWS} steps · 6 words after step 0 · ${MOCK_ROWS - 5} of ${MOCK_ROWS - 1} later steps silent`);
  });

  it("reads out the labeller's verdicts in one chip, what each leaves out in its tooltip", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(screen.getByText("heading 2/3 · capture ✓ · altitude 1/2 · speed 1/1")).toBeTruthy();
    select(1);
    render(<TrainingSentenceBar />);
    const chip = screen.getByText("heading 0/0 · capture — · altitude 2/2 · speed 1/1");
    expect(chip.getAttribute("title")).toMatch(/\(1 more with no row of their own: the lead reaches the clearance\).*none, on the final at entry/);
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

  it("carries a legend of what is drawn, folded until opened: only the switches that are on, the reading in each tooltip", () => {
    select();
    render(<TrainingSentenceBar />);
    const legend = screen.getByLabelText("What the 3D scene shows");
    expect(legend.textContent).toBe("Legend ▸");
    fireEvent.click(screen.getByText("Legend ▸"));
    expect(legend.textContent).toMatch(/heading word: judged rows/);
    expect(screen.getByText("heading word: judged rows").closest("li")!.getAttribute("title"))
      .toMatch(/from 4 s after it is said to the next word's, where the track must stay within ±4\.5° of it/);
    expect(legend.textContent).toMatch(/capture turn/);
    expect(legend.textContent).not.toMatch(/funnel|turn region|autopilot/);
    // each row's full reading, as text behind ⓘ
    fireEvent.click(screen.getByRole("button", { name: "What each line in 3D is" }));
    expect(screen.getByRole("note", { name: "What each line in 3D is" }).textContent)
      .toMatch(/heading word: judged rowsa heading word's judged rows, on the ground: from 4 s after it is said/);
    appState.trainingLayers = { ...appState.trainingLayers, headingBands: false };
    render(<TrainingSentenceBar />);
    const other = screen.getAllByLabelText("What the 3D scene shows")[1];
    fireEvent.click(other.querySelector("button")!);
    expect(other.textContent).not.toMatch(/judged rows/);
  });

  it("opens the read-back check, and puts the notes behind ⓘ", () => {
    select();
    render(<TrainingSentenceBar />);
    expect(document.querySelector(".training-sentence-legend")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "How to read the bar" }));
    expect(document.querySelector(".training-sentence-legend")!.textContent).toMatch(/The sentence ends [\d.]+ km before the threshold/);
    fireEvent.click(screen.getByRole("button", { name: "Read-back" }));
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
