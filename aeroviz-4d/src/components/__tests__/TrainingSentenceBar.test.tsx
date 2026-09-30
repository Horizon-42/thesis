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
      trainingPick: null as unknown,
      trainingGenerations: [] as unknown[], trainingSource: null as unknown, trainingWindow: null as unknown,
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
import { parseTrainingExecutorOverlay, parseTrainingPriorOverlay, type TrainingGenerationView } from "../../data/trainingOverlays";
import {
  AUGMENTED_R1_ID, AUGMENTED_R2_ID, AUGSTART_BASE_ID, AUGSTART_POST_ID, BASE_MODEL_ID, POST_TRAINED_ID, mockGenerationViews,
} from "../../data/__tests__/trainingOverlays.fixture";
import {
  AUTOPILOT_PLAYBACK_MIN_SPEEDUP, AUTOPILOT_TAIL_OPACITY, parseTrainingAutopilot,
} from "../../data/trainingAutopilot";
import { VECTORED_KEY, WORD } from "../../data/__tests__/trainingSample.fixture";
import { TRAINING_MODEL_COLOR, TRAINING_OUTSIDE_COLOR, TRAINING_REPLAY_COLOR } from "../../utils/trainingWordColors";
import {
  failedAnswer, mockAutopilotAnswer, mockAutopilotRequest, mockSelection,
} from "../../data/__tests__/trainingAutopilot.fixture";

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

/** Publish models' own sentences over the selected flight, as the panel does (``change``: of each raw payload; ``ids``:
 *  base and landing r1 by default). */
function generations(position = 0, change: (raw: any) => void = () => undefined, ids?: string[]) {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  appState.trainingGenerations = mockGenerationViews(parsed.value, position, change, ids);
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
    expect(names).toEqual(["Truth", "base", "landing r1"]);
    expect(tabs.querySelector("button")!.getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByRole("group", { name: /rounds$/ })).toBeNull();                  // one round each: no round chips
    const post = screen.getAllByRole("button", { name: "landing r1" });
    fireEvent.click(post[post.length - 1]);
    expect(setTrainingSource).toHaveBeenCalledWith({ overlayId: POST_TRAINED_ID, sample: 0 });
  });

  it("starts another model at its first sample, whichever sample the last one was read at", () => {
    select();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 1 };
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: "landing r1" }));
    expect(setTrainingSource).toHaveBeenLastCalledWith({ overlayId: POST_TRAINED_ID, sample: 0 });
  });

  it("gives a model published at several rounds one tab and its rounds beside it, a round keeping the sample read", () => {
    select();
    generations(0, undefined, [AUGMENTED_R2_ID, BASE_MODEL_ID, AUGMENTED_R1_ID, POST_TRAINED_ID]);
    const { rerender } = render(<TrainingSentenceBar />);
    const tabs = screen.getByRole("group", { name: "Which sentence is read" });
    expect([...tabs.querySelectorAll("button")].map((button) => button.textContent)).toEqual(["Truth", "base", "landing r1", "augmented"]);
    expect(screen.queryByRole("group", { name: "augmented's rounds" })).toBeNull();        // only the model read shows its rounds
    // the tab opens its first round at the first sample
    fireEvent.click(screen.getByRole("button", { name: "augmented" }));
    expect(setTrainingSource).toHaveBeenLastCalledWith({ overlayId: AUGMENTED_R1_ID, sample: 0 });
    appState.trainingSource = { overlayId: AUGMENTED_R1_ID, sample: 1 };
    rerender(<TrainingSentenceBar />);
    const rounds = screen.getByRole("group", { name: "augmented's rounds" });
    expect([...rounds.querySelectorAll("button")].map((button) => [button.textContent, button.getAttribute("aria-pressed")])).toEqual([
      ["r1", "true"], ["r2", "false"]]);
    expect(rounds.querySelector("button")!.getAttribute("title")).toBe(
      "augmented r1 (v3_stage2/aug_s1, post-trained from landing r1 by ts-prior-augmented-reward-v5): 1 of 2 of its sentences " +
      "for this flight landed");
    expect(screen.getByRole("button", { name: "augmented" }).getAttribute("aria-pressed")).toBe("true");
    // another round: the same sample number, so one flight is compared round by round
    fireEvent.click(screen.getByRole("button", { name: "augmented r2" }));
    expect(setTrainingSource).toHaveBeenLastCalledWith({ overlayId: AUGMENTED_R2_ID, sample: 1 });
    // away and back: the tab returns to the round last read in it
    appState.trainingSource = { overlayId: AUGMENTED_R2_ID, sample: 1 };
    rerender(<TrainingSentenceBar />);
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    rerender(<TrainingSentenceBar />);
    expect(screen.queryByRole("group", { name: "augmented's rounds" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "augmented" }));
    expect(setTrainingSource).toHaveBeenLastCalledWith({ overlayId: AUGMENTED_R2_ID, sample: 0 });
  });

  it("names two runs of one stage by their runs, each a tab of its own even with one round", () => {
    select();
    generations(0, undefined, [AUGMENTED_R1_ID]);
    const [clip] = appState.trainingGenerations as TrainingGenerationView[];
    const restart = { ...clip, overlay: { ...clip.overlay, overlayId: "restart_r1",
      model: { ...clip.overlay.model, run: "4dTrajectory/outputs/POOLED/prior/v3_restart/aug_s1" } } };
    appState.trainingGenerations = [clip, restart];
    render(<TrainingSentenceBar />);
    const tabs = screen.getByRole("group", { name: "Which sentence is read" });
    expect([...tabs.querySelectorAll("button")].map((button) => button.textContent)).toEqual([
      "Truth", "augmented r1 · v3_restart", "augmented r1 · v3_stage2"]);
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
      Object.assign(raw.flights[0].samples[1], { outcome: "crossed_without_capture", crossing: { crossM: 80, heightM: 30, atS: 149.6, runway: 0 },
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

  it("draws a model's sample in the truth's flat bands, framed in its colour, over the span it only observed, with the truth's words ticked under each row", () => {
    select();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    // the model's words, said by it — never the labeller's kinds
    expect(screen.getByLabelText(/^heading 225° — said by base at step 12 \(24 s\), in force to 32 s$/)).toBeTruthy();
    expect(screen.queryByLabelText(/— the heading the track reaches a lead later/)).toBeNull();
    expect(document.querySelectorAll(".training-sentence-band.model").length).toBe(12);          // its twelve words
    // flat: no hatching, no dashed edge — what says it is a model's is the frame, a strip down its rows in its colour
    expect(document.querySelectorAll("pattern")).toHaveLength(0);
    expect([...document.querySelectorAll(".training-sentence-band-fill")].every((band) => !band.hasAttribute("stroke-dasharray"))).toBe(true);
    expect(document.querySelector(".training-sentence-model-strip")!.getAttribute("fill")).toBe(TRAINING_MODEL_COLOR.base);
    expect(screen.getByLabelText("steps 0–3: observed only, the model speaks from step 4")).toBeTruthy();
    // a tick under a row wherever the truth says a word of that column after step 0
    const truth = (appState.trainingSelection as { flight: { words: { events: Array<{ row: number }> } } }).flight.words.events;
    expect(document.querySelectorAll(".training-sentence-truth-tick").length).toBe(truth.filter((event) => event.row > 0).length);
    // its own moments: its clearance, its end, and where the observed flight's sentence ends
    expect(screen.getByLabelText("base cleared the flight to join the final at 48 s")).toBeTruthy();
    expect(screen.getByLabelText("base's flight landed at 110 s")).toBeTruthy();
    expect(screen.getByLabelText("the observed flight's sentence ends at 120 s")).toBeTruthy();
    // where its flight ended, its time written on the axis under it, in its colour
    const end = screen.getByLabelText("base's flight ended at 110 s");
    expect([end.textContent, end.getAttribute("fill")]).toEqual(["110 s", TRAINING_MODEL_COLOR.base]);
    // how the sample ended, and no truth-only chip; its words fly like the truth's
    expect(screen.getByText("#1: landed on 09 at 110 s (observed 120 s)")).toBeTruthy();
    expect(screen.queryByText(/^heading \d+\/\d+ · capture/)).toBeNull();
    expect(screen.getByRole("button", { name: "▶ Fly" })).toBeTruthy();
  });

  it("writes a model's end time even where its flight ran to the axis's end — in its colour, the axis's tick giving way — and none under the truth", () => {
    select();
    generations();
    const { unmount } = render(<TrainingSentenceBar />);
    expect(document.querySelector(".training-sentence-model-end")).toBeNull();
    expect(document.querySelector(".training-sentence-model-strip")).toBeNull();
    unmount();
    // the second sample timed out at 150 s on its last step: the axis ends there
    generations(0, (raw) => { Object.assign(raw.flights[0].samples[1], { rows: 75 }); });
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 1 };
    render(<TrainingSentenceBar />);
    const end = screen.getByLabelText("base's flight ended at 150 s");
    expect([end.textContent, end.getAttribute("text-anchor")]).toEqual(["150 s", "end"]);
    expect(screen.getAllByText("150 s")).toHaveLength(1);
  });

  it("picks a model's word for the live executor — its sample named — from its band and its Fly button", () => {
    select();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    const source = { overlayId: BASE_MODEL_ID, sample: 0 };
    fireEvent.click(screen.getByLabelText(/^heading 225° — said by base at step 12/));
    expect(setTrainingPick).toHaveBeenLastCalledWith({ source, column: "heading", row: 12, attempt: 0 });
    const fly = screen.getByRole("button", { name: "▶ Fly" }) as HTMLButtonElement;
    expect(fly.title).toMatch(/^The executor flies the model's sentence again from its first step \(4\)/);
    fireEvent.click(fly);
    expect(setTrainingPick).toHaveBeenLastCalledWith({ source, column: "heading", row: 12, attempt: 0 });
  });

  it("offers nothing to fly for a flight the backend does not fly live — a window's aircraft: no Fly button, a band only selects", () => {
    select();
    appState.trainingSelection = { ...(appState.trainingSelection as object), liveExecutor: false };
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    expect(screen.queryByRole("button", { name: "▶ Fly" })).toBeNull();
    fireEvent.click(screen.getByLabelText(/^heading 225° — said by base at step 12/));
    expect(setTrainingPick).not.toHaveBeenCalled();
  });

  it("offers nothing to fly for a word a model said as or after its flight ended — the backend refuses the same words", () => {
    select();
    // the first sample speaks on past its landing at 110 s: level at step 55 (110 s, the end itself), a speed word at
    // step 58 (116 s)
    generations(0, (raw) => {
      const landed = raw.flights[0].samples[0];
      Object.assign(landed, { rows: 60, events: [...landed.events, { row: 55, column: 4, value: WORD.level },
        { row: 58, column: 5, value: WORD.speed110 }] });
    });
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    for (const label of [/^angle level — said by base at step 55/, /^speed 110 m\/s — said by base at step 58/]) {
      fireEvent.click(screen.getByLabelText(label));
      expect(setTrainingPick).not.toHaveBeenCalled();
      const fly = screen.getByRole("button", { name: "▶ Fly" }) as HTMLButtonElement;
      expect(fly.disabled).toBe(true);
      expect(fly.title).toBe("The model said this word after its flight had ended (110 s): there is no flight to fly.");
    }
    // a step before the end is flown
    fireEvent.click(screen.getByLabelText(/^speed unspecified — said by base at step 40/));
    expect(setTrainingPick).toHaveBeenLastCalledWith({ source: { overlayId: BASE_MODEL_ID, sample: 0 }, column: "speed", row: 40,
      attempt: 0 });
  });

  it("numbers a model's samples, marking the ones whose flight did not land, and reads the one chosen", () => {
    select();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 1 };
    render(<TrainingSentenceBar />);
    const samples = screen.getByRole("group", { name: "base's samples" });
    expect([...samples.querySelectorAll("button")].map((button) => [button.textContent, button.getAttribute("aria-pressed")]))
      .toEqual([["1", "false"], ["2 ✗", "true"]]);
    expect(screen.getByText("#2: timed out, pointing at 27 at 150 s (observed 120 s)")).toBeTruthy();
    // its rows run past the observed flight's: the axis runs on to its end (152 s), whose tick gives way to the flight's end
    expect(screen.getByLabelText("base's flight ended at 150 s").textContent).toBe("150 s");
    expect(screen.queryByText("152 s")).toBeNull();
    fireEvent.click(samples.querySelector("button")!);
    expect(setTrainingSource).toHaveBeenCalledWith({ overlayId: BASE_MODEL_ID, sample: 0 });
  });

  it("names the runway a sample crossed — for another runway's threshold, that runway, not the one it pointed at", () => {
    select();
    // the second sample pointed at 27 to its end but crossed 09's threshold lined up with it
    generations(0, (raw) => {
      Object.assign(raw.flights[0].samples[1], { outcome: "crossed_other_runway",
        crossing: { crossM: 12, heightM: 140, atS: 149.6, runway: 0 } });
    });
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 1 };
    render(<TrainingSentenceBar />);
    expect(screen.getByText("#2: other runway on 09 at 150 s (observed 120 s)")).toBeTruthy();
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
    expect(screen.getByText("base: not flown (stand-in dynamics) — the truth is shown")).toBeTruthy();
    // the truth is drawn whole: the labeller's tallies at its rows' ends, and the Fly button for its words
    expect(document.querySelectorAll(".training-sentence-tally").length).toBeGreaterThan(0);
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
    expect(setTrainingPick).toHaveBeenLastCalledWith({ source: null, column: "heading", row: 8, attempt: 0 });
    fireEvent.click(band);
    expect(setTrainingPick).toHaveBeenLastCalledWith(null);
  });

  it("flies the selected word from its Fly button too, which waits for a word to be selected", () => {
    select();
    render(<TrainingSentenceBar />);
    const fly = screen.getByRole("button", { name: "▶ Fly" }) as HTMLButtonElement;
    expect(fly.disabled).toBe(true);
    expect(fly.title).toMatch(/^Select a word/);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    setTrainingPick.mockClear();
    fireEvent.click(screen.getByRole("button", { name: "▶ Fly" }));
    expect(setTrainingPick).toHaveBeenLastCalledWith({ source: null, column: "heading", row: 8, attempt: 0 });
  });

  it("offers to fly the same segment again once it has flown, and waits while it flies", () => {
    select();
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const request = mockAutopilotRequest(parsed.value, VECTORED_KEY, "heading", 8);
    appState.trainingPick = { source: null, column: "heading", row: 8, attempt: 2 };
    appState.trainingAutopilot = { status: "flying", request };
    const { unmount } = render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    expect((screen.getByRole("button", { name: "Flying …" }) as HTMLButtonElement).disabled).toBe(true);
    unmount();
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(parsed.value, request), request, mockSelection(parsed.value, request));
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: answer.value, playedAt: 0 };
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    setTrainingPick.mockClear();
    fireEvent.click(screen.getByRole("button", { name: "↻ Fly again" }));
    expect(setTrainingPick).toHaveBeenLastCalledWith({ source: null, column: "heading", row: 8, attempt: 3 });
  });

  it("reads out the live executor's segment in one short line: its verdict and the two times, the word only off the selection", () => {
    select();
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const request = mockAutopilotRequest(parsed.value, VECTORED_KEY, "heading", 8);
    appState.trainingAutopilot = { status: "flying", request };
    const { unmount } = render(<TrainingSentenceBar />);
    // no band selected: the line names the word it flies
    expect(screen.getByText("Autopilot · heading 225° · flying …").title).toBe("flying heading 225°");
    unmount();
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(parsed.value, request), request, mockSelection(parsed.value, request));
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: answer.value, playedAt: 0 };
    const { unmount: done } = render(<TrainingSentenceBar />);
    // its band selected: the band names it, the line does not
    fireEvent.click(screen.getByLabelText(/^heading 225° .* issued at step 8 /));
    const line = document.querySelector(".training-sentence-autopilot")!;
    expect(line.textContent).toBe("Autopilot · in envelope · 8.00 s flown · computed 1.24 s");
    expect((line as HTMLElement).title).toMatch(/^heading 225°: in envelope; the flight reached .*; 8\.00 s flown, computed 1\.24 s$/);
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
    appState.trainingAutopilot = { status: "ready", request, segment: flown.value, playedAt: 0 };
    render(<TrainingSentenceBar />);
    const red = document.querySelector(".training-sentence-autopilot strong") as HTMLElement;
    expect(red.textContent).toBe("out of envelope");
    expect(red.style.color).toBe("rgb(255, 45, 45)");
  });

  it("puts a cursor on the flown word's row: at its step while it flies on the backend, then with the 3D aircraft, faded in the tail, left at the end", () => {
    select();
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    // heading 225° said at step 8 (16 s), the next heading word at step 10 (20 s): flown 16–24 s, the tail from 20 s — 8 s,
    // flown out at the least speed-up
    const request = mockAutopilotRequest(parsed.value, VECTORED_KEY, "heading", 8);
    const realMs = (flownS: number) => (flownS / AUTOPILOT_PLAYBACK_MIN_SPEEDUP) * 1000;
    const frames: FrameRequestCallback[] = [];
    const raf = vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => frames.push(callback));
    const now = vi.spyOn(Date, "now").mockReturnValue(1_000_000);
    const nextFrame = (ms: number) => {
      now.mockReturnValue(1_000_000 + ms);
      frames.shift()!(0);
    };
    const issue = (pattern: RegExp) => screen.getByLabelText(pattern).querySelector(".training-sentence-issue")!;
    const cursor = () => document.querySelector(".training-sentence-autopilot-cursor") as SVGRectElement | null;
    const at = () => Number(/^translate\(([-\d.]+) 0\)$/.exec(cursor()!.getAttribute("transform")!)![1]);
    appState.trainingAutopilot = { status: "flying", request };
    const { rerender, unmount } = render(<TrainingSentenceBar />);
    const said = issue(/^heading 225° .* issued at step 8 /);
    const x16 = Number(said.getAttribute("x"));
    const x20 = Number(issue(/^heading 180° .* issued at step 10 /).getAttribute("x"));
    // on the heading word's row, at its step, pulsing, no frame loop
    expect(Number(cursor()!.getAttribute("y"))).toBe(Number(said.getAttribute("y")) - 1);
    expect(cursor()!.classList.contains("waiting")).toBe(true);
    expect(at()).toBeCloseTo(x16, 6);
    expect(frames).toHaveLength(0);
    // the answer arrives: the same cursor flies it out from the word's step
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(parsed.value, request), request, mockSelection(parsed.value, request));
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: answer.value, playedAt: 1_000_000 };
    rerender(<TrainingSentenceBar />);
    expect(cursor()!.classList.contains("waiting")).toBe(false);
    expect(at()).toBeCloseTo(x16, 6);
    // 4 s along: where the next heading word is heard, not yet faded
    nextFrame(realMs(4));
    expect(at()).toBeCloseTo(x20, 6);
    expect(cursor()!.style.opacity).toBe("");
    // in the tail, faded as the 3D tail is
    nextFrame(realMs(6));
    expect(at()).toBeCloseTo(x16 + 1.5 * (x20 - x16), 6);
    expect(cursor()!.style.opacity).toBe(String(AUTOPILOT_TAIL_OPACITY));
    // flown out: left at the segment's end (24 s), and the loop stops
    nextFrame(realMs(60));
    expect(at()).toBeCloseTo(x16 + 2 * (x20 - x16), 6);
    expect(frames).toHaveLength(0);
    // flown again (↻ Fly again): back at the word's step, pulsing, while the backend flies it; its answer a new start,
    // from the word's step again, unfaded
    appState.trainingAutopilot = { status: "flying", request };
    rerender(<TrainingSentenceBar />);
    expect(cursor()!.classList.contains("waiting")).toBe(true);
    expect(at()).toBeCloseTo(x16, 6);
    expect(cursor()!.style.opacity).toBe("");
    appState.trainingAutopilot = { status: "ready", request, segment: answer.value, playedAt: 1_000_000 + realMs(60) };
    rerender(<TrainingSentenceBar />);
    expect(at()).toBeCloseTo(x16, 6);
    expect(cursor()!.style.opacity).toBe("");
    unmount();
    raf.mockRestore();
    now.mockRestore();
    // a refusal flew nothing; nor did a flight that failed in its first cycle (one state: 3D flies nothing either)
    appState.trainingAutopilot = { status: "failed", request, problem: "the backend did not answer" };
    const refused = render(<TrainingSentenceBar />);
    expect(cursor()).toBeNull();
    refused.unmount();
    const one = parseTrainingAutopilot(failedAnswer(mockAutopilotAnswer(parsed.value, request), 1), request,
      mockSelection(parsed.value, request));
    if (!one.ok) throw new Error(one.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: one.value, playedAt: 1 };
    render(<TrainingSentenceBar />);
    expect(cursor()).toBeNull();
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
    // the header says only what went wrong, in the fewest words, in the replay's verdict colour; which words, in its
    // tooltip and in the notes
    const chip = screen.getByText("Replay · 2 words out");
    const amber = document.createElement("span");
    amber.style.color = TRAINING_REPLAY_COLOR.flawed;
    expect(chip.style.color).toBe(amber.style.color);
    expect(chip.title).toBe("The executor's replay of this sentence: landed; outside their envelopes: heading 180° at step 10, " +
      "approach cleared at step 20 (a red dot at the band's left). The full reading is behind ⓘ.");
    fireEvent.click(screen.getByRole("button", { name: "How to read the bar" }));
    const notes = document.querySelector(".training-sentence-legend")!.textContent!;
    expect(notes).toMatch(/The executor \(own dynamics\) flew this sentence from row 0.*: landed, 1\.5 m right of the centreline, 20\.8 m above the threshold/);
    expect(notes).toMatch(/5\/7 words inside their envelopes.* · outside: heading 180° at step 10, approach cleared at step 20 · evaluation pass \(observed pass\)\./);
  });

  it("says nothing of the replay in the header when it landed with every word inside, nor over a model's sentence", () => {
    select();
    overlays();
    // every word inside, as its words and the gate's counts both say
    const flown = appState.trainingExecutor as { flight: { words: Array<{ status: string }>; counts: { wordsJudged: number } } };
    flown.flight = {
      ...flown.flight, words: flown.flight.words.map((word) => (word.status === "outside" ? { ...word, status: "inside" } : word)),
      counts: { ...flown.flight.counts, wordsInside: flown.flight.counts.wordsJudged },
    } as typeof flown.flight;
    const { unmount } = render(<TrainingSentenceBar />);
    expect(document.querySelector(".training-sentence-head")!.textContent).not.toMatch(/Replay/);
    unmount();
    overlays();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    expect(document.querySelector(".training-sentence-head")!.textContent).not.toMatch(/Replay/);
  });

  it("says why a flight the replay does not fly has no verdicts", () => {
    select(1);
    overlays(1);
    render(<TrainingSentenceBar />);
    expect(document.querySelectorAll(".training-sentence-verdict")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "How to read the bar" }));
    expect(document.querySelector(".training-sentence-legend")!.textContent).toMatch(/The executor's replay does not fly it: no identified type\./);
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

  it("writes the labeller's verdicts at the end of their rows, lined up on the slash, what each leaves out in its tooltip", () => {
    select();
    const { unmount } = render(<TrainingSentenceBar />);
    const tallies = () => [...document.querySelectorAll(".training-sentence-tally")] as SVGTextElement[];
    const written = () => tallies().map((tally) => [...tally.querySelectorAll("tspan")].map((span) => span.textContent).join(""));
    expect(tallies().map((tally) => tally.getAttribute("aria-label"))).toEqual([
      "approach: the capture turn: monotone ✓, rate and bank ✓",
      "heading: 2 of 3 heading words inside their bands",
      "altitude: 1 of 2 altitude tubes held",
      "speed: 1 of 1 speed words held",
    ]);
    expect(written()).toEqual(["✓", "2/3", "1/2", "1/1"]);
    // red where a word did not hold; every slash at one x
    expect(tallies().map((tally) => tally.style.fill)).toEqual(["", TRAINING_OUTSIDE_COLOR, TRAINING_OUTSIDE_COLOR, ""]);
    expect(new Set(tallies().flatMap((tally) => [...tally.querySelectorAll("tspan")].map((span) => span.getAttribute("x")))).size).toBe(1);
    unmount();
    // the straight-in flight: no capture turn (no mark), a heading word with no row of its own said in the tooltip
    select(1);
    render(<TrainingSentenceBar />);
    expect(written()).toEqual(["0/0", "2/2", "1/1"]);
    expect(tallies()[0].querySelector("title")!.textContent).toBe(
      "0 of 0 heading words inside their bands (1 more with no row of their own: the lead reaches the clearance)");
  });

  it("writes no tallies over a model's sentence: the labeller's checks are the truth's", () => {
    select();
    generations();
    appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    expect(document.querySelectorAll(".training-sentence-tally")).toHaveLength(0);
  });

  it("offers the augmented starts beside the real ones: a family of its own, read against nothing of the truth's start", () => {
    select();
    generations(0, () => undefined, [BASE_MODEL_ID, POST_TRAINED_ID, AUGSTART_BASE_ID, AUGSTART_POST_ID]);
    // reading the truth: the real start's tabs (Truth, base, landing r1), the switch on "Real start"
    const { unmount } = render(<TrainingSentenceBar />);
    expect(screen.getByRole("button", { name: "Real start" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "Truth" })).toBeTruthy();
    // the switch takes the model read to its augmented start (none read: the first model there), keeping the sample
    fireEvent.click(screen.getByRole("button", { name: "Augmented start" }));
    expect(setTrainingSource).toHaveBeenLastCalledWith({ overlayId: AUGSTART_BASE_ID, sample: 0 });
    unmount();
    appState.trainingSource = { overlayId: POST_TRAINED_ID, sample: 1 };
    const { unmount: fromLanding } = render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: "Augmented start" }));
    expect(setTrainingSource).toHaveBeenLastCalledWith({ overlayId: AUGSTART_POST_ID, sample: 1 });
    fromLanding();
    // reading landing r1 from its augmented start: no Truth tab, the moved observed steps, no truth ticks or observed end,
    // the move on the chip — and its words flown live from that start
    appState.trainingSource = { overlayId: AUGSTART_POST_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    expect(screen.getByRole("button", { name: "Augmented start" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByRole("button", { name: "Truth" })).toBeNull();
    expect(screen.getAllByRole("button", { pressed: true }).map((button) => button.textContent)).toContain("landing r1");
    expect(screen.getByText("observed, moved")).toBeTruthy();
    expect(document.querySelectorAll(".training-sentence-truth-tick")).toHaveLength(0);
    expect(screen.queryByLabelText(/the observed flight's sentence ends/)).toBeNull();
    expect(screen.getByText(/^#1: landed on 09 at 110 s · moved \+7\.2°, \+84 m, ×1\.030$/).getAttribute("title"))
      .toMatch(/^From an augmented start: .* drawn with seed 1337 in 2 draws; its time limit 2× .*the source flight landed on 09/);
    fireEvent.click(screen.getByLabelText(/^heading 225° .* said by landing r1 at step 12 /));
    const fly = screen.getByRole("button", { name: "▶ Fly" }) as HTMLButtonElement;
    expect(fly.disabled).toBe(false);
    expect(fly.title).toMatch(/from its first step \(4\), from its augmented start, to the end of this heading word's segment/);
    fireEvent.click(fly);
    expect(setTrainingPick).toHaveBeenLastCalledWith({ source: { overlayId: AUGSTART_POST_ID, sample: 0 }, column: "heading",
      row: 12, attempt: 0 });
    // back to the real start: the same model there
    fireEvent.click(screen.getByRole("button", { name: "Real start" }));
    expect(setTrainingSource).toHaveBeenLastCalledWith({ overlayId: POST_TRAINED_ID, sample: 0 });
  });

  it("goes back to the truth from an augmented start whose model has no real-start overlay", () => {
    select();
    generations(0, () => undefined, [BASE_MODEL_ID, AUGSTART_POST_ID]);
    appState.trainingSource = { overlayId: AUGSTART_POST_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    fireEvent.click(screen.getByRole("button", { name: "Real start" }));
    expect(setTrainingSource).toHaveBeenLastCalledWith(null);
  });

  it("says why an augmented start's model does not fly a flight, and reads the truth", () => {
    select(1);
    generations(1, () => undefined, [AUGSTART_BASE_ID]);
    appState.trainingSource = { overlayId: AUGSTART_BASE_ID, sample: 0 };
    render(<TrainingSentenceBar />);
    expect(screen.getByText("base: not flown (stand-in dynamics) — the truth is shown")).toBeTruthy();
    expect(screen.getByRole("button", { name: "▶ Fly" })).toBeTruthy();
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
