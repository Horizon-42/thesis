/**
 * The Training panel over a prior set of stage B: the switch to the prior's sets appears only where the airport has
 * `index_prior_v3.json`; a set is read, its flights and — side by side — the closed-loop sentence and each sentence the prior
 * said are listed with their outcomes; a click publishes that sentence as the flight on screen; the row inspector shows the
 * probability of go-around and the words the procedure blocked at the cursor's row; a set of another schema is refused by name.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

const { appState, setTrainingSelection, setTrainingIntervalS, setTrainingAutopilot, setTrainingCursorS, fetchMock } = vi.hoisted(() => ({
  appState: {
    activeAirportCode: "KXXX" as string,
    trainingLayers: { headingBands: true, vertical: true, candidates: true },
    trainingIntervalS: 4 as number | null,
    trainingSelection: null as unknown, trainingPick: null,
    cursorS: 0,
  },
  setTrainingSelection: vi.fn(),
  setTrainingIntervalS: vi.fn(),
  setTrainingAutopilot: vi.fn(),
  setTrainingCursorS: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingSelection, setTrainingIntervalS, setTrainingAutopilot, setTrainingLayer: vi.fn() }),
  useTrainingCursor: () => ({ trainingCursorS: appState.cursorS, setTrainingCursorS }),
}));

import TrainingPanel from "../TrainingPanel";
import { TRAINING_PRIOR_SAMPLE_SCHEMA } from "../../data/trainingPriorSample";
import { stageBIndex, stageBSample, stageBSampleFile } from "../../data/__tests__/stageB";
import { chooseTrainingTab, useTrainingTabs, type TrainingTabs } from "../../data/trainingTabs";

/** The tabs the session gives the bar (the bar is not rendered here: a spy reads them). */
let shownTabs: TrainingTabs | null = null;
function TabsSpy() {
  shownTabs = useTrainingTabs();
  return null;
}

const INDEX_PATH = "data/airports/KXXX/training/index_prior_v3.json";
const SAMPLE_PATH = "data/airports/KXXX/training/fixture_set/sample.json";

function jsonResponse(body: unknown) {
  return { ok: true, headers: { get: () => "application/json" }, text: async () => JSON.stringify(body) };
}
const notFound = () => ({ ok: false, status: 404, headers: { get: () => "text/html" }, text: async () => "<!doctype html>" });
const serve = (files: Record<string, unknown>) =>
  fetchMock.mockImplementation(async (url: string) => (url in files ? jsonResponse(files[url]) : notFound()));
const lastPublished = (): any => {
  const calls = setTrainingSelection.mock.calls;
  return calls.length ? calls[calls.length - 1][0] : null;
};

async function openPriorSets() {
  render(<><TrainingPanel hidden={false} /><TabsSpy /></>);
  fireEvent.change(await screen.findByLabelText("Sets of"), { target: { value: "prior" } });
}

describe("the prior's sets in the Training panel", () => {
  beforeEach(() => {
    appState.cursorS = 0;
    for (const mock of [setTrainingSelection, setTrainingIntervalS, setTrainingCursorS]) mock.mockClear();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("offers nothing where the airport has no prior index", async () => {
    serve({});
    render(<TrainingPanel hidden={false} />);
    expect(await screen.findByText(/No Training export for KXXX yet/)).toBeTruthy();
    expect(screen.queryByLabelText("Sets of")).toBeNull();
  });

  it("gives the bar the tabs Labelled, Closed loop and each sample, and publishes the sentence the bar chooses", async () => {
    serve({ [INDEX_PATH]: stageBIndex(), [SAMPLE_PATH]: stageBSampleFile() });
    await openPriorSets();
    const sample = stageBSample();
    const { head, sentences } = sample.flights[0];
    expect(await screen.findByText(head.callsign)).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();                                     // no table in the left panel
    // the set opens on the first flight's first prior sentence, read at the prior's Δ
    await waitFor(() => expect(lastPublished()?.flight.flightKey).toBe(`${head.flightKey}~prior-0`));
    expect(shownTabs!.tabs.map((tab) => tab.label)).toEqual(["Labelled", "Closed loop", ...sentences.map((s) => `${s.sample}`)]);
    expect(shownTabs!.tabs[2].title).toContain("A sentence the prior said (sample 0");
    expect(shownTabs!.chosen).toBe("sample-0");
    expect(shownTabs!.tabs.map((tab) => tab.outcome)).toEqual([null, head.closedLoop["4"].replay.outcome, ...sentences.map((s) => s.outcome)]);
    expect(setTrainingIntervalS).toHaveBeenCalledWith(4);
    expect(lastPublished().vocabulary.rowIntervalsS).toEqual([4]);
    act(() => chooseTrainingTab("sample-1"));
    await waitFor(() => expect(lastPublished().flight.flightKey).toBe(`${head.flightKey}~prior-1`));
    act(() => chooseTrainingTab("closed-loop"));
    await waitFor(() => expect(lastPublished().flight.flightKey).toBe(`${head.flightKey}~closed-loop`));
    expect(screen.getByTitle(/^At the cursor/).textContent).toContain("choose a sample's tab (0, 1 …)");
    act(() => chooseTrainingTab("labelled"));
    await waitFor(() => expect(setTrainingIntervalS).toHaveBeenLastCalledWith(null));
    expect(screen.getByRole("button", { name: /^This sentence/ }).textContent).toContain("not flown");
  });

  it("shows, at the cursor's row, the go-around probability and the words the procedure blocked", async () => {
    serve({ [INDEX_PATH]: stageBIndex(), [SAMPLE_PATH]: stageBSampleFile() });
    const sample = stageBSample();
    const sentence = sample.flights[0].sentences[0];
    appState.cursorS = sample.flights[0].head.closedLoop["4"].startS + 4 * 5;        // Δ row 5
    await openPriorSets();
    // the line under the list and its probability strip stay in the panel; the line opens no page (D134: no row inspector)
    const line = await screen.findByTitle(/^At the cursor/);
    expect(line.textContent).toContain(`row 5: go-around ${(sentence.goAroundProbability[5] * 100).toFixed(1)} %`);
    expect(line.textContent).toContain(`altitude ${sentence.blocked.altitude[5].length}`);
    expect(screen.queryByRole("button", { name: /^At the cursor/ })).toBeNull();
    expect(screen.getByRole("img", { name: /Probability of go-around/ })).toBeTruthy();
  });

  it("refuses a set of another schema by name and keeps the others", async () => {
    serve({ [INDEX_PATH]: stageBIndex(), [SAMPLE_PATH]: { ...stageBSampleFile(), schema: "aeroviz-training-sample-v9" } });
    await openPriorSets();
    expect(await screen.findByText(/Set fixture_set cannot be read/)).toBeTruthy();
    expect(screen.getByText(new RegExp(`not one of ${TRAINING_PRIOR_SAMPLE_SCHEMA}`))).toBeTruthy();
  });
});

describe("choices that outlive a set or an airport", () => {
  beforeEach(() => {
    appState.cursorS = 0;
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    appState.activeAirportCode = "KXXX";
  });

  it("switches to a set with fewer sentences of a shared flight without a render that has the old choice", async () => {
    const index = stageBIndex();
    index.sets.push({ ...index.sets[0], id: "set_b", file: "set_b/sample.json" });
    const other = stageBSampleFile();
    other.setId = "set_b";
    other.flights[0].prior = other.flights[0].prior.slice(0, 1);          // the same flight, one sentence
    serve({ [INDEX_PATH]: index, [SAMPLE_PATH]: stageBSampleFile(), "data/airports/KXXX/training/set_b/sample.json": other });
    await openPriorSets();
    await waitFor(() => expect(shownTabs?.tabs.some((tab) => tab.id === "sample-1")).toBe(true));
    act(() => chooseTrainingTab("sample-1"));
    await waitFor(() => expect(lastPublished().flight.flightKey).toContain("~prior-1"));
    fireEvent.change(screen.getByLabelText("Set"), { target: { value: "set_b" } });
    await waitFor(() => expect(lastPublished()?.setId).toBe("set_b"));
    expect(lastPublished().flight.flightKey).toContain("~prior-0");
    expect(shownTabs!.tabs.some((tab) => tab.id === "sample-1")).toBe(false);
  });

  it("falls back to stage A at an airport that has no prior sets", async () => {
    serve({ [INDEX_PATH]: stageBIndex(), [SAMPLE_PATH]: stageBSampleFile() });
    const view = render(<TrainingPanel hidden={false} />);
    fireEvent.change(await screen.findByLabelText("Sets of"), { target: { value: "prior" } });
    await waitFor(() => expect(lastPublished()?.flight.flightKey).toContain("~prior-0"));
    appState.activeAirportCode = "KYYY";
    view.rerender(<TrainingPanel hidden={false} key="other" />);
    expect(await screen.findByText(/No Training export for KYYY yet/)).toBeTruthy();
    expect(screen.queryByLabelText("Sets of")).toBeNull();
  });
});
