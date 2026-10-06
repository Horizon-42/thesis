/**
 * One layout for the Training views of the three stages (outline §6.2): the set chooser with the set's own intent line
 * (the backend's answer, or the problem by name) and a SMOKE tag; the item list with four columns; one line per readout;
 * the details page in every stage — its header ⓘ never disabled, on "The set and the experiment" — opened too from outside
 * the dock (the sentence bar's notes); the session's tabs (stage C: Labelled and its rounds).
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
import { AEROVIZ_BACKEND_URL } from "../../pilot/pilotClient";
import { stageBIndex, stageBSampleFile, stageBValSampleFile } from "../../data/__tests__/stageB";
import { stageCIndex, stageCSample, stageCSampleFile, stageCSelection, WINDOW_SET_ID } from "../../data/__tests__/stageC";
import { chooseTrainingTab, useTrainingTabs, type TrainingTabs } from "../../data/trainingTabs";
import { requestTrainingDetails } from "../training/PanelParts";
import { resultsAnswer, resultsUrl } from "../../data/__tests__/trainingResults";
import { roundTabLabel } from "../training/TrainingWindowSession";
import { otherOf } from "../../data/trainingWindowSample";

const intentUrl = (setId: string) => `${AEROVIZ_BACKEND_URL.replace(/\/+$/, "")}/experiments/intent?run=${encodeURIComponent(setId)}`;
const json = (body: unknown, ok = true) => ({ ok, status: ok ? 200 : 404, headers: { get: () => "application/json" }, text: async () => JSON.stringify(body) });
const notFound = () => ({ ok: false, status: 404, headers: { get: () => "text/html" }, text: async () => "<!doctype html>" });
function serve(files: Record<string, unknown>, intents: Record<string, unknown> = {}) {
  fetchMock.mockImplementation(async (url: string) => {
    if (url in files) return json(files[url]);
    if (url in intents) return json(intents[url], (intents[url] as { ok: boolean }).ok);
    return notFound();
  });
}
const lastPublished = (): any => {
  const calls = setTrainingSelection.mock.calls;
  return calls.length ? calls[calls.length - 1][0] : null;
};
let shownTabs: TrainingTabs | null = null;
function TabsSpy() {
  shownTabs = useTrainingTabs();
  return null;
}

const C_FILES = {
  "data/airports/KXXX/training/index_post_v3.json": stageCIndex(),
  [`data/airports/KXXX/training/${WINDOW_SET_ID}/sample.json`]: stageCSampleFile(),
};
const B_FILES = {
  "data/airports/KXXX/training/index_prior_v3.json": stageBIndex(),
  "data/airports/KXXX/training/fixture_set/sample.json": stageBSampleFile(),
  "data/airports/KXXX/training/fixture_val/sample.json": stageBValSampleFile(),
};

async function openStage(value: "prior" | "window") {
  render(<><TrainingPanel hidden={false} /><TabsSpy /></>);
  fireEvent.change(await screen.findByLabelText("Sets of"), { target: { value } });
}

describe("one layout for the three stages", () => {
  beforeEach(() => {
    appState.trainingSelection = null;
    for (const mock of [setTrainingSelection, setTrainingIntervalS, setTrainingCursorS]) mock.mockClear();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("stage C: the windows in four columns, its tabs Labelled and the rounds, one line per readout, its details page", async () => {
    serve(C_FILES, { [intentUrl(WINDOW_SET_ID)]: { ok: true, run: WINDOW_SET_ID, campaign: "post_fixture", title: "The windows",
      intent: "What the post-training's rounds say in recorded traffic.", design: "post_training.md §6", line: "The fixture's windows." } });
    await openStage("window");
    const sample = stageCSample();
    const head = sample.windows[0].commanded[0].head;
    expect((await screen.findAllByText(head.callsign)).length).toBe(sample.windows.length);
    expect(screen.queryByRole("table")).toBeNull();                                      // no table in the left panel
    const column = screen.getAllByText(/landed ·/).find((node) => node.classList.contains("training-flight-executor"))!;
    expect(column.textContent).toMatch(/^\d+\/1 landed · \d+ other$/);
    await waitFor(() => expect(shownTabs?.tabs.map((tab) => tab.label)).toEqual(["Labelled", "Start (base)"]));
    expect(shownTabs!.chosen).toBe("start");
    // window B's Labelled tab says the window starts from a moved start
    expect(sample.windows[2].kind).toBe("B");
    fireEvent.click(screen.getByRole("list", { name: "The set's windows" }).querySelectorAll("button")[2]);
    await waitFor(() => expect(shownTabs!.tabs[0].title).toContain("moved start"));
    act(() => chooseTrainingTab("labelled"));
    await waitFor(() => expect(setTrainingIntervalS).toHaveBeenLastCalledWith(null));
    expect(screen.getByRole("button", { name: /^This round/ }).textContent).toContain("not flown");
    expect(screen.getByRole("button", { name: /^The window/ }).textContent).toContain("B: the commanded aircraft's start moved");
    // the set's own line, and the header's ⓘ on "The set and the experiment"
    expect(await screen.findByRole("button", { name: /The fixture's windows/ })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Training details" }));
    const page = await screen.findByRole("dialog");
    expect(page.textContent).toContain("The windows");
    expect(page.textContent).toContain("What the post-training's rounds say in recorded traffic.");
    expect(page.textContent).toContain("post_training.md §6");
    expect(screen.getByRole("tab", { name: /Rounds/ })).toBeTruthy();
  });

  it("stage C: the cursor slider carries the window's first predicted step and, on a round's tab, its loss of separation", async () => {
    serve(C_FILES);
    const sample = stageCSample();
    const lost = sample.windows.findIndex((window) => window.rounds[0].losses.length > 0);
    appState.trainingSelection = stageCSelection(sample, lost);                       // the bar's state: that window's round
    await openStage("window");
    await screen.findAllByText(sample.windows[0].commanded[0].head.callsign);
    fireEvent.click(screen.getByRole("list", { name: "The set's windows" }).querySelectorAll("button")[lost]);
    const [aircraft] = sample.windows[lost].commanded;
    const [loss] = sample.windows[lost].rounds[0].losses;
    await waitFor(() => expect(document.querySelector('[data-mark^="loss-"] title')?.textContent)
      .toContain(`with ${otherOf(loss, aircraft.datasetId)}`));
    expect(document.querySelector('[data-mark="first-step"] title')!.textContent).toContain(`${aircraft.firstStepS} s`);
    act(() => chooseTrainingTab("labelled"));
    await waitFor(() => expect(document.querySelector('[data-mark^="loss-"]')).toBeNull());
    expect(document.querySelector('[data-mark="first-step"]')).not.toBeNull();
  });

  it("a results section without fields says why, and still shows the set's own rounds", async () => {
    serve(C_FILES, { [resultsUrl("C", "KXXX", WINDOW_SET_ID)]: resultsAnswer("Cmissing") });
    await openStage("window");
    fireEvent.click(await screen.findByRole("button", { name: "Training details" }));
    fireEvent.click(await screen.findByRole("tab", { name: /Rounds/ }));
    const page = await screen.findByRole("dialog");
    await waitFor(() => expect(page.textContent).toContain("campaign.json does not exist"));
    expect(screen.getByRole("table", { name: "Every window's rounds" })).toBeTruthy();
  });

  it("stage B: the ⓘ is never disabled; the claimed val set says its readout; a set without an intent says so by name", async () => {
    serve(B_FILES, { [intentUrl("fixture_set")]: { ok: false, run: "fixture_set", campaigns: [], error: "fixture_set is a run of no campaign in intents.json" },
      [resultsUrl("B", "KXXX", "fixture_val")]: resultsAnswer("Bval") });
    await openStage("prior");
    expect(await screen.findByRole("button", { name: /No intent for fixture_set: fixture_set is a run of no campaign/ })).toBeTruthy();
    const info = screen.getByRole("button", { name: "Training details" }) as HTMLButtonElement;
    expect(info.disabled).toBe(false);
    fireEvent.change(screen.getByLabelText("Set"), { target: { value: "fixture_val" } });
    await waitFor(() => expect(lastPublished()?.setId).toBe("fixture_val"));
    fireEvent.click(info);
    const page = await screen.findByRole("dialog");
    // the first section: the intent and one line of provenance (D134); the results: the base's validation, for this set only
    expect(page.textContent).toContain("made from fixture/readout_val");
    expect(page.textContent).not.toContain("What this view shows");
    fireEvent.click(await screen.findByRole("tab", { name: /Validation/ }));
    expect((await screen.findByRole("dialog")).textContent).toContain("1.0800 per step");
  });

  it("the header's ⓘ opens a page where no session is on screen (stage B's index cannot be read), saying why", async () => {
    serve({ "data/airports/KXXX/training/index_prior_v3.json": { schema: "aeroviz-training-prior-index-v0", airport: "KXXX", sets: [] } });
    await openStage("prior");
    expect(await screen.findByText(/cannot be read/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Training details" }));
    const page = await screen.findByRole("dialog");
    expect(page.textContent).toContain("no prior set of KXXX can be read");
  });

  it("the details page opens from outside the dock (the sentence bar's notes) on its section", async () => {
    serve(B_FILES, { [resultsUrl("B", "KXXX", "fixture_set")]: resultsAnswer("B") });
    await openStage("prior");
    await waitFor(() => expect(lastPublished()?.flight.flightKey).toContain("~prior-0"));
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    act(() => requestTrainingDetails("speed", opener));
    const page = await screen.findByRole("dialog");
    await waitFor(() => expect(page.textContent).toContain("flight rows a second"));
    expect(page.querySelector('[aria-label="The words at a row"]')).toBeNull();           // no row inspector (D134)
  });
});

describe("the tabs' short labels (the user, 2026-10-06)", () => {
  it("a round is r1, r2 …; the base's start is Start (base)", () => {
    expect([1, 2, 12].map(roundTabLabel)).toEqual(["r1", "r2", "r12"]);
    expect(roundTabLabel("start")).toBe("Start (base)");
  });
});
