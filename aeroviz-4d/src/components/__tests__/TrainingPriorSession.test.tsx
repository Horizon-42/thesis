/**
 * The Training panel over a prior set of stage B: the switch to the prior's sets appears only where the airport has
 * `index_prior_v1.json`; a set is read, its flights and — side by side — the closed-loop sentence and each sentence the prior
 * said are listed with their outcomes; a click publishes that sentence as the flight on screen; the row inspector shows the
 * probability of go-around and the words the procedure blocked at the cursor's row; a set of another schema is refused by name.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

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

const INDEX_PATH = "data/airports/KXXX/training/index_prior_v1.json";
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
  render(<TrainingPanel hidden={false} />);
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

  it("lists the closed-loop sentence and the prior's sentences side by side and publishes the one chosen", async () => {
    serve({ [INDEX_PATH]: stageBIndex(), [SAMPLE_PATH]: stageBSampleFile() });
    await openPriorSets();
    const sample = stageBSample();
    const { head, sentences } = sample.flights[0];
    expect(await screen.findByText(head.callsign)).toBeTruthy();
    const table = await screen.findByRole("table", { name: "The flight's sentences" });
    expect(table.querySelectorAll("tbody tr")).toHaveLength(1 + sentences.length);
    // the set opens on the first flight's first prior sentence, read at the prior's Δ
    await waitFor(() => expect(lastPublished()?.flight.flightKey).toBe(`${head.flightKey}~prior-0`));
    expect(setTrainingIntervalS).toHaveBeenCalledWith(4);
    expect(lastPublished().vocabulary.rowIntervalsS).toEqual([4]);
    fireEvent.click(screen.getByRole("button", { name: "prior · sample 1" }));
    await waitFor(() => expect(lastPublished().flight.flightKey).toBe(`${head.flightKey}~prior-1`));
    fireEvent.click(screen.getByRole("button", { name: "closed-loop reading" }));
    await waitFor(() => expect(lastPublished().flight.flightKey).toBe(`${head.flightKey}~closed-loop`));
    expect(screen.getByText(/pick a prior sentence for them/)).toBeTruthy();
  });

  it("shows, at the cursor's row, the go-around probability and the words the procedure blocked", async () => {
    serve({ [INDEX_PATH]: stageBIndex(), [SAMPLE_PATH]: stageBSampleFile() });
    const sample = stageBSample();
    const sentence = sample.flights[0].sentences[0];
    appState.cursorS = sample.flights[0].head.closedLoop["4"].startS + 4 * 5;        // Δ row 5
    await openPriorSets();
    const inspector = await screen.findByLabelText("The words at a row");
    expect(inspector.textContent).toContain(`${(sentence.goAroundProbability[5] * 100).toFixed(1)} % said`);
    expect(inspector.textContent).toContain(`${sentence.blocked.altitude[5].length} blocked`);
    expect(inspector.textContent).toContain("Blocked · angle");
    fireEvent.change(screen.getByLabelText("Word row"), { target: { value: "9" } });
    expect(setTrainingCursorS).toHaveBeenCalledWith(sample.flights[0].head.closedLoop["4"].startS + 9 * 4);
  });

  it("refuses a set of another schema by name and keeps the others", async () => {
    serve({ [INDEX_PATH]: stageBIndex(), [SAMPLE_PATH]: { ...stageBSampleFile(), schema: "aeroviz-training-sample-v9" } });
    await openPriorSets();
    expect(await screen.findByText(/Set fixture_set cannot be read/)).toBeTruthy();
    expect(screen.getByText(new RegExp(`not one of ${TRAINING_PRIOR_SAMPLE_SCHEMA}`))).toBeTruthy();
  });
});
