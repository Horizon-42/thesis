/**
 * Stage D's window view (frontend §5, F3) on the set of two commanded aircraft (`stageD.ts`): the stage switch offers D
 * where the airport has its index; the list names the anchor "+1" and the landed of both; the aircraft strip shows both,
 * the anchor silent, and a click, `[` and `]` or a click in the scene select an aircraft (the tab stays); D's readouts;
 * the slider's slate band of the anchor's silence; a window of one commanded aircraft shows no strip.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

const { appState, setTrainingSelection, fetchMock } = vi.hoisted(() => ({
  appState: {
    activeAirportCode: "KXXX" as string,
    mode: "training" as string,
    trainingLayers: { headingBands: true, vertical: true, candidates: true },
    trainingIntervalS: 4 as number | null,
    trainingSelection: null as unknown, trainingPick: null,
    cursorS: 0,
  },
  setTrainingSelection: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingSelection, setTrainingIntervalS: vi.fn(), setTrainingAutopilot: vi.fn(), setTrainingLayer: vi.fn() }),
  useTrainingCursor: () => ({ trainingCursorS: appState.cursorS, setTrainingCursorS: vi.fn() }),
}));

import TrainingPanel from "../TrainingPanel";
import { stageDIndex, stageDSample, stageDSampleFile, stageDSelection, TWO_SET_ID } from "../../data/__tests__/stageD";
import { stageCIndex, stageCSampleFile, WINDOW_SET_ID } from "../../data/__tests__/stageC";
import { trainingWindowOriginOf } from "../../data/trainingWindowSample";
import { pickTrainingWindowAircraft } from "../../data/trainingWindowLayers";

const json = (body: unknown) => ({ ok: true, status: 200, headers: { get: () => "application/json" }, text: async () => JSON.stringify(body) });
const notFound = () => ({ ok: false, status: 404, headers: { get: () => "text/html" }, text: async () => "<!doctype html>" });
const D_FILES: Record<string, unknown> = {
  "data/airports/KXXX/training/index_multi_v1.json": stageDIndex(),
  [`data/airports/KXXX/training/${TWO_SET_ID}/sample.json`]: stageDSampleFile(),
  "data/airports/KXXX/training/index_post_v3.json": stageCIndex(),
  [`data/airports/KXXX/training/${WINDOW_SET_ID}/sample.json`]: stageCSampleFile(),
};
const lastOrigin = () => {
  const calls = setTrainingSelection.mock.calls.filter((call) => call[0] !== null);
  return trainingWindowOriginOf(calls[calls.length - 1][0].flight)!;
};

async function openStage(value: "window" | "multi") {
  const view = render(<TrainingPanel hidden={false} />);
  fireEvent.change(await screen.findByLabelText("Sets of"), { target: { value } });
  return view;
}

describe("stage D's window view", () => {
  beforeEach(() => {
    appState.trainingSelection = null;
    setTrainingSelection.mockClear();
    fetchMock.mockImplementation(async (url: string) => (url in D_FILES ? json(D_FILES[url]) : notFound()));
    vi.stubGlobal("fetch", fetchMock);
  });

  it("is offered by the stage switch where the airport has stage D's index, and lists the anchor +1", async () => {
    await openStage("multi");
    const select = screen.getByLabelText("Sets of") as HTMLSelectElement;
    expect([...select.options].map((option) => option.value)).toEqual(["stageA", "window", "multi"]);
    const list = await screen.findByRole("list", { name: "The set's windows" });
    const sample = stageDSample();
    const [anchor] = sample.windows[0].commanded;
    expect(within(list).getByText(`${anchor.head.callsign} +1`)).toBeTruthy();
  });

  it("the strip shows both aircraft, the anchor silent; a click, ] and [ select one, the tab staying", async () => {
    const view = await openStage("multi");
    const strip = await screen.findByRole("group", { name: "The window's commanded aircraft" });
    const squares = within(strip).getAllByRole("button");
    expect(squares).toHaveLength(2);
    expect(squares[0].getAttribute("aria-pressed")).toBe("true");
    expect(squares[0].dataset.silent).toBe("true");                       // the anchor answered its loss
    expect(strip.textContent).toContain("1 of 2");
    await waitFor(() => expect(lastOrigin().aircraft.place).toBe(0));
    fireEvent.click(squares[1]);
    await waitFor(() => expect(lastOrigin().aircraft.place).toBe(1));
    expect(lastOrigin().round).toBe("start");                             // the tab stays
    fireEvent.keyDown(document.body, { key: "[" });
    await waitFor(() => expect(lastOrigin().aircraft.place).toBe(0));
    fireEvent.keyDown(document.body, { key: "]" });
    await waitFor(() => expect(lastOrigin().aircraft.place).toBe(1));
    // in another task the panel stays mounted, hidden: the keys are not Training's there (AV28)
    appState.mode = "fly";
    view.rerender(<TrainingPanel hidden />);
    fireEvent.keyDown(document.body, { key: "[" });
    expect(lastOrigin().aircraft.place).toBe(1);
    appState.mode = "training";
    view.rerender(<TrainingPanel hidden={false} />);
    // a click on the anchor in the scene (the layer's pick) selects it
    act(() => pickTrainingWindowAircraft(stageDSample().windows[0].commanded[0].datasetId));
    await waitFor(() => expect(lastOrigin().aircraft.place).toBe(0));
  });

  it("gives D's readouts and the slider's slate band of the anchor's silence", async () => {
    appState.trainingSelection = stageDSelection(stageDSample(), 0, 0);      // the slider draws for a sentence on screen
    await openStage("multi");
    const readouts = await screen.findByRole("list", { name: "Readouts" });
    const sample = stageDSample();
    const [anchor] = sample.windows[0].commanded;
    expect(readouts.textContent).toMatch(/W [\d.]+ of 2 · \d landed · \d+ go-arounds · 1 silent · 2 losses of separation between commanded aircraft, 0 with recorded/);
    expect(readouts.textContent).toContain(`${anchor.head.callsign} · lost separation · r 0.00 · silent from row`);
    expect(readouts.textContent).toContain("joins at 0 s");
    expect(readouts.textContent).toContain("2 commanded · at most 0 recorded at once");
    expect(document.querySelector('[data-mark="silent"]')).not.toBeNull();
  });

  it("a window of one commanded aircraft (stage C's) shows no strip", async () => {
    await openStage("window");
    await screen.findByRole("list", { name: "Readouts" });
    expect(screen.queryByRole("group", { name: "The window's commanded aircraft" })).toBeNull();
  });
});
