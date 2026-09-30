/**
 * The Training panel over a WINDOW SET (the Training module §4.11): the windows and their aircraft, the aircraft on
 * screen published on the window's clock, the models' sentences projected onto it, the window published for the strip
 * and 3D — and the read-back set beside it untouched.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";

const {
  appState, setTrainingSelection, setTrainingGenerations, setTrainingWindow, fetchMock,
} = vi.hoisted(() => ({
  appState: {
    activeAirportCode: "KXXX" as string,
    trainingLayers: { headingBands: true, corridor: true, vertical: true, candidates: true },
    trainingSelection: null, trainingColumn: null, trainingPick: null,
    trainingSource: null as unknown,
    trainingGenerations: [] as unknown[],
  },
  setTrainingSelection: vi.fn(),
  setTrainingGenerations: vi.fn(),
  setTrainingWindow: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    ...appState, setTrainingSelection, setTrainingGenerations, setTrainingWindow, setTrainingLayer: vi.fn(),
    setTrainingExecutor: vi.fn(), setTrainingPrior: vi.fn(), setTrainingAutopilot: vi.fn(),
  }),
}));

import TrainingPanel from "../TrainingPanel";
import { mockIndex, mockSample, SET_ID } from "../../data/__tests__/trainingSample.fixture";
import {
  STRAIGHT_ID, TRAFFIC_SET_ID, VECTORED_ID, WINDOW_MODEL_ID, mockTrafficEntry, mockTrafficSet, mockWindowOverlay, mockWindowOverlays,
} from "../../data/__tests__/trainingTraffic.fixture";

function jsonResponse(body: unknown) {
  return { ok: true, headers: { get: () => "application/json" }, text: async () => JSON.stringify(body) };
}

function notFound() {
  return { ok: false, status: 404, headers: { get: () => "text/html" }, text: async () => "<!doctype html>" };
}

const ROOT = "data/airports/KXXX/training";

function serve(): void {
  const index: any = mockIndex();
  index.sets.push(mockTrafficEntry());
  const files: Record<string, unknown> = {
    [`${ROOT}/index.json`]: index,
    [`${ROOT}/${SET_ID}/sample.json`]: mockSample(),
    [`${ROOT}/${TRAFFIC_SET_ID}/traffic.json`]: mockTrafficSet(),
    [`${ROOT}/overlays.json`]: mockWindowOverlays(),
    [`${ROOT}/${WINDOW_MODEL_ID}/window_generation.json`]: mockWindowOverlay(),
  };
  fetchMock.mockImplementation(async (url: string) => (url in files ? jsonResponse(files[url]) : notFound()));
}

function last(mock: ReturnType<typeof vi.fn>): any {
  const calls = mock.mock.calls;
  return calls.length ? calls[calls.length - 1][0] : undefined;
}

async function openWindows() {
  render(<TrainingPanel hidden={false} />);
  const chooser = await screen.findByRole("combobox");
  fireEvent.change(chooser, { target: { value: TRAFFIC_SET_ID } });
  return screen.findByRole("list", { name: "Windows" });
}

describe("TrainingPanel over a window set", () => {
  beforeEach(() => {
    setTrainingSelection.mockClear();
    setTrainingGenerations.mockClear();
    setTrainingWindow.mockClear();
    appState.trainingSource = null;
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
    serve();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("lists the set by its windows, and a window's aircraft — the commanded ones to pick, the others by role", async () => {
    const windows = await openWindows();
    expect(within(screen.getByRole("combobox")).getByText(/traffic_windows_select · 1 windows, select/)).toBeTruthy();
    expect(within(windows).getByText("01-01 00:00Z")).toBeTruthy();
    expect(within(windows).getByText("2 by model · 2 others")).toBeTruthy();
    const aircraft = screen.getByRole("list", { name: "Aircraft of the window" });
    expect(within(aircraft).getAllByRole("button").map((button) => button.textContent)).toEqual([
      expect.stringContaining("TST1"), expect.stringContaining("TST2")]);
    expect(within(aircraft).getByText("ABC1")).toBeTruthy();
    expect(within(aircraft).getByText("background")).toBeTruthy();
    // as recorded (the truth read): both land, in the record's order
    expect(within(aircraft).getByText("landed 1st")).toBeTruthy();
  });

  it("puts the first commanded aircraft on screen on the window's clock, and another at the same moment", async () => {
    await openWindows();
    await waitFor(() => expect(last(setTrainingSelection)?.flight.datasetId).toBe(VECTORED_ID));
    const first = last(setTrainingSelection);
    expect(first.clock).toEqual({ scope: `KXXX/${TRAFFIC_SET_ID}/window 0`, offsetS: 100 });
    expect(first.liveExecutor).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: /TST2/ }));
    await waitFor(() => expect(last(setTrainingSelection)?.flight.datasetId).toBe(STRAIGHT_ID));
    expect(last(setTrainingSelection).clock).toEqual({ scope: first.clock.scope, offsetS: 160 });
    expect(last(setTrainingWindow).window.index).toBe(0);
  });

  it("projects each model onto the aircraft on screen, and says in the list how each sample of the window went", async () => {
    appState.trainingSource = { overlayId: WINDOW_MODEL_ID, sample: 0 };
    await openWindows();
    await waitFor(() => expect(last(setTrainingGenerations)).toHaveLength(1));
    const [projected] = last(setTrainingGenerations);
    expect(projected.overlay.overlayId).toBe(WINDOW_MODEL_ID);
    expect(projected.flight.samples.map((sample: any) => sample.outcome)).toEqual(["lost_separation", "timeout"]);
    const aircraft = screen.getByRole("list", { name: "Aircraft of the window" });
    expect(within(aircraft).getByText("lost separation · TST2")).toBeTruthy();
    expect(within(aircraft).getByText("landed 1st (recorded 2nd)")).toBeTruthy();
    const marks = within(screen.getByRole("list", { name: "Windows" })).getByTitle(/sample by sample/);
    expect(marks.getAttribute("title")).toMatch(/#1 an aircraft lost separation.*#2 no loss of separation, but not every/);
    expect(screen.getByRole("button", { name: /Windows/ }).textContent).toMatch(/recorded 0\.0 % · base 25\.0 % lost separation/);
  });

  it("unpublishes the window when another set is opened", async () => {
    await openWindows();
    await waitFor(() => expect(last(setTrainingWindow)).not.toBeNull());
    fireEvent.change(screen.getByRole("combobox"), { target: { value: SET_ID } });
    await screen.findByText("TST1");
    await waitFor(() => expect(last(setTrainingWindow)).toBeNull());
  });
});
