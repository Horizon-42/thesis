/**
 * TrainingPanel: the empty state, the readable set, the sets it refuses BY NAME from the manifest
 * alone (never downloaded), a readable set that fails with its field, and the overlays drawn over the
 * set — the executor's replay and the prior's predictions — or, without them, the switches that say so.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const {
  appState, setTrainingSelection, setTrainingLayer, setTrainingExecutor, setTrainingPrior, setTrainingAutopilot,
  setTrainingAutopilotAuto, fetchMock,
} = vi.hoisted(() => ({
  appState: {
    activeAirportCode: "KXXX" as string,
    trainingLayers: { headingBands: true, corridor: true, vertical: true, candidates: true },
    // no word selected: the live executor asks for nothing
    trainingSelection: null, trainingColumn: null, trainingCursorS: 0, trainingAutopilot: null, trainingPick: null,
    trainingAutopilotAuto: true,
  },
  setTrainingSelection: vi.fn(),
  setTrainingLayer: vi.fn(),
  setTrainingExecutor: vi.fn(),
  setTrainingPrior: vi.fn(),
  setTrainingAutopilot: vi.fn(),
  setTrainingAutopilotAuto: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    ...appState, setTrainingSelection, setTrainingLayer, setTrainingExecutor, setTrainingPrior, setTrainingAutopilot,
    setTrainingAutopilotAuto,
  }),
}));

import TrainingPanel from "../TrainingPanel";
import { SET_ID, STRAIGHT_KEY, VECTORED_KEY, mockIndex, mockSample } from "../../data/__tests__/trainingSample.fixture";
import {
  EXECUTOR_ID, PRIOR_ID, mockExecutorOverlay, mockOverlays, mockPriorOverlay,
} from "../../data/__tests__/trainingOverlays.fixture";

function jsonResponse(body: unknown) {
  return { ok: true, headers: { get: () => "application/json" }, text: async () => JSON.stringify(body) };
}

function notFound() {
  return { ok: false, status: 404, headers: { get: () => "text/html" }, text: async () => "<!doctype html>" };
}

function serve(files: Record<string, unknown>) {
  fetchMock.mockImplementation(async (url: string) => (url in files ? jsonResponse(files[url]) : notFound()));
}

function lastPublished(): any {
  const calls = setTrainingSelection.mock.calls;
  return calls.length ? calls[calls.length - 1][0] : null;
}

function lastOf(mock: ReturnType<typeof vi.fn>): any {
  const calls = mock.mock.calls;
  return calls.length ? calls[calls.length - 1][0] : undefined;
}

const INDEX_PATH = "data/airports/KXXX/training/index.json";
const SAMPLE_PATH = `data/airports/KXXX/training/${SET_ID}/sample.json`;
const OLD_PATH = "data/airports/KXXX/training/box_v3/sample.json";
const FIRST_PATH = "data/airports/KXXX/training/instruction_v1/sample.json";
const SECOND_PATH = "data/airports/KXXX/training/instruction_v2/sample.json";
const OVERLAYS_PATH = "data/airports/KXXX/training/overlays.json";
const EXECUTOR_PATH = `data/airports/KXXX/training/${EXECUTOR_ID}/executor.json`;
const PRIOR_PATH = `data/airports/KXXX/training/${PRIOR_ID}/prior.json`;

describe("TrainingPanel", () => {
  beforeEach(() => {
    appState.activeAirportCode = "KXXX";
    setTrainingSelection.mockClear();
    setTrainingLayer.mockClear();
    setTrainingExecutor.mockClear();
    setTrainingPrior.mockClear();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("with no export on disk", () => {
    beforeEach(() => serve({}));

    it("names the airport, the path it reads and the command that writes it", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText(/No Training export for KXXX yet/)).toBeTruthy();
      expect(screen.getByText(INDEX_PATH)).toBeTruthy();
      expect(screen.getByText(/run_ts\.py instruction_training_export .*--airport KXXX/)).toBeTruthy();
    });

    // AV5: vite does not watch public/data, so a directory created after boot is the SPA fallback.
    it("warns that the dev server must be restarted after the first export", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText(/restart the dev server/i)).toBeTruthy();
    });
  });

  describe("with an export", () => {
    beforeEach(() => serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample() }));

    it("opens on the set it can read, not the first one listed, and downloads only that", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText("TST1")).toBeTruthy();
      expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe(SET_ID);
      const fetched = fetchMock.mock.calls.map(([url]) => url);
      expect(fetched).toContain(SAMPLE_PATH);
      expect(fetched).not.toContain(OLD_PATH);
      expect(fetched).not.toContain(FIRST_PATH);
      expect(fetched).not.toContain(SECOND_PATH);
    });

    it("publishes the flight with the vocabulary and the candidate runways", async () => {
      render(<TrainingPanel />);
      await waitFor(() => expect(lastPublished()?.flight.flightKey).toBe(VECTORED_KEY));
      expect(lastPublished().candidates.map((c: any) => c.ident)).toEqual(["09", "27"]);
      fireEvent.click(screen.getByText("TST2"));
      await waitFor(() => expect(lastPublished()?.flight.flightKey).toBe(STRAIGHT_KEY));
    });

    it("refuses a superseded set BY NAME from the manifest alone, without downloading it", async () => {
      render(<TrainingPanel />);
      await screen.findByText("TST1");
      fireEvent.change(screen.getByRole("combobox"), { target: { value: "box_v3" } });
      expect(await screen.findByText(/read under box-v3, a superseded vocabulary/)).toBeTruthy();
      expect(fetchMock.mock.calls.map(([url]) => url)).not.toContain(OLD_PATH);
      await waitFor(() => expect(lastPublished()).toBeNull());
    });

    it("marks the refused sets in the chooser", async () => {
      render(<TrainingPanel />);
      await screen.findByText("TST1");
      const options = [...(screen.getByRole("combobox") as HTMLSelectElement).options].map((option) => option.text);
      expect(options.find((text) => text.startsWith("box_v3"))).toMatch(/box-v3 — refused/);
      expect(options.find((text) => text.startsWith("instruction_v1"))).toMatch(/instruction-v1 — refused/);
      expect(options.find((text) => text.startsWith("instruction_v2"))).toMatch(/instruction-v2 — refused/);
      expect(options.find((text) => text.startsWith(SET_ID))).not.toMatch(/refused/);
    });

    it("with no overlay published, keeps both switches off and names the commands that write them", async () => {
      render(<TrainingPanel />);
      await screen.findByText("TST1");
      const replay = screen.getByLabelText("Executor replay") as HTMLInputElement;
      const prior = screen.getByLabelText("Prior predictions") as HTMLInputElement;
      expect(replay.disabled && prior.disabled).toBe(true);
      expect(replay.checked || prior.checked).toBe(false);
      // the commands fold away under "none published"
      expect(screen.getAllByText("none published")).toHaveLength(2);
      expect(screen.getByText(new RegExp(`run_ts\\.py executor_training_export .*--set ${SET_ID} --airport KXXX`)).closest("details")).not.toBeNull();
      expect(screen.getByText(new RegExp(`run_ts\\.py prior_training_export .*--set ${SET_ID} --airport KXXX`))).toBeTruthy();
      await waitFor(() => expect(lastOf(setTrainingExecutor)).toBeNull());
    });

    it("switches the envelopes everywhere through the shared layers", async () => {
      render(<TrainingPanel />);
      await screen.findByText("TST1");
      fireEvent.click(screen.getByLabelText("Altitude tubes + speed bands"));
      expect(setTrainingLayer).toHaveBeenCalledWith("vertical", false);
      fireEvent.click(screen.getByLabelText("Heading bands"));
      expect(setTrainingLayer).toHaveBeenCalledWith("headingBands", false);
      fireEvent.click(screen.getByLabelText("Capture turn + corridor"));
      expect(setTrainingLayer).toHaveBeenCalledWith("corridor", false);
      // what each shows is its tooltip
      expect(screen.getByLabelText("Altitude tubes + speed bands").closest("label")!.getAttribute("title"))
        .toMatch(/each altitude word's tube, and on the speed chart each speed word's transition and band/);
    });

    it("states the draw it came from, in full in its tooltip", async () => {
      render(<TrainingPanel />);
      const note = await screen.findByText(/^2 flights · val, 1 per stratum/);
      expect(note.getAttribute("title")).toMatch(/a test draw/);
    });
  });

  describe("with the executor's replay and the prior's predictions over the set", () => {
    beforeEach(() => serve({
      [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample(), [OVERLAYS_PATH]: mockOverlays(),
      [EXECUTOR_PATH]: mockExecutorOverlay(), [PRIOR_PATH]: mockPriorOverlay(),
    }));

    it("publishes both for the selected flight, and follows the selection", async () => {
      render(<TrainingPanel />);
      await waitFor(() => expect(lastOf(setTrainingExecutor)?.flight.flightKey).toBe(VECTORED_KEY));
      await waitFor(() => expect(lastOf(setTrainingPrior)?.flight.flightKey).toBe(VECTORED_KEY));
      fireEvent.click(screen.getByText("TST2"));
      await waitFor(() => expect(lastOf(setTrainingExecutor)?.flight.flightKey).toBe(STRAIGHT_KEY));
      expect(lastOf(setTrainingExecutor).flight.flown).toBe(false);
    });

    it("says under each flight what the executor made of it", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText("landed · 5/7")).toBeTruthy();
      expect(screen.getByText("not flown")).toBeTruthy();
    });

    it("folds the replay's gate table and the prior's readout under the switches", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText(/The executor's val replay gate · spec 777777777777/)).toBeTruthy();
      expect(screen.getByText(/The prior's val readout · 0\.1778 per step against 0\.3444 \/ 0\.3260/)).toBeTruthy();
      expect(screen.getByText(/own dynamics at KXXX — gated, each share ≥ 95 %/)).toBeTruthy();
      expect(screen.getByText(/stand-in dynamics at KXXX — reported, not gated: a stand-in's dynamics/)).toBeTruthy();
    });

    it("stops publishing an overlay switched off", async () => {
      render(<TrainingPanel />);
      await waitFor(() => expect(lastOf(setTrainingExecutor)?.flight.flightKey).toBe(VECTORED_KEY));
      fireEvent.click(screen.getByLabelText("Executor replay"));
      await waitFor(() => expect(lastOf(setTrainingExecutor)).toBeNull());
      expect(lastOf(setTrainingPrior)?.flight.flightKey).toBe(VECTORED_KEY);
    });

    it("never fetches the last airport's overlay under the next airport's path", async () => {
      const { rerender } = render(<TrainingPanel />);
      await waitFor(() => expect(lastOf(setTrainingExecutor)?.flight.flightKey).toBe(VECTORED_KEY));
      appState.activeAirportCode = "KYYY";
      rerender(<TrainingPanel />);
      await waitFor(() => expect(fetchMock.mock.calls.map(([url]) => url)).toContain("data/airports/KYYY/training/index.json"));
      const fetched = fetchMock.mock.calls.map(([url]) => url as string);
      expect(fetched.filter((url) => url.startsWith("data/airports/KYYY/training/executor_test"))).toEqual([]);
      expect(fetched.filter((url) => url.startsWith("data/airports/KYYY/training/prior_test"))).toEqual([]);
    });

    it("refuses an overlay drawn over the set before it was re-exported, and says why", async () => {
      const stale: any = mockExecutorOverlay();
      stale.base.sampleWrittenUtc = "2026-09-01T00:00:00+00:00";
      serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample(), [OVERLAYS_PATH]: mockOverlays(), [EXECUTOR_PATH]: stale,
              [PRIOR_PATH]: mockPriorOverlay() });
      render(<TrainingPanel />);
      expect(await screen.findByText(`Overlay ${EXECUTOR_ID} cannot be read.`)).toBeTruthy();
      expect(screen.getByText(/the set was re-exported after the overlay/)).toBeTruthy();
      await waitFor(() => expect(lastOf(setTrainingPrior)?.flight.flightKey).toBe(VECTORED_KEY));
    });
  });

  it("names the field when a readable set fails, and leaves the manifest standing", async () => {
    const broken: any = mockSample();
    broken.flights[0].envelopes.altitude[0].check.inside = 3;
    serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: broken });
    render(<TrainingPanel />);
    expect(await screen.findByText(new RegExp(`Set ${SET_ID} cannot be read`))).toBeTruthy();
    expect(screen.getByText(/says 3 rows inside, but the tube's own flags count 20/)).toBeTruthy();
    expect(screen.getByRole("combobox")).toBeTruthy();
  });

  it("greys out a malformed entry by name while the others load", async () => {
    const index: any = mockIndex();
    delete index.sets[0].title;
    serve({ [INDEX_PATH]: index, [SAMPLE_PATH]: mockSample() });
    render(<TrainingPanel />);
    expect(await screen.findByText(/Entry box_v3 was rejected/)).toBeTruthy();
    expect(await screen.findByText("TST1")).toBeTruthy();
  });
});
