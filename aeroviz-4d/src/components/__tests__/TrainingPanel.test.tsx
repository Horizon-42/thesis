/**
 * TrainingPanel: the empty state, the readable set, the sets it refuses BY NAME from the manifest
 * alone (never downloaded), a readable set that fails with its field, and the two empty slots.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const { appState, setTrainingSelection, setTrainingLayer, fetchMock } = vi.hoisted(() => ({
  appState: {
    activeAirportCode: "KXXX" as string,
    trainingLayers: { lateral: true, vertical: true, candidates: true },
  },
  setTrainingSelection: vi.fn(),
  setTrainingLayer: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingSelection, setTrainingLayer }),
}));

import TrainingPanel from "../TrainingPanel";
import { STRAIGHT_KEY, VECTORED_KEY, mockIndex, mockSample } from "../../data/__tests__/trainingSample.fixture";

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

const INDEX_PATH = "data/airports/KXXX/training/index.json";
const SAMPLE_PATH = "data/airports/KXXX/training/instruction_v1/sample.json";
const OLD_PATH = "data/airports/KXXX/training/box_v3/sample.json";

describe("TrainingPanel", () => {
  beforeEach(() => {
    appState.activeAirportCode = "KXXX";
    setTrainingSelection.mockClear();
    setTrainingLayer.mockClear();
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
      expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe("instruction_v1");
      const fetched = fetchMock.mock.calls.map(([url]) => url);
      expect(fetched).toContain(SAMPLE_PATH);
      expect(fetched).not.toContain(OLD_PATH);
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
      expect(options.find((text) => text.startsWith("instruction_v1"))).not.toMatch(/refused/);
    });

    it("shows the two slots that are empty on purpose, disabled", async () => {
      render(<TrainingPanel />);
      await screen.findByText("TST1");
      const replay = screen.getByLabelText(/executor replay — not built yet/) as HTMLInputElement;
      const prior = screen.getByLabelText(/prior-generated sentence — no prior yet/) as HTMLInputElement;
      expect(replay.disabled && prior.disabled).toBe(true);
    });

    it("switches the envelopes everywhere through the shared layers", async () => {
      render(<TrainingPanel />);
      await screen.findByText("TST1");
      fireEvent.click(screen.getByLabelText(/vertical: the altitude tubes/));
      expect(setTrainingLayer).toHaveBeenCalledWith("vertical", false);
    });

    it("states the draw it came from", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText(/2 flights · a test draw\./)).toBeTruthy();
    });
  });

  it("names the field when a readable set fails, and leaves the manifest standing", async () => {
    const broken: any = mockSample();
    broken.flights[0].envelopes.altitude[0].check.inside = 3;
    serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: broken });
    render(<TrainingPanel />);
    expect(await screen.findByText(/Set instruction_v1 cannot be read/)).toBeTruthy();
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
