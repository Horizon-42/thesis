/**
 * TrainingPanel: the empty state, the readable set (read from `training/index_v4.json` and nothing else), an index or a set
 * of another schema refused by name — with the schema found and the one expected — an entry that is not a set entry, and the
 * flown flight's outcome and DA check in the dock.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const { appState, setTrainingSelection, setTrainingIntervalS, setTrainingAutopilot, setTrainingLayer, fetchMock } = vi.hoisted(() => ({
  appState: {
    activeAirportCode: "KXXX" as string,
    trainingLayers: { headingBands: true, vertical: true, candidates: true },
    trainingIntervalS: 2 as number | null,
    // no word picked: the live executor asks for nothing
    trainingSelection: null, trainingPick: null,
  },
  setTrainingSelection: vi.fn(),
  setTrainingIntervalS: vi.fn(),
  setTrainingAutopilot: vi.fn(),
  setTrainingLayer: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingSelection, setTrainingIntervalS, setTrainingAutopilot, setTrainingLayer }),
}));

import TrainingPanel from "../TrainingPanel";
import { TRAINING_INDEX_SCHEMA, TRAINING_SAMPLE_SCHEMA } from "../../data/trainingSample";
import { FLIGHT_KEY, stageAIndex, stageASampleFile } from "../../data/__tests__/stageA";

const INDEX_PATH = "data/airports/KXXX/training/index_v4.json";
const OLD_INDEX_PATH = "data/airports/KXXX/training/index.json";
const SAMPLE_PATH = "data/airports/KXXX/training/fixture_set/sample.json";

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

describe("TrainingPanel", () => {
  beforeEach(() => {
    appState.activeAirportCode = "KXXX";
    appState.trainingIntervalS = 2;
    for (const mock of [setTrainingSelection, setTrainingIntervalS, setTrainingLayer]) mock.mockClear();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("with no export on disk", () => {
    beforeEach(() => serve({}));

    it("names the airport, the path it reads and the command that writes it", async () => {
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText(/No Training export for KXXX yet/)).toBeTruthy();
      expect(screen.getByText(INDEX_PATH)).toBeTruthy();
      expect(screen.getByText(/run_ts\.py training_export .*--airports KXXX/)).toBeTruthy();
    });

    // AV5: vite does not watch public/data, so a directory created after boot is the SPA fallback.
    it("warns that the dev server must be restarted after the first export", async () => {
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText(/restart the dev server/i)).toBeTruthy();
    });
  });

  describe("with an export", () => {
    beforeEach(() => serve({ [INDEX_PATH]: stageAIndex(), [SAMPLE_PATH]: stageASampleFile() }));

    it("reads index_v4.json and never the old view's index.json", async () => {
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("KXXX:test")).toBeTruthy();
      const asked = fetchMock.mock.calls.map(([url]) => url);
      expect(asked).toContain(INDEX_PATH);
      expect(asked).not.toContain(OLD_INDEX_PATH);
    });

    it("lists the flights, publishes the first, and puts the set's first Δ on screen", async () => {
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("KXXX:test")).toBeTruthy();
      // the flight is published by an effect after the list renders
      await waitFor(() => expect(lastPublished()?.flight.flightKey).toBe(FLIGHT_KEY));
      expect(lastPublished()).toMatchObject({ airport: "KXXX", setId: "fixture_set" });
      expect(setTrainingIntervalS).toHaveBeenCalledWith(2);
      expect(screen.getByText(/1 flights · train 1, select 0, 1 per stratum/)).toBeTruthy();
    });

    it("says in the flight list how the flown flight ended, its DA check and how many words were added", async () => {
      render(<TrainingPanel hidden={false} />);
      const tag = (await screen.findAllByText(/unstable at minimums/)).find((node) => node.classList.contains("training-flight-executor"))!;
      expect(tag.title).toContain("9 of the");
      expect(tag.textContent).toContain("DA✗");
      expect(tag.textContent).toContain("9c");
    });

    it("shows the flown flight's outcome, crossing and the DA check's values in the dock, for the Δ read", async () => {
      const { rerender } = render(<TrainingPanel hidden={false} />);
      const block = await screen.findByLabelText("The flown flight at Δ 2 s");
      expect(block.textContent).toContain("unstable at minimums at 511 s");
      expect(block.textContent).toContain("15.5 m right of the centreline, 41.8 m above the threshold");
      expect(block.textContent).toContain("failed");
      expect(block.textContent).toContain("cone ±116.3 m");
      expect(block.textContent).toContain("26.8 m above the glidepath");
      appState.trainingIntervalS = 8;
      rerender(<TrainingPanel hidden={false} />);
      expect(screen.getByLabelText("The flown flight at Δ 8 s")).toBeTruthy();
      appState.trainingIntervalS = null;
      rerender(<TrainingPanel hidden={false} />);
      expect(screen.queryByLabelText(/The flown flight at/)).toBeNull();
      expect(screen.getByText("47 words")).toBeTruthy();
    });

    it("offers the three switches the view has, no more", async () => {
      render(<TrainingPanel hidden={false} />);
      await screen.findByText("KXXX:test");
      expect(screen.getAllByRole("checkbox").map((box) => box.parentElement!.textContent)).toEqual([
        "Heading bands", "Altitude tubes + speed bands", "Other candidate runways"]);
      fireEvent.click(screen.getByLabelText("Heading bands"));
      expect(setTrainingLayer).toHaveBeenCalledWith("headingBands", false);
    });

    it("opens the details page on the flown flights, with every flight at every Δ", async () => {
      render(<TrainingPanel hidden={false} />);
      fireEvent.click(await screen.findByRole("button", { name: /Flown flights/ }));
      const dialog = screen.getByRole("dialog", { name: "Training details" });
      expect(dialog.textContent).toContain("Δ 2 s");
      expect(dialog.textContent).toContain("Δ 8 s");
      expect(dialog.textContent).toContain("unstable at minimums · DA ✗ · 9 added");
    });

    it("keeps its session while hidden in another task: the panel hides, nothing is torn down", async () => {
      const { container, rerender } = render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("KXXX:test")).toBeTruthy();
      rerender(<TrainingPanel hidden />);
      expect((container.querySelector("section.training-panel") as HTMLElement).hidden).toBe(true);
      expect(setTrainingSelection).not.toHaveBeenLastCalledWith(null);
      rerender(<TrainingPanel hidden={false} />);
      expect((container.querySelector("section.training-panel") as HTMLElement).hidden).toBe(false);
      expect(screen.getByText("KXXX:test")).toBeTruthy();
    });
  });

  describe("with files it refuses", () => {
    it("refuses an index of another schema whole, naming the schema found and the one expected", async () => {
      serve({ [INDEX_PATH]: { ...stageAIndex(), schema: "aeroviz-training-index-v1" } });
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText(`${INDEX_PATH} cannot be read.`)).toBeTruthy();
      expect(screen.getByText(`schema is "aeroviz-training-index-v1", expected "${TRAINING_INDEX_SCHEMA}"`)).toBeTruthy();
      expect(setTrainingSelection).not.toHaveBeenCalledWith(expect.objectContaining({ flight: expect.anything() }));
    });

    it("names an entry that is not a set entry and the field, and still lists the others", async () => {
      const index = stageAIndex();
      index.sets.push({ ...index.sets[0], id: "old_set", kind: "vocabulary-readback" });
      serve({ [INDEX_PATH]: index, [SAMPLE_PATH]: stageASampleFile() });
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("Entry old_set was rejected.")).toBeTruthy();
      expect(screen.getByText(/old_set\.kind is "vocabulary-readback", not one of closed-loop-readback/)).toBeTruthy();
      expect(await screen.findByText("KXXX:test")).toBeTruthy();
    });

    it("refuses a set of another schema by name, and the others stay untouched", async () => {
      serve({ [INDEX_PATH]: stageAIndex(), [SAMPLE_PATH]: { ...stageASampleFile(), schema: "aeroviz-training-sample-v8" } });
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("Set fixture_set cannot be read.")).toBeTruthy();
      expect(screen.getByText(`sample.schema is "aeroviz-training-sample-v8", not one of ${TRAINING_SAMPLE_SCHEMA}`)).toBeTruthy();
      expect(lastPublished()).toBeNull();
    });
  });
});
