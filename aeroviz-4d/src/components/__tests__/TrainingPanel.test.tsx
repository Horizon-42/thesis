/**
 * TrainingPanel: the empty state, the readable set (read from `training/index_v5.json` and nothing else), an index or a set
 * of another schema refused by name — with the schema found and the one expected — an entry that is not a set entry, and the
 * flown flight's outcome and DA check in the dock.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

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
import { chooseTrainingTab } from "../../data/trainingTabs";

const INDEX_PATH = "data/airports/KXXX/training/index_v5.json";
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

    it("reads index_v5.json and never the old view's index.json", async () => {
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
      // the set chooser's line: the set's intent (none here: the test has no backend) opens "The set and the experiment"
      fireEvent.click(await screen.findByRole("button", { name: /No intent for fixture_set/ }));
      const page = await screen.findByRole("dialog");
      expect(page.textContent).toContain("made from fixture/instruction_language");    // one line of provenance (D134)
    });

    it("says in the flight list how many of the flight's closed-loop sentences landed, each one's outcome in its tooltip", async () => {
      render(<TrainingPanel hidden={false} />);
      const tag = (await screen.findAllByText(/landed/)).find((node) => node.classList.contains("training-flight-executor"))!;
      expect(tag.textContent).toBe("0/3 landed");
      expect(tag.title).toContain("Δ 2 s: unstable at minimums");
    });

    it("says the flown flight at the tab's Δ in one line, and follows the bar's tab", async () => {
      render(<TrainingPanel hidden={false} />);
      const line = await screen.findByRole("button", { name: /^Flown flight[^s]/ });
      expect(line.textContent).toContain("Δ 2 s: unstable at minimums at 511 s · DA ✗");
      act(() => chooseTrainingTab("interval-8"));
      await waitFor(() => expect(screen.getByRole("button", { name: /^Flown flight[^s]/ }).textContent).toContain("Δ 8 s"));
      expect(setTrainingIntervalS).toHaveBeenLastCalledWith(8);
      act(() => chooseTrainingTab("labelled"));
      await waitFor(() => expect(screen.getByRole("button", { name: /^Flown flight[^s]/ }).textContent).toContain("not flown"));
      expect(setTrainingIntervalS).toHaveBeenLastCalledWith(null);
    });

    it("offers the three switches the view has, no more", async () => {
      render(<TrainingPanel hidden={false} />);
      await screen.findByText("KXXX:test");
      expect(screen.getAllByRole("checkbox").map((box) => box.parentElement!.textContent)).toEqual([
        "Heading bands", "Altitude tubes", "Other candidate runways"]);
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

    it("refuses a set with a val flight: the panel opens stage A's sets, train and select (D109)", async () => {
      const sample = stageASampleFile();
      sample.flights = sample.flights.map((flight: Record<string, unknown>) => ({ ...flight, split: "val" }));
      serve({ [INDEX_PATH]: stageAIndex(), [SAMPLE_PATH]: sample });
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("Set fixture_set cannot be read.")).toBeTruthy();
      expect(screen.getByText(/split is "val", not one of train, select/)).toBeTruthy();
      expect(lastPublished()).toBeNull();
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
