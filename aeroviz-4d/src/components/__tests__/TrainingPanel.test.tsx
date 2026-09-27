/**
 * TrainingPanel: the empty state, the readable set, the sets it refuses BY NAME from the manifest
 * alone (never downloaded), a readable set that fails with its field, and the overlays drawn over the
 * set — the executor's replay and the prior's predictions — or, without them, the switches that say so.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";

const {
  appState, setTrainingSelection, setTrainingLayer, setTrainingExecutor, setTrainingPrior, setTrainingAutopilot,
  setTrainingAutopilotAuto, setTrainingGenerations, fetchMock,
} = vi.hoisted(() => ({
  appState: {
    activeAirportCode: "KXXX" as string,
    trainingLayers: { headingBands: true, corridor: true, vertical: true, candidates: true },
    // no word selected: the live executor asks for nothing
    trainingSelection: null, trainingColumn: null, trainingAutopilot: null, trainingPick: null,
    trainingAutopilotAuto: true, trainingSource: null as unknown,
    // the models' sentences the panel publishes go to the mocked setter: the card reads none
    trainingGenerations: [] as unknown[],
  },
  setTrainingGenerations: vi.fn(),
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
    setTrainingAutopilotAuto, setTrainingGenerations,
  }),
}));

import TrainingPanel from "../TrainingPanel";
import { SET_ID, STRAIGHT_KEY, VECTORED_KEY, mockIndex, mockSample } from "../../data/__tests__/trainingSample.fixture";
import {
  AUGSTART_BASE_ID, AUGSTART_POST_ID, BASE_MODEL_ID, EXECUTOR_ID, POST_TRAINED_ID, PRIOR_ID, mockExecutorOverlay, mockGenerationOverlay, mockOverlays,
  mockOverlaysWithGenerations, mockPriorOverlay,
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
    appState.trainingSource = null;
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
      expect(screen.getByText(/run_ts\.py instruction_training_export .*--airport KXXX/)).toBeTruthy();
    });

    // AV5: vite does not watch public/data, so a directory created after boot is the SPA fallback.
    it("warns that the dev server must be restarted after the first export", async () => {
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText(/restart the dev server/i)).toBeTruthy();
    });
  });

  describe("with an export", () => {
    beforeEach(() => serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample() }));

    it("keeps its session while hidden in another task: the panel hides, nothing is torn down", async () => {
      const { container, rerender } = render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("TST1")).toBeTruthy();
      rerender(<TrainingPanel hidden />);
      expect((container.querySelector("section.training-panel") as HTMLElement).hidden).toBe(true);
      expect(setTrainingSelection).not.toHaveBeenLastCalledWith(null);
      rerender(<TrainingPanel hidden={false} />);
      expect((container.querySelector("section.training-panel") as HTMLElement).hidden).toBe(false);
      expect(screen.getByText("TST1")).toBeTruthy();
    });

    it("opens on the set it can read, not the first one listed, and downloads only that", async () => {
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("TST1")).toBeTruthy();
      expect((screen.getByRole("combobox") as HTMLSelectElement).value).toBe(SET_ID);
      const fetched = fetchMock.mock.calls.map(([url]) => url);
      expect(fetched).toContain(SAMPLE_PATH);
      expect(fetched).not.toContain(OLD_PATH);
      expect(fetched).not.toContain(FIRST_PATH);
      expect(fetched).not.toContain(SECOND_PATH);
    });

    it("publishes the flight with the vocabulary and the candidate runways", async () => {
      render(<TrainingPanel hidden={false} />);
      await waitFor(() => expect(lastPublished()?.flight.flightKey).toBe(VECTORED_KEY));
      expect(lastPublished().candidates.map((c: any) => c.ident)).toEqual(["09", "27"]);
      fireEvent.click(screen.getByText("TST2"));
      await waitFor(() => expect(lastPublished()?.flight.flightKey).toBe(STRAIGHT_KEY));
    });

    it("refuses a superseded set BY NAME from the manifest alone, without downloading it", async () => {
      render(<TrainingPanel hidden={false} />);
      await screen.findByText("TST1");
      fireEvent.change(screen.getByRole("combobox"), { target: { value: "box_v3" } });
      expect(await screen.findByText(/read under box-v3, a superseded vocabulary/)).toBeTruthy();
      expect(fetchMock.mock.calls.map(([url]) => url)).not.toContain(OLD_PATH);
      await waitFor(() => expect(lastPublished()).toBeNull());
    });

    it("marks the refused sets in the chooser", async () => {
      render(<TrainingPanel hidden={false} />);
      await screen.findByText("TST1");
      const options = [...(screen.getByRole("combobox") as HTMLSelectElement).options].map((option) => option.text);
      expect(options.find((text) => text.startsWith("box_v3"))).toMatch(/box-v3 — refused/);
      expect(options.find((text) => text.startsWith("instruction_v1"))).toMatch(/instruction-v1 — refused/);
      expect(options.find((text) => text.startsWith("instruction_v2"))).toMatch(/instruction-v2 — refused/);
      expect(options.find((text) => text.startsWith(SET_ID))).not.toMatch(/refused/);
    });

    it("with no overlay published, keeps both switches off and names the commands that write them", async () => {
      render(<TrainingPanel hidden={false} />);
      await screen.findByText("TST1");
      const replay = screen.getByLabelText("Executor replay") as HTMLInputElement;
      const prior = screen.getByLabelText("Prior predictions") as HTMLInputElement;
      expect(replay.disabled && prior.disabled).toBe(true);
      expect(replay.checked || prior.checked).toBe(false);
      // the commands fold away under "none published", under each switch
      expect(screen.getAllByText("none published")).toHaveLength(2);
      expect(screen.getByText(new RegExp(`run_ts\\.py executor_training_export .*--set ${SET_ID} --airport KXXX`)).closest("details")).not.toBeNull();
      expect(screen.getByText(new RegExp(`run_ts\\.py prior_training_export .*--set ${SET_ID} --airport KXXX`))).toBeTruthy();
      await waitFor(() => expect(lastOf(setTrainingExecutor)).toBeNull());
    });

    it("switches the envelopes everywhere through the shared layers", async () => {
      render(<TrainingPanel hidden={false} />);
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

    it("says on its details page what each switch shows, and the vocabulary, as text", async () => {
      render(<TrainingPanel hidden={false} />);
      await screen.findByText("TST1");
      const open = screen.getByRole("button", { name: "Training details" });
      fireEvent.click(open);
      const page = screen.getByRole("dialog", { name: "Training details" });
      const about = page.querySelector(".training-notes-list")!.textContent!;
      expect(about).toMatch(/Altitude tubes \+ speed bandseach altitude word's tube, and on the speed chart each speed word's/);
      expect(about).toMatch(/Fly on band clickOn: clicking a word's band in the sentence bar flies its segment at once/);
      // the sections this set has nothing for stay listed, disabled, and say why
      expect((within(page).getByRole("tab", { name: /The executor's replay gate/ }) as HTMLButtonElement).disabled).toBe(true);
      expect(within(page).getByRole("tab", { name: /The executor's replay gate/ }).textContent).toMatch(/none published for this set/);
      fireEvent.click(within(page).getByRole("tab", { name: "Vocabulary" }));
      expect(within(page).getByRole("tabpanel").textContent).toMatch(/Runway pointer/);
      // Escape closes it, and the focus goes back to what opened it
      fireEvent.keyDown(page, { key: "Escape" });
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(document.activeElement).toBe(open);
    });

    it("states the draw it came from, in full in its tooltip", async () => {
      render(<TrainingPanel hidden={false} />);
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
      render(<TrainingPanel hidden={false} />);
      await waitFor(() => expect(lastOf(setTrainingExecutor)?.flight.flightKey).toBe(VECTORED_KEY));
      await waitFor(() => expect(lastOf(setTrainingPrior)?.flight.flightKey).toBe(VECTORED_KEY));
      fireEvent.click(screen.getByText("TST2"));
      await waitFor(() => expect(lastOf(setTrainingExecutor)?.flight.flightKey).toBe(STRAIGHT_KEY));
      expect(lastOf(setTrainingExecutor).flight.flown).toBe(false);
    });

    it("says under each flight what the executor made of it", async () => {
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("landed · 5/7")).toBeTruthy();
      expect(screen.getByText("not flown")).toBeTruthy();
    });

    it("lists the replay's gate and the prior's readout by their conclusions, and opens each on the details page", async () => {
      render(<TrainingPanel hidden={false} />);
      const links = await screen.findByRole("list", { name: "Readouts" });
      expect((await within(links).findByRole("button", { name: /Replay gate/ })).textContent).toBe("Replay gateval · spec 777777777777›");
      expect(within(links).getByRole("button", { name: /Prior readout/ }).textContent)
        .toBe("Prior readoutval · 0.1778 per step against 0.3444 / 0.3260›");
      // no table in the dock
      expect(document.querySelector(".training-panel table")).toBeNull();

      fireEvent.click(within(links).getByRole("button", { name: /Replay gate/ }));
      let page = screen.getByRole("dialog", { name: "Training details" });
      expect(within(page).getByRole("tab", { selected: true }).textContent).toBe("The executor's replay gate");
      expect(within(page).getByText(/own dynamics/).closest("caption")!.textContent).toMatch(/own dynamics at KXXX\s*gated — each share ≥ 95 %/);
      expect(within(page).getByText(/stand-in dynamics/).closest("caption")!.textContent)
        .toMatch(/stand-in dynamics at KXXX\s*reported, not gated: a stand-in's dynamics/);
      fireEvent.click(within(page).getByRole("button", { name: "Close the details" }));

      fireEvent.click(within(links).getByRole("button", { name: /Prior readout/ }));
      page = screen.getByRole("dialog", { name: "Training details" });
      expect(within(page).getByRole("tab", { selected: true }).textContent).toBe("The prior's readout");
      // the lowest of each row is marked; what a row's tooltip said is a table of its own
      expect(within(page).getByText("Where the truth says a word")).toBeTruthy();
      expect(page.querySelectorAll(".training-results-best").length).toBeGreaterThan(0);
    });

    it("says on the details page to switch an overlay on when its readout is switched off", async () => {
      render(<TrainingPanel hidden={false} />);
      await screen.findByRole("list", { name: "Readouts" });
      fireEvent.click(screen.getByLabelText("Executor replay"));
      await waitFor(() => expect(screen.queryByRole("button", { name: /Replay gate/ })).toBeNull());
      fireEvent.click(screen.getByRole("button", { name: "Training details" }));
      const tab = within(screen.getByRole("dialog")).getByRole("tab", { name: /The executor's replay gate/ }) as HTMLButtonElement;
      expect(tab.disabled).toBe(true);
      expect(tab.title).toBe("switch on Executor replay under Draw");
    });

    it("closes its details page when the panel is hidden, and does not bring it back", async () => {
      const { rerender } = render(<TrainingPanel hidden={false} />);
      fireEvent.click(await within(await screen.findByRole("list", { name: "Readouts" })).findByRole("button", { name: /Prior readout/ }));
      expect(screen.getByRole("dialog")).toBeTruthy();
      rerender(<TrainingPanel hidden />);
      expect(screen.queryByRole("dialog")).toBeNull();
      rerender(<TrainingPanel hidden={false} />);
      expect(screen.queryByRole("dialog")).toBeNull();
    });

    it("stops publishing an overlay switched off", async () => {
      render(<TrainingPanel hidden={false} />);
      await waitFor(() => expect(lastOf(setTrainingExecutor)?.flight.flightKey).toBe(VECTORED_KEY));
      fireEvent.click(screen.getByLabelText("Executor replay"));
      await waitFor(() => expect(lastOf(setTrainingExecutor)).toBeNull());
      expect(lastOf(setTrainingPrior)?.flight.flightKey).toBe(VECTORED_KEY);
    });

    it("never fetches the last airport's overlay under the next airport's path", async () => {
      const { rerender } = render(<TrainingPanel hidden={false} />);
      await waitFor(() => expect(lastOf(setTrainingExecutor)?.flight.flightKey).toBe(VECTORED_KEY));
      appState.activeAirportCode = "KYYY";
      rerender(<TrainingPanel hidden={false} />);
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
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText(`Overlay ${EXECUTOR_ID} cannot be read.`)).toBeTruthy();
      expect(screen.getByText(/the set was re-exported after the overlay/)).toBeTruthy();
      await waitFor(() => expect(lastOf(setTrainingPrior)?.flight.flightKey).toBe(VECTORED_KEY));
    });
  });

  describe("with the models' own sentences over the set", () => {
    const BASE_PATH = `data/airports/KXXX/training/${BASE_MODEL_ID}/generation.json`;
    const POST_PATH = `data/airports/KXXX/training/${POST_TRAINED_ID}/generation.json`;
    beforeEach(() => serve({
      [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample(), [OVERLAYS_PATH]: mockOverlaysWithGenerations(),
      [EXECUTOR_PATH]: mockExecutorOverlay(), [PRIOR_PATH]: mockPriorOverlay(),
      [BASE_PATH]: mockGenerationOverlay(BASE_MODEL_ID), [POST_PATH]: mockGenerationOverlay(POST_TRAINED_ID),
    }));

    it("marks each flight with the chosen model's samples, filled where its flight landed", async () => {
      appState.trainingSource = { overlayId: BASE_MODEL_ID, sample: 1 };
      render(<TrainingPanel hidden={false} />);
      const marks = await screen.findByText("●○");
      expect(marks.getAttribute("title")).toBe("1 of 2 of the model's sentences landed: #1 landed, #2 timed out");
      const unflown = [...document.querySelectorAll(".training-flight-samples")].find((item) => item.textContent === "—");
      expect(unflown?.getAttribute("title")).toBe("the model's sentences do not fly it: stand-in dynamics");
    });

    it("says how each model's samples landed, beside its formal readout here and at every airport, on the details page", async () => {
      render(<TrainingPanel hidden={false} />);
      const link = await screen.findByText(/^base 50% \(val 90\.0%\) · landing r1 50% \(val 97\.0%\)$/);
      fireEvent.click(link);
      const page = screen.getByRole("dialog", { name: "Training details" });
      expect(within(page).getAllByText("val · KXXX")).toHaveLength(2);
      expect(within(page).getAllByText("val · all airports")).toHaveLength(2);
      // what every model shares is said once; what differs is a column
      const shared = within(page).getByLabelText("Shared by every model").textContent!;
      expect(shared).toMatch(/samples a flight2/);
      expect(shared).toMatch(/procedure masksnone \(the vocabulary's rules alone\)/);
      const base = within(page).getAllByRole("row").find((row) => row.querySelector("th[scope='row']")?.textContent === "base")!;
      expect(base.textContent).toMatch(/v3_step1\/full_s1/);
      expect(base.textContent).toMatch(/trained on data alone/);
    });

    it("counts the models flown from augmented starts apart: in the readout line and on the details page", async () => {
      const path = (id: string) => `data/airports/KXXX/training/${id}/generation.json`;
      serve({
        [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample(),
        [OVERLAYS_PATH]: mockOverlaysWithGenerations([BASE_MODEL_ID, POST_TRAINED_ID, AUGSTART_BASE_ID, AUGSTART_POST_ID]),
        ...Object.fromEntries([BASE_MODEL_ID, POST_TRAINED_ID, AUGSTART_BASE_ID, AUGSTART_POST_ID]
          .map((id) => [path(id), mockGenerationOverlay(id)])),
      });
      // a model read from an augmented start (chosen in the sentence bar): each flight shows its samples
      appState.trainingSource = { overlayId: AUGSTART_BASE_ID, sample: 0 };
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText("●○")).toBeTruthy();
      // the straight-in flight is not on its own dynamics: never drawn
      const unflown = [...document.querySelectorAll(".training-flight-samples")].find((item) => item.textContent === "—");
      expect(unflown?.getAttribute("title")).toBe("the model's sentences do not fly it: stand-in dynamics");
      // the readout line: the set's own starts, then the augmented ones; the details page: a table each
      const link = await screen.findByText(/^base 50% \(val 90\.0%\) · landing r1 50% \(val 97\.0%\) · augmented starts: base 50% · landing r1 50%$/);
      fireEvent.click(link);
      const page = screen.getByRole("dialog", { name: "Training details" });
      expect(within(page).getByText("From augmented starts")).toBeTruthy();
      expect(within(page).getByText(/Every flight on its own dynamics found a plausible start within 10 draws\./)).toBeTruthy();
      expect([...page.querySelectorAll(".training-results-models caption strong")].map((cell) => cell.textContent))
        .toEqual(["Landed", "Landed from augmented starts"]);
      const shared = [...page.querySelectorAll("[aria-label='Shared by every model']")].map((item) => item.textContent!);
      expect(shared[1]).toMatch(/startsaugmented, seed 1337: rotated within ±15°, raised within ±150 m, sped up within ±5%, up to 10 draws/);
      expect(shared[1]).toMatch(/time limit2× the observed remaining time/);
      expect(shared[1]).toMatch(/formal readoutnone \(augmented starts\)/);
    });

    it("drops the column naming whose sentences a row counts when no model has a formal readout", async () => {
      const bare = (id: string) => ({ ...(mockGenerationOverlay(id) as any), readout: null });
      serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample(), [OVERLAYS_PATH]: mockOverlaysWithGenerations(),
              [BASE_PATH]: bare(BASE_MODEL_ID), [POST_PATH]: bare(POST_TRAINED_ID) });
      render(<TrainingPanel hidden={false} />);
      fireEvent.click(await screen.findByText(/^base 50% · landing r1 50%$/));
      const page = screen.getByRole("dialog", { name: "Training details" });
      expect([...page.querySelectorAll(".training-results-models thead th")].map((cell) => cell.textContent))
        .toEqual(["model", "all", "straight-in", "vectored"]);
      expect(within(page).getByText(/no formal readout was given/)).toBeTruthy();
    });

    it("reads a model that flies none of the set's flights without failing: nothing to count", async () => {
      const none: any = mockGenerationOverlay(BASE_MODEL_ID);
      none.flights[0] = { ...none.flights[0], flown: false, group: "stand-in dynamics", samples: [] };
      serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample(), [OVERLAYS_PATH]: mockOverlaysWithGenerations(),
              [BASE_PATH]: none, [POST_PATH]: mockGenerationOverlay(POST_TRAINED_ID) });
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText(/^base — \(val 90\.0%\)/)).toBeTruthy();
    });

    it("says why a model's sentences cannot be read, and asks again on Retry", async () => {
      const broken: any = mockGenerationOverlay(BASE_MODEL_ID);
      broken.schema = "aeroviz-training-generation-v0";
      const files: Record<string, unknown> = {
        [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample(), [OVERLAYS_PATH]: mockOverlaysWithGenerations(),
        [BASE_PATH]: broken, [POST_PATH]: mockGenerationOverlay(POST_TRAINED_ID),
      };
      serve(files);
      render(<TrainingPanel hidden={false} />);
      expect(await screen.findByText(`Overlay ${BASE_MODEL_ID} cannot be read.`)).toBeTruthy();
      files[BASE_PATH] = mockGenerationOverlay(BASE_MODEL_ID);
      serve(files);
      fireEvent.click(screen.getByRole("button", { name: "Retry" }));
      await waitFor(() => expect(screen.queryByText(`Overlay ${BASE_MODEL_ID} cannot be read.`)).toBeNull());
      // read now: counted in the readout line (and offered in the sentence bar's tabs)
      expect(await screen.findByText(/^base 50% \(val 90\.0%\) · landing r1 50% \(val 97\.0%\)$/)).toBeTruthy();
    });
  });

  it("with no model's sentences published, lists no readout of them and says so on the details page", async () => {
    serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample(), [OVERLAYS_PATH]: mockOverlays() });
    render(<TrainingPanel hidden={false} />);
    await screen.findByText("TST1");
    expect(screen.queryByText("Models' sentences")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Training details" }));
    const page = screen.getByRole("dialog", { name: "Training details" });
    const models = within(page).getByRole("tab", { name: /The models' own sentences/ }) as HTMLButtonElement;
    expect(models.disabled).toBe(true);
    expect(models.textContent).toMatch(/none published for this set/);
  });

  it("names the field when a readable set fails, and leaves the manifest standing", async () => {
    const broken: any = mockSample();
    broken.flights[0].envelopes.altitude[0].check.inside = 3;
    serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: broken });
    render(<TrainingPanel hidden={false} />);
    expect(await screen.findByText(new RegExp(`Set ${SET_ID} cannot be read`))).toBeTruthy();
    expect(screen.getByText(/says 3 rows inside, but the tube's own flags count 20/)).toBeTruthy();
    expect(screen.getByRole("combobox")).toBeTruthy();
  });

  it("greys out a malformed entry by name while the others load", async () => {
    const index: any = mockIndex();
    delete index.sets[0].title;
    serve({ [INDEX_PATH]: index, [SAMPLE_PATH]: mockSample() });
    render(<TrainingPanel hidden={false} />);
    expect(await screen.findByText(/Entry box_v3 was rejected/)).toBeTruthy();
    expect(await screen.findByText("TST1")).toBeTruthy();
  });
});
