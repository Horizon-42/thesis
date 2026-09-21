/**
 * TrainingPanel: the empty state (T1) and the three states of design §4.5 that
 * arrived with the reader (T4a).
 *
 * The states are kept apart deliberately: "not exported yet" is the normal state
 * on a fresh machine and must stay actionable (path + command + the vite
 * restart, AV5), while "exported but wrong" must name the field — and must not
 * take the other sets down with it (AV6, in reverse).
 */
import { beforeEach, describe, expect, it, vi, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const { appState, setTrainingSelection, setTrainingLayer, fetchMock } = vi.hoisted(() => ({
  appState: {
    activeAirportCode: "KRDU" as string,
    trainingLayers: { flown: true, model: true },
  },
  setTrainingSelection: vi.fn(),
  setTrainingLayer: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingSelection, setTrainingLayer }),
}));

import TrainingPanel from "../TrainingPanel";
import { trainingIndexPath } from "../../data/trainingSample";
import { mockIndex, mockSample } from "../../data/__tests__/trainingSample.fixture";

function jsonResponse(body: unknown) {
  return {
    ok: true,
    headers: { get: () => "application/json" },
    text: async () => JSON.stringify(body),
  };
}

function notFound() {
  return { ok: false, status: 404, headers: { get: () => "text/html" }, text: async () => "<!doctype html>" };
}

/** Serve the manifest and the sample from a per-test table, by path. */
function serve(files: Record<string, unknown>) {
  fetchMock.mockImplementation(async (url: string) =>
    url in files ? jsonResponse(files[url]) : notFound(),
  );
}

/** The most recent thing the panel published for the sentence bar. */
function lastPublished(): any {
  const calls = setTrainingSelection.mock.calls;
  return calls.length ? calls[calls.length - 1][0] : null;
}

const INDEX_PATH = "data/airports/KRDU/training/index.json";
const SAMPLE_PATH = "data/airports/KRDU/training/vocabulary_tau10/sample.json";

describe("TrainingPanel", () => {
  beforeEach(() => {
    appState.activeAirportCode = "KRDU";
    setTrainingSelection.mockClear();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // ── ① nothing exported ────────────────────────────────────────────────────
  describe("with no export on disk", () => {
    beforeEach(() => serve({}));

    it("says there is no export yet, for the ACTIVE airport", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText(/No Training export for KRDU yet/)).toBeTruthy();
    });

    it("names the exact path it reads, so the message is actionable", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText(INDEX_PATH)).toBeTruthy();
    });

    it("names the command that writes it", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText(/run_ts\.py instruction_sample_export/)).toBeTruthy();
    });

    // AV5: vite does not watch public/data, so a directory created after the dev
    // server booted is served as the SPA fallback. Without this line the first
    // person to export hits a "received HTML" error about a file that is on disk.
    it("warns that the dev server must be restarted after the first export", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText(/restart the dev server/i)).toBeTruthy();
      expect(screen.getByText(/npm run dev/)).toBeTruthy();
    });

    it("follows the active airport rather than hardcoding one", async () => {
      appState.activeAirportCode = "KSJC";
      render(<TrainingPanel />);
      expect(await screen.findByText(/No Training export for KSJC yet/)).toBeTruthy();
      expect(screen.getByText("data/airports/KSJC/training/index.json")).toBeTruthy();
    });

    it("exports the path helper the reader shares with the message", () => {
      expect(trainingIndexPath("KSTL")).toBe("data/airports/KSTL/training/index.json");
    });
  });

  // ── a good export ─────────────────────────────────────────────────────────
  describe("with an export", () => {
    beforeEach(() => serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample() }));

    // The flight list and the set's title are always out; everything read ONCE —
    // what the module is, the shas, the word counts — folds behind the ⓘ so the
    // list keeps the height the sentence bar would otherwise take.
    it("lists the flights without making the reader open anything", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText("DAL123")).toBeTruthy();
      expect(screen.queryByText("segment-v14")).toBeNull();
    });

    it("shows the vocabulary the words were read under, behind the ⓘ", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText("DAL123")).toBeTruthy();
      fireEvent.click(screen.getByRole("button", { name: /What does this panel show/ }));
      expect(screen.getByText("segment-v14")).toBeTruthy();
      // the runway classes come from the file, never from the airport's runways
      expect(screen.getByText("05L 05R 23L 23R")).toBeTruthy();
      expect(screen.getByText(/heading 72 · vertical 6 · speed 16 · runway 4/)).toBeTruthy();
    });

    // A word is a BAND: the panel lists the vertical words with their tolerances
    // and the speed range with its percentage, because a panel that gave only
    // the centres would be stating half of what the vocabulary says (§5.6).
    it("states the tolerance each word carries, not only its centre", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText("DAL123")).toBeTruthy();
      fireEvent.click(screen.getByRole("button", { name: /What does this panel show/ }));
      expect(screen.getByText(/↑3\.0°±0\.21 · level±0\.10 · ↓1\.4°±0\.10/)).toBeTruthy();
      expect(screen.getByText(/44…157 m\/s, ±3 %/)).toBeTruthy();
      expect(screen.getByText(/not the joint envelope/)).toBeTruthy();
    });

    // The spec's sha and the runway classes' sha are separate fields: two
    // artefacts with the same spec can carry different runway lists (§4.4-2).
    it("shows the vocabulary sha and the runway sha apart", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText("DAL123")).toBeTruthy();
      fireEvent.click(screen.getByRole("button", { name: /What does this panel show/ }));
      expect(screen.getByText("c7a4f4239f52…")).toBeTruthy();
      expect(screen.getByText("aa11bb22cc33…")).toBeTruthy();
    });

    it("publishes the selected flight for the sentence bar to draw", async () => {
      render(<TrainingPanel />);
      await waitFor(() => {
        const published = lastPublished();
        expect(published?.flight?.flightKey).toBe("DAL123_05L_a1b2c3_1699999999");
        expect(published?.vocabulary?.readingRule).toBe("segment-v14");
        // the views that draw the corridor need the rule it was drawn under
        expect(published?.geometry?.bandsAreJoint).toBe(false);
      });
    });

    // ② the second layer has no artefact until stage B3′
    it("says why there is no prior-generated set rather than leaving a blank", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText("DAL123")).toBeTruthy();
      fireEvent.click(screen.getByRole("button", { name: /What does this panel show/ }));
      expect(screen.getByText(/No prior-generated set for KRDU/)).toBeTruthy();
      expect(screen.getByText(/stage B3′/)).toBeTruthy();
    });
  });

  // ── ③ one bad set, and one bad manifest entry ─────────────────────────────
  it("names the field when a set's sample is wrong, and keeps the panel up", async () => {
    const broken = mockSample() as any;
    // one row per event still, but two events at the same instant
    broken.flights[0].sentence.eventTimesS = [0, 26, 26, 108, 130, 188, 222];
    serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: broken });

    render(<TrainingPanel />);
    expect(await screen.findByText(/Set vocabulary_tau10 cannot be read/)).toBeTruthy();
    expect(screen.getByText(/strictly increasing/)).toBeTruthy();
    // the panel itself is still there, with its heading
    expect(screen.getByRole("heading", { name: "Training" })).toBeTruthy();
  });

  // AV6 in reverse: one rejected entry names itself, the good set still loads.
  it("greys a rejected manifest entry without emptying the list", async () => {
    const index = mockIndex() as any;
    index.sets.push({ ...index.sets[0], id: "broken_set", kind: "not-a-kind" });
    serve({ [INDEX_PATH]: index, [SAMPLE_PATH]: mockSample() });

    render(<TrainingPanel />);
    expect(await screen.findByText(/Set broken_set was rejected/)).toBeTruthy();
    expect(screen.getByText(/not-a-kind/)).toBeTruthy();
    expect(await screen.findByText("DAL123")).toBeTruthy();
  });

  it("reports a manifest that is not a manifest, by field", async () => {
    serve({ [INDEX_PATH]: { schema: "something-else", airport: "KRDU", sets: [] } });
    render(<TrainingPanel />);
    expect(await screen.findByText(new RegExp(`${INDEX_PATH} cannot be read`))).toBeTruthy();
    expect(screen.getByText(/schema is "something-else"/)).toBeTruthy();
  });

  it("selects another flight when its row is clicked", async () => {
    const sample = mockSample() as any;
    const second = structuredClone(sample.flights[0]);
    second.flightKey = "AAL456_23R_b2c3d4_1700000000";
    second.callsign = "AAL456";
    second.runway = "23R";
    second.stratum = "straight-in";
    sample.flights.push(second);
    serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: sample });

    render(<TrainingPanel />);
    fireEvent.click(await screen.findByText("AAL456"));
    await waitFor(() => {
      expect(lastPublished()?.flight?.callsign).toBe("AAL456");
    });
  });
});

// ── the two switches, and the experiment picker (2026-09-21) ────────────────

describe("TrainingPanel's switches", () => {
  beforeEach(() => {
    appState.activeAirportCode = "KRDU";
    setTrainingLayer.mockClear();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
    serve({ [INDEX_PATH]: mockIndex(), [SAMPLE_PATH]: mockSample() });
  });
  afterEach(() => vi.unstubAllGlobals());

  // The observed track has NO switch: it is the aircraft that was actually
  // there, and every other line is read against it.
  it("offers a switch for each line a sentence draws, and none for the measured one", async () => {
    render(<TrainingPanel />);
    expect(await screen.findByText("DAL123")).toBeTruthy();
    expect(screen.getByLabelText(/the words flown by rule/)).toBeTruthy();
    expect(screen.getByLabelText(/what the model said/)).toBeTruthy();
    expect(screen.queryByLabelText(/measured/)).toBeNull();
  });

  it("switches a line off through the shared state, not its own", async () => {
    render(<TrainingPanel />);
    expect(await screen.findByText("DAL123")).toBeTruthy();
    fireEvent.click(screen.getByLabelText(/the words flown by rule/));
    expect(setTrainingLayer).toHaveBeenCalledWith("flown", false);
  });

  // A set with no model has nothing to switch; the box says so rather than
  // toggling a line that does not exist.
  // The question is about the set that is OPEN, not about the airport: a switch
  // enabled by a set nobody is looking at toggles a line that is not there.
  it("disables the model switch when the OPEN set carries none", async () => {
    const index = mockIndex() as any;
    index.sets.push({ ...index.sets[0], id: "a_prior_set", kind: "prior-generated",
                      prior: { sha256: "abc", seed: 1, method: "teacher-forced-next-word" } });
    serve({ [INDEX_PATH]: index, [SAMPLE_PATH]: mockSample() });

    render(<TrainingPanel />);
    expect(await screen.findByText("DAL123")).toBeTruthy();
    // the open set is the read-back one, even though the manifest holds a prior set
    expect(screen.getByLabelText(/what the model said/)).toHaveProperty("disabled", true);
  });
});

describe("a manifest holding a superseded set", () => {
  beforeEach(() => {
    appState.activeAirportCode = "KRDU";
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  // A vocabulary bump leaves older sets listed — they are real exports, and the
  // panel says why they cannot be read when one is picked. What it must not do
  // is OPEN on one: the manifest states each set's reading rule, so which are
  // current is known before any sample is fetched.
  it("opens on a set this reader can read, not on the first by id", async () => {
    const index = mockIndex() as any;
    const stale = { ...index.sets[0], id: "aaa_older", readingRule: "segment-v13" };
    index.sets = [stale, index.sets[0]];
    serve({ [INDEX_PATH]: index, [SAMPLE_PATH]: mockSample() });

    render(<TrainingPanel />);
    expect(await screen.findByText("DAL123")).toBeTruthy();
    expect(screen.queryByText(/cannot be read/)).toBeNull();
  });

  it("says which sets are superseded, in the picker itself", async () => {
    const index = mockIndex() as any;
    index.sets = [{ ...index.sets[0], id: "aaa_older", readingRule: "segment-v13" }, index.sets[0]];
    serve({ [INDEX_PATH]: index, [SAMPLE_PATH]: mockSample() });

    render(<TrainingPanel />);
    expect(await screen.findByText("DAL123")).toBeTruthy();
    expect(screen.getByRole("option", { name: /aaa_older.*segment-v13, superseded/ })).toBeTruthy();
  });
});
