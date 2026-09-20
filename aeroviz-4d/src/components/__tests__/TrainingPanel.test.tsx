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

const { appState, setTrainingSelection, fetchMock } = vi.hoisted(() => ({
  appState: { activeAirportCode: "KRDU" as string },
  setTrainingSelection: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingSelection }),
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
      expect(screen.queryByText("plateau-v11")).toBeNull();
    });

    it("shows the vocabulary the words were read under, behind the ⓘ", async () => {
      render(<TrainingPanel />);
      expect(await screen.findByText("DAL123")).toBeTruthy();
      fireEvent.click(screen.getByRole("button", { name: /What does this panel show/ }));
      expect(screen.getByText("plateau-v11")).toBeTruthy();
      // the runway classes come from the file, never from the airport's runways
      expect(screen.getByText("05L 05R 23L 23R")).toBeTruthy();
      expect(screen.getByText(/heading 36 · altitude 11 · speed 23 · runway 4/)).toBeTruthy();
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
        expect(published?.vocabulary?.readingRule).toBe("plateau-v11");
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
