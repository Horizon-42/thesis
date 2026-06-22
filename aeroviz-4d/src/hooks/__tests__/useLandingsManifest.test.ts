import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useLandingsManifest } from "../useLandingsManifest";

const fetchMock = vi.fn();

function jsonResponse(body: unknown) {
  return { ok: true, headers: { get: () => "application/json" }, text: async () => JSON.stringify(body) };
}

function notFound() {
  return { ok: false, status: 404, headers: { get: () => "application/json" }, text: async () => "" };
}

const MANIFEST = {
  airport: "KRDU",
  combined: "trajectories.czml",
  runways: [
    { runway: "23R", file: "landings/KRDU_23R.czml", count: 40 },
    { runway: "05L", file: "landings/KRDU_05L.czml", count: 12 },
  ],
};

describe("useLandingsManifest", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockReset();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("loads a valid manifest as ready", async () => {
    fetchMock.mockResolvedValue(jsonResponse(MANIFEST));

    const { result } = renderHook(() => useLandingsManifest("KRDU"));

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.manifest?.runways.map((r) => r.runway)).toEqual(["23R", "05L"]);
    expect(fetchMock).toHaveBeenCalledWith("/data/airports/KRDU/landings/index.json");
  });

  it("reports a missing manifest as empty, not error", async () => {
    fetchMock.mockResolvedValue(notFound());

    const { result } = renderHook(() => useLandingsManifest("KMSY"));

    await waitFor(() => expect(result.current.status).toBe("empty"));
    expect(result.current.manifest).toBeNull();
  });

  it("is idle with no airport selected", () => {
    const { result } = renderHook(() => useLandingsManifest(""));
    expect(result.current.status).toBe("idle");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
