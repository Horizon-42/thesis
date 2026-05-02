import type { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppProvider, useApp } from "../AppContext";

const fetchMock = vi.fn();

function jsonResponse(body: unknown) {
  return {
    ok: true,
    headers: { get: () => "application/json" },
    text: async () => JSON.stringify(body),
  };
}

describe("AppContext", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(
      jsonResponse({
        defaultAirport: "KRDU",
        airports: [
          {
            code: "CYVR",
            name: "Vancouver International Airport",
            lat: 49.193901,
            lon: -123.183998,
          },
          {
            code: "KRDU",
            name: "Raleigh-Durham International Airport",
            lat: 35.878659,
            lon: -78.7873,
          },
        ],
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resets airport-scoped selection state when switching airports", async () => {
    const wrapper = ({ children }: { children: ReactNode }) => (
      <AppProvider>{children}</AppProvider>
    );

    const { result } = renderHook(() => useApp(), { wrapper });

    await waitFor(() => expect(result.current.activeAirportCode).toBe("KRDU"));

    const viewer = {
      trackedEntity: { id: "flight-1" },
      isDestroyed: () => false,
    } as any;

    act(() => {
      result.current.setViewer(viewer);
      result.current.setAirport({ code: "KRDU", lon: -78.7873, lat: 35.878659, height: 15000 });
      result.current.setSelectedFlightId("flight-1");
      result.current.setProcedureBranchVisible("branch:R", true);
      result.current.setProcedureDisplayLevel("DEBUG");
      result.current.setPlaybackSpeed(120);
    });

    act(() => {
      result.current.setActiveAirportCode("CYVR");
    });

    expect(result.current.activeAirportCode).toBe("CYVR");
    expect(result.current.selectedFlightId).toBeNull();
    expect(result.current.procedureVisibility).toEqual({});
    expect(result.current.procedureDisplayLevel).toBe("PROTECTION");
    expect(result.current.airport).toBeNull();
    expect(result.current.playbackSpeed).toBe(120);
    expect(result.current.layers.runways).toBe(true);
    expect(viewer.trackedEntity).toBeUndefined();
  });
});
