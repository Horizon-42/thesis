import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const { appState } = vi.hoisted(() => ({
  appState: {
    layers: { airportLocalTerrain: true },
    airportLocalTerrain: {
      status: "missing",
      airportCode: "KRDU",
      sourceLabel: null,
      minimumHeightM: null,
      maximumHeightM: null,
      loadedTiles: 0,
      totalTiles: 0,
      error: null,
    },
  } as any,
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => appState,
}));

import AirportLocalTerrainAlert from "../AirportLocalTerrainAlert";

describe("AirportLocalTerrainAlert", () => {
  it("opens a dialog when local terrain data is missing", () => {
    appState.layers.airportLocalTerrain = true;
    appState.airportLocalTerrain = {
      ...appState.airportLocalTerrain,
      status: "missing",
      airportCode: "KRDU",
      error: null,
    };

    render(<AirportLocalTerrainAlert />);

    expect(screen.getByRole("alertdialog").textContent).toContain("KRDU");
    expect(screen.queryByText(/metadata\.json/)).not.toBeNull();
  });

  it("shows precision regeneration guidance for terrain load errors", () => {
    appState.layers.airportLocalTerrain = true;
    appState.airportLocalTerrain = {
      ...appState.airportLocalTerrain,
      status: "error",
      airportCode: "KSJC",
      error: "missing precision.horizontalResolutionM",
    };

    render(<AirportLocalTerrainAlert />);

    expect(screen.getByRole("alertdialog").textContent).toContain("regenerate");
    expect(screen.queryByText("missing precision.horizontalResolutionM")).not.toBeNull();
  });

  it("can be dismissed for the current airport and error", () => {
    appState.layers.airportLocalTerrain = true;
    appState.airportLocalTerrain = {
      ...appState.airportLocalTerrain,
      status: "missing",
      airportCode: "CYVR",
      error: null,
    };

    render(<AirportLocalTerrainAlert />);

    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));

    expect(screen.queryByRole("alertdialog")).toBeNull();
  });
});
