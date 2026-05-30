import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { toggleLayer, setPlaybackSpeed, setActiveAirportCode } = vi.hoisted(() => ({
  toggleLayer: vi.fn(),
  setPlaybackSpeed: vi.fn(),
  setActiveAirportCode: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: null,
    layers: {
      satelliteImagery: true,
      terrain: false,
      dsmTerrain: true,
      runways: true,
      waypoints: false,
      ocsSurfaces: true,
      trajectories: true,
      obstacles: true,
      obstacleLabels: false,
      procedures: true,
    },
    toggleLayer,
    playbackSpeed: 60,
    setPlaybackSpeed,
    airports: [
      { code: "KRDU", name: "Raleigh-Durham International Airport", lat: 35.878659, lon: -78.7873 },
      { code: "CYVR", name: "Vancouver International Airport", lat: 49.193901, lon: -123.183998 },
    ],
    activeAirportCode: "KRDU",
    setActiveAirportCode,
  }),
}));

import ControlPanel from "../ControlPanel";

describe("ControlPanel", () => {
  it("switches the active airport from the selector", () => {
    render(<ControlPanel />);

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "CYVR" },
    });

    expect(setActiveAirportCode).toHaveBeenCalledWith("CYVR");
  });

  it("toggles obstacle labels independently", () => {
    render(<ControlPanel />);

    const checkbox = screen.getByLabelText("Obstacle Labels") as HTMLInputElement;
    expect(checkbox.checked).toBe(false);

    fireEvent.click(checkbox);

    expect(toggleLayer).toHaveBeenCalledWith("obstacleLabels");
  });

  it("places satellite imagery before terrain in the layer toggles", () => {
    render(<ControlPanel />);

    const layerLabels = screen.getAllByRole("checkbox").map((checkbox) => {
      return checkbox.closest("label")?.textContent;
    });

    expect(layerLabels.indexOf("Satellite Imagery")).toBeLessThan(
      layerLabels.indexOf("Terrain"),
    );
  });
});
