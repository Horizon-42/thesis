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
      terrain: false,
      dsmTerrain: false,
      runways: true,
      waypoints: false,
      ocsSurfaces: true,
      trajectories: true,
      obstacles: true,
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
});
