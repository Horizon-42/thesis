import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, rangeRing, toggleLayer, setRangeRingRadiusKm } = vi.hoisted(() => {
  const defaultLayers = {
    satelliteImagery: true,
    terrain: false,
    airportLocalTerrain: false,
    terrainHillshade: false,
    terrainHeightTint: false,
    runways: true,
    waypoints: false,
    trajectories: false,
    obstacles: false,
    obstacleLabels: false,
    procedures: false,
    rangeRing: false,
  };
  const defaultAirportLocalTerrain = {
    status: "disabled",
    airportCode: "KRDU",
    sourceLabel: null,
    sourceKind: null,
    sourceName: null,
    horizontalResolutionM: null,
    sourceCrsCode: null,
    sourceCrsName: null,
    minimumHeightM: null,
    maximumHeightM: null,
    error: null,
  };
  return {
    appState: { layers: { ...defaultLayers }, airportLocalTerrain: { ...defaultAirportLocalTerrain }, defaultLayers } as any,
    // the ring's radius is its own context (it moves on every slider step), read through its own hook
    rangeRing: { radiusKm: 5 },
    toggleLayer: vi.fn(),
    setRangeRingRadiusKm: vi.fn(),
  };
});

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, toggleLayer, setRangeRingRadiusKm }),
  useRangeRingRadiusKm: () => rangeRing.radiusKm,
}));

import HudLayers from "../HudLayers";

const expand = () => fireEvent.click(screen.getByRole("button", { name: /Layers/ }));

describe("HudLayers", () => {
  beforeEach(() => {
    appState.layers = { ...appState.defaultLayers };
    appState.airportLocalTerrain = { ...appState.airportLocalTerrain, status: "disabled" };
    vi.clearAllMocks();
  });

  it("starts collapsed, with the number of layers that are on", () => {
    render(<HudLayers />);
    const head = screen.getByRole("button", { name: /Layers/ });
    expect(head.getAttribute("aria-expanded")).toBe("false");
    expect(head.textContent).toContain("2/9");
    expect(screen.queryByLabelText("Imagery")).toBeNull();
  });

  it("does not include the RNAV procedures toggle (it lives in the Procedure panel)", () => {
    render(<HudLayers />);
    expand();
    expect(screen.queryByLabelText("RNAV Procedures")).toBeNull();
    expect(screen.getAllByRole("checkbox")).toHaveLength(9);
  });

  it("has no legacy FAF OCS debug toggle", () => {
    render(<HudLayers />);
    expand();
    expect(screen.queryByLabelText(/Legacy FAF/i)).toBeNull();
  });

  it("toggles obstacle labels independently", () => {
    render(<HudLayers />);
    expand();
    const checkbox = screen.getByLabelText("Labels") as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
    fireEvent.click(checkbox);
    expect(toggleLayer).toHaveBeenCalledWith("obstacleLabels");
  });

  it("places satellite imagery before terrain in the layer toggles", () => {
    render(<HudLayers />);
    expand();
    const labels = screen.getAllByRole("checkbox").map((box) => box.closest("label")?.textContent);
    expect(labels.indexOf("Imagery")).toBeLessThan(labels.indexOf("Terrain"));
  });

  it("shows no range-ring radius while the ring is off", () => {
    render(<HudLayers />);
    expand();
    expect(screen.queryByLabelText("Range ring radius")).toBeNull();
  });

  it("commits a radius typed into the ring's number field, clamped to 1–50 km", () => {
    appState.layers.rangeRing = true;
    render(<HudLayers />);
    expand();
    const number = screen.getByRole("spinbutton") as HTMLInputElement;
    expect(number.value).toBe("5");
    fireEvent.change(number, { target: { value: "80" } });
    expect(setRangeRingRadiusKm).toHaveBeenLastCalledWith(50);
    fireEvent.change(number, { target: { value: "0.2" } });
    expect(setRangeRingRadiusKm).toHaveBeenLastCalledWith(1);
  });

  it("summarises the local terrain's source on one line while it is on", () => {
    appState.layers.airportLocalTerrain = true;
    appState.airportLocalTerrain = {
      status: "active",
      airportCode: "KRDU",
      sourceLabel: "Airport local heightmap terrain",
      sourceKind: "dsm",
      sourceName: "USGS TNM DSM",
      horizontalResolutionM: 2,
      sourceCrsCode: "EPSG:26917",
      sourceCrsName: "EPSG:26917 / UTM zone 17 projected metres",
      minimumHeightM: 89,
      maximumHeightM: 243,
      error: null,
    };
    render(<HudLayers />);
    expand();
    expect(screen.getByText("DSM (USGS TNM DSM) · 2 m spacing · EPSG:26917")).toBeTruthy();
  });
});
