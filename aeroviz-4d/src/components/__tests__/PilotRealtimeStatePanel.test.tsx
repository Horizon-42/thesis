import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import PilotRealtimeStatePanel from "../PilotRealtimeStatePanel";
import type { PilotSnapshot } from "../../pilot/pilotClient";

const snapshot: PilotSnapshot = {
  ok: true,
  elapsedS: 12.4,
  state: {
    lon: -114.05,
    lat: 51.02,
    altM: 1234,
    speedMps: 121.5,
    headingDeg: -14.2,
    flightPathDeg: 2.25,
    massKg: 78000,
    aircraftType: "A320",
  },
  control: {
    thrustN: 12000,
    bankDeg: 0,
    attackDeg: 3.5,
  },
  aero: {
    liftCoefficient: 0.412,
    dragCoefficient: 0.038,
    actualLoadFactor: 1.234,
  },
};

describe("PilotRealtimeStatePanel", () => {
  beforeEach(() => {
    document.body.innerHTML = '<div class="cesium-overlay-container"></div>';
  });

  it("renders the realtime readouts into the overlay container", async () => {
    const { container } = render(
      <PilotRealtimeStatePanel snapshot={snapshot} visible={true} />,
    );

    const panel = await screen.findByRole("complementary", {
      name: "Realtime aircraft state",
    });
    const overlay = document.querySelector(".cesium-overlay-container");

    expect(overlay?.contains(panel)).toBe(true);
    expect(container.contains(panel)).toBe(false);
    expect(screen.getByText("Live State")).toBeTruthy();
    expect(screen.getByText("Altitude")).toBeTruthy();
    expect(screen.getByText("Heading Angle (psi)")).toBeTruthy();
    expect(screen.getByText("Flight Path Angle (gamma)")).toBeTruthy();
    expect(screen.getByText("Attack Angle (alpha)")).toBeTruthy();
    expect(screen.getByText("Actual n")).toBeTruthy();
    expect(screen.getByText("Drag Coefficient")).toBeTruthy();
    expect(screen.getByText("1234 m")).toBeTruthy();
    expect(screen.getByText("-14.2 deg")).toBeTruthy();
    expect(screen.getByText("3.50 deg")).toBeTruthy();
    expect(screen.getByText("1.234 g")).toBeTruthy();
  });

  it("stays hidden when simulation is not flying", () => {
    render(<PilotRealtimeStatePanel snapshot={snapshot} visible={false} />);

    expect(
      screen.queryByRole("complementary", { name: "Realtime aircraft state" }),
    ).toBeNull();
  });

  it("can add a separate trajectory control readout row", async () => {
    render(
      <PilotRealtimeStatePanel
        snapshot={snapshot}
        visible={true}
        showControlReadout={true}
      />,
    );

    expect(await screen.findByText("Control")).toBeTruthy();
    expect(screen.getByText("bank 0.0 deg | alpha 3.50 deg | thrust 12000 N")).toBeTruthy();
  });

  it("renders load-factor control readouts when requested", async () => {
    render(
      <PilotRealtimeStatePanel
        snapshot={{
          ...snapshot,
          control: {
            thrustN: 12000,
            bankDeg: 0,
            attackDeg: 0,
            loadFactor: 1.2,
          },
        }}
        visible={true}
        showControlReadout={true}
        simulationMode="loadFactor"
      />,
    );

    expect(await screen.findByText("Load Factor")).toBeTruthy();
    expect(screen.getByText("1.20 g")).toBeTruthy();
    expect(screen.getByText("bank 0.0 deg | n 1.20 | thrust 12000 N")).toBeTruthy();
    expect(screen.queryByText("Attack Angle (alpha)")).toBeNull();
  });

  it("overlays A/C/D deviations vs B in compare mode, keeping B as the main value", async () => {
    const systems = [
      { key: "A", label: "A · fixed tangent", colorRgba: [244, 114, 22, 240] as [number, number, number, number], isReference: false },
      { key: "B", label: "B · reference", colorRgba: [226, 232, 240, 245] as [number, number, number, number], isReference: true },
      { key: "C", label: "C · geodetic", colorRgba: [56, 189, 248, 240] as [number, number, number, number], isReference: false },
      { key: "D", label: "D · no transport", colorRgba: [250, 204, 21, 240] as [number, number, number, number], isReference: false },
    ];
    render(
      <PilotRealtimeStatePanel
        snapshot={snapshot}
        visible={true}
        showControlReadout={true}
        simulationMode="casadi"
        comparisonDeltas={{
          A: { horiz: 335.2, alt: -3.4, head: 0.14, speed: 0.2, fpa: 0.27 },
          C: { horiz: 0.03, alt: 0, head: 0, speed: 0, fpa: 0 },
          D: { horiz: 145.6, alt: -5.1, head: 0.08, speed: -0.1, fpa: -0.12 },
        }}
        comparisonSystems={systems}
      />,
    );

    // The main value is still the reference B's state (altitude here).
    expect(await screen.findByText("1234 m")).toBeTruthy();
    // The new horizontal-error row appears (B is the zero reference).
    expect(screen.getByText("Horiz Err (vs B)")).toBeTruthy();

    const chips = Array.from(
      document.querySelectorAll<HTMLElement>(".pilot-realtime-delta"),
    );
    // Altitude row carries the signed A delta (-3.4), tinted with A's colour.
    const altA = chips.find((c) => c.textContent?.includes("-3.4"));
    expect(altA?.textContent).toContain("A");
    expect(altA?.style.color).toBe("rgb(244, 114, 22)");

    // Horizontal deltas are magnitudes (no sign): A 335.2, D 145.6.
    expect(chips.find((c) => c.textContent?.includes("335.2"))?.textContent).toContain("A");
    expect(chips.find((c) => c.textContent?.includes("145.6"))?.textContent).toContain("D");

    // Flight-path-angle deltas are signed: A +0.27, D -0.12.
    expect(chips.find((c) => c.textContent?.includes("+0.27"))?.textContent).toContain("A");
    expect(chips.find((c) => c.textContent?.includes("-0.12"))?.textContent).toContain("D");

    // B is the reference, so it never gets a delta chip.
    expect(chips.some((c) => c.title === "B · reference")).toBe(false);
  });

  it("does not render delta chips or the horiz-error row outside compare mode", () => {
    render(<PilotRealtimeStatePanel snapshot={snapshot} visible={true} />);
    expect(screen.queryByText("Horiz Err (vs B)")).toBeNull();
    expect(document.querySelector(".pilot-realtime-delta")).toBeNull();
  });

  it("shows actual minus target position errors when a target state is provided", async () => {
    render(
      <PilotRealtimeStatePanel
        snapshot={snapshot}
        visible={true}
        targetState={{ lat: 51, lon: -114.1, altM: 1200 }}
      />,
    );

    expect(await screen.findByText("Lat Error")).toBeTruthy();
    expect(screen.getByText("+0.020000 deg")).toBeTruthy();
    expect(screen.getByText("+0.050000 deg")).toBeTruthy();
    expect(screen.getByText("+34.0 m")).toBeTruthy();
  });
});
