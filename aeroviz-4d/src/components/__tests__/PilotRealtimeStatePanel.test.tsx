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
    massKg: 10000,
  },
  control: {
    thrustN: 12000,
    bankDeg: 0,
    attackDeg: 3.5,
  },
  aero: {
    liftCoefficient: 0.412,
    dragCoefficient: 0.038,
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
    expect(screen.getByText("Drag Coefficient")).toBeTruthy();
    expect(screen.getByText("1234 m")).toBeTruthy();
    expect(screen.getByText("-14.2 deg")).toBeTruthy();
    expect(screen.getByText("3.50 deg")).toBeTruthy();
  });

  it("stays hidden when simulation is not flying", () => {
    render(<PilotRealtimeStatePanel snapshot={snapshot} visible={false} />);

    expect(
      screen.queryByRole("complementary", { name: "Realtime aircraft state" }),
    ).toBeNull();
  });
});
