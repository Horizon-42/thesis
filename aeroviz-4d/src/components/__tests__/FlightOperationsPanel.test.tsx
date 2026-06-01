import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("../PilotPanel", () => ({
  default: () => <section data-testid="pilot-panel">Pilot Mode</section>,
}));

vi.mock("../FlightTable", () => ({
  default: () => <div data-testid="flight-table">Flight Table</div>,
}));

import FlightOperationsPanel from "../FlightOperationsPanel";

describe("FlightOperationsPanel", () => {
  it("places the flight list controls after the pilot mode block", () => {
    render(<FlightOperationsPanel flightIds={["AAL123"]} />);

    const pilotPanel = screen.getByTestId("pilot-panel");
    const flightOpsHeading = screen.getByRole("heading", { name: "Flight Ops" });

    expect(
      pilotPanel.compareDocumentPosition(flightOpsHeading) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});
