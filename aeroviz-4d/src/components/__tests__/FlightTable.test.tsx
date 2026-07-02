import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

type Datum = {
  initialVMps: number | null;
  massKg: number | null;
  optimizedTimeS: number | null;
  failed: boolean;
};

const { appState, optimizerData } = vi.hoisted(() => ({
  appState: { viewer: null, selectedFlightId: null as string | null, setSelectedFlightId: vi.fn() },
  optimizerData: {
    byFlightId: new Map<string, Datum>(),
    comparisonActive: false,
  },
}));

vi.mock("../../context/AppContext", () => ({ useApp: () => appState }));
vi.mock("../../hooks/useFlightOptimizerData", () => ({
  useFlightOptimizerData: () => optimizerData,
}));

import FlightTable from "../FlightTable";

const flightIds = ["UPS1276", "FDX1738"];
const flightSummaries = {
  UPS1276: { durationS: 562 },
  FDX1738: { durationS: 309 },
};

describe("FlightTable", () => {
  beforeEach(() => {
    appState.selectedFlightId = null;
    optimizerData.byFlightId = new Map<string, Datum>([
      ["UPS1276", { initialVMps: 141.85, massKg: 66300, optimizedTimeS: 576, failed: false }],
      ["FDX1738", { initialVMps: 148.7, massKg: 77800, optimizedTimeS: 309, failed: false }],
    ]);
    optimizerData.comparisonActive = false;
    vi.clearAllMocks();
  });

  it("is collapsed by default — only the toggle (with the count) shows", () => {
    render(<FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />);
    expect(screen.getByRole("button", { name: /Flights \(2\)/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /Flights/ }).getAttribute("aria-expanded")).toBe("false");
    // No table rows while collapsed.
    expect(screen.queryByText("UPS1276")).toBeNull();
  });

  it("expands to show V, Mass and Time (no Opt column when comparison is off)", () => {
    render(<FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />);
    fireEvent.click(screen.getByRole("button", { name: /Flights/ }));

    expect(screen.getByText("UPS1276")).toBeTruthy();
    expect(screen.getByText("142")).toBeTruthy(); // V from the optimizer initial state (unit in header)
    expect(screen.getByText("66.3")).toBeTruthy(); // mass from the optimizer (unit in header)
    expect(screen.getByText("9:22")).toBeTruthy(); // 562 s duration
    expect(screen.queryByText("Opt")).toBeNull(); // comparison off → no optimized column
  });

  it("adds the optimized-time column when the comparison is active", () => {
    optimizerData.comparisonActive = true;
    render(<FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />);
    fireEvent.click(screen.getByRole("button", { name: /Flights/ }));

    expect(screen.getByText("Opt")).toBeTruthy();
    expect(screen.getByText("9:36")).toBeTruthy(); // 576 s optimized final time
  });

  it("shows V + mass for a FAILED flight and marks its id red (no optimized time)", () => {
    optimizerData.comparisonActive = true;
    optimizerData.byFlightId = new Map<string, Datum>([
      // failed: still has V + mass (from the scenario), but no optimized time.
      ["UPS1276", { initialVMps: 134.9, massKg: 136000, optimizedTimeS: null, failed: true }],
      ["FDX1738", { initialVMps: 148.7, massKg: 77800, optimizedTimeS: 309, failed: false }],
    ]);
    render(<FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />);
    fireEvent.click(screen.getByRole("button", { name: /Flights/ }));

    const failedId = screen.getByText("UPS1276");
    expect(failedId.className).toContain("flight-table-failed");
    expect(failedId.getAttribute("title")).toContain("failed");
    // V + mass still show for the failed flight.
    expect(screen.getByText("135")).toBeTruthy();
    expect(screen.getByText("136.0")).toBeTruthy();
    // The non-failed flight's id is not flagged.
    expect(screen.getByText("FDX1738").className).not.toContain("flight-table-failed");
  });

  it("shows a dash where the optimizer has no data for a flight", () => {
    optimizerData.byFlightId = new Map<string, Datum>(); // no optimizer record for any flight
    render(<FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />);
    fireEvent.click(screen.getByRole("button", { name: /Flights/ }));
    // Time still comes from the track; V + mass are blank without an optimizer record.
    expect(screen.getByText("9:22")).toBeTruthy();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });
});
