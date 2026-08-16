import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

type Datum = {
  initialVMps: number | null;
  massKg: number | null;
  resultTimeS: number | null;
  status: "solved" | "offTarget" | "indeterminate" | "failed";
};

const { appState, comparisonData } = vi.hoisted(() => ({
  appState: { viewer: null, selectedFlightId: null as string | null, setSelectedFlightId: vi.fn() },
  comparisonData: {
    byFlightKey: new Map<string, Datum>(),
    comparisonActive: false,
    resultKind: "optimization" as "prediction" | "optimization" | null,
  },
}));

vi.mock("../../context/AppContext", () => ({ useApp: () => appState }));
vi.mock("../../hooks/useFlightComparisonData", () => ({
  useFlightComparisonData: () => comparisonData,
}));

import FlightTable from "../FlightTable";

// Entity ids are flight_keys (id_runway_icao24_landingTime); the callsign is display-only
// and lives on the summary. The optimizer map is keyed by the SAME flight_key (the
// comparison group), so the join is exact — never a callsign match.
const UPS = "UPS1276_05L_a1b2c3_20260614T101112Z";
const FDX = "FDX1738_05L_d4e5f6_20260614T112233Z";
const flightIds = [UPS, FDX];
const flightSummaries = {
  [UPS]: { durationS: 562, callsign: "UPS1276" },
  [FDX]: { durationS: 309, callsign: "FDX1738" },
};

describe("FlightTable", () => {
  beforeEach(() => {
    appState.selectedFlightId = null;
    comparisonData.byFlightKey = new Map<string, Datum>([
      [UPS, { initialVMps: 141.85, massKg: 66300, resultTimeS: 576, status: "solved" }],
      [FDX, { initialVMps: 148.7, massKg: 77800, resultTimeS: 309, status: "solved" }],
    ]);
    comparisonData.comparisonActive = false;
    comparisonData.resultKind = "optimization";
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
    comparisonData.comparisonActive = true;
    render(<FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />);
    fireEvent.click(screen.getByRole("button", { name: /Flights/ }));

    expect(screen.getByText("Opt")).toBeTruthy();
    expect(screen.getByText("9:36")).toBeTruthy(); // 576 s optimized final time
  });

  it("shows V + mass for a FAILED flight and marks its id red (no optimized time)", () => {
    comparisonData.comparisonActive = true;
    comparisonData.byFlightKey = new Map<string, Datum>([
      // failed: still has V + mass (from the scenario), but no optimized time.
      [UPS, { initialVMps: 134.9, massKg: 136000, resultTimeS: null, status: "failed" }],
      [FDX, { initialVMps: 148.7, massKg: 77800, resultTimeS: 309, status: "solved" }],
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

  it("marks an OFF-TARGET flight's id yellow but keeps its optimized time", () => {
    comparisonData.comparisonActive = true;
    comparisonData.byFlightKey = new Map<string, Datum>([
      // solved but missed the evaluation gates: yellow flag, optimized time still shown.
      [UPS, { initialVMps: 141.85, massKg: 66300, resultTimeS: 576, status: "offTarget" }],
      [FDX, { initialVMps: 148.7, massKg: 77800, resultTimeS: 309, status: "solved" }],
    ]);
    render(<FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />);
    fireEvent.click(screen.getByRole("button", { name: /Flights/ }));

    const offTargetId = screen.getByText("UPS1276");
    expect(offTargetId.className).toContain("flight-table-offtarget");
    expect(offTargetId.className).not.toContain("flight-table-failed");
    expect(offTargetId.getAttribute("title")).toContain("off target");
    expect(screen.getByText("9:36")).toBeTruthy(); // 576 s optimized time still shown
    expect(screen.getByText("FDX1738").className).not.toContain("flight-table-offtarget");
  });

  it("shows a dash where the optimizer has no data for a flight", () => {
    comparisonData.byFlightKey = new Map<string, Datum>(); // no comparison record for any flight
    render(<FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />);
    fireEvent.click(screen.getByRole("button", { name: /Flights/ }));
    // Time still comes from the track; V + mass are blank without an optimizer record.
    expect(screen.getByText("9:22")).toBeTruthy();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("keeps namesake flights' optimizer facts apart (join by flight_key, not callsign)", () => {
    // Two landings by the same callsign on different days: distinct entity ids/keys, same
    // display name. Keying the optimizer map by callsign served ONE flight's numbers for
    // both rows — the join must be the flight_key.
    const A = "ASA677_05R_a54aae_20260629T093123Z";
    const B = "ASA677_05R_a9e8ce_20260630T093925Z";
    comparisonData.comparisonActive = true;
    comparisonData.byFlightKey = new Map<string, Datum>([
      [A, { initialVMps: 130, massKg: 78000, resultTimeS: 400, status: "solved" }],
      [B, { initialVMps: 118, massKg: 64000, resultTimeS: 350, status: "failed" }],
    ]);
    render(
      <FlightTable
        flightIds={[A, B]}
        flightSummaries={{
          [A]: { durationS: 500, callsign: "ASA677" },
          [B]: { durationS: 480, callsign: "ASA677" },
        }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Flights/ }));

    const rows = screen.getAllByText("ASA677");
    expect(rows.length).toBe(2);   // two rows, same displayed callsign
    // Each row carries its OWN V/mass, and only B is flagged failed.
    expect(screen.getByText("130")).toBeTruthy();
    expect(screen.getByText("118")).toBeTruthy();
    expect(rows.filter((el) => el.className.includes("flight-table-failed")).length).toBe(1);
  });

  it("uses Pred plus baseline green/red for prediction pass and fail", () => {
    comparisonData.comparisonActive = true;
    comparisonData.resultKind = "prediction";
    comparisonData.byFlightKey = new Map<string, Datum>([
      [UPS, { initialVMps: 141.85, massKg: 66300, resultTimeS: 576, status: "solved" }],
      [FDX, { initialVMps: 148.7, massKg: 77800, resultTimeS: 309, status: "offTarget" }],
    ]);

    render(<FlightTable flightIds={flightIds} flightSummaries={flightSummaries} />);
    fireEvent.click(screen.getByRole("button", { name: /Flights/ }));

    expect(screen.getByText("Pred")).toBeTruthy();
    expect(screen.getByText("UPS1276").className).toContain("flight-table-pass");
    expect(screen.getByText("UPS1276").getAttribute("title")).toContain("prediction passed");
    expect(screen.getByText("FDX1738").className).toContain("flight-table-failed");
    expect(screen.getByText("FDX1738").getAttribute("title")).toContain("prediction failed");
  });
});
